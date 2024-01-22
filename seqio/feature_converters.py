# Copyright 2024 The SeqIO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Feature converters for common architectures.

In short, feature converters carry out additional data processing to the
tf.data.Dataset out of the Task API. They convert the features of the input
dataset into more descriptive features (e.g., "decoder_target_tokens" instead of
"targets") as well as pad and/or pack them. The features of the input dataset
are referred to as "task_features" because they are the output of the Task API.
Those of the output dataset are referred to as "model_features" as they are the
features directly fed to the model implementation.

We provide feature converters for the following three architectures:

  - encoder-decoder
  - decoder-only
  - encoder-only

Each of these feature converters inherit the base class FeatureConverter and
override two methods `_convert_features` and `get_model_feature_lengths` to
define how task features are mapped to the model features including the length
relationships. Other model architectures can be supported by subclassing the
FeatureConverter class in a similar manner.


Definition: standard_features

Throughout this module, we refer to the following 10 fields as
standard_features. Depending on the model architecture, a subset of them will
be returned by the feature converter.

  - encoder_input_tokens
  - encoder_target_tokens
  - encoder_loss_weights
  - encoder_positions
  - encoder_segment_ids
  - decoder_input_tokens
  - decoder_target_tokens
  - decoder_loss_weights
  - decoder_positions
  - decoder_segment_ids

  *_segment_ids and *_positions fields are only relevant for packed dataset.

  *_segment_ids is a tf.Tensor of integer which is aligned with
  *_input_tokens. Positive integers represent the sequence membership in
  the packed examples. 0 represents padding. For example, encoder_segment_ids =
  [1, 1, 2, 2, 2, 0] means that the first two positions belong to the first
  sequence, the next three to the second sequence and the last position is a
  padding.

  *_positions is a tf.Tensor of integer representing the position index in the
  original sequence before packing. For example, consider
  encoder_positions = [0, 1, 0, 1, 2, 0]. The first two tokens were the 0th and
  1st tokens of the first sequence and next three tokens are the 0th, 1st and
  2nd tokens of the second sequence before packing.

  *_loss_weights is used to indicate which positions should be used for the loss
  calculation.


Underlying assumptions

The feature converters implemented in this module assume the following about the
input dataset.

  - If EOS tokens are required, they are already appended in the input dataset.
  - The input dataset is not batched.
"""

import abc
import dataclasses
import functools
from typing import Mapping, Optional, Sequence
from absl import logging
from seqio import utils
import tensorflow.compat.v2 as tf

# TODO(hwchung): remove this.
# pointer for backward compatilbility.
autoregressive_inputs = utils.make_autoregressive_inputs


def _check_lengths(
    ds: tf.data.Dataset,
    expected_lengths: Mapping[str, int],
    sequence_axis_mapping: Mapping[str, int],
    strict: bool,
    error_label: str,
) -> tf.data.Dataset:
  """Check the length of each feature in `ds` against `expected_lengths`.

  There are two checking criteria controlled by `strict` arg.

  If strict = True,
  for each feature in ds, check len(feature) == expected_lengths[feature].

  If strict = False,
  for each feature in ds, check len(feature) <= expected_lengths[feature].

  Features of the input dataset may have [None] shape. The assertion is run at
  the graph execution time when the length is determined.

  Args:
    ds: a tf.data.Dataset to be checked.
    expected_lengths: a mapping from a feature name to an expected length.
    sequence_axis_mapping: a mapping from feature name to its sequence
      dimension.
    strict: if true, the length of each feature should exactly match the
      expected length whereas false condition allows the length to be less than
      or equal to the expected length.
    error_label: a label used to indicate the validation stage

  Returns:
    ds: the same dataset as but with the assertion ops attached.
  """

  def _check_length(feat, v):
    if feat not in expected_lengths:
      return v

    if strict:
      error_message = (
          f"Feature '{feat}' has length not equal to the expected length of "
          f"{expected_lengths[feat]} during {error_label} validation"
      )
      assertion_op = functools.partial(
          tf.debugging.assert_equal, message=error_message
      )
    else:
      error_message = (
          f"Feature '{feat}' has length not less than or equal to the expected "
          f"length of {expected_lengths[feat]} during {error_label} validation"
      )
      assertion_op = functools.partial(
          tf.debugging.assert_less_equal, message=error_message
      )

    expected_length = tf.constant(expected_lengths[feat], dtype=tf.int64)
    sequence_axis = sequence_axis_mapping[feat]
    if isinstance(v, tf.RaggedTensor):
      if strict:
        # For strict mode we require all sequence dim lengths be equal.
        # We multiply by [1] to (potentially) broadcast to 1d since non-ragged
        # dimensions will return scalar rather than vector row_lengths.
        lengths = v.row_lengths(sequence_axis) * tf.constant(
            [1], dtype=tf.int64
        )
        tf.debugging.assert_equal(
            len(tf.unique(lengths)[0]),
            1,
            "Strict length check requires all RaggedTensor dimensions to be"
            f" equal, got {v.row_lengths()}",
        )
      actual_length = v.bounding_shape(axis=sequence_axis, out_type=tf.int64)
    else:
      actual_length = tf.shape(v, out_type=tf.int64)[sequence_axis]
    assertion_op(actual_length, expected_length)
    return v

  ds = ds.map(
      lambda ex: {k: _check_length(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )

  return ds


def non_padding_position(
    tensor: tf.Tensor, dtype: tf.dtypes.DType = tf.int32, pad_id: int = 0
) -> tf.Tensor:
  """Return a tensor with 1 on non-padding and 0 on padding positions."""
  return tf.cast(tf.not_equal(tensor, pad_id), dtype=dtype)


def _check_exact_match(
    expected_features: Sequence[str],
    actual_features: Sequence[str],
    expected_feature_source: str,
    actual_feature_source: str,
) -> None:
  """Check whether expected and actual features match one-to-one."""
  expected_features = set(expected_features)
  actual_features = set(actual_features)

  if expected_features != actual_features:
    if actual_features - expected_features:
      extra_features = actual_features - expected_features
      raise ValueError(
          f"The {actual_feature_source} contains extra features not specified "
          f"in the {expected_feature_source}: {extra_features}"
      )
    else:
      missing_features = expected_features - actual_features
      raise ValueError(
          f"The {actual_feature_source} is missing features specified "
          f"in the {expected_feature_source}: {missing_features}"
      )


class FeatureConverter(abc.ABC):
  """Abstract base class for feature converters.

  Subclasses of FeatureConverter are used to convert the tf.data.Dataset
  instance from the Task API to features that are passed to the
  model implementation. Note that Task API has an attribute "output_features",
  which is referred to as "task features" in the context of FeatureConverter.

  Typically the task features contain keys: "inputs" and "targets". The model
  features are constructed based on what is consumed by the model architecture.
  For custom model architectures that require additional model features, one
  needs to subclass FeatureConverter.

  This conversion is fully specified by

    1. defining the mapping of the features in `_convert_features` method and
    2. defining the relationship between sequence lengths of input and output
       features in `get_model_feature_lengths` which is a function of
       task_feature_lengths.

  Therefore, a subclass of FeatureConverter should override `_convert_features`
  and `get_model_feature_lengths` methods.

  The actual feature conversion is done in the `__call__` method, which
  wraps around the `_convert_features` method in order to provide useful checks
  and ensure compatibilities. See `_validate_dataset` and `__call__` methods
  for more details.

  Other notes:

    If pack = True, each feature in the task features should be packable,
    i.e., 1-dimensional.

    Subclasses must override TASK_FEATURES and MODEL_FEATURES. If packing is
    used, they must override PACKING_FEATURE_DTYPES as well. These are the
    packing-specific features such as "*_segment_ids".

    Pass-through features are incompatible with packing and should not be used
    in that case. FeatureConverter only implements the scaffolding, but the real
    pass-through should be implemented in each sub-class inside
    `_convert_features` and `get_model_feature_lengths`.

  Attributes:
    pack: whether to pack the dataset.
    use_custom_packing_ops: whether to use custom ops for packing.
    apply_length_check: if True, it checks whether output feature lengths are
      less than the lengths given by `sequence_length`.
    bos_id: bos id for decoder inputs.
    passthrough_features: a mapping that extends the `TASK_FEATURES` and
      `MODEL_FEATURES` including features that will pass through without any
      processing.
  """

  @dataclasses.dataclass(frozen=True)
  class FeatureSpec:
    """Rank and dtype specifications for features."""

    dtype: tf.dtypes.DType
    rank: int = 1
    sequence_dim: int = 0

  TASK_FEATURES: Mapping[str, "FeatureConverter.FeatureSpec"]
  MODEL_FEATURES: Mapping[str, "FeatureConverter.FeatureSpec"]
  PACKING_FEATURE_DTYPES: Mapping[str, tf.dtypes.DType]

  def __init__(
      self,
      pack: bool = True,
      use_custom_packing_ops: bool = False,
      apply_length_check: bool = True,
      bos_id: int = 0,
      passthrough_features: Optional[
          Mapping[str, "FeatureConverter.FeatureSpec"]
      ] = None,
  ):
    self._pack = pack
    self._use_custom_packing_ops = use_custom_packing_ops
    self._apply_length_check = apply_length_check
    self._bos_id = bos_id

    if self.TASK_FEATURES is None:
      raise ValueError("TASK_FEATURES must be defined in the subclass.")

    if self.MODEL_FEATURES is None:
      raise ValueError("MODEL_FEATURES must be defined in the subclass.")

    if self.pack and self.PACKING_FEATURE_DTYPES is None:
      raise ValueError(
          "PACKING_FEATURE_DTYPES must be defined in the subclass if pack=True."
      )

    if passthrough_features is not None:
      if self.pack:
        raise ValueError("Packing is incompatible with pass-through features.")
      self._passthrough_features = passthrough_features
    else:
      self._passthrough_features = {}

  def _validate_dataset(
      self,
      ds: tf.data.Dataset,
      expected_features: Mapping[str, "FeatureConverter.FeatureSpec"],
      expected_lengths: Mapping[str, int],
      strict: bool,
      error_label: str,
  ) -> tf.data.Dataset:
    """Validate properties of the dataset, raising Exceptions if needed.

    This method is used to validate whether the input dataset is compatible
    with the desired specifications. In particular, the following aspects are
    checked.

    Each feature in `expected_features`
      - is also in `ds`
      - is also in expected_lengths
      - is compatible with the expected lengths

    The compatibility of the length is controlled by strict. If true, the length
    of each feature should exactly match the expected length whereas false
    condition allows the length to be less than or equal to the expected length.

    Args:
      ds: a tf.data.Dataset to be validated
      expected_features: expected features
      expected_lengths: a mapping from feature to its length
      strict: whether the lengths should be strictly equal or a length less than
        or equal to expected length is allowed.
      error_label: a label used to indicate the validation stage

    Returns:
      ds: the same dataset as but with the assertion ops attached.
    """
    element_spec = ds.element_spec
    for feat in expected_features:
      if feat not in element_spec:
        raise ValueError(
            "Dataset is missing an expected feature during "
            f"{error_label} validation: '{feat}'"
        )

      if expected_features[feat].dtype != element_spec[feat].dtype:
        actual_dtype = element_spec[feat].dtype.name
        raise ValueError(
            f"Dataset has incorrect type for feature '{feat}' during "
            f"{error_label} validation: Got {actual_dtype}, expected "
            f"{expected_features[feat].dtype.name}"
        )

      if expected_features[feat].rank != len(element_spec[feat].shape):
        actual_rank = len(element_spec[feat].shape)
        raise ValueError(
            f"Dataset has incorrect rank for feature '{feat}' during "
            f"{error_label} validation: "
            f"Got {actual_rank}, expected {expected_features[feat].rank}"
        )

    sequence_axis_mapping = {
        feat: expected_features[feat].sequence_dim for feat in expected_features
    }
    # Remove rank-0 features from expected lengths to bypass the length check.
    expected_lengths = {
        k: v
        for k, v in expected_lengths.items()
        if k in expected_features and expected_features[k].rank != 0
    }
    if self._apply_length_check:
      ds = _check_lengths(
          ds, expected_lengths, sequence_axis_mapping, strict, error_label
      )
    else:
      logging.info(
          "Length validation is skipped since `apply_length_check=False`"
      )
    return ds

  def __call__(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    r"""Convert the features of `ds` into output features.

    This method should not be overridden by subclasses.

    There are two conversion steps and five validation steps.

    Conversion 1: task features are converted to model features in
                  `_convert_features

    Conversion 2: task_feature_lengths are converted to model_feature_lengths in
                  `get_model_feature_lengths`

    Validation 1: verifies that the user input `task_feature_lengths` only
                  contains the required features.

    Validation 2: verifies whether the input dataset has same or more features,
                  same dtype, and length that is less than or equal compared to
                  input_ds.

    Validation 3: partially verifies the behavior of overridden
                  `get_model_feature_lengths`.

    Validation 4: check whether the output dataset has expected features (extra
                  features are allowed), dtype, rank and lengths (exact match).

    Validation 5: check one-to-one match between the output dataset and
                  `expected_dtypes`. Extra features are not allowed.

    The following diagram describes the validation and conversion processes. We
    treat features in the TASK_FEATURES and MODEL_FEATURES specified as class
    variables as the ground-truth. For validations 3, 4 and 5, we define
    `expected_dtypes`.

    There are 5 validation steps. features (<=) means that features of the
    variable on the left is a subset of those of the variable on the right. For
    example, validation 2 guarantees that TASK_FEATURES has features that are a
    subset of the features of input_ds. Validation 4 has length (==), which
    means that it ensures that each feature in MODEL_FEATURES has the same
    length as the corresponding feature in output_ds.

    Overall, these 5 validations ensures that the output_ds has the expected
    features with exact length, dtype and rank. Again, these validations assume
    that TASK_FEATURES and MODEL_FEATURES are correct.


                        Validation 1                     Validation 2
    task_feature_lengths <-----------> TASK_FEATURES <----------------> input_ds
    |                   features (==)                    features (<=)        |
    |                                                    dtype (==)           |
    |                                                    length (<=)          |
    |                                                    rank (==1)           |
    |                                                                         |
    |   Conversion 2                                           Conversion 1   |
    | get_model_feature_lengths                             _convert_features |
    |                                                                         |
    |                                              Validation 5               |
    |                                           <-------------------->        |
    |                                                 features (==)           |
    |                                                                         |
    \/                    Validation 3                    Validation 4        \/
    model_feature_lengths <-------> expected_dtypes <----------------> output_ds
                        features (==)                     features (<=)
                                                          dtype (==)
                                                          length (==)
                                                          rank (==1)

    Args:
      ds: a tf.data.Dataset to be validated
      task_feature_lengths: a mapping from a task feature to its length

    Returns:
      ds: the converted dataset.
    """
    # Validation 1
    task_features_with_passthrough = dict(self.TASK_FEATURES)
    task_features_with_passthrough.update(self._passthrough_features)
    _check_exact_match(
        expected_features=list(task_features_with_passthrough),
        actual_features=list(task_feature_lengths),
        expected_feature_source="TASK_FEATURES",
        actual_feature_source="task_feature_lengths",
    )

    # Validation 2
    ds = self._validate_dataset(
        ds,
        expected_features=task_features_with_passthrough,
        expected_lengths=task_feature_lengths,
        # Before pack/pad, check feature (of ds) length <= task feature length
        strict=False,
        error_label="input_validation",
    )

    # Conversion 1: implemented by subclass
    ds = self._convert_features(ds, task_feature_lengths)

    expected_features = dict(self.MODEL_FEATURES)
    expected_features.update(self._passthrough_features)
    if self.pack:
      for k, v in expected_features.items():
        # Packing requires rank 1.
        if v.rank != 1 and not self._use_custom_packing_ops:
          raise ValueError(
              "When packing is enabled, expected ranks must be 1 or "
              f"use_custom_packing_ops must be set. Got expected rank {v.rank} "
              f"for feature {k}."
          )
      for k, v in self.PACKING_FEATURE_DTYPES.items():
        expected_features[k] = FeatureConverter.FeatureSpec(rank=1, dtype=v)

    # Conversion 2: implemented by subclasses
    model_feature_lengths = self.get_model_feature_lengths(task_feature_lengths)

    # Validation 3
    _check_exact_match(
        expected_features=list(expected_features),
        actual_features=list(model_feature_lengths),
        expected_feature_source="model_feature_names",
        actual_feature_source="model_feature_lengths",
    )

    # Validation 4
    ds = self._validate_dataset(
        ds,
        expected_features=expected_features,
        expected_lengths=model_feature_lengths,
        # After pack/pad, check feature (of ds) length == model feature length
        strict=True,
        error_label="output_validation",
    )

    # Validation 5
    _check_exact_match(
        expected_features=list(expected_features),
        actual_features=list(ds.element_spec),
        expected_feature_source="model_feature_names",
        actual_feature_source="output_dataset",
    )
    return ds

  def _pack_or_pad(
      self, ds: tf.data.Dataset, packed_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Trim/pad to packed_lengths and optionally pack the input dataset."""
    if self.pack:
      ds = utils.trim_and_pack_dataset(
          ds, packed_lengths, self._use_custom_packing_ops
      )
    else:
      ds = utils.trim_and_pad_dataset(ds, packed_lengths)
    return ds

  @abc.abstractmethod
  def _convert_features(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Main feature conversion method to be overridden.."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    raise NotImplementedError

  @property
  def pack(self) -> bool:
    return self._pack

  @property
  def bos_id(self) -> int:
    return self._bos_id


class EncDecFeatureConverter(FeatureConverter):
  """Feature converter for an encoder-decoder architecture.

  The input dataset has "inputs" and "targets" field. These will be converted
  to a subset of standard features.

  To use packing, pass pack = True argument to the FeatureConverter's
  constructor. When packing is done, two additional fields are added for each of
  "inputs" and "targets" fields.

  Example for a packed dataset:

  The input dataset has two examples each with "inputs" and "targets".

  ds = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]

  task_feature_lengths = {"inputs": 10, "targets": 7}

  First, the `inputs` are packed together, padded to length 10 and assigned to
  "encoder_input_tokens" field. The `targets` are processed similarly.

  The "*_segment_id" fields are generated from the packing operation. For the
  explanation of these fields, see the module docstring.

  The "decoder_loss_weights" is a binary mask indicating where non-padding
  positions are, i.e., value of 1 indicates non-padding and 0 for padding. This
  class assumes that the loss is taken only on the decoder side.

  converted_ds = [{
       "encoder_input_tokens": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
          "encoder_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
      "decoder_target_tokens": [3, 9, 1, 4, 1, 0, 0],
       "decoder_input_tokens": [0, 3, 9, 0, 4, 0, 0],
       "decoder_loss_weights": [1, 1, 1, 1, 1, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0, 0],
          "decoder_positions": [0, 1, 2, 0, 1, 0, 0],
  }]

  Note that two examples are packed together into one example.
  """

  TASK_FEATURES = {
      "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "encoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "encoder_segment_ids": tf.int32,
      "decoder_segment_ids": tf.int32,
      "encoder_positions": tf.int32,
      "decoder_positions": tf.int32,
  }

  def _convert_example(
      self, features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    """Convert a seq2seq example into an example with model features."""
    # targets_segment_id is present only for a packed dataset.
    decoder_input_tokens = utils.make_autoregressive_inputs(
        features["targets"],
        sequence_id=features.get("targets_segment_ids", None),
        bos_id=self.bos_id,
    )

    d = {
        "encoder_input_tokens": features["inputs"],
        "decoder_target_tokens": features["targets"],
        "decoder_input_tokens": decoder_input_tokens,
        # Loss is computed for all but the padding positions.
        "decoder_loss_weights": non_padding_position(features["targets"]),
    }
    d.update({k: features[k] for k in self._passthrough_features})

    if self.pack:
      d["encoder_segment_ids"] = features["inputs_segment_ids"]
      d["decoder_segment_ids"] = features["targets_segment_ids"]
      d["encoder_positions"] = features["inputs_positions"]
      d["decoder_positions"] = features["targets_positions"]

    return d

  def _convert_features(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Convert the dataset to be fed to the encoder-decoder model.

    The conversion process involves two steps

    1. Each feature in the `task_feature_lengths` is trimmed/padded and
       optionally packed depending on the value of self.pack.
    2. "inputs" fields are mapped to the encoder input and "targets" are mapped
       to decoder input (after being shifted) and target.

    All the keys in the `task_feature_lengths` should be present in the input
    dataset, which may contain some extra features that are not in the
    `task_feature_lengths`. They will not be included in the output dataset.
    One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
    fields.

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from feature to its length.

    Returns:
      ds: the converted dataset.
    """
    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(
        self._convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    encoder_length = task_feature_lengths["inputs"]
    decoder_length = task_feature_lengths["targets"]

    model_feature_lengths = {
        "encoder_input_tokens": encoder_length,
        "decoder_target_tokens": decoder_length,
        "decoder_input_tokens": decoder_length,
        "decoder_loss_weights": decoder_length,
    }
    for k in self._passthrough_features:
      model_feature_lengths[k] = task_feature_lengths[k]

    if self.pack:
      model_feature_lengths["encoder_segment_ids"] = encoder_length
      model_feature_lengths["decoder_segment_ids"] = decoder_length
      model_feature_lengths["encoder_positions"] = encoder_length
      model_feature_lengths["decoder_positions"] = decoder_length

    return model_feature_lengths


class PrePackedEncDecFeatureConverter(EncDecFeatureConverter):
  """Feature converter for encoder-decoder with pre-packed examples.

  The input dataset has "inputs", "targets", "inputs_segment_ids",
  "inputs_positions", "targets_segment_ids", and "targets_positions". These will
  be converted to a subset of standard features.

  Since this feature converter assumes the data is already pre-packed, setting
  'pack=True' is not allowed.
  """

  TASK_FEATURES = {
      "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "inputs_positions": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "inputs_segment_ids": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets_positions": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets_segment_ids": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "encoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "encoder_positions": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "encoder_segment_ids": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_positions": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_segment_ids": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }

  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
    if self.pack:
      raise ValueError(
          "'pack=True' is not allowed in PrePackedEncDecFeatureConverter."
      )

  def _convert_features(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Convert the dataset to be fed to the encoder-decoder model.

    See "PrePackedEncDecFeatureConverter._convert.features".

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from feature to its length.

    Returns:
      ds: the converted dataset.
    """

    def convert_example(
        features: Mapping[str, tf.Tensor]
    ) -> Mapping[str, tf.Tensor]:
      # targets_segment_id is present only for a packed dataset.
      decoder_input_tokens = utils.make_autoregressive_inputs(
          features["targets"],
          sequence_id=features.get("targets_segment_ids", None),
          bos_id=self.bos_id,
      )

      d = {
          "encoder_input_tokens": features["inputs"],
          "decoder_target_tokens": features["targets"],
          "decoder_input_tokens": decoder_input_tokens,
          # Loss is computed for all but the padding positions.
          "decoder_loss_weights": non_padding_position(features["targets"]),
          "encoder_segment_ids": features["inputs_segment_ids"],
          "decoder_segment_ids": features["targets_segment_ids"],
          "encoder_positions": features["inputs_positions"],
          "decoder_positions": features["targets_positions"],
      }
      return d

    ds = utils.trim_and_pad_dataset(ds, task_feature_lengths)

    return ds.map(
        convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    encoder_length = task_feature_lengths["inputs"]
    decoder_length = task_feature_lengths["targets"]

    model_feature_lengths = {
        "encoder_input_tokens": encoder_length,
        "decoder_target_tokens": decoder_length,
        "decoder_input_tokens": decoder_length,
        "decoder_loss_weights": decoder_length,
        "encoder_segment_ids": encoder_length,
        "decoder_segment_ids": decoder_length,
        "encoder_positions": encoder_length,
        "decoder_positions": decoder_length,
    }
    return model_feature_lengths


class LMFeatureConverter(FeatureConverter):
  """Feature converter for a language model (decoder-only) architecture.

  The input dataset must have "targets" field only.

  One common usecase is to pre-train a decoder-only model with the standard
  language modeling objective (i.e., predict the next token given the previous
  ones) on a unlabeled text corpus which only has "targets". Then the
  pre-trained model can be fine-tuned on a supervised task, e.g., machine
  translation by concatenating "inputs" and "targets". For this use case,
  pre-train with LMFeatureConverter and fine-tune with PrefixLMFeatureConverter.

  Example: a packed dataset.

    ds = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]

    input_lengths = {"targets": 6}

    converted_ds = {
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0],
         "decoder_input_tokens": [0, 3, 9, 0, 4, 0],
         "decoder_loss_weights": [1, 1, 1, 1, 1, 0],
            "decoder_positions": [0, 1, 2, 0, 1, 0],
          "decoder_segment_ids": [1, 1, 1, 2, 2, 0]
    }
  Note that two examples are packed together into one example.
  """

  TASK_FEATURES = {"targets": FeatureConverter.FeatureSpec(dtype=tf.int32)}
  MODEL_FEATURES = {
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "decoder_segment_ids": tf.int32,
      "decoder_positions": tf.int32,
  }

  def _convert_example(
      self, features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    """Convert an LM example into an example with model features."""
    # targets_segment_id is present only for a packed dataset.
    decoder_input_tokens = utils.make_autoregressive_inputs(
        features["targets"],
        sequence_id=features.get("targets_segment_ids", None),
        bos_id=self.bos_id,
    )

    d = {
        "decoder_target_tokens": features["targets"],
        "decoder_input_tokens": decoder_input_tokens,
        "decoder_loss_weights": non_padding_position(features["targets"]),
    }

    if self.pack:
      d["decoder_segment_ids"] = features["targets_segment_ids"]
      d["decoder_positions"] = features["targets_positions"]

    return d

  def _convert_features(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Convert the dataset to be fed to a language model."""
    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(
        self._convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    decoder_length = task_feature_lengths["targets"]
    model_feature_lengths = {
        "decoder_target_tokens": decoder_length,
        "decoder_input_tokens": decoder_length,
        "decoder_loss_weights": decoder_length,
    }
    if self.pack:
      model_feature_lengths["decoder_segment_ids"] = decoder_length
      model_feature_lengths["decoder_positions"] = decoder_length

    return model_feature_lengths


class PrefixLMFeatureConverter(LMFeatureConverter):
  """Feature converter for a prefix language model architecture.

  The input dataset must have both "inputs" and "targets" fields. For language
  modeling objective with "targets" only dataset, use LMFeatureConverter.

  A decoder is a network which autoregressively produces an output sequence. It
  can be used for an input dataset which has a notion of "inputs" as well as
  "targets", (e.g., machine translation) by concatenating them to form the new
  targets. See Raffel et al. (2020), https://arxiv.org/abs/1910.10683, Section
  3.2.1 for a more detailed take on this topic.

  In the Prefix LM architecture discussed in Raffel et al. (2020), the tokens
  from the "inputs" portion are applied a fully visible self attention whereas
  those from "targets" are applied the causal self attention. This makes the
  contextual representation of the tokens from "inputs" bidirectional.

  In order to provide this information, this class provides an additional
  feature "decoder_causal_attention" on top of the model features returned by
  LMFeatureConverter. "decoder_causal_attention" is a binary mask where a value
  of 1 represents that the corresponding input token to the decoder belongs to
  the "inputs" before concatenation. Note that this attention mask is optional.
  For a model that does not require this feature, e.g., a fully causal masking
  on the concatenated sequence, the attention mask can be simply ignored.

  Note that "decoder_causal_attention" includes one additional position to the
  right. This is the position where the final token of the "inputs" (often an
  EOS) is read and the first "targets" token is predicted. This follows
  mesh_tensorflow/transformer/transformer.py

  Since "inputs" and "targets" are concatenated to form the new targets for the
  decoder, we might want to compute the loss only on the tokens that belong to
  "targets" before concatenation. This behavior is controlled by
  "loss_on_targets_only" attribute, which is passed to the constructor. By
  default, it is set to True. The resulting "decoder_loss_weights" therefore
  zeros out "inputs" portion as well as the padding tokens while having 1's on
  the targets token.

  Example 1: a packed dataset
  ```
  ds = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]

  task_feature_lengths = {"inputs": 7, "targets": 8}

  converted_ds = {
      "decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
       "decoder_input_tokens": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
       "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
          "decoder_positions": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
        "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
   "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  }
  ```

  Example 2: unpacked dataset with extra long "inputs" `task_feature_lengths`
  ```
  ds = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]

  task_feature_lengths = {"inputs": 10, "targets": 4}

  converted_ds = {
         "decoder_target_tokens": [9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0, 0],
          "decoder_input_tokens": [0, 9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0],
          "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
      "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  }

  Note that if the inputs length specified in `task_feature_lengths` is longer
  than the actual example length, the padding tokens are added after
  concatenation.
  ```

  Attributes:
    loss_on_targets_only: whether to compute loss on tokens which belonged to
      "targets" before concatenation.
  """

  TASK_FEATURES = {
      "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_causal_attention": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "decoder_segment_ids": tf.int32,
      "decoder_positions": tf.int32,
  }

  def __init__(self, loss_on_targets_only: bool = True, **kwargs) -> None:
    self._loss_on_targets_only = loss_on_targets_only
    super().__init__(**kwargs)

  def _convert_example(
      self, features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    """Convert a Prefix LM example into an example with model features.

    Example:
    ```

    Suppose the original dataset is

    ds = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]

    Then the input features to this method (after padding) are

    features = {
                   "targets" = [9, 4, 6, 1, 3, 9, 1, 0, 0]
              "inputs_width" = [4, 4, 4, 4, 4, 4, 4, 0, 0]
      "inputs_width_add_pos" = [5, 5, 5, 5, 5, 5, 5, 0, 0]
    }

    where "inputs_width" is length of "inputs" tiled across length dimension and
    "inputs_width_add_pos" is the same except that it has one additional
    position.

    First the parent class's _convert_example method is used to obtain the
    standard LM features. Then we compute "decoder_causal_attention". For an
    unpacked dataset, we need to define the "positions" feature. Then,
    `positions < inputs_width_add_pos` gives the decoder_causal_attention.

        "inputs_width_add_pos" = [5, 5, 5, 5, 5, 5, 5, 0, 0]
                   "positions" = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                           <     ---------------------------
    "decoder_causal_attention" = [1, 1, 1, 1, 1, 0, 0, 0, 0]

    Then, we compute the loss weights, which requires isolating the "targets"
    position. Here we use "inputs_width" feature to filter out the "inputs"
    portion. `padding_mask` has 1's on inputs and targets and 0's on padding.
    Taking XOR filters out the targets portion.

          "inputs_width" = [4, 4, 4, 4, 4, 4, 4, 0, 0]
             "positions" = [0, 1, 2, 3, 4, 5, 6, 0, 0]
                     <     ---------------------------
                  inputs = [1, 1, 1, 1, 0, 0, 0, 0, 0]
            padding_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0]
                    xor    ---------------------------
    decoder_loss_weights = [0, 0, 0, 0, 1, 1, 1, 0, 0]

    Note that decoder_loss_weights is computed by the LMFeatureConverter.
    ```

    Args:
      features: an input tf.data.Dataset to be converted.

    Returns:
      d: the converted features.
    """
    # First use the standard LM conversion.
    lm_features = super()._convert_example(features)

    # Initialize the return dictionary with the lm features.
    d = dict(lm_features)

    if self.pack:
      positions = features["targets_positions"]
    # Without packing, targets_positions field does not exist.
    else:
      positions = tf.range(tf.size(features["targets"]))

    inputs_width = features["inputs_width_add_pos"]
    # Binary mask where 1 represents a position in a non-causal attention region
    d["decoder_causal_attention"] = tf.cast(
        positions < inputs_width, dtype=features["targets"].dtype
    )

    # When computing the loss weights with self.loss_on_targets_only = True, we
    # use features["inputs_width"], which encodes the number of "inputs" tokens.
    if self.loss_on_targets_only:
      # 1's on inputs and 0's on targets and padding.
      inputs = positions < features["inputs_width"]

      # 1's on inputs and targets and 0's on padding.
      padding_mask = tf.cast(d["decoder_loss_weights"], dtype=tf.bool)

      # XOR picks targets only. See docstring for an example.
      d["decoder_loss_weights"] = tf.cast(
          tf.math.logical_xor(inputs, padding_mask),
          dtype=features["targets"].dtype,
      )

    d.update({k: features[k] for k in self._passthrough_features})
    return d

  def _concat_and_add_masks(
      self, features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    """Creates concatenated inputs and targets fields and adds masks."""
    inputs = features["inputs"]
    targets = features["targets"]
    # If the targets are empty, we add one padding target.
    targets = tf.cond(
        tf.size(targets) > 0,
        lambda: targets,
        lambda: tf.zeros(1, dtype="int32"),
    )

    # Width of the "inputs" portion in the concatenated sequence.
    width = tf.size(inputs)
    inputs_width = tf.fill([tf.size(inputs) + tf.size(targets)], width)

    # Width with an extra position to the right in the inputs mask. See
    # docstring for details.
    inputs_width_add_pos = tf.fill(
        [tf.size(inputs) + tf.size(targets)], width + 1
    )

    d = {
        "targets": tf.concat([inputs, targets], axis=-1),
        "inputs_width": inputs_width,
        "inputs_width_add_pos": inputs_width_add_pos,
    }
    d.update({k: features[k] for k in self._passthrough_features})
    return d

  def _concat_task_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    concat_length = sum(
        v
        for k, v in task_feature_lengths.items()
        if k not in self._passthrough_features
    )
    task_lengths = {
        "targets": concat_length,
        "inputs_width": concat_length,
        "inputs_width_add_pos": concat_length,
    }
    for k in self._passthrough_features:
      task_lengths[k] = task_feature_lengths[k]
    return task_lengths

  def _convert_features(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Convert the input dataset to an output dataset to be fed to the model.

    The "inputs" and "targets" are concatenated to form the new targets. In
    addition, the binary mask to distinguish "inputs" and "targets" token are
    concatenated as well.

    We define inputs_width to be a width (or a number of tokens) of "inputs" in
    the concatenated sequence. This method computes the width corresponding with
    and without additional position. Both of these are necessary
    `_convert_example`.

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from task feature name to its length.

    Returns:
      ds: the converted dataset.
    """

    def swap_vals(t, old_val, new_val):
      return tf.where(tf.equal(t, old_val), tf.fill([tf.size(t)], new_val), t)

    def swap_inputs_width(ex, old_val, new_val):
      ex["inputs_width"] = swap_vals(ex["inputs_width"], old_val, new_val)
      return ex

    replace_0s = functools.partial(swap_inputs_width, old_val=0, new_val=-1)
    restore_0s = functools.partial(swap_inputs_width, old_val=-1, new_val=-0)

    ds = ds.map(
        self._concat_and_add_masks,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    concat_task_feature_lengths = self._concat_task_feature_lengths(
        task_feature_lengths
    )

    ds = ds.map(replace_0s, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = self._pack_or_pad(ds, concat_task_feature_lengths)
    ds = ds.map(restore_0s, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds.map(
        self._convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    decoder_length = sum(
        v
        for k, v in task_feature_lengths.items()
        if k not in self._passthrough_features
    )
    concat_length = {"targets": decoder_length}
    lm_model_feature_lengths = super().get_model_feature_lengths(concat_length)
    model_feature_lengths = dict(lm_model_feature_lengths)
    model_feature_lengths["decoder_causal_attention"] = decoder_length
    for k in self._passthrough_features:
      model_feature_lengths[k] = task_feature_lengths[k]
    return model_feature_lengths

  @property
  def loss_on_targets_only(self) -> bool:
    return self._loss_on_targets_only


class PrefixSuffixLMFeatureConverter(PrefixLMFeatureConverter):
  """Feature converter for a input + target + suffix language model.

  When "suffixes" is an empty list, it is identical as PrefixLMFeatureConverter.
  When "suffixes" is not empty, it merges "targets" and "suffixes" but
  computes the loss only over tokens from "suffixes".

  Example: a packed dataset
  ```
  ds = [{"inputs": [9, 4, 6], "targets": [3, 9], "suffixes": [2, 1]},
        {"inputs": [3, 2,], "targets": [4,], "suffixes": []}]

  task_feature_lengths = {"inputs": 7, "targets": 8}

  converted_ds = {
      "decoder_target_tokens": [9, 4, 6, 3, 9, 2, 1, 3, 2, 4, 0, 0, 0, 0, 0],
      "decoder_input_tokens": [0, 9, 4, 6, 3, 9, 2, 0, 3, 2, 0, 0, 0, 0, 0],
      "decoder_loss_weights": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
      "target_suffix_weights": [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
      "decoder_positions": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 0, 0, 0, 0, 0],
      "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0],
      "decoder_causal_attention": [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
  }
  ```
  """

  TASK_FEATURES = {
      "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "suffixes": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_causal_attention": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "target_suffix_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "decoder_segment_ids": tf.int32,
      "decoder_positions": tf.int32,
  }

  def __init__(self, loss_on_targets_only: bool = True, **kwargs) -> None:
    self._loss_on_targets_only = loss_on_targets_only
    super().__init__(**kwargs)

  def _convert_example(
      self, features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    """Convert a example into an example with model features."""
    # First use the standard LM conversion.
    lm_features = super()._convert_example(features)
    d = dict(lm_features)
    target_suffix_weights = tf.cast(
        tf.equal(features["target_suffix_weights"], 2),
        dtype=d["decoder_loss_weights"].dtype,
    )
    d["target_suffix_weights"] = target_suffix_weights
    return d

  def _concat_and_add_masks(
      self, features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    """Creates concatenated inputs and targets fields and adds masks."""
    inputs = features["inputs"]
    targets = features["targets"]
    suffixes = features["suffixes"]
    target_suffixes = tf.concat([targets, suffixes], axis=0)
    # If the targets are empty, we add one padding target.
    target_suffixes = tf.cond(
        tf.size(target_suffixes) > 0,
        lambda: target_suffixes,
        lambda: tf.zeros(1, dtype="int32"),
    )

    # Width of the "inputs" portion in the concatenated sequence.
    width = tf.size(inputs)
    inputs_width = tf.fill([tf.size(inputs) + tf.size(target_suffixes)], width)

    # Width with an extra position to the right in the inputs mask. See
    # docstring for PrefixSuffixLMFeatureConverter class for details.
    inputs_width_add_pos = tf.fill(
        [tf.size(inputs) + tf.size(target_suffixes)], width + 1
    )

    target_weights_with_suffix = tf.concat(
        [
            tf.ones_like(inputs),
            tf.ones_like(targets),
            tf.fill(
                [
                    tf.size(suffixes),
                ],
                2,
            ),
        ],
        axis=-1,
    )
    target_weights_without_suffix = tf.concat(
        [
            tf.ones_like(inputs),
            tf.fill(
                [
                    tf.size(target_suffixes),
                ],
                2,
            ),
        ],
        axis=-1,
    )

    target_weights = tf.cond(
        tf.size(suffixes) > 0,
        lambda: target_weights_with_suffix,
        lambda: target_weights_without_suffix,
    )
    return {
        "targets": tf.concat([inputs, target_suffixes], axis=-1),
        "inputs_width": inputs_width,
        "inputs_width_add_pos": inputs_width_add_pos,
        "target_suffix_weights": target_weights,
    }

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    decoder_length = sum(task_feature_lengths.values())
    concat_length = {"targets": decoder_length}
    lm_model_feature_lengths = super().get_model_feature_lengths(concat_length)
    model_feature_lengths = dict(lm_model_feature_lengths)
    model_feature_lengths["decoder_causal_attention"] = decoder_length
    model_feature_lengths["target_suffix_weights"] = decoder_length
    return model_feature_lengths

  def _concat_task_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    concat_length = sum(task_feature_lengths.values())
    return {
        "targets": concat_length,
        "inputs_width": concat_length,
        "inputs_width_add_pos": concat_length,
        "target_suffix_weights": concat_length,
    }


class DecoderFeatureConverter(FeatureConverter):
  """Wrapper of FeatureConverter that handles both LM and PrefixLM tasks.

  The converter to choose depends on the keys of `task_feature_lengths`.
  """

  TASK_FEATURES = {
      "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
      # Optional fields:
      #   "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32) - Runs
      #     PrefixLMFeatureConverter if present
      #   "suffixes": FeatureConverter.FeatureSpec(dtype=tf.int32) - Runs
      #     PrefixSuffixLMFeatureConverter is present. Runs LMFeatureConverter
      #     if neither are present.
  }
  MODEL_FEATURES = {
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
      # Only if `inputs` field is present:
      "decoder_causal_attention": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "decoder_segment_ids": tf.int32,
      "decoder_positions": tf.int32,
  }

  def __init__(
      self,
      loss_on_targets_only: bool = True,
      pack: bool = True,
      use_custom_packing_ops: bool = False,
      apply_length_check: bool = True,
      bos_id: int = 0,
      passthrough_features: Optional[
          Mapping[str, FeatureConverter.FeatureSpec]
      ] = None,
  ) -> None:
    self._loss_on_targets_only = loss_on_targets_only
    super().__init__(
        pack=pack,
        use_custom_packing_ops=use_custom_packing_ops,
        apply_length_check=apply_length_check,
        bos_id=bos_id,
        passthrough_features=passthrough_features,
    )
    self.prefixlm_feature_converter = PrefixLMFeatureConverter(
        loss_on_targets_only=loss_on_targets_only,
        pack=pack,
        use_custom_packing_ops=use_custom_packing_ops,
        apply_length_check=apply_length_check,
        bos_id=bos_id,
        passthrough_features=passthrough_features,
    )
    self.strictlm_feature_converter = LMFeatureConverter(
        pack=pack,
        use_custom_packing_ops=use_custom_packing_ops,
        apply_length_check=apply_length_check,
        bos_id=bos_id,
    )
    self.prefixsuffixlm_feature_converter = PrefixSuffixLMFeatureConverter(
        loss_on_targets_only=loss_on_targets_only,
        pack=pack,
        use_custom_packing_ops=use_custom_packing_ops,
        apply_length_check=apply_length_check,
        bos_id=bos_id,
    )

  def __call__(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    # NOTE: __call__ can safely be overridden here because it is delegated to
    # other FeatureConverters - invoking their __call__ and associated checks.
    if "suffixes" in task_feature_lengths:
      return self.prefixsuffixlm_feature_converter(ds, task_feature_lengths)
    if "inputs" in task_feature_lengths:
      return self.prefixlm_feature_converter(ds, task_feature_lengths)
    else:
      return self.strictlm_feature_converter(ds, task_feature_lengths)

  def _convert_features(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """DecoderFeatureConverter does not have this method."""
    raise Exception("DecoderFeaturerConverter does not have this method.")

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    if "suffixes" in task_feature_lengths:
      model_feature_lengths = (
          self.prefixsuffixlm_feature_converter.get_model_feature_lengths(
              task_feature_lengths
          )
      )
    elif "inputs" in task_feature_lengths:
      model_feature_lengths = (
          self.prefixlm_feature_converter.get_model_feature_lengths(
              task_feature_lengths
          )
      )
    else:
      model_feature_lengths = (
          self.strictlm_feature_converter.get_model_feature_lengths(
              task_feature_lengths
          )
      )
    return model_feature_lengths


class EncoderFeatureConverter(FeatureConverter):
  """Feature converter for encoder-only achitecture such as BERT.

  The inputs and targets to the encoder are expected to be aligned.

  Just like BERT (Devlin et al. 2019, https://arxiv.org/abs/1810.04805), a
  sentinel token (e.g., [CLS]) is expected to be prepended to the inputs and
  targets sequences. This ensures that the model can be used for a
  classification task. For a packed dataset, each sequence has separate sentinel
  tokens. In terms of segment_id, the classification sentinel is considered as a
  part of the sequence to which it is appended.

  Example for a packed dataset:

    The input dataset has two examples each with algined "inputs" and "targets".

    Here assume that mask_id = 9 and cls_id = 8

    ds = [{"inputs": [8, 9, 9, 3, 4, 1], "targets": [8, 7, 4, 3, 4, 1]},
          {"inputs": [8, 3, 9, 1], "targets": [8, 3, 6, 1]}]

    task_feature_lengths = {"inputs": 11, "targets": 11}

    converted_ds = [{
         "encoder_input_tokens": [8, 9, 9, 3, 4, 1, 8, 3, 9, 1, 0],
        "encoder_target_tokens": [8, 7, 4, 3, 4, 1, 8, 3, 6, 1, 0],
          "encoder_segment_ids": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0],
            "encoder_positions": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0],
         "encoder_loss_weights": [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    }]

    Note that two examples are packed together into one example.

  Attributes:
    mask_id: an integer indicating the mask sentinel token. This id is used to
      find the positions where the loss is taken.
  """

  TASK_FEATURES = {
      "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "encoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "encoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "encoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "encoder_segment_ids": tf.int32,
      "encoder_positions": tf.int32,
  }

  def __init__(self, mask_id: int, **kwargs):
    self._mask_id = mask_id
    super().__init__(**kwargs)

  def _convert_features(
      self, ds: tf.data.Dataset, input_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Convert the input dataset to an output dataset to be fed to the model.

    The conversion process involves three steps

    1. Each feature in the `input_lengths` is packed.
    2. "inputs" fields are mapped to the encoder input and "targets" are mapped
       to encoder target. Loss is taken only on the masked positions just as in
       Masked Language Modeling objective.

    Args:
      ds: an input tf.data.Dataset to be converted.
      input_lengths: a mapping from a feature to its length

    Returns:
      ds: the converted dataset.
    """

    @utils.map_over_dataset
    def convert_example(
        features: Mapping[str, tf.Tensor]
    ) -> Mapping[str, tf.Tensor]:
      inputs = features["inputs"]
      d = {
          "encoder_input_tokens": inputs,
          "encoder_target_tokens": features["targets"],
          "encoder_loss_weights": tf.cast(
              tf.equal(inputs, self.mask_id), tf.int32
          ),
      }

      if self.pack:
        d["encoder_segment_ids"] = features["inputs_segment_ids"]
        d["encoder_positions"] = features["inputs_positions"]

      return d

    ds = self._pack_or_pad(ds, input_lengths)
    return convert_example(ds)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    encoder_length = task_feature_lengths["inputs"]
    model_feature_lengths = {
        "encoder_target_tokens": encoder_length,
        "encoder_input_tokens": encoder_length,
        "encoder_loss_weights": encoder_length,
    }
    if self.pack:
      model_feature_lengths["encoder_segment_ids"] = encoder_length
      model_feature_lengths["encoder_positions"] = encoder_length

    return model_feature_lengths

  @property
  def mask_id(self):
    return self._mask_id


class PassThroughFeatureConverter(FeatureConverter):
  """This feature converter pass through the dataset without any processing."""

  def __init__(self, **unused_kwargs):  # pylint: disable=super-init-not-called
    pass

  def __call__(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    del task_feature_lengths
    return ds

  def _convert_features(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ):
    """This method is required to be overridden but unused."""
    pass

  def get_model_feature_lengths(self, task_feature_lengths: Mapping[str, int]):
    """This method is required to be overridden but unused."""
    pass


class PrePackedLMFeatureConverter(PassThroughFeatureConverter):
  """This feature converter fixes length and filters batch features."""

  BATCH_FEATURES = (
      "decoder_input_tokens",
      "decoder_loss_weights",
      "decoder_positions",
      "decoder_segment_ids",
      "decoder_target_tokens",
  )

  def _set_shape_and_filter(self, ex, task_feature_lengths):
    shaped_filtered_ex = {}
    for feature in self.BATCH_FEATURES:
      ex[feature].set_shape(shape=task_feature_lengths["targets"])
      shaped_filtered_ex[feature] = ex[feature]
    return shaped_filtered_ex

  def __call__(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Returns input dataset filtered for BATCH_FEATURES with fixed lengths.

    Args:
      ds: Prepacked dataset containing decoder BATCH_FEATURES and potentially
        additional features to be filtered out.
      task_feature_lengths: Dictionary of feature lengths. Must include a
        targets key, used for all decoder features.

    Returns: A dataset filtered to BATCH_FEATURES and with fixed set length.
    """
    return ds.map(
        functools.partial(
            self._set_shape_and_filter,
            task_feature_lengths=task_feature_lengths,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


class PrePackedPrefixLMFeatureConverter(PrePackedLMFeatureConverter):
  """Prefix LM variant of PrePackedLMFeatureConverter.

  The pass through feature converter fixes feature lengths and filters batch
  features. For Prefix LM, the inputs and targets are combined into each
  feature.
  """

  BATCH_FEATURES = PrePackedLMFeatureConverter.BATCH_FEATURES + (
      "decoder_causal_attention",
  )

  def _set_shape_and_filter(self, ex, task_feature_lengths):
    shaped_filtered_ex = {}
    for feature in self.BATCH_FEATURES:
      ex[feature].set_shape(
          shape=task_feature_lengths["inputs"] + task_feature_lengths["targets"]
      )
      shaped_filtered_ex[feature] = ex[feature]
    return shaped_filtered_ex


class GrainFeatureConverter(FeatureConverter):
  """Feature converter for Grain data pipeline."""

  def get_grain_transforms(
      self, task_feature_lengths: Mapping[str, int], batch_size: int
  ):
    raise NotImplementedError(
        "Need to implement the `get_grain_transforms` method which returns "
        "grain.Transformations."
    )
