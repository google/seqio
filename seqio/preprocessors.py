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

"""Preprocessors for SeqIO Tasks."""

import dataclasses
import functools
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from seqio import feature_converters
from seqio import utils
import tensorflow.compat.v2 as tf

OutputFeaturesType = Mapping[str, utils.Feature]
SequenceLengthType = Mapping[str, int]


@utils.map_over_dataset
def rekey(x, key_map=None):
  """Replace the feature keys according to the mapping in `key_map`.

  For example, if the dataset returns examples of the format:
  {'foo': 'something', 'bar': 'something else'}
  and key_map = {'boo': 'foo', 'spar': 'bar'} then this function will return
  examples with the format
  {'boo': 'something', 'spar': 'something else'}

  If a mapping is to an empty key name or None, the new value is set to an empty
  string.

  Args:
    x: an example to process.
    key_map: dictionary mapping new keys to original keys

  Returns:
    A preprocessed example with the format listed above.
  """
  if key_map:
    return {
        new_key: x[old_key] if old_key else ''
        for new_key, old_key in key_map.items()
    }
  return x


def tokenize(
    dataset: tf.data.Dataset,
    output_features: OutputFeaturesType,
    copy_pretokenized: bool = True,
    with_eos: bool = False,
) -> tf.data.Dataset:
  """Encode output features with specified vocabularies.

  Passes through other features unchanged. Optionally passes through copy
  of original features with "_pretokenized" suffix added to the key.

  When `with_eos` is True and input features are ranked > 1, then an EOS is
  appended only to the last item of each 1-D sequence.

  Args:
    dataset: a tf.data.Dataset of examples to tokenize.
    output_features: a dict of Feature objects; their vocabulary attribute will
      be used to tokenize the specified features.
    copy_pretokenized: bool, whether to pass through copies of original features
      with "_pretokenized" suffix added to the key.
    with_eos: bool, whether to append EOS to the end of the sequence.

  Returns:
    a tf.data.Dataset
  """
  tokenize_fn = functools.partial(
      tokenize_impl,
      output_features=output_features,
      copy_pretokenized=copy_pretokenized,
      with_eos=with_eos,
  )
  # Set a high parallelism to mitigate head-of-line blocking due to occasional
  # long tokenizer execution times. This value has been experimentally
  # determined to work well. Setting it higher could potentially help
  # performance but will also require extra memory.
  # TODO(b/277349572) Switch back to autotune when the linked issue is resolved.
  return utils.map_over_dataset(fn=tokenize_fn, num_parallel_calls=256)(dataset)


def tokenize_impl(
    features: Mapping[str, tf.Tensor],
    output_features: OutputFeaturesType,
    copy_pretokenized: bool = True,
    with_eos: bool = False,
) -> Mapping[str, tf.Tensor]:
  """Encode output features with specified vocabularies.

  Passes through other features unchanged. Optionally passes through copy
  of original features with "_pretokenized" suffix added to the key.

  When `with_eos` is True and input features are ranked > 1, then an EOS is
  appended only to the last item of each 1-D sequence.

  Args:
    features: a string-keyed dict of tensors to tokenize.
    output_features: a dict of Feature objects; their vocabulary attribute will
      be used to tokenize the specified features.
    copy_pretokenized: bool, whether to pass through copies of original features
      with "_pretokenized" suffix added to the key.
    with_eos: bool, whether to append EOS to the end of the sequence.

  Returns:
    a string-keyed dict of Tensors
  """

  ret = {}
  for k, v in features.items():
    if k in output_features:
      if copy_pretokenized:
        ret[f'{k}_pretokenized'] = v
      vocab = output_features[k].vocabulary
      v = vocab.encode_tf(v)
      if with_eos and output_features[k].add_eos:
        # Expand dims here so that the below code can work with 1-d tensors.
        v = tf.expand_dims(v, 0)
        # Make sure we keep tensor as ragged to allow for uneven concat.
        if isinstance(v, tf.Tensor):
          v = tf.RaggedTensor.from_tensor(v)

        # Append eos to the last item of every sequence.
        eos_shape = tf.concat([v.bounding_shape()[:-2], [1, 1]], axis=0)
        eos_id = tf.broadcast_to(vocab.eos_id, eos_shape)
        last_in_sequence = tf.concat([v[..., -1:, :], eos_id], axis=-1)
        # Concat back the newly modified final sequence item.
        v = tf.concat([v[..., :-1, :], last_in_sequence], axis=-2)
        # Un-expand outer dimension.
        v = v[0]

    ret[k] = v
  return ret


def tokenize_and_append_eos(
    dataset: tf.data.Dataset,
    output_features: OutputFeaturesType,
    copy_pretokenized: bool = True,
) -> tf.data.Dataset:
  """Encode output features with specified vocabularies and append EOS.

  Passes through non-string features unchanged. Optionally passes through copy
  of original features with "_pretokenized" suffix added to the key.

  Args:
    dataset: a tf.data.Dataset of examples to tokenize.
    output_features: a dict of Feature objects; their vocabulary attribute will
      be used to tokenize the specified features.
    copy_pretokenized: bool, whether to pass through copies of original features
      with "_pretokenized" suffix added to the key.

  Returns:
    a tf.data.Dataset
  """
  return tokenize(dataset, output_features, copy_pretokenized, with_eos=True)


@utils.map_over_dataset
def print_dataset(features, summarize=3):
  """Print dataset fields for debugging purposes."""
  return {
      k: tf.keras.backend.print_tensor(v, f'{k}: ', summarize=summarize)
      for k, v in features.items()
  }


def append_eos(
    dataset: tf.data.Dataset,
    output_features: OutputFeaturesType,
) -> tf.data.Dataset:
  """Appends EOS to output feature token sequences with `add_eos` set to True.

  Respects the `add_eos` field of the seqio.Features in `output_features`.

  Args:
    dataset: a tf.data.Dataset of tokenized examples to preprocess.
    output_features: a mapping of output feature names to Feature objects.

  Returns:
    a tf.data.Dataset of tokenized examples with EOS added to specified output
    features.
  """

  def _maybe_add_eos(key: str, value: tf.Tensor) -> tf.Tensor:
    if key not in output_features or not output_features[key].add_eos:
      return value
    else:
      eos_id = output_features[key].vocabulary.eos_id
      return _append_to_innermost_axis(value, eos_id)

  return dataset.map(
      lambda ex: {k: _maybe_add_eos(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )


def append_eos_after_trim(
    dataset: tf.data.Dataset,
    output_features: OutputFeaturesType,
    sequence_length: Optional[SequenceLengthType] = None,
) -> tf.data.Dataset:
  """Trims output feature token sequences and then appends EOS.

  Respects the `add_eos` field of the seqio.Features in `output_features`.
  Truncates features before adding the EOS to ensure they fit in the max length
  specified by `sequence_length` once the EOS is added. If `sequence_length` is
  None, no trimming is performed.

  Note that sequences are automatically trimmed at the end of the Task pipeline,
  so unless you want the features to always end in EOS, use `append_eos`
  instead.

  Args:
    dataset: a tf.data.Dataset of tokenized examples to preprocess.
    output_features: a mapping of output feature names to Feature objects.
    sequence_length: a mapping from output feature names to max lengths. If
      provided, output feature sequences will be trimmed to ensure they are not
      longer than this length once EOS is added.

  Returns:
    a tf.data.Dataset of tokenized examples with EOS added to specified output
    features.
  """
  trim_fn = functools.partial(
      append_eos_after_trim_impl,
      output_features=output_features,
      sequence_length=sequence_length,
  )
  return utils.map_over_dataset(fn=trim_fn)(dataset)


def append_eos_after_trim_impl(
    features: Dict[str, tf.Tensor],
    output_features: OutputFeaturesType,
    sequence_length: Optional[SequenceLengthType] = None,
) -> Dict[str, tf.Tensor]:
  """Trims output feature token sequences and then appends EOS.

  Respects the `add_eos` field of the seqio.Features in `output_features`.
  Truncates features before adding the EOS to ensure they fit in the max length
  specified by `sequence_length` once the EOS is added. If `sequence_length` is
  None, no trimming is performed.

  Note that sequences are automatically trimmed at the end of the Task pipeline,
  so unless you want the features to always end in EOS, use `append_eos`
  instead.

  Args:
    features: a dict of tokenized examples to preprocess.
    output_features: a mapping of output feature names to Feature objects.
    sequence_length: a mapping from output feature names to max lengths. If
      provided, output feature sequences will be trimmed to ensure they are not
      longer than this length once EOS is added.

  Returns:
    a tf.data.Dataset of tokenized examples with EOS added to specified output
    features.
  """
  for key, value in features.items():
    if key not in output_features or not output_features[key].add_eos:
      pass
    else:
      eos_id = output_features[key].vocabulary.eos_id
      if (
          sequence_length is not None
          and sequence_length.get(key, None) is not None
      ):
        max_length = sequence_length[key]
        value = value[..., : max_length - 1]

      features[key] = _append_to_innermost_axis(value, eos_id)
  return features


def _append_to_innermost_axis(
    tensor: tf.Tensor, scalar: tf.Tensor
) -> tf.Tensor:
  """Appends `scalar` to each slice in the innermost axis of `tensor`.

  >>> _append_to_innermost_axis([1, 2, 3], -1)
  [1, 2, 3, -1]
  >>> _append_to_innermost_axis([[1, 2], [3, 4]], -1)
  [[1, 2, -1], [3, 4, -1]]
  >>> _append_to_innermost_axis(tf.ragged.constant([[1, 2], [3]]), -1)
  [[1, 2, -1], [3, -1]]

  Args:
    tensor: The tensor that should have a value appended.
    scalar: The value to append.

  Returns:
    A copy of `tensor` with `scalar` appended to each slice along
    the innermost axis.
  """
  if isinstance(tensor, tf.RaggedTensor):
    if tensor.shape.rank > 2:
      return tensor.with_values(
          _append_to_innermost_axis(tensor.values, scalar)
      )
    else:
      return tf.concat([tensor, tf.fill([tensor.nrows(), 1], scalar)], axis=1)
  else:
    ndims = tf.rank(tensor)
    paddings = tf.concat(
        [tf.zeros((ndims - 1, 2), dtype=tf.int32), tf.constant([[0, 1]])],
        axis=0,
    )
    return tf.pad(tensor, paddings=paddings, constant_values=scalar)


@utils.map_over_dataset
def truncate_inputs_left(example, sequence_length):
  """Pre-processor for truncation of inputs sequences from the left.

  Default seqio truncation always removes the overflow on the right, which may
  not be optimal for decoder only models.
  Applying this pre-processor truncates the 'inputs' from the left according to
  sequence_length['inputs'].
  This pre-processor should be applied after [seqio.preprocessors.tokenize,
  seqio.preprocessors.append_eos].

  Args:
    example: an example to process.
    sequence_length: dictionary with token sequence length for the inputs and
      targets.

  Returns:
    Example with truncated 'inputs' sequence.
  """
  if sequence_length is None or 'inputs' not in sequence_length:
    return example

  example['inputs'] = example['inputs'][-sequence_length['inputs'] :]

  return example


def apply_feature_converter(
    dataset: tf.data.Dataset,
    sequence_length: Dict[str, int],
    feature_converter: feature_converters.FeatureConverter,
) -> tf.data.Dataset:
  """Applies feature converter on the dataset.

  Example:
    Apply `EncDecFeatureConverter` with `pack` set to True to convert
    sequence-to-sequence examples to 'packed examples'.
    preprocessors = [
      functools.partial(
          apply_feature_converter,
          feature_converter=feature_converters.EncDecFeatureConverter(pack=True)),
    ]

  Args:
    dataset: a tf.data.Dataset of tokenized examples to pack.
    sequence_length: a mapping from output feature names to max lengths.
    feature_converter: an instance of feature_converters.FeatureConverter.

  Returns:
    a tf.data.Dataset of packed examples.
  """
  return feature_converter(dataset, sequence_length)


@utils.map_over_dataset
def preprocess_tensorflow_examples(example, inputs_format, targets_format):
  """Parse dict of tf.tensors into inputs and targets.

  This function takes a tf.data.Dataset of strings. The function returns a
  tf.data.Dataset of feature dictionaries of the form {"inputs": string,
  "targets": string}.

  inputs_format contains a template string used to produce the "inputs" string.
  targets_format contains a template string used to produce the "targets"
  string.

  Args:
    example (Dict[str, tf.Tensor]): The example to preprocess, represented as a
      dictionary where the keys are the feature names and the values are
      TensorFlow tensors.
    inputs_format (str): A format string specifying how to preprocess the
      inputs. The format string can include placeholder fields surrounded by
      curly braces, which will be replaced by the corresponding values from the
      example dictionary. For example, if the inputs_format is "Summarize this
      {text}", the format string "{text}" will be replaced by the actual tensor
      value.
    targets_format (str): A format string specifying how to preprocess the
      targets. It follows the same rules as the `inputs_format`.

  Returns:
    Dict[str, tf.Tensor]: The preprocessed example, where the inputs and targets
    have been formatted according to the provided format strings.
  """

  def _format(format_string, ex):
    field_names = ex.keys()
    field_names_re = (
        '(' + '|'.join(['{{{}}}'.format(x) for x in field_names]) + ')'
    )
    parts = []
    for p in re.split(field_names_re, format_string):
      p_name = p[1:-1]
      if p_name in field_names:
        parts.append(ex[p_name])
      else:
        parts.append(p)
    return tf.strings.join(parts)

  return {
      'inputs': _format(inputs_format, example),
      'targets': _format(targets_format, example),
  }


