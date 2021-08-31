# Copyright 2021 The SeqIO Authors.
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

"""Preprocessors for Tensor Inputs."""

import functools
from typing import Dict, Optional, Union, Mapping

from seqio import dataset_providers
import tensorflow.compat.v2 as tf

OutputFeaturesType = Mapping[str, dataset_providers.Feature]
SequenceLengthType = Mapping[str, int]


def tokenize_tensor(k: str,
                    v: tf.Tensor,
                    output_features: OutputFeaturesType,
                    with_eos: bool = False):
  """Tokenize a single input tensor.

  Args:
    k: name of feature.
    v: feature input tensor.
    output_features: a dict of Feature objects; their vocabulary attribute will
      be used to tokenize the specified features.
    with_eos: bool, whether to append EOS to the end of the sequence.
  Returns:
    a tokeinzed tf.RaggedTensor.
  """

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

  return v


def tokenize(features: Dict[str, tf.Tensor],
             output_features: OutputFeaturesType,
             copy_pretokenized: bool = True,
             with_eos: bool = False):
  """Tokenize features with batched tensor inputs.

  Note: Feature value tensors must have a batch dimension, such as [batch_size,
  None]. It will be splitted and tokenized with tokenize_tensor. For case of
  single tensor as input, a batch dimension has to be added by tf.expand_dim(x,
  axis=0).

  Args:
    features: a dictionary of input features to tokenize.
    output_features: a dict of Feature objects; their vocabulary attribute will
      be used to tokenize the specified features.
    copy_pretokenized: bool, whether to pass through copies of original features
      with "_pretokenized" suffix added to the key.
    with_eos: bool, whether to append EOS to the end of the sequence.

  Returns:
    A Dictionary with tokenized features or also including pretokenized inputs.
  """

  tokenize_tensor_fn = functools.partial(
      tokenize_tensor, output_features=output_features, with_eos=with_eos)

  def _tokenize_batch(k, v):
    """Tokenize a batched tensor.

    process each row by `tokenize_tensor` using tf.map_fn, the output should be
    a tf.RaggedTensor. Input `v` can be tf.Tensor or tf.RaggedTensor.

    Args:
      k: str, key of feature name.
      v: tf.Tensor, inputs to be tokenized with a batch dimension at axis=0.

    Returns:
      a tf.RaggedTensor.
    """
    # convert a batched input tensor to a list of tensors to tokenize.
    ragged_rank = None
    if isinstance(v, tf.RaggedTensor):
      ragged_rank = v.ragged_rank
    else:
      # ragged_rank for a tf.Tensor is rank - 1.
      ragged_rank = v.shape.rank - 1

    return tf.map_fn(
        fn=lambda t: tokenize_tensor_fn(k, t), elems=v,
        fn_output_signature=tf.RaggedTensorSpec(
            dtype=tf.int32, ragged_rank=ragged_rank))

  ret = {}
  for k, v in features.items():
    if k in output_features:
      if copy_pretokenized:
        ret[f"{k}_pretokenized"] = v
      v = _tokenize_batch(k, v)
    ret[k] = v
  return ret


def tokenize_and_append_eos(
    features: Dict[str, tf.Tensor],
    output_features: OutputFeaturesType,
    copy_pretokenized: bool = True) -> Dict[str, tf.Tensor]:
  return tokenize(
      features,
      output_features,
      copy_pretokenized,
      with_eos=True)


def maybe_add_eos(
    key: str,
    value: tf.Tensor,
    output_features: OutputFeaturesType) -> tf.Tensor:
  """Add_eos if needed."""
  if key not in output_features or not output_features[key].add_eos:
    return value
  else:
    eos_id = output_features[key].vocabulary.eos_id
    return tf.concat([value, [eos_id]], axis=0)


def append_eos(
    features: Dict[str, Union[tf.RaggedTensor, tf.Tensor]],
    output_features: OutputFeaturesType,
) -> Dict[str, Union[tf.RaggedTensor, tf.Tensor]]:
  """Appends EOS to output feature token sequences with `add_eos` set to True.

  Respects the `add_eos` field of the seqio.Features in `output_features`.

  Args:
    features: a tf.data.Dataset of tokenized examples to preprocess.
    output_features: a mapping of output feature names to Feature objects.

  Returns:
    a tf.data.Dataset of tokenized examples with EOS added to specified output
    features.
  """
  maybe_add_eos_fn = functools.partial(
      maybe_add_eos, output_features=output_features)

  return {
      k: tf.map_fn(lambda t: maybe_add_eos_fn(k, t), v)
      for k, v in features.items()
  }


def maybe_add_eos_and_trim(
    key: str,
    value: tf.Tensor,
    output_features: OutputFeaturesType,
    sequence_length: Optional[SequenceLengthType] = None):
  """Add eos and trim if needed."""
  if key not in output_features or not output_features[key].add_eos:
    return value
  eos_id = output_features[key].vocabulary.eos_id
  if (sequence_length is not None and
      sequence_length.get(key, None) is not None):
    max_length = sequence_length[key]
    return tf.concat([value[:max_length - 1], [eos_id]], axis=0)
  else:
    return tf.concat([value, [eos_id]], axis=0)


def append_eos_after_trim(
    features: Dict[str, Union[tf.RaggedTensor, tf.Tensor]],
    output_features: OutputFeaturesType,
    sequence_length: Optional[SequenceLengthType] = None):
  """Implementation of trim output featur token sequeneces and appends EOS.

  Args:
    features: A dictionary of tensor inputs.
    output_features: a mapping of output feature names to Feature objects.
    sequence_length: a mapping from output feature names to max lengths. If
      provided, output feature sequences will be trimmed to ensure they are not
      longer than this length once EOS is added.

  Returns:
    tokenized examples with EOS added to specified output.
  """

  maybe_add_eos_and_trim_fn = functools.partial(
      maybe_add_eos_and_trim,
      output_features=output_features,
      sequence_length=sequence_length)

  return {
      k: tf.map_fn(lambda t: maybe_add_eos_and_trim_fn(k, t), v)
      for k, v in features.items()
  }
