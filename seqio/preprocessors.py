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

"""Preprocessors for SeqIO Tasks."""

import functools
from typing import Mapping, Optional

from seqio import dataset_providers
from seqio import tensor_preprocessors
from seqio import utils
import tensorflow.compat.v2 as tf


OutputFeaturesType = Mapping[str, dataset_providers.Feature]
SequenceLengthType = Mapping[str, int]


@utils.map_over_dataset
def rekey(x, key_map=None):
  """Replace the feature keys according to the mapping in `key_map`.

  For example, if the dataset returns examples of the format:
  {'foo': 'something', 'bar': 'something else'}
  and key_map = {'boo': 'foo', 'spar': 'bar'} then this function will return
  examples with the format
  {'boo': 'something', 'spar': 'something else'}

  If a mapping is to an empty key or None, set the new key to an empty string.
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
    with_eos: bool = False
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

  tokenize_tensor_fn = functools.partial(
      tensor_preprocessors.tokenize_tensor,
      output_features=output_features,
      with_eos=with_eos)

  def _tokenize(features):
    ret = {}
    for k, v in features.items():
      if k in output_features:
        if copy_pretokenized:
          ret[f'{k}_pretokenized'] = v
        v = tokenize_tensor_fn(k, v)
      ret[k] = v
    return ret

  return dataset.map(
      _tokenize, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def tokenize_and_append_eos(
    dataset: tf.data.Dataset,
    output_features: OutputFeaturesType,
    copy_pretokenized: bool = True,
) -> tf.data.Dataset:
  """Encode output features with specified vocbularies and append EOS.

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
def print_dataset(features):
  """tf.Print dataset fields for debugging purposes."""
  return {k: tf.Print(v, [v], k + ': ') for k, v in features.items()}


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
  maybe_add_eos_fn = functools.partial(
      tensor_preprocessors.maybe_add_eos, output_features=output_features)

  return dataset.map(
      lambda ex: {k: maybe_add_eos_fn(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE)


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
    sequence_length: a mapping from output feature names to max lengths.
      If provided, output feature sequences will be trimmed to ensure they are
      not longer than this length once EOS is added.

  Returns:
    a tf.data.Dataset of tokenized examples with EOS added to specified output
    features.
  """

  maybe_add_eos_and_trim_fn = functools.partial(
      tensor_preprocessors.maybe_add_eos_and_trim,
      output_features=output_features,
      sequence_length=sequence_length)

  return dataset.map(
      lambda ex: {k: maybe_add_eos_and_trim_fn(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
