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

"""Tests for tensor_preprocessors."""

import functools
from absl.testing import parameterized

from seqio import dataset_providers
from seqio import tensor_preprocessors
from seqio import test_utils
import tensorflow.compat.v2 as tf


Feature = dataset_providers.Feature


def _add_batch_dim(inputs, batch_size):
  """Add batch dim to each value of inputs."""
  def _batch_fn(v):
    if isinstance(v, tf.RaggedTensor):
      return tf.ragged.stack([v] * batch_size, axis=0)
    else:
      return tf.broadcast_to(v, [batch_size] + v.shape.as_list())
  return tf.nest.map_structure(_batch_fn, inputs)


class TensorPreprocessorsTest(tf.test.TestCase, parameterized.TestCase):

  def _assert_dict_tensors_equal(self, actual, expected):
    self.assertEqual(actual.keys(), expected.keys())
    for k, v in expected.items():
      a = actual[k]
      if isinstance(a, tf.RaggedTensor) and not isinstance(v, tf.RaggedTensor):
        # Convert expect value to ragged tensor when actual is ragged.
        v = tf.RaggedTensor.from_tensor(v)
      self.assertAllEqual(a, v)

  def test_tokenize(self):
    add_batch_dim = functools.partial(_add_batch_dim, batch_size=16)
    og_features = {
        'prefix': tf.convert_to_tensor('This is', dtype=tf.string),
        'suffix': tf.convert_to_tensor('a test.', dtype=tf.string)
    }
    output_features = {
        'prefix':
            Feature(
                test_utils.MockVocabulary({'This is': [0, 1]}), add_eos=True),
        'suffix':
            Feature(
                test_utils.MockVocabulary({'a test.': [2, 3]}), add_eos=False),
    }

    self._assert_dict_tensors_equal(
        tensor_preprocessors.tokenize(
            add_batch_dim(og_features),
            output_features=output_features),
        add_batch_dim({
            'prefix': tf.constant([0, 1], dtype=tf.int32),
            'prefix_pretokenized': tf.constant('This is', dtype=tf.string),
            'suffix': tf.constant([2, 3], dtype=tf.int32),
            'suffix_pretokenized': tf.constant('a test.', dtype=tf.string)
        }))

    self._assert_dict_tensors_equal(
        tensor_preprocessors.tokenize(
            add_batch_dim(og_features),
            output_features=output_features,
            copy_pretokenized=False),
        add_batch_dim({
            'prefix': tf.constant([0, 1], dtype=tf.int32),
            'suffix': tf.constant([2, 3], dtype=tf.int32),
        }))

    self._assert_dict_tensors_equal(
        tensor_preprocessors.tokenize_and_append_eos(
            add_batch_dim(og_features),
            output_features=output_features,
            copy_pretokenized=False),
        add_batch_dim({
            'prefix': tf.constant([0, 1, 1], dtype=tf.int32),
            'suffix': tf.constant([2, 3], dtype=tf.int32),
        }))

  def test_tokenize_multiple_ranks(self):
    add_batch_dim = functools.partial(_add_batch_dim, batch_size=16)
    vocab = test_utils.sentencepiece_vocab()
    output_features = {
        'prefix': Feature(vocab, add_eos=True),
        'suffix': Feature(vocab, add_eos=False),
    }

    # Test for 1-rank features.
    og_features_1d = {
        'prefix': tf.constant(['This is', 'this is'], dtype=tf.string),
        'suffix': tf.constant(['a test.', 'another'], dtype=tf.string)
    }
    self._assert_dict_tensors_equal(
        tensor_preprocessors.tokenize(
            add_batch_dim(og_features_1d),
            output_features=output_features),
        add_batch_dim({
            'prefix': tf.ragged.constant(
                [[3, 2, 20, 8, 6, 3, 8, 6], [11, 8, 6, 3, 8, 6]], tf.int32),
            'prefix_pretokenized': tf.constant(
                ['This is', 'this is'], tf.string),
            'suffix': tf.ragged.constant(
                [[3, 5, 10, 2], [3, 5, 22, 7, 24, 20, 4, 23]], tf.int32),
            'suffix_pretokenized': tf.constant(
                ['a test.', 'another'], tf.string)
            }))
    self._assert_dict_tensors_equal(
        tensor_preprocessors.tokenize(
            add_batch_dim(og_features_1d),
            output_features=output_features,
            with_eos=True),
        add_batch_dim({
            'prefix': tf.ragged.constant(
                [[3, 2, 20, 8, 6, 3, 8, 6], [11, 8, 6, 3, 8, 6, 1]], tf.int32),
            'prefix_pretokenized': tf.constant(
                ['This is', 'this is'], tf.string),
            'suffix': tf.ragged.constant(
                [[3, 5, 10, 2], [3, 5, 22, 7, 24, 20, 4, 23]], tf.int32),
            'suffix_pretokenized': tf.constant(
                ['a test.', 'another'], tf.string)
            }))

    # Test for 2-rank features.
    og_features_2d = {
        'prefix': tf.constant([['This is'], ['this is']], dtype=tf.string),
        'suffix': tf.constant([['a test.'], ['another']], dtype=tf.string)
    }
    self._assert_dict_tensors_equal(
        tensor_preprocessors.tokenize(
            add_batch_dim(og_features_2d),
            output_features=output_features),
        add_batch_dim({
            'prefix':
                tf.ragged.constant(
                    [[[3, 2, 20, 8, 6, 3, 8, 6]], [[11, 8, 6, 3, 8, 6]]],
                    tf.int32),
            'prefix_pretokenized':
                tf.constant([['This is'], ['this is']], tf.string),
            'suffix':
                tf.ragged.constant(
                    [[[3, 5, 10, 2]], [[3, 5, 22, 7, 24, 20, 4, 23]]],
                    tf.int32),
            'suffix_pretokenized':
                tf.constant([['a test.'], ['another']], tf.string)
        }))
    self._assert_dict_tensors_equal(
        tensor_preprocessors.tokenize(
            add_batch_dim(og_features_2d),
            output_features=output_features,
            with_eos=True),
        add_batch_dim({
            'prefix':
                tf.ragged.constant(
                    [[[3, 2, 20, 8, 6, 3, 8, 6, 1]], [[11, 8, 6, 3, 8, 6, 1]]],
                    tf.int32),
            'prefix_pretokenized':
                tf.constant([['This is'], ['this is']], tf.string),
            'suffix':
                tf.ragged.constant(
                    [[[3, 5, 10, 2]], [[3, 5, 22, 7, 24, 20, 4, 23]]],
                    tf.int32),
            'suffix_pretokenized':
                tf.constant([['a test.'], ['another']], tf.string)
        }))

    # Test for 3-rank features.
    og_features_3d = {
        'prefix': tf.ragged.constant(
            [[['a', 'b'], ['c']], [['d', 'e'], ['f']], [['g', 'h'], ['i']]],
            dtype=tf.string),
        'suffix': tf.ragged.constant(
            [[['j'], ['k', 'l', 'm']], [['n'], ['o', 'p']]], dtype=tf.string)
    }
    self._assert_dict_tensors_equal(
        tensor_preprocessors.tokenize(
            add_batch_dim(og_features_3d),
            output_features=output_features),
        add_batch_dim({
            'prefix': tf.ragged.constant(
                [[[[3, 5], [3, 2]], [[3, 13]]],
                 [[[3, 21], [3, 4]], [[3, 2]]],
                 [[[3, 2], [3, 20]], [[3, 8]]]], dtype=tf.int32),
            'prefix_pretokenized': tf.ragged.constant(
                [[['a', 'b'], ['c']], [['d', 'e'], ['f']], [['g', 'h'], ['i']]],
                dtype=tf.string),
            'suffix': tf.ragged.constant(
                [[[[3, 2]], [[3, 2], [3, 9], [3, 14]]],
                 [[[3, 22]], [[3, 7], [3, 15]]]], dtype=tf.int32),
            'suffix_pretokenized': tf.ragged.constant(
                [[['j'], ['k', 'l', 'm']], [['n'], ['o', 'p']]],
                dtype=tf.string)
        }))
    self._assert_dict_tensors_equal(
        tensor_preprocessors.tokenize(
            add_batch_dim(og_features_3d),
            output_features=output_features,
            with_eos=True),
        add_batch_dim({
            'prefix': tf.ragged.constant(
                [[[[3, 5], [3, 2, 1]], [[3, 13, 1]]],
                 [[[3, 21], [3, 4, 1]], [[3, 2, 1]]],
                 [[[3, 2], [3, 20, 1]], [[3, 8, 1]]]], dtype=tf.int32),
            'prefix_pretokenized': tf.ragged.constant(
                [[['a', 'b'], ['c']], [['d', 'e'], ['f']], [['g', 'h'], ['i']]],
                dtype=tf.string),
            'suffix': tf.ragged.constant(
                [[[[3, 2]], [[3, 2], [3, 9], [3, 14]]],
                 [[[3, 22]], [[3, 7], [3, 15]]]], dtype=tf.int32),
            'suffix_pretokenized': tf.ragged.constant(
                [[['j'], ['k', 'l', 'm']], [['n'], ['o', 'p']]],
                dtype=tf.string)
        }))

  def test_append_eos(self):
    og_features = {
        'inputs': tf.constant([1, 2, 3], tf.int32),
        'targets': tf.constant([4, 5, 6, 7], tf.int32),
        'arrows': tf.constant([8, 9, 10, 11], tf.int32),
        'bows': tf.constant([12, 13], tf.int32)
    }
    vocab = test_utils.sentencepiece_vocab()
    output_features = {
        'inputs': Feature(vocab, add_eos=False),
        'targets': Feature(vocab, add_eos=True),
        'arrows': Feature(vocab, add_eos=True),
    }
    sequence_length = {
        'inputs': 4,
        'targets': 3,
        'arrows': 5,
        'bows': 1
    }
    add_batch_dim = functools.partial(_add_batch_dim, batch_size=16)

    # Add eos only.
    self._assert_dict_tensors_equal(
        tensor_preprocessors.append_eos(
            add_batch_dim(og_features),
            output_features=output_features),
        add_batch_dim({
            'inputs': tf.constant([1, 2, 3], tf.int32),
            'targets': tf.constant([4, 5, 6, 7, 1], tf.int32),
            'arrows': tf.constant([8, 9, 10, 11, 1], tf.int32),
            'bows': tf.constant([12, 13], tf.int32)
        }))

    # Trim to sequence lengths.
    self._assert_dict_tensors_equal(
        tensor_preprocessors.append_eos_after_trim(
            add_batch_dim(og_features),
            output_features=output_features,
            sequence_length=sequence_length),
        add_batch_dim({
            'inputs': tf.constant([1, 2, 3], tf.int32),
            'targets': tf.constant([4, 5, 1], tf.int32),
            'arrows': tf.constant([8, 9, 10, 11, 1], tf.int32),
            'bows': tf.constant([12, 13], tf.int32)
        }))

    # Trim to sequence lengths (but with targets=None).
    sequence_length['targets'] = None
    self._assert_dict_tensors_equal(
        tensor_preprocessors.append_eos_after_trim(
            add_batch_dim(og_features),
            output_features=output_features,
            sequence_length=sequence_length),
        add_batch_dim({
            'inputs': tf.constant([1, 2, 3], tf.int32),
            'targets': tf.constant([4, 5, 6, 7, 1], tf.int32),
            'arrows': tf.constant([8, 9, 10, 11, 1], tf.int32),
            'bows': tf.constant([12, 13], tf.int32)
        }))

    # Don't trim to sequence lengths.
    self._assert_dict_tensors_equal(
        tensor_preprocessors.append_eos_after_trim(
            add_batch_dim(og_features),
            output_features=output_features),
        add_batch_dim({
            'inputs': tf.constant([1, 2, 3], tf.int32),
            'targets': tf.constant([4, 5, 6, 7, 1], tf.int32),
            'arrows': tf.constant([8, 9, 10, 11, 1], tf.int32),
            'bows': tf.constant([12, 13], tf.int32)
        }))

  def test_append_eos_ragged_tensor(self):
    # Test a feature with batch size = 4 but as RaggedTensor.
    og_features = {
        'inputs':
            tf.ragged.range(starts=1, limits=[4, 4, 5, 5], dtype=tf.int32),
        'targets':
            tf.ragged.range(starts=4, limits=[6, 6, 8, 8], dtype=tf.int32),
        'arrows':
            tf.ragged.range(starts=8, limits=[11, 11, 12, 12], dtype=tf.int32),
        'bows':
            tf.ragged.range(starts=12, limits=[13, 13, 14, 14], dtype=tf.int32)
    }
    vocab = test_utils.sentencepiece_vocab()
    output_features = {
        'inputs': Feature(vocab, add_eos=False),
        'targets': Feature(vocab, add_eos=True),
        'arrows': Feature(vocab, add_eos=True),
    }
    sequence_length = {
        'inputs': 4,
        'targets': 3,
        'arrows': 5,
        'bows': 1
    }

    # Add eos only.
    self._assert_dict_tensors_equal(
        tensor_preprocessors.append_eos(
            og_features,
            output_features=output_features),
        {
            'inputs':
                tf.ragged.constant(
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4]],
                    dtype=tf.int32),
            'targets':
                tf.ragged.constant(
                    [[4, 5, 1], [4, 5, 1], [4, 5, 6, 7, 1], [4, 5, 6, 7, 1]],
                    dtype=tf.int32),
            'arrows':
                tf.ragged.constant([[8, 9, 10, 1], [8, 9, 10, 1],
                                    [8, 9, 10, 11, 1], [8, 9, 10, 11, 1]],
                                   dtype=tf.int32),
            'bows':
                tf.ragged.constant([[12], [12], [12, 13], [12, 13]],
                                   dtype=tf.int32)
        })

    # Trim to sequence lengths.
    self._assert_dict_tensors_equal(
        tensor_preprocessors.append_eos_after_trim(
            og_features,
            output_features=output_features,
            sequence_length=sequence_length),
        {
            'inputs':
                tf.ragged.constant(
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4]],
                    dtype=tf.int32),
            'targets':
                tf.ragged.constant([[4, 5, 1], [4, 5, 1], [4, 5, 1], [4, 5, 1]],
                                   dtype=tf.int32),
            'arrows':
                tf.ragged.constant([[8, 9, 10, 1], [8, 9, 10, 1],
                                    [8, 9, 10, 11, 1], [8, 9, 10, 11, 1]],
                                   dtype=tf.int32),
            'bows':
                tf.ragged.constant([[12], [12], [12, 13], [12, 13]],
                                   dtype=tf.int32)
        })

    # Trim to sequence lengths (but with targets=None).
    sequence_length['targets'] = None
    self._assert_dict_tensors_equal(
        tensor_preprocessors.append_eos_after_trim(
            og_features,
            output_features=output_features,
            sequence_length=sequence_length),
        {
            'inputs':
                tf.ragged.constant(
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4]],
                    dtype=tf.int32),
            'targets':
                tf.ragged.constant(
                    [[4, 5, 1], [4, 5, 1], [4, 5, 6, 7, 1], [4, 5, 6, 7, 1]],
                    dtype=tf.int32),
            'arrows':
                tf.ragged.constant([[8, 9, 10, 1], [8, 9, 10, 1],
                                    [8, 9, 10, 11, 1], [8, 9, 10, 11, 1]],
                                   dtype=tf.int32),
            'bows':
                tf.ragged.constant([[12], [12], [12, 13], [12, 13]],
                                   dtype=tf.int32)
        })

    # Don't trim to sequence lengths.


if __name__ == '__main__':
  tf.test.main()
