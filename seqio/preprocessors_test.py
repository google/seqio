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

"""Tests for seqio.preprocessors."""

from absl.testing import absltest
from seqio import dataset_providers
from seqio import feature_converters
from seqio import preprocessors
from seqio import test_utils
import tensorflow.compat.v2 as tf

assert_dataset = test_utils.assert_dataset
Feature = dataset_providers.Feature


class PreprocessorsTest(tf.test.TestCase):

  def test_tokenize(self):
    og_dataset = tf.data.Dataset.from_tensors(
        {'prefix': 'This is', 'suffix': 'a test.'}
    )
    output_features = {
        'prefix': Feature(
            test_utils.MockVocabulary({'This is': [0, 1]}), add_eos=True
        ),
        'suffix': Feature(
            test_utils.MockVocabulary({'a test.': [2, 3]}), add_eos=False
        ),
    }

    assert_dataset(
        preprocessors.tokenize(og_dataset, output_features=output_features),
        {
            'prefix': [0, 1],
            'prefix_pretokenized': 'This is',
            'suffix': [2, 3],
            'suffix_pretokenized': 'a test.',
        },
    )
    assert_dataset(
        preprocessors.tokenize(
            og_dataset, output_features=output_features, copy_pretokenized=False
        ),
        {'prefix': [0, 1], 'suffix': [2, 3]},
    )

    assert_dataset(
        preprocessors.tokenize_and_append_eos(
            og_dataset, output_features=output_features, copy_pretokenized=False
        ),
        {'prefix': [0, 1, 1], 'suffix': [2, 3]},
    )

  def test_tokenize_multiple_ranks(self):
    vocab = test_utils.sentencepiece_vocab()
    output_features = {
        'prefix': Feature(vocab, add_eos=True),
        'suffix': Feature(vocab, add_eos=False),
    }

    # Test for 1-rank features.
    og_dataset_1d = tf.data.Dataset.from_tensors(
        {'prefix': ['This is', 'this is'], 'suffix': ['a test.', 'another']}
    )
    assert_dataset(
        preprocessors.tokenize(og_dataset_1d, output_features=output_features),
        {
            'prefix': [[3, 2, 20, 8, 6, 3, 8, 6], [11, 8, 6, 3, 8, 6]],
            'prefix_pretokenized': ['This is', 'this is'],
            'suffix': [[3, 5, 10, 2], [3, 5, 22, 7, 24, 20, 4, 23]],
            'suffix_pretokenized': ['a test.', 'another'],
        },
    )
    assert_dataset(
        preprocessors.tokenize(
            og_dataset_1d, output_features=output_features, with_eos=True
        ),
        {
            'prefix': [[3, 2, 20, 8, 6, 3, 8, 6], [11, 8, 6, 3, 8, 6, 1]],
            'prefix_pretokenized': ['This is', 'this is'],
            'suffix': [[3, 5, 10, 2], [3, 5, 22, 7, 24, 20, 4, 23]],
            'suffix_pretokenized': ['a test.', 'another'],
        },
    )

    # Test for 2-rank features.
    og_dataset_2d = tf.data.Dataset.from_tensors({
        'prefix': [['This is'], ['this is']],
        'suffix': [['a test.'], ['another']],
    })

    assert_dataset(
        preprocessors.tokenize(og_dataset_2d, output_features=output_features),
        {
            'prefix': [[[3, 2, 20, 8, 6, 3, 8, 6]], [[11, 8, 6, 3, 8, 6]]],
            'prefix_pretokenized': [['This is'], ['this is']],
            'suffix': [[[3, 5, 10, 2]], [[3, 5, 22, 7, 24, 20, 4, 23]]],
            'suffix_pretokenized': [['a test.'], ['another']],
        },
    )
    assert_dataset(
        preprocessors.tokenize(
            og_dataset_2d, output_features=output_features, with_eos=True
        ),
        {
            'prefix': [
                [[3, 2, 20, 8, 6, 3, 8, 6, 1]],
                [[11, 8, 6, 3, 8, 6, 1]],
            ],
            'prefix_pretokenized': [['This is'], ['this is']],
            'suffix': [[[3, 5, 10, 2]], [[3, 5, 22, 7, 24, 20, 4, 23]]],
            'suffix_pretokenized': [['a test.'], ['another']],
        },
    )

    # Test for 3-rank features.
    og_dataset_3d = tf.data.Dataset.from_tensors({
        'prefix': tf.ragged.constant(
            [[['a', 'b'], ['c']], [['d', 'e'], ['f']], [['g', 'h'], ['i']]]
        ),
        'suffix': tf.ragged.constant(
            [[['j'], ['k', 'l', 'm']], [['n'], ['o', 'p']]]
        ),
    })
    assert_dataset(
        preprocessors.tokenize(og_dataset_3d, output_features=output_features),
        {
            'prefix': [
                [[[3, 5], [3, 2]], [[3, 13]]],
                [[[3, 21], [3, 4]], [[3, 2]]],
                [[[3, 2], [3, 20]], [[3, 8]]],
            ],
            'prefix_pretokenized': [
                [['a', 'b'], ['c']],
                [['d', 'e'], ['f']],
                [['g', 'h'], ['i']],
            ],
            'suffix': [
                [[[3, 2]], [[3, 2], [3, 9], [3, 14]]],
                [[[3, 22]], [[3, 7], [3, 15]]],
            ],
            'suffix_pretokenized': [
                [['j'], ['k', 'l', 'm']],
                [['n'], ['o', 'p']],
            ],
        },
    )
    assert_dataset(
        preprocessors.tokenize(
            og_dataset_3d, output_features=output_features, with_eos=True
        ),
        {
            'prefix': [
                [[[3, 5], [3, 2, 1]], [[3, 13, 1]]],
                [[[3, 21], [3, 4, 1]], [[3, 2, 1]]],
                [[[3, 2], [3, 20, 1]], [[3, 8, 1]]],
            ],
            'prefix_pretokenized': [
                [['a', 'b'], ['c']],
                [['d', 'e'], ['f']],
                [['g', 'h'], ['i']],
            ],
            'suffix': [
                [[[3, 2]], [[3, 2], [3, 9], [3, 14]]],
                [[[3, 22]], [[3, 7], [3, 15]]],
            ],
            'suffix_pretokenized': [
                [['j'], ['k', 'l', 'm']],
                [['n'], ['o', 'p']],
            ],
        },
    )

  def test_append_eos(self):
    # Features for this test:
    #    name     |   shape   | add_eos | seq_length
    #    ---------+-----------+---------+-----------
    #    inputs   |       [3] |   False |         4
    #    targets  |       [4] |    True |         3
    #    arrows   |       [4] |    True |         5
    #    strings  |    [3, 2] |    True |         3
    #    feathers | [3, None] |    True |         4
    #    bows     |       [2] |     n/a |         1
    og_dataset = tf.data.Dataset.from_tensors({
        'inputs': [1, 2, 3],
        'targets': [4, 5, 6, 7],
        'arrows': [8, 9, 10, 11],
        'strings': [[14, 15], [16, 17], [18, 19]],
        'feathers': tf.ragged.constant([[20, 21], [], [22, 23, 24, 25, 26]]),
        'bows': [12, 13],
    })
    vocab = test_utils.sentencepiece_vocab()
    output_features = {
        'inputs': Feature(vocab, add_eos=False),
        'targets': Feature(vocab, add_eos=True),
        'arrows': Feature(vocab, add_eos=True),
        'strings': Feature(vocab, add_eos=True),
        'feathers': Feature(vocab, add_eos=True),
    }
    sequence_length = {
        'inputs': 4,
        'targets': 3,
        'arrows': 5,
        'strings': 3,
        'feathers': 4,
        'bows': 1,  # note: ignored, since bows is not in output_features.
    }

    # Add eos only.
    assert_dataset(
        preprocessors.append_eos(og_dataset, output_features),
        {
            'inputs': [1, 2, 3],
            'targets': [4, 5, 6, 7, 1],
            'arrows': [8, 9, 10, 11, 1],
            'strings': [[14, 15, 1], [16, 17, 1], [18, 19, 1]],
            'feathers': [[20, 21, 1], [1], [22, 23, 24, 25, 26, 1]],
            'bows': [12, 13],
        },
    )

    # Trim to sequence lengths.
    assert_dataset(
        preprocessors.append_eos_after_trim(
            og_dataset,
            output_features=output_features,
            sequence_length=sequence_length,
        ),
        {
            'inputs': [1, 2, 3],
            'targets': [4, 5, 1],
            'arrows': [8, 9, 10, 11, 1],
            'strings': [[14, 15, 1], [16, 17, 1], [18, 19, 1]],
            'feathers': [[20, 21, 1], [1], [22, 23, 24, 1]],
            'bows': [12, 13],
        },
    )

    # Trim to sequence lengths (but with targets=None).
    sequence_length['targets'] = None
    assert_dataset(
        preprocessors.append_eos_after_trim(
            og_dataset,
            output_features=output_features,
            sequence_length=sequence_length,
        ),
        {
            'inputs': [1, 2, 3],
            'targets': [4, 5, 6, 7, 1],
            'arrows': [8, 9, 10, 11, 1],
            'strings': [[14, 15, 1], [16, 17, 1], [18, 19, 1]],
            'feathers': [[20, 21, 1], [1], [22, 23, 24, 1]],
            'bows': [12, 13],
        },
    )

    # Don't trim to sequence lengths.
    assert_dataset(
        preprocessors.append_eos_after_trim(
            og_dataset, output_features=output_features
        ),
        {
            'inputs': [1, 2, 3],
            'targets': [4, 5, 6, 7, 1],
            'arrows': [8, 9, 10, 11, 1],
            'strings': [[14, 15, 1], [16, 17, 1], [18, 19, 1]],
            'feathers': [[20, 21, 1], [1], [22, 23, 24, 25, 26, 1]],
            'bows': [12, 13],
        },
    )

  def test_append_to_innermost_axis(self):
    # Skip this test until a new version of TF is released.
    self.skipTest('b/273289576')
    test_cases = [
        ([1, 2, 3], -1, [1, 2, 3, -1]),
        ([[1, 2], [3, 4]], -1, [[1, 2, -1], [3, 4, -1]]),
        (tf.ragged.constant([[1, 2], [3]]), -1, [[1, 2, -1], [3, -1]]),
        (
            tf.ragged.constant([[[1, 2], [3]], [[4, 5, 6]]]),
            -1,
            [[[1, 2, -1], [3, -1]], [[4, 5, 6, -1]]],
        ),
        (
            tf.ragged.constant([[[1, 2], [3, 4]], [[5, 6]]], ragged_rank=1),
            -1,
            [[[1, 2, -1], [3, 4, -1]], [[5, 6, -1]]],
        ),
    ]
    for tensor, scalar, expected in test_cases:
      with self.subTest(f'({tensor}, {scalar}) -> {expected}'):
        actual = preprocessors._append_to_innermost_axis(tensor, scalar)
        self.assertAllEqual(actual, expected)

  def test_rekey(self):
    og_dataset = tf.data.Dataset.from_tensors(
        {'text': 'That is good.', 'other': 'That is bad.'}
    )
    dataset = preprocessors.rekey(
        og_dataset, {'inputs': 'other', 'targets': 'text'}
    )
    assert_dataset(
        dataset, {'inputs': 'That is bad.', 'targets': 'That is good.'}
    )

    dataset = preprocessors.rekey(og_dataset, {'targets': 'text'})
    assert_dataset(dataset, {'targets': 'That is good.'})

    dataset = preprocessors.rekey(og_dataset, {'inputs': 'text'})
    assert_dataset(dataset, {'inputs': 'That is good.'})

    dataset = preprocessors.rekey(og_dataset)
    assert_dataset(dataset, {'text': 'That is good.', 'other': 'That is bad.'})

    dataset = preprocessors.rekey(
        og_dataset, {'inputs': 'text', 'targets': None}
    )
    assert_dataset(dataset, {'inputs': 'That is good.', 'targets': ''})

  def test_truncate_length_two(self):
    self.og_tokenized_dataset = tf.data.Dataset.from_tensors({
        'inputs': [1, 2, 3],
        'targets': [4, 5, 6, 7],
    })

    # Truncate inputs from left according to sequence_length.
    sequence_length = {'inputs': 2, 'targets': 4}
    assert_dataset(
        preprocessors.truncate_inputs_left(
            self.og_tokenized_dataset, sequence_length
        ),
        {
            'inputs': [2, 3],
            'targets': [4, 5, 6, 7],
        },
    )

  def test_dont_truncate(self):
    self.og_tokenized_dataset = tf.data.Dataset.from_tensors({
        'inputs': [1, 2, 3],
        'targets': [4, 5, 6, 7],
    })
    # Don't truncate inputs when sequence_length is None.
    assert_dataset(
        preprocessors.truncate_inputs_left(self.og_tokenized_dataset, None),
        {
            'inputs': [1, 2, 3],
            'targets': [4, 5, 6, 7],
        },
    )

    assert_dataset(
        preprocessors.truncate_inputs_left(
            self.og_tokenized_dataset, {'targets': 4}
        ),
        {
            'inputs': [1, 2, 3],
            'targets': [4, 5, 6, 7],
        },
    )

  def test_apply_encdec_feature_converter(self):
    x = [
        {
            # first example
            'inputs': [8, 7, 1, 0],
            'targets': [4, 1, 0],
            'redundant_feature': [0],
        },
        {
            # second example
            'inputs': [2, 3, 4, 1],
            'targets': [5, 6, 1],
            'redundant_feature': [1],
        },
        {
            # third example
            'inputs': [9, 2, 13, 4, 6],
            'targets': [14, 3, 9, 17, 0],
            'redundant_feature': [2],
        },
        {
            # fourth example
            'inputs': [31, 2, 0, 0, 0],
            'targets': [1, 21, 6, 2, 0],
            'redundant_feature': [3],
        },
        {
            # fifth example
            'inputs': [111, 21, 131, 41, 26, 0],
            'targets': [114, 56, 12, 0, 0],
            'redundant_feature': [4],
        },
    ]

    ds = test_utils.create_default_dataset(
        x, feature_names=('inputs', 'targets', 'redundant_feature')
    )
    sequence_length = {'inputs': 8, 'targets': 7}
    feature_converter = feature_converters.EncDecFeatureConverter(pack=True)
    packed_ds = preprocessors.apply_feature_converter(
        ds, sequence_length=sequence_length, feature_converter=feature_converter
    )
    expected = [
        {
            # first and second examples are packed here.
            'encoder_input_tokens': [8, 7, 1, 2, 3, 4, 1, 0],
            'encoder_segment_ids': [1, 1, 1, 2, 2, 2, 2, 0],
            'encoder_positions': [0, 1, 2, 0, 1, 2, 3, 0],
            'decoder_target_tokens': [4, 1, 5, 6, 1, 0, 0],
            'decoder_input_tokens': [0, 4, 0, 5, 6, 0, 0],
            'decoder_loss_weights': [1, 1, 1, 1, 1, 0, 0],
            'decoder_segment_ids': [1, 1, 2, 2, 2, 0, 0],
            'decoder_positions': [0, 1, 0, 1, 2, 0, 0],
        },
        {
            # third example is packed here.
            'encoder_input_tokens': [9, 2, 13, 4, 6, 0, 0, 0],
            'encoder_segment_ids': [1, 1, 1, 1, 1, 0, 0, 0],
            'encoder_positions': [0, 1, 2, 3, 4, 0, 0, 0],
            'decoder_target_tokens': [14, 3, 9, 17, 0, 0, 0],
            'decoder_input_tokens': [0, 14, 3, 9, 0, 0, 0],
            'decoder_loss_weights': [1, 1, 1, 1, 0, 0, 0],
            'decoder_segment_ids': [1, 1, 1, 1, 0, 0, 0],
            'decoder_positions': [0, 1, 2, 3, 0, 0, 0],
        },
        {
            # fourth and fifth examples are packed here.
            'encoder_input_tokens': [31, 2, 111, 21, 131, 41, 26, 0],
            'encoder_segment_ids': [1, 1, 2, 2, 2, 2, 2, 0],
            'encoder_positions': [0, 1, 0, 1, 2, 3, 4, 0],
            'decoder_target_tokens': [1, 21, 6, 2, 114, 56, 12],
            'decoder_input_tokens': [0, 1, 21, 6, 0, 114, 56],
            'decoder_loss_weights': [1, 1, 1, 1, 1, 1, 1],
            'decoder_segment_ids': [1, 1, 1, 1, 2, 2, 2],
            'decoder_positions': [0, 1, 2, 3, 0, 1, 2],
        },
    ]
    # It tests packing and also tests that packed_dataset has examples
    # in the same order as they are in original unpacked_datatset.
    assert_dataset(packed_ds, expected)

  def test_preprocess_tensorflow_examples(self):
    og_dataset = tf.data.Dataset.from_tensors(
        {'text': 'Hello', 'label': 'World'}
    )

    inputs_format = 'Input: {text}'
    targets_format = 'Output: {label}'

    dataset = preprocessors.preprocess_tensorflow_examples(
        og_dataset, inputs_format, targets_format
    )

    assert_dataset(
        dataset, {'inputs': 'Input: Hello', 'targets': 'Output: World'}
    )

  def test_preprocess_tensorflow_examples_with_special_chars(self):
    og_dataset = tf.data.Dataset.from_tensors(
        {'text': 'Hello\\nWorld', 'label': 'Hello\\tGoogle'}
    )

    inputs_format = 'Input: {text}'
    targets_format = 'Output: {label}'

    dataset = preprocessors.preprocess_tensorflow_examples(
        og_dataset, inputs_format, targets_format
    )

    assert_dataset(
        dataset,
        {'inputs': 'Input: Hello\\nWorld', 'targets': 'Output: Hello\\tGoogle'},
    )


if __name__ == '__main__':
  absltest.main()
