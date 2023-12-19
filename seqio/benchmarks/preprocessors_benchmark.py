# Copyright 2023 The SeqIO Authors.
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

"""Microbenchmarks for SeqIO preprocessors functions."""

import os

import google_benchmark
from seqio import dataset_providers
from seqio import feature_converters
from seqio import preprocessors
from seqio import test_utils
from seqio import vocabularies
import tensorflow.compat.v2 as tf


Feature = dataset_providers.Feature

_TEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_data'
)
_SENTENCEPIECE_VOCAB = vocabularies.SentencePieceVocabulary(
    os.path.join(_TEST_DIR, 'sentencepiece', 'sentencepiece.model')
)
_OUTPUT_FEATURES = {
    'prefix': Feature(_SENTENCEPIECE_VOCAB, add_eos=True),
    'suffix': Feature(_SENTENCEPIECE_VOCAB, add_eos=False),
}


@google_benchmark.register
def rekey(state):
  og_dataset = tf.data.Dataset.from_tensors(
      {'text': 'That is good.', 'other': 'That is bad.'}
  )
  while state:
    _ = preprocessors.rekey(og_dataset, {'inputs': 'other', 'targets': 'text'})


@google_benchmark.register
def tokenize(state):
  og_dataset = tf.data.Dataset.from_tensors(
      {'prefix': 'This is', 'suffix': 'a test.'}
  )
  while state:
    preprocessors.tokenize(og_dataset, output_features=_OUTPUT_FEATURES)


@google_benchmark.register
def tokenize_3_rank(state):
  og_dataset = tf.data.Dataset.from_tensors({
      'prefix': tf.ragged.constant(
          [[['a', 'b'], ['c']], [['d', 'e'], ['f']], [['g', 'h'], ['i']]]
      ),
      'suffix': tf.ragged.constant(
          [[['j'], ['k', 'l', 'm']], [['n'], ['o', 'p']]]
      ),
  })
  while state:
    preprocessors.tokenize(og_dataset, output_features=_OUTPUT_FEATURES)


@google_benchmark.register
def tokenize_and_append_eos(state):
  og_dataset = tf.data.Dataset.from_tensors(
      {'prefix': 'This is', 'suffix': 'a test.'}
  )
  while state:
    preprocessors.tokenize_and_append_eos(
        og_dataset, output_features=_OUTPUT_FEATURES
    )


@google_benchmark.register
def append_eos(state):
  """Microbenchmark for appending EOS."""
  og_dataset = tf.data.Dataset.from_tensors({
      'inputs': [1, 2, 3],
      'targets': [4, 5, 6, 7],
      'arrows': [8, 9, 10, 11],
      'strings': [[14, 15], [16, 17], [18, 19]],
      'feathers': tf.ragged.constant([[20, 21], [], [22, 23, 24, 25, 26]]),
      'bows': [12, 13],
  })
  output_features = {
      'inputs': Feature(_SENTENCEPIECE_VOCAB, add_eos=False),
      'targets': Feature(_SENTENCEPIECE_VOCAB, add_eos=True),
      'arrows': Feature(_SENTENCEPIECE_VOCAB, add_eos=True),
      'strings': Feature(_SENTENCEPIECE_VOCAB, add_eos=True),
      'feathers': Feature(_SENTENCEPIECE_VOCAB, add_eos=True),
  }
  while state:
    _ = preprocessors.append_eos(og_dataset, output_features)


@google_benchmark.register
def append_eos_after_trim(state):
  """Microbenchmark for appending EOS after trimming."""
  og_dataset = tf.data.Dataset.from_tensors({
      'inputs': [1, 2, 3],
      'targets': [4, 5, 6, 7],
      'arrows': [8, 9, 10, 11],
      'strings': [[14, 15], [16, 17], [18, 19]],
      'feathers': tf.ragged.constant([[20, 21], [], [22, 23, 24, 25, 26]]),
      'bows': [12, 13],
  })
  output_features = {
      'inputs': Feature(_SENTENCEPIECE_VOCAB, add_eos=False),
      'targets': Feature(_SENTENCEPIECE_VOCAB, add_eos=True),
      'arrows': Feature(_SENTENCEPIECE_VOCAB, add_eos=True),
      'strings': Feature(_SENTENCEPIECE_VOCAB, add_eos=True),
      'feathers': Feature(_SENTENCEPIECE_VOCAB, add_eos=True),
  }
  sequence_length = {
      'inputs': 4,
      'targets': 3,
      'arrows': 5,
      'strings': 3,
      'feathers': 4,
  }
  while state:
    _ = preprocessors.append_eos_after_trim(
        og_dataset,
        output_features=output_features,
        sequence_length=sequence_length,
    )


@google_benchmark.register
def truncate_inputs_left(state):
  og_dataset = tf.data.Dataset.from_tensors({
      'inputs': [1, 2, 3],
      'targets': [4, 5, 6, 7],
  })
  sequence_length = {'inputs': 2, 'targets': 4}
  while state:
    _ = preprocessors.truncate_inputs_left(og_dataset, sequence_length)


@google_benchmark.register
def apply_feature_converter(state):
  """Microbenchmark for applying feature converter."""
  x = {'inputs': [8, 7, 1, 0], 'targets': [4, 1, 0], 'redundant_feature': [0]}
  ds = test_utils.create_default_dataset(
      [x], feature_names=('inputs', 'targets', 'redundant_feature')
  )
  sequence_length = {'inputs': 8, 'targets': 7}
  feature_converter = feature_converters.EncDecFeatureConverter()
  while state:
    _ = preprocessors.apply_feature_converter(
        ds, sequence_length=sequence_length, feature_converter=feature_converter
    )


# TODO(b/315985098): Ask mishragaurav@ for a good example and create a test.
# @google_benchmark.register
# def hash_and_tile_subtask_id(state):
#   og_dataset = tf.data.Dataset.from_tensors({
#       'inputs': 'This is',
#       'targets': 'a test.',
#       'provenance/task': 'test_task_name',
#   })
#   while state:
#     _ = preprocessors.hash_and_tile_subtask_id(og_dataset)


@google_benchmark.register
def preprocess_tensorflow_examples(state):
  og_dataset = tf.data.Dataset.from_tensors({'text': 'Hello', 'label': 'World'})
  while state:
    _ = preprocessors.preprocess_tensorflow_examples(
        og_dataset, 'Input: {text}', 'Output: {label}'
    )


if __name__ == '__main__':
  google_benchmark.main()
