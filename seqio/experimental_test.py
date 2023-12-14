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

"""Tests for seqio.preprocessors."""

import contextlib
from unittest import mock

from absl.testing import absltest
from seqio import dataset_providers
from seqio import experimental
from seqio import test_utils
from seqio import utils
from seqio import vocabularies
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


assert_dataset = test_utils.assert_dataset
Feature = dataset_providers.Feature
CacheDatasetPlaceholder = dataset_providers.CacheDatasetPlaceholder
MixtureRegistry = dataset_providers.MixtureRegistry
TaskRegistry = dataset_providers.TaskRegistry
ShardInfo = dataset_providers.ShardInfo


class FullyCachedTaskTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    TaskRegistry.reset()
    MixtureRegistry.reset()

    self.fake_source = dataset_providers.FunctionDataSource(
        lambda split, shuffle_files: tf.data.Dataset.range(2), ['train']
    )

    self.vocabulary = vocabularies.PassThroughVocabulary(100)

    self.metrics_fns = [lambda targets, predictions: 0]

    def fake_preprocessor(ds):
      """Adds one and casts to int32."""
      return ds.map(lambda x: tf.cast(x + 1, tf.int32))

    def fake_preprocessor_of(ds, output_features):
      """Creates output feature dict from scalar input."""
      return ds.map(lambda x: {k: [x] for k in output_features})

    def fake_preprocessor_sl(ds, sequence_length):
      """Concatenates the sequence length to each feature."""
      return ds.map(
          lambda x: {  # pylint:disable=g-long-lambda
              k: tf.concat([v, [sequence_length[k]]], 0) for k, v in x.items()
          }
      )

    def fake_preprocessor_sl_of(ds, sequence_length, output_features):
      """Adds the sequence length to each feature with `add_eos` enabled."""
      return ds.map(
          lambda x: {  # pylint:disable=g-long-lambda
              k: (
                  tf.concat([v, [sequence_length[k]]], 0)
                  if output_features[k].add_eos
                  else v
              )
              for k, v in x.items()
          }
      )

    self.preprocessors = [
        fake_preprocessor,
        fake_preprocessor_of,
        fake_preprocessor_sl,
        fake_preprocessor_sl_of,
    ]

  def validate_fully_cached_task(
      self, name, sequence_length, actual_sequence_length, expected_dataset
  ):
    new_task = TaskRegistry.get(name)
    self.assertLen(new_task.preprocessors, 6)
    self.assertEqual(new_task.metric_fns, self.metrics_fns)
    self.assertIsInstance(new_task.preprocessors[-2], CacheDatasetPlaceholder)
    self.assertTrue(new_task.preprocessors[-2].required)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            f"Task '{name}' requires caching, but was called with "
            '`use_cached=False`.'
        ),
    ):
      new_task.get_dataset(None)

    # Disable caching restriction to verify dataset is correct.
    new_task.preprocessors[-2]._required = False

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            f"Fully-cached task '{name}' can only be loaded with "
            f'`sequence_length={sequence_length}` or `None`. '
            f'Given sequence_length={actual_sequence_length}.'
        ),
    ):
      new_task.get_dataset(
          {k: v + 1 for k, v in sequence_length.items()}, use_cached=False
      )

    assert_dataset(new_task.get_dataset(None, shuffle=False), expected_dataset)
    assert_dataset(
        new_task.get_dataset(sequence_length, shuffle=False), expected_dataset
    )

  def test_add_fully_cached_task(self):
    preprocessors = list(self.preprocessors)
    preprocessors.insert(2, CacheDatasetPlaceholder())

    TaskRegistry.add(
        'encoder_decoder_task',
        source=self.fake_source,
        preprocessors=preprocessors,
        output_features={
            'inputs': Feature(self.vocabulary, add_eos=True),
            'targets': Feature(self.vocabulary, add_eos=False),
        },
        metric_fns=self.metrics_fns,
    )

    sequence_length = {'inputs': 5, 'targets': 6}
    actual_sequence_length = {'inputs': 6, 'targets': 7}
    experimental.add_fully_cached_task('encoder_decoder_task', sequence_length)
    self.validate_fully_cached_task(
        'encoder_decoder_task_i5_t6',
        sequence_length,
        actual_sequence_length,
        [
            {'inputs': [1, 5, 5], 'targets': [1, 6]},
            {'inputs': [2, 5, 5], 'targets': [2, 6]},
        ],
    )

  def test_add_fully_cached_task_single_feature(self):
    TaskRegistry.add(
        'decoder_task',
        source=self.fake_source,
        preprocessors=self.preprocessors,
        output_features={'targets': Feature(self.vocabulary, add_eos=True)},
        metric_fns=self.metrics_fns,
    )

    sequence_length = {'targets': 6}
    actual_sequence_length = {'targets': 7}
    experimental.add_fully_cached_task('decoder_task', sequence_length)
    self.validate_fully_cached_task(
        'decoder_task_6',
        sequence_length,
        actual_sequence_length,
        [
            {'targets': [1, 6, 6]},
            {'targets': [2, 6, 6]},
        ],
    )

  def test_add_fully_cached_task_unique_prefix(self):
    TaskRegistry.add(
        'feature_prefix_task',
        source=self.fake_source,
        preprocessors=self.preprocessors,
        output_features={
            'tar': Feature(self.vocabulary, add_eos=True),
            'targets': Feature(self.vocabulary, add_eos=False),
        },
        metric_fns=self.metrics_fns,
    )

    sequence_length = {'tar': 5, 'targets': 6}
    actual_sequence_length = {'tar': 6, 'targets': 7}
    experimental.add_fully_cached_task('feature_prefix_task', sequence_length)
    self.validate_fully_cached_task(
        'feature_prefix_task_tar5_targ6',
        sequence_length,
        actual_sequence_length,
        [
            {'tar': [1, 5, 5], 'targets': [1, 6]},
            {'tar': [2, 5, 5], 'targets': [2, 6]},
        ],
    )

  def test_add_fully_cached_task_disallow_shuffling(self):
    TaskRegistry.add(
        'decoder_task',
        source=self.fake_source,
        preprocessors=self.preprocessors,
        output_features={'targets': Feature(self.vocabulary, add_eos=True)},
        metric_fns=self.metrics_fns,
    )

    sequence_length = {'targets': 6}
    new_task = experimental.add_fully_cached_task(
        'decoder_task', sequence_length, disallow_shuffling=True
    )

    # Disable caching restriction to get past cache check.
    new_task.preprocessors[-2]._required = False

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Shuffling is disallowed for Task 'decoder_task_6' since its "
            '`shuffle_buffer_size` was set to `None` on construction.'
        ),
    ):
      new_task.get_dataset(None, shuffle=True, use_cached=False)

    new_task.get_dataset(None, shuffle=False, use_cached=False)

  def test_add_fully_cached_mixture(self):
    TaskRegistry.add(
        'task1',
        source=self.fake_source,
        preprocessors=self.preprocessors,
        output_features={'targets': Feature(self.vocabulary, add_eos=False)},
        metric_fns=self.metrics_fns,
    )

    TaskRegistry.add(
        'task2',
        source=self.fake_source,
        preprocessors=self.preprocessors,
        output_features={'targets': Feature(self.vocabulary, add_eos=True)},
        metric_fns=self.metrics_fns,
    )

    MixtureRegistry.add('mix', [('task1', 2), ('task2', lambda x: 1)])

    experimental.add_fully_cached_mixture('mix', sequence_length={'targets': 6})

    new_mix = MixtureRegistry.get('mix_6')
    new_task_names = ('task1_6', 'task2_6')
    self.assertContainsSubset(new_task_names, TaskRegistry.names())

    new_tasks = [TaskRegistry.get(n) for n in new_task_names]

    self.assertCountEqual(new_tasks, new_mix.tasks)
    self.assertEqual(new_mix.get_rate(new_tasks[0]), 2)
    self.assertEqual(new_mix.get_rate(new_tasks[1]), 1)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Task 'task1_6' requires caching, but was called with "
            '`use_cached=False`.'
        ),
    ):
      new_mix.get_dataset(None)

    # Disable caching restriction to get past cache check.
    for t in new_tasks:
      t.preprocessors[-2]._required = False

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Fully-cached task 'task1_6' can only be loaded with "
            "`sequence_length={'targets': 6}` or `None`. "
            "Given sequence_length={'targets': 7}."
        ),
    ):
      new_mix.get_dataset({'targets': 7}, use_cached=False)

    expected_dataset = [
        {'targets': [1, 6, 6]},
        {'targets': [2, 6, 6]},
        {'targets': [1, 6]},
        {'targets': [1, 6, 6]},
        {'targets': [2, 6]},
        {'targets': [2, 6, 6]},
    ]

    assert_dataset(
        new_mix.get_dataset(None, shuffle=False).take(6), expected_dataset
    )
    assert_dataset(
        new_mix.get_dataset({'targets': 6}, shuffle=False).take(6),
        expected_dataset,
    )

  def test_add_fully_cached_mixture_disallow_shuffling(self):
    TaskRegistry.add(
        'task1',
        source=self.fake_source,
        preprocessors=self.preprocessors,
        output_features={'targets': Feature(self.vocabulary, add_eos=False)},
        metric_fns=self.metrics_fns,
    )

    TaskRegistry.add(
        'task2',
        source=self.fake_source,
        preprocessors=self.preprocessors,
        output_features={'targets': Feature(self.vocabulary, add_eos=True)},
        metric_fns=self.metrics_fns,
    )

    MixtureRegistry.add('mix', [('task1', 2), ('task2', lambda x: 1)])

    new_mixture = experimental.add_fully_cached_mixture(
        'mix', sequence_length={'targets': 6}, disallow_shuffling=True
    )

    # Disable caching restriction to get past cache check.
    for t in new_mixture.tasks:
      t.preprocessors[-2]._required = False

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Shuffling is disallowed for Task 'task1_6' since its "
            '`shuffle_buffer_size` was set to `None` on construction.'
        ),
    ):
      new_mixture.get_dataset(None, shuffle=True, use_cached=False)

    new_mixture.get_dataset(None, shuffle=False, use_cached=False)


class FewshotTest(absltest.TestCase):

  def test_fewshot_data_source(self):
    def fake_dataset_fn(split, shuffle_files, seed=None):
      # Note that for the purposes of this unit test, fake_dataset_fn
      # deliberately does not properly implement shuffling. We test whether
      # FewShotDataSource is robust to this.
      del shuffle_files
      del seed
      return tf.data.Dataset.range(
          *((0, 2) if split == 'validation' else (3, 7))
      )

    # 0 shot
    src = experimental.FewshotDataSource(
        dataset_providers.FunctionDataSource(
            dataset_fn=fake_dataset_fn, splits=['train', 'validation']
        ),
        num_shots=0,
    )
    dataset = src.get_dataset('validation', shuffle=False)
    assert_dataset(
        dataset,
        [
            {
                'eval': 0,
            },
            {'eval': 1},
        ],
    )

    # 1 shot
    preprocessors = [
        utils.map_over_dataset(lambda x: {'inputs': 0, 'targets': x})
    ]
    src = experimental.FewshotDataSource(
        dataset_providers.FunctionDataSource(
            dataset_fn=fake_dataset_fn, splits=['train', 'validation']
        ),
        train_preprocessors=preprocessors,
        eval_preprocessors=preprocessors,
        num_shots=1,
    )

    # When split is 'train', check that 'train' and 'eval' fields of each
    # example are NOT always the same -- this can happen if the underlying
    # dataset_fn does not implement shuffling, causing identical examples from
    # the same split to be zipped together.

    def train_and_eval_fields_always_same(dataset):
      for ex in tfds.as_numpy(dataset):
        if ex['train'] != ex['eval']:
          return False
      return True

    # As long as train and eval fields aren't the same for SOME random seed, we
    # have achieved the desired behavior. We fix the seed for this test because
    # there are some seeds where train and eval fields DO happen to be the same
    # by random chance, which would break this test.
    self.assertFalse(
        train_and_eval_fields_always_same(
            src.get_dataset(split='train', shuffle=True, seed=123)
        )
    )

    # Even when shuffle is off, we don't want the train and eval fields to be
    # the same. Instead, the 'train' field should be deterministically shuffled.
    self.assertFalse(
        train_and_eval_fields_always_same(
            src.get_dataset(split='train', shuffle=False)
        )
    )

    # 3 shot
    src = experimental.FewshotDataSource(
        dataset_providers.FunctionDataSource(
            dataset_fn=fake_dataset_fn, splits=['train', 'validation']
        ),
        train_preprocessors=[
            utils.map_over_dataset(lambda x: {'inputs': 0, 'targets': x})
        ],
        num_shots=3,
    )
    dataset = src.get_dataset('validation', shuffle=False)
    assert_dataset(
        dataset,
        [
            {'eval': 0, 'train': {'inputs': [0, 0, 0], 'targets': [3, 5, 4]}},
            {'eval': 1, 'train': {'inputs': [0, 0, 0], 'targets': [6, 6, 3]}},
        ],
    )
    # Note: the train split has been deterministically shuffled, so the values
    # of the 'targets' field that we test for are deterministic but arbitrary.

    # 3-shot, sharded.
    assert_dataset(
        src.get_dataset(
            'validation', shuffle=False, shard_info=ShardInfo(0, 2)
        ),
        [
            {'eval': 0, 'train': {'inputs': [0, 0, 0], 'targets': [3, 5, 5]}},
        ],
    )
    assert_dataset(
        src.get_dataset(
            'validation', shuffle=False, shard_info=ShardInfo(1, 2)
        ),
        [
            {'eval': 1, 'train': {'inputs': [0, 0, 0], 'targets': [4, 6, 6]}},
        ],
    )
    # Note: the train split has been deterministically shuffled, so the values
    # of the 'targets' field that we test for are deterministic but arbitrary.

    # Missing train
    src = experimental.FewshotDataSource(
        dataset_providers.FunctionDataSource(
            dataset_fn=fake_dataset_fn, splits=['validation']
        ),
        num_shots=3,
    )
    with self.assertRaisesRegex(
        ValueError,
        "Train split 'train' is not one of the original source splits: "
        r'\(\'validation\',\)',
    ):
      dataset = src.get_dataset('validation')

  def test_fewshot_data_source_eval_on_fixed_exemplars(self):
    def fake_dataset_fn(split, shuffle_files, seed=None):
      # Note that for the purposes of this unit test, fake_dataset_fn
      # deliberately does not properly implement shuffling. We test whether
      # FewShotDataSource is robust to this.
      del shuffle_files
      del seed
      return tf.data.Dataset.range(
          *((0, 2) if split == 'validation' else (3, 7))
      )

    # 1 shot
    preprocessors = [
        utils.map_over_dataset(lambda x: {'inputs': 0, 'targets': x})
    ]
    src = experimental.FewshotDataSource(
        dataset_providers.FunctionDataSource(
            dataset_fn=fake_dataset_fn, splits=['train', 'validation']
        ),
        train_preprocessors=preprocessors,
        num_shots=1,
        eval_on_fixed_exemplars=True,
    )

    def exemplars_always_same(dataset):
      """Checks if exemplars are always the same."""
      train_ex = None
      for ex in dataset:
        if train_ex is None:
          train_ex = ex['train']
          continue
        if ex['train'] != train_ex:
          return False
      return True

    # Use 'validation' split for `eval_ds`. Since `train_ds` is initiated from
    # the 'train' split and `eval_on_fixed_exemplars=True`, always use the fixed
    # set of exemplars.
    self.assertTrue(
        exemplars_always_same(
            src.get_dataset(split='validation', shuffle=True, seed=123)
        )
    )

    assert_dataset(
        src.get_dataset('validation', shuffle=False),
        [
            {
                'eval': 0,
                'train': {
                    'inputs': [0],
                    'targets': [3],
                },
            },
            {
                'eval': 1,
                'train': {
                    'inputs': [0],
                    'targets': [3],
                },
            },
        ],
    )

    # `eval_on_fixed_exemplars` is ignored when `split` equals `train_split`.
    self.assertFalse(
        exemplars_always_same(
            src.get_dataset(split='train', shuffle=True, seed=123)
        )
    )

  def test_fewshot_preprocessor(self):
    train_examples = [
        {
            'inputs': 'How many states in the US?',
            'targets': '50',
        },
        {
            'inputs': 'How many cents in a dollar?',
            'targets': '100',
        },
        {
            'inputs': 'How many cents in a quarter?',
            'targets': '25',
        },
    ]

    eval_examples = [
        {
            'inputs': 'Who was in the Beatles?',
            'targets': 'John',
            'answers': ['John', 'Paul', 'George', 'Ringo'],
        },
        {
            'inputs': 'When did the Beatles break up?',
            'targets': '1970',
            'answers': ['1970', 'April 10, 1970', 'April 10', '4/10/1970'],
        },
    ]

    def _from_generator(examples):
      return tf.data.Dataset.from_generator(
          lambda: (x for x in examples),
          output_types={k: tf.string for k in examples[0].keys()},
          output_shapes={
              k: [None] if isinstance(v, list) else []
              for k, v in examples[0].items()
          },
      )

    train_ds = _from_generator(train_examples).repeat()
    eval_ds = _from_generator(eval_examples)

    # 0-shot
    dataset = experimental.fewshot_preprocessor(
        tf.data.Dataset.zip({'eval': eval_ds}),
        inputs_prefix='0 ',
        targets_prefix=' X 1 ',
        example_separator=' X ',
    )
    assert_dataset(
        dataset,
        [
            {
                'inputs': '0 Who was in the Beatles? X 1',
                'targets': 'John',
                'answers': ['John', 'Paul', 'George', 'Ringo'],
            },
            {
                'inputs': '0 When did the Beatles break up? X 1',
                'targets': '1970',
                'answers': ['1970', 'April 10, 1970', 'April 10', '4/10/1970'],
            },
        ],
    )

    # 2-shot
    dataset = experimental.fewshot_preprocessor(
        tf.data.Dataset.zip({'train': train_ds.batch(2), 'eval': eval_ds}),
        inputs_prefix='0 ',
        targets_prefix=' X 1 ',
        example_separator=' X ',
    )
    assert_dataset(
        dataset,
        [
            {
                'inputs': (
                    '0 How many states in the US? X 1 50 X 0 How many cents in '
                    'a dollar? X 1 100 X 0 Who was in the Beatles? X 1'
                ),
                'targets': 'John',
                'answers': ['John', 'Paul', 'George', 'Ringo'],
            },
            {
                'inputs': (
                    '0 How many cents in a quarter? X 1 25 X 0 How many states '
                    'in the US? X 1 50 X 0 When did the Beatles break up? X 1'
                ),
                'targets': '1970',
                'answers': ['1970', 'April 10, 1970', 'April 10', '4/10/1970'],
            },
        ],
    )

    # 1-shot, batched eval
    dataset = experimental.fewshot_preprocessor(
        tf.data.Dataset.zip(
            {'train': train_ds.batch(1), 'eval': eval_ds.batch(2)}
        ),
        inputs_prefix='0 ',
        targets_prefix=' X 1 ',
        example_separator=' X ',
    )
    assert_dataset(
        dataset,
        [
            {
                'inputs': (
                    '0 How many states in the US? X 1 50 X 0 Who was in the '
                    'Beatles? X 1'
                ),
                'targets': 'John',
                'answers': ['John', 'Paul', 'George', 'Ringo'],
            },
            {
                'inputs': (
                    '0 How many states in the US? X 1 50 X 0 When did the '
                    'Beatles break up? X 1'
                ),
                'targets': '1970',
                'answers': ['1970', 'April 10, 1970', 'April 10', '4/10/1970'],
            },
        ],
    )

    # 0-shot + prompt
    dataset = experimental.fewshot_preprocessor(
        tf.data.Dataset.zip({'eval': eval_ds}),
        inputs_prefix='0 ',
        targets_prefix=' X 1 ',
        example_separator=' X ',
        prompt='Answer the question',
    )
    assert_dataset(
        dataset,
        [
            {
                'inputs': 'Answer the question X 0 Who was in the Beatles? X 1',
                'targets': 'John',
                'answers': ['John', 'Paul', 'George', 'Ringo'],
            },
            {
                'inputs': (
                    'Answer the question X 0 When did the Beatles break up? X 1'
                ),
                'targets': '1970',
                'answers': ['1970', 'April 10, 1970', 'April 10', '4/10/1970'],
            },
        ],
    )

    # 2-shot + prompt
    dataset = experimental.fewshot_preprocessor(
        tf.data.Dataset.zip({'train': train_ds.batch(2), 'eval': eval_ds}),
        inputs_prefix='0 ',
        targets_prefix=' X 1 ',
        example_separator=' X ',
        prompt='Answer the question',
    )
    assert_dataset(
        dataset,
        [
            {
                'inputs': (
                    'Answer the question X 0 How many states in the US? X 1 50'
                    ' X 0 How many cents in a dollar? X 1 100 X 0 Who was in'
                    ' the Beatles? X 1'
                ),
                'targets': 'John',
                'answers': ['John', 'Paul', 'George', 'Ringo'],
            },
            {
                'inputs': (
                    'Answer the question X 0 How many cents in a quarter? X 1'
                    ' 25 X 0 How many states in the US? X 1 50 X 0 When did the'
                    ' Beatles break up? X 1'
                ),
                'targets': '1970',
                'answers': ['1970', 'April 10, 1970', 'April 10', '4/10/1970'],
            },
        ],
    )

    # 1-shot, batched eval + prompt
    dataset = experimental.fewshot_preprocessor(
        tf.data.Dataset.zip(
            {'train': train_ds.batch(1), 'eval': eval_ds.batch(2)}
        ),
        inputs_prefix='0 ',
        targets_prefix=' X 1 ',
        example_separator=' X ',
        prompt='Answer the question',
    )
    assert_dataset(
        dataset,
        [
            {
                'inputs': (
                    'Answer the question X 0 How many states in the US? X 1 50'
                    ' X 0 Who was in the Beatles? X 1'
                ),
                'targets': 'John',
                'answers': ['John', 'Paul', 'George', 'Ringo'],
            },
            {
                'inputs': (
                    'Answer the question X 0 How many states in the US? X 1 50'
                    ' X 0 When did the Beatles break up? X 1'
                ),
                'targets': '1970',
                'answers': ['1970', 'April 10, 1970', 'April 10', '4/10/1970'],
            },
        ],
    )

    # 1-shot + eval_on_fixed_exemplars + reverse
    dataset = experimental.fewshot_preprocessor(
        tf.data.Dataset.zip(
            {'train': train_ds.batch(1).take(1).repeat(), 'eval': eval_ds}
        ),
        inputs_prefix='0 ',
        targets_prefix=' X 1 ',
        example_separator=' X ',
        reverse=True,
    )
    assert_dataset(
        dataset,
        [
            {
                'inputs': (
                    '0 How many states in the US? X 1 50 X 0 Who was in the'
                    ' Beatles? X 1'
                ),
                'targets': 'John',
                'answers': ['John', 'Paul', 'George', 'Ringo'],
            },
            {
                'inputs': (
                    '0 How many states in the US? X 1 50 X 0 When did the'
                    ' Beatles break up? X 1'
                ),
                'targets': '1970',
                'answers': ['1970', 'April 10, 1970', 'April 10', '4/10/1970'],
            },
        ],
    )

    # 2-shot + eval_on_fixed_exemplars + reverse
    dataset = experimental.fewshot_preprocessor(
        tf.data.Dataset.zip(
            {'train': train_ds.batch(2).take(1).repeat(), 'eval': eval_ds}
        ),
        inputs_prefix='0 ',
        targets_prefix=' X 1 ',
        example_separator=' X ',
        reverse=True,
    )
    assert_dataset(
        dataset,
        [
            {
                'inputs': (
                    '0 How many cents in a dollar? X 1 100 X 0 How many states '
                    'in the US? X 1 50 X 0 Who was in the Beatles? X 1'
                ),
                'targets': 'John',
                'answers': ['John', 'Paul', 'George', 'Ringo'],
            },
            {
                'inputs': (
                    '0 How many cents in a dollar? X 1 100 X 0 How many states '
                    'in the US? X 1 50 X 0 When did the Beatles break up? X 1'
                ),
                'targets': '1970',
                'answers': ['1970', 'April 10, 1970', 'April 10', '4/10/1970'],
            },
        ],
    )



class SentinelTaskTest(FullyCachedTaskTest):

  def validate_sentinel_task(self, name, sequence_length, expected_dataset):
    new_task = TaskRegistry.get(name)
    # With sentinels inserted we want +1 processors.
    self.assertLen(new_task.preprocessors, 5)
    self.assertEqual(new_task.metric_fns, self.metrics_fns)
    self.assertIsNotNone(new_task.postprocessor)

    assert_dataset(
        new_task.get_dataset(sequence_length, shuffle=False), expected_dataset
    )

  def test_add_sentinel_task(self):
    preprocessors = list(self.preprocessors)

    TaskRegistry.add(
        'encoder_decoder_task',
        source=self.fake_source,
        preprocessors=preprocessors,
        output_features={
            'inputs': Feature(self.vocabulary, add_eos=True),
            'targets': Feature(self.vocabulary, add_eos=False),
        },
        metric_fns=self.metrics_fns,
    )

    sequence_length = {'inputs': 10, 'targets': 11}
    for num_sentinels in [1, 2, 4]:
      experimental.add_task_with_sentinels(
          'encoder_decoder_task', num_sentinels=num_sentinels
      )

    for sentinel_num in [1, 2, 4]:
      sentinel_ids = [
          self.vocabulary.vocab_size - (i + 1) for i in range(sentinel_num)
      ]
      self.validate_sentinel_task(
          f'encoder_decoder_task_{sentinel_num}_sentinel',
          sequence_length,
          [
              {
                  'inputs': [1, 10, 10] + sentinel_ids,
                  'targets': sentinel_ids + [1, 11],
              },
              {
                  'inputs': [2, 10, 10] + sentinel_ids,
                  'targets': sentinel_ids + [2, 11],
              },
          ],
      )


class DisableRegistryTest(absltest.TestCase):

  @contextlib.contextmanager
  def _restore_registry_after_test(self):
    og_tr_add = TaskRegistry.add
    og_tr_add_p = TaskRegistry.add_provider
    og_tr_get = TaskRegistry.get
    og_mr_add = MixtureRegistry.add
    og_mr_add_p = MixtureRegistry.add_provider
    og_mr_get = MixtureRegistry.get
    try:
      yield
    finally:
      TaskRegistry.add = og_tr_add
      TaskRegistry.add_provider = og_tr_add_p
      TaskRegistry.get = og_tr_get
      MixtureRegistry.add = og_mr_add
      MixtureRegistry.add_provider = og_mr_add_p
      MixtureRegistry.get = og_mr_get

  def test_enforce_task_registry_empty(self):
    TaskRegistry.reset()
    MixtureRegistry.reset()
    function_source = dataset_providers.FunctionDataSource(
        dataset_fn=test_utils.get_fake_dataset, splits=['train', 'validation']
    )
    TaskRegistry.add(
        'dummy_task',
        source=function_source,
        output_features=test_utils.FakeTaskTest.DEFAULT_OUTPUT_FEATURES,
    )
    with self.assertRaisesRegex(ValueError, 'The TaskRegistry is non-empty.*'):
      with self._restore_registry_after_test():
        experimental.disable_registry()

  def test_enforce_both_registries_empty(self):
    TaskRegistry.reset()
    MixtureRegistry.reset()
    function_source = dataset_providers.FunctionDataSource(
        dataset_fn=test_utils.get_fake_dataset, splits=['train', 'validation']
    )
    TaskRegistry.add(
        'dummy_task',
        source=function_source,
        output_features=test_utils.FakeTaskTest.DEFAULT_OUTPUT_FEATURES,
    )
    MixtureRegistry.add('dummy_mixture', ['dummy_task'], default_rate=1.0)
    with self.assertRaisesRegex(
        ValueError, 'The TaskRegistry, MixtureRegistry is non-empty.*'
    ):
      with self._restore_registry_after_test():
        experimental.disable_registry()

  def test_task_registry_add_no_op(self):
    TaskRegistry.reset()
    MixtureRegistry.reset()
    with self._restore_registry_after_test():
      experimental.disable_registry()
      function_source = dataset_providers.FunctionDataSource(
          dataset_fn=test_utils.get_fake_dataset, splits=['train', 'validation']
      )
      TaskRegistry.add(
          'dummy_task',
          source=function_source,
          output_features=test_utils.FakeTaskTest.DEFAULT_OUTPUT_FEATURES,
      )
      task = dataset_providers.Task(
          'dummy_task2',
          source=function_source,
          output_features=test_utils.FakeTaskTest.DEFAULT_OUTPUT_FEATURES,
      )
      TaskRegistry.add_provider('dummy_task2', task)
      self.assertEmpty(TaskRegistry.names())

  def test_task_registry_get_error(self):
    TaskRegistry.reset()
    MixtureRegistry.reset()
    with self._restore_registry_after_test():
      experimental.disable_registry()
      with self.assertRaisesRegex(
          ValueError, 'Disabled TaskRegistry.get call.*'
      ):
        TaskRegistry.get('dummy_task')

  def test_mixture_registry_add_no_op(self):
    TaskRegistry.reset()
    MixtureRegistry.reset()
    with self._restore_registry_after_test():
      experimental.disable_registry()
      function_source = dataset_providers.FunctionDataSource(
          dataset_fn=test_utils.get_fake_dataset, splits=['train', 'validation']
      )
      task = dataset_providers.Task(
          'dummy_task',
          source=function_source,
          output_features=test_utils.FakeTaskTest.DEFAULT_OUTPUT_FEATURES,
      )
      TaskRegistry.add_provider('dummy_task2', task)
      MixtureRegistry.add('dummy_mixture', [task], default_rate=1.0)
      mixture = dataset_providers.Mixture('dummy_mixture2', [[task, 1.0]])
      MixtureRegistry.add_provider('dummy_mixture2', mixture)
      self.assertEmpty(TaskRegistry.names())
      self.assertEmpty(MixtureRegistry.names())

  def test_mixture_registry_get_error(self):
    TaskRegistry.reset()
    MixtureRegistry.reset()
    with self.assertRaisesRegex(
        ValueError, 'Disabled MixtureRegistry.get call.*'
    ):
      with self._restore_registry_after_test():
        experimental.disable_registry()
        MixtureRegistry.get('dummy_mixture')



if __name__ == '__main__':
  absltest.main()
