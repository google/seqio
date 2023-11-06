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

"""Tests for seqio.dataset_providers."""

import copy
import functools
import os
import shutil
from typing import Any, Callable, Mapping, Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pyglove as pg
from seqio import dataset_providers
from seqio import feature_converters
from seqio import metrics as metrics_lib
from seqio import preprocessors
from seqio import test_utils
from seqio import utils
from seqio import vocabularies
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()

TaskRegistry = dataset_providers.TaskRegistry
MixtureRegistry = dataset_providers.MixtureRegistry
mock = absltest.mock
assert_dataset = test_utils.assert_dataset
create_default_dataset = test_utils.create_default_dataset


class TasksTest(test_utils.FakeTaskTest):

  def setUp(self):
    super(TasksTest, self).setUp()
    self._sequence_length = {"inputs": 13, "targets": 13}

  def test_invalid_name(self):
    with self.assertRaisesRegex(
        ValueError,
        (
            "Task name 'invalid/name' contains invalid characters. "
            "Must match regex: .*"
        ),
    ):
      self.add_task("invalid/name", self.function_source)

  def test_repeat_name(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Attempting to register duplicate provider: text_line_task"
    ):
      self.add_task("text_line_task", self.text_line_source)

  def test_metric_obj_arg_for_task(self):
    def score_metric_fn_1(targets, scores):
      del targets, scores
      return {}

    def score_metric_fn_2(targets, scores):
      del targets, scores
      return {}

    input_metric_objs = [
        metrics_lib.LegacyMetric.empty(
            metric_fn=score_metric_fn_1, postprocess_fn=None
        )
    ]
    input_metric_fns = [score_metric_fn_2]
    task = self.add_task(
        name="test_metric_obj_arg_for_task",
        source=self.function_source,
        preprocessors=self.DEFAULT_PREPROCESSORS,
        output_features=self.DEFAULT_OUTPUT_FEATURES,
        metric_fns=input_metric_fns,
        metric_objs=input_metric_objs,
    )

    actual_metric_objs = list(task.metric_objs)
    self.assertLen(actual_metric_objs, 2)

  def test_metric_fn_signature(self):
    # pylint:disable=unused-argument

    add_task = functools.partial(self.add_task, source=self.function_source)

    def score_metric_fn(targets, scores):
      return {}

    def predict_metric_fn(targets, predictions):
      return {}

    def predict_with_aux_metric_fn(targets, predictions, aux_values):
      return {}

    valid_task = add_task(
        "valid_metrics",
        metric_fns=[
            score_metric_fn,
            predict_metric_fn,
            predict_with_aux_metric_fn,
        ],
    )

    self.assertSameElements(
        [score_metric_fn, predict_metric_fn, predict_with_aux_metric_fn],
        valid_task.metric_fns,
    )
    self.assertSameElements([score_metric_fn], valid_task.score_metric_fns)
    self.assertSameElements([predict_metric_fn], valid_task.predict_metric_fns)
    self.assertSameElements(
        [predict_with_aux_metric_fn], valid_task.predict_with_aux_metric_fns
    )

    def extra_arg_metric_fn(targets, predictions, extra_param):
      return {}

    expected_error_message_prefix = (
        "Metric functions must have positional arguments matching either "
        "('targets', 'scores'), ('targets', 'predictions') or ('targets', "
        "'predictions', 'aux_values'). Got: "
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_error_message_prefix
        + "('targets', 'predictions', 'extra_param')",
    ):
      valid_task = add_task(
          "extra_arg_metric", metric_fns=[extra_arg_metric_fn]
      )
      # Construction (and thus validation) of metric functions and metric
      # objects is deferred until access time, so we need to actually access
      # them in order to trigger the error.
      _ = valid_task.metric_objs

    def bad_order_metric_fn(predictions, targets):
      return {}

    with self.assertRaisesWithLiteralMatch(
        ValueError, expected_error_message_prefix + "('predictions', 'targets')"
    ):
      valid_task = add_task(
          "bad_order_metric", metric_fns=[bad_order_metric_fn]
      )
      # Construction (and thus validation) of metric functions and metric
      # objects is deferred until access time, so we need to actually access
      # them in order to trigger the error.
      _ = valid_task.metric_objs

    def bad_default_metric_fn(targets, predictions=(0)):
      return {}

    with self.assertRaisesWithLiteralMatch(
        ValueError, expected_error_message_prefix + "('targets',)"
    ):
      valid_task = add_task(
          "bad_default_metric", metric_fns=[bad_default_metric_fn]
      )
      # Construction (and thus validation) of metric functions and metric
      # objects is deferred until access time, so we need to actually access
      # them in order to trigger the error.
      _ = valid_task.metric_objs

    def ok_default_metric_fn(targets, predictions, extra_param=3):
      return {}

    valid_task_2 = add_task(
        "valid_metrics_2", metric_fns=[ok_default_metric_fn]
    )
    self.assertSameElements([ok_default_metric_fn], valid_task_2.metric_fns)
    self.assertEmpty(valid_task_2.score_metric_fns)
    self.assertSameElements(
        [ok_default_metric_fn], valid_task_2.predict_metric_fns
    )

    def predict_metric_fn_with_types(
        targets: Sequence[Mapping[str, Any]],
        predictions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, metrics_lib.MetricValue]:
      return {}

    valid_task_with_types = TaskRegistry.add(
        "valid_metrics_with_types",
        source=self.function_source,
        output_features={
            "inputs": dataset_providers.Feature(
                test_utils.sentencepiece_vocab()
            ),
            "targets": dataset_providers.Feature(
                test_utils.sentencepiece_vocab()
            ),
        },
        metric_fns=[predict_metric_fn_with_types],
    )

    self.assertSameElements(
        [predict_metric_fn_with_types], valid_task_with_types.metric_fns
    )

    # pylint:enable=unused-argument

  def test_tfds_task(self):
    self.verify_task_matches_fake_datasets("tfds_task", use_cached=False)

  def test_function_task(self):
    self.verify_task_matches_fake_datasets("function_task", use_cached=False)

  def test_text_line_task(self):
    self.verify_task_matches_fake_datasets(
        "text_line_task", use_cached=False, splits=["train"]
    )

  def test_tf_example_task(self):
    self.verify_task_matches_fake_datasets(
        "tf_example_task", use_cached=False, splits=["train"]
    )

  def _get_preps_with_cache_placeholder_buffer_size(self, buffer_size):
    preps = list(self.DEFAULT_PREPROCESSORS)
    for i, p in enumerate(preps):
      if isinstance(p, dataset_providers.CacheDatasetPlaceholder):
        preps[i] = dataset_providers.CacheDatasetPlaceholder(
            file_shuffle_buffer_size=buffer_size
        )
    return preps

  def _mock_and_assert_cached_source(self, task_name):
    cached_task = dataset_providers.get_mixture_or_task(task_name)
    cached_task._get_cached_source = mock.MagicMock(
        side_effect=cached_task._get_cached_source  # pytype: disable=attribute-error  # always-use-return-annotations
    )
    _ = cached_task.get_dataset(None, "train", use_cached=True)
    cached_task._get_cached_source.assert_called_once_with("train")

  def test_cached_data_source_shuffle_buffer_default(self):
    self._mock_and_assert_cached_source("cached_task")

  def test_cached_data_source_shuffle_buffer_set(self):
    self.add_task(
        "cached_task_buf_2",
        self.tfds_source,
        self._get_preps_with_cache_placeholder_buffer_size(2),
    )
    shutil.copytree(
        self.cached_task_dir,
        os.path.join(self.test_data_dir, "cached_task_buf_2"),
    )
    self._mock_and_assert_cached_source("cached_task_buf_2")

  def test_proto_task(self):
    self.verify_task_matches_fake_datasets(
        "proto_task", use_cached=False, splits=["train"]
    )

  def test_num_input_examples(self):
    self.assertEqual(30, self.cached_task.num_input_examples("train"))
    self.assertEqual(10, self.cached_task.num_input_examples("validation"))

  def test_disallow_shuffle(self):
    task = dataset_providers.Task(
        "no_shuffle",
        source=self.function_source,
        output_features=self.DEFAULT_OUTPUT_FEATURES,
        preprocessors=self.DEFAULT_PREPROCESSORS,
        shuffle_buffer_size=None,
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Shuffling is disallowed for Task 'no_shuffle' since its "
            "`shuffle_buffer_size` was set to `None` on construction."
        ),
    ):
      task.get_dataset(None, shuffle=True)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Shuffling is disallowed for Task 'no_shuffle' since its "
            "`shuffle_buffer_size` was set to `None` on construction."
        ),
    ):
      task.get_dataset(None, shuffle=True, shuffle_buffer_size=100)

    task.get_dataset(None, shuffle=False)

  def test_supports_caching(self):
    self.assertFalse(
        dataset_providers.Task(
            "nosupports_cache",
            source=self.function_source,
            output_features=self.DEFAULT_OUTPUT_FEATURES,
            preprocessors=[],
        ).supports_caching
    )

    self.assertFalse(
        dataset_providers.Task(
            "nosupports_cache",
            source=self.function_source,
            output_features=self.DEFAULT_OUTPUT_FEATURES,
            preprocessors=[preprocessors.tokenize],
        ).supports_caching
    )

    self.assertTrue(
        dataset_providers.Task(
            "supports_cache",
            source=self.function_source,
            output_features=self.DEFAULT_OUTPUT_FEATURES,
            preprocessors=[
                preprocessors.tokenize,
                dataset_providers.CacheDatasetPlaceholder(),
            ],
        ).supports_caching
    )

    self.assertTrue(
        dataset_providers.Task(
            "supports_cache",
            source=self.function_source,
            output_features=self.DEFAULT_OUTPUT_FEATURES,
            preprocessors=[
                dataset_providers.CacheDatasetPlaceholder(required=True),
                preprocessors.tokenize,
            ],
        ).supports_caching
    )

    self.assertTrue(
        dataset_providers.Task(
            "supports_cache",
            source=self.function_source,
            output_features=self.DEFAULT_OUTPUT_FEATURES,
            preprocessors=[
                dataset_providers.CacheDatasetPlaceholder(),
            ],
        ).supports_caching
    )

  def test_requires_caching(self):
    self.assertFalse(
        dataset_providers.Task(
            "nosupports_cache",
            output_features=self.DEFAULT_OUTPUT_FEATURES,
            source=self.function_source,
            preprocessors=[preprocessors.tokenize],
        ).requires_caching
    )

    self.assertFalse(
        dataset_providers.Task(
            "supports_cache",
            output_features=self.DEFAULT_OUTPUT_FEATURES,
            source=self.function_source,
            preprocessors=[
                preprocessors.tokenize,
                dataset_providers.CacheDatasetPlaceholder(),
            ],
        ).requires_caching
    )

    task = dataset_providers.Task(
        "requires_cache",
        output_features=self.DEFAULT_OUTPUT_FEATURES,
        source=self.function_source,
        preprocessors=[
            dataset_providers.CacheDatasetPlaceholder(required=True),
            preprocessors.tokenize,
        ],
    )

    self.assertTrue(task.requires_caching)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Task 'requires_cache' requires caching, but was called with "
            "`use_cached=False`."
        ),
    ):
      task.get_dataset({"inputs": 512, "targets": 512}, use_cached=False)

    # We haven't actually cached the task, so it still fails but with a
    # different error.
    with self.assertRaisesWithLiteralMatch(
        AssertionError,
        "'requires_cache' does not exist in any of the task cache directories.",
    ):
      task.get_dataset({"inputs": 512, "targets": 512}, use_cached=True)

  def test_datasource_prohibits_caching(self):
    function_source_no_cache = dataset_providers.FunctionDataSource(
        dataset_fn=test_utils.get_fake_dataset,
        splits=["train", "validation"],
        caching_permitted=False,
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Caching was requested for 'prohibits_cache', but the underlying"
            " data source prohibits caching. Please remove"
            " `CacheDatasetPlaceholder` and try again."
        ),
    ):
      task = dataset_providers.Task(
          "prohibits_cache",
          output_features=self.DEFAULT_OUTPUT_FEATURES,
          source=function_source_no_cache,
          preprocessors=[
              dataset_providers.CacheDatasetPlaceholder(required=True),
              preprocessors.tokenize,
          ],
      )
      task._validate_preprocessors()

  def test_cache_exists(self):
    self.assertTrue(self.cached_task.cache_dir)
    self.cached_task.assert_cached()
    self.assertEqual(
        os.path.join(self.test_data_dir, "cached_task"),
        self.cached_task.cache_dir,
    )

    self.assertFalse(self.uncached_task.cache_dir)
    with self.assertRaisesWithLiteralMatch(
        AssertionError,
        "'tfds_task' does not exist in any of the task cache directories.",
    ):
      TaskRegistry.get("tfds_task").assert_cached()

  def test_get_cached_stats(self):
    expected_train_stats = {
        "examples": 3,
        "inputs_tokens": 36,
        "inputs_max_tokens": 13,
        "targets_tokens": 18,
        "targets_max_tokens": 6,
    }
    self.assertEqual(
        expected_train_stats, self.cached_task.get_cached_stats("train")
    )
    # Check repeated call.
    self.assertEqual(
        expected_train_stats, self.cached_task.get_cached_stats("train")
    )
    expected_validation_stats = {
        "examples": 2,
        "inputs_tokens": 23,
        "inputs_max_tokens": 12,
        "targets_tokens": 36,
        "targets_max_tokens": 21,
    }
    self.assertEqual(
        expected_validation_stats,
        self.cached_task.get_cached_stats("validation"),
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Stats do not exist for 'cached_task' split: fake"
    ):
      self.cached_task.get_cached_stats("fake")
    with self.assertRaisesWithLiteralMatch(
        AssertionError,
        "'uncached_task' does not exist in any of the task cache directories.",
    ):
      self.uncached_task.get_cached_stats("train")

  def test_set_global_cache_dirs(self):
    utils.set_global_cache_dirs([])
    self.assertFalse(self.cached_task.cache_dir)

    utils.set_global_cache_dirs([self.test_data_dir])
    self.assertTrue(self.cached_task.cache_dir)

  def test_get_dataset_cached(self):
    self.verify_task_matches_fake_datasets(
        "cached_task", use_cached=True, token_preprocessed=False
    )

    # Test with token preprocessor.
    self.cached_task = self.cached_task.replace(
        preprocessors=(
            self.DEFAULT_PREPROCESSORS + (test_utils.test_token_preprocessor,)
        )
    )

    self.verify_task_matches_fake_datasets(
        "cached_task",
        use_cached=True,
        token_preprocessed=False,
    )

  def test_get_dataset_onthefly(self):
    self.verify_task_matches_fake_datasets("uncached_task", use_cached=False)

    # Test with token preprocessor.
    self.cached_task = self.cached_task.replace(
        preprocessors=(
            self.DEFAULT_PREPROCESSORS + (test_utils.test_token_preprocessor,)
        )
    )

    self.verify_task_matches_fake_datasets(
        "cached_task",
        use_cached=False,
        token_preprocessed=False,
    )

  def test_get_dataset_no_truncation(self):
    self.verify_task_matches_fake_datasets(
        "uncached_task", use_cached=False, sequence_length=None
    )

  def test_sharding(self):
    for i in range(3):
      self.verify_task_matches_fake_datasets(
          "cached_task",
          use_cached=False,
          num_shards=i,
          token_preprocessed=False,
      )
      self.verify_task_matches_fake_datasets(
          "cached_task", use_cached=True, num_shards=i, token_preprocessed=False
      )

  def test_feature_validation(self):
    default_vocab = test_utils.sentencepiece_vocab()
    features = {
        "inputs": dataset_providers.Feature(
            vocabulary=default_vocab, required=False
        ),
        "targets": dataset_providers.Feature(
            vocabulary=default_vocab, required=True
        ),
        "inputs_rank2": dataset_providers.Feature(
            vocabulary=vocabularies.PassThroughVocabulary(5),
            required=False,
            rank=2,
        ),
        "continuous_features": dataset_providers.ContinuousFeature(
            required=False, rank=2
        ),
    }

    def _materialize(output):
      task = dataset_providers.Task(
          "feature_validation_task",
          self.function_source,
          output_features=features,
          preprocessors=(lambda _: tf.data.Dataset.from_tensors(output),),
          metric_fns=[],
      )
      sequence_length = copy.deepcopy(self._sequence_length)
      sequence_length["inputs_rank2"] = 13

      list(
          task.get_dataset(
              sequence_length, "train", use_cached=False
          ).as_numpy_iterator()
      )

    # Missing optional feature: OK
    _materialize({"targets": [0]})

    # Missing required feature.
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Task dataset is missing expected output feature after"
            " preprocessing: targets"
        ),
    ):
      _materialize({"inputs": [0]})

    # Wrong type.
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Task dataset has incorrect type for feature 'targets' after "
            "preprocessing: Got string, expected int32"
        ),
    ):
      _materialize({"targets": ["wrong type"]})

    # Wrong rank.
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Task dataset has incorrect rank for feature 'targets' after "
            "preprocessing: Got 0, expected 1"
        ),
    ):
      _materialize({"targets": 0})

    # Verify rank > 1 works.
    _materialize({"targets": [0], "inputs_rank2": [[0, 0, 0], [0, 0, 0]]})

    # Wrong rank (1 when 2 is expected).
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Task dataset has incorrect rank for feature 'inputs_rank2' after "
            "preprocessing: Got 1, expected 2"
        ),
    ):
      _materialize({"targets": [0], "inputs_rank2": [0]})
    # Test ContinuousFeature
    _materialize({"targets": [0], "continuous_features": [[1, 1], [0, 1]]})

  def test_value_errors(self):
    dataset_fn = lambda split, shuffle_files: tf.data.Dataset.from_tensors(
        ["test"]
    )
    output_features = {
        "inputs": dataset_providers.Feature(test_utils.sentencepiece_vocab())
    }

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "`CacheDatasetPlaceholder` can appear at most once in the"
            " preprocessing pipeline. Found multiple in"
            " 'multiple_cache_placeholders'."
        ),
    ):
      _ = dataset_providers.Task(
          "multiple_cache_placeholders",
          source=dataset_providers.FunctionDataSource(
              dataset_fn=dataset_fn, splits=["train", "validation"]
          ),
          preprocessors=[
              test_utils.test_text_preprocessor,
              preprocessors.tokenize,
              dataset_providers.CacheDatasetPlaceholder(),
              test_utils.test_token_preprocessor,
              dataset_providers.CacheDatasetPlaceholder(),
          ],
          output_features=output_features,
          metric_fns=[],
      )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "'test_token_preprocessor' has a `sequence_length` argument but"
            " occurs before `CacheDatasetPlaceholder` in"
            " 'sequence_length_pre_cache'. This is not allowed since the"
            " sequence length is specified at run time."
        ),
    ):
      task = dataset_providers.Task(
          "sequence_length_pre_cache",
          dataset_providers.FunctionDataSource(
              dataset_fn=dataset_fn,
              splits=["train"],
          ),
          preprocessors=[
              test_utils.test_text_preprocessor,
              preprocessors.tokenize,
              test_utils.test_token_preprocessor,
              dataset_providers.CacheDatasetPlaceholder(),
          ],
          output_features=output_features,
          metric_fns=[],
      )
      task._validate_preprocessors()

  def test_no_eos(self):
    default_vocab = test_utils.sentencepiece_vocab()
    features = {
        "inputs": dataset_providers.Feature(
            add_eos=True, vocabulary=default_vocab
        ),
        "targets": dataset_providers.Feature(
            add_eos=False, vocabulary=default_vocab
        ),
    }
    self.add_task("task_no_eos", self.function_source, output_features=features)
    self.verify_task_matches_fake_datasets("task_no_eos", use_cached=False)

  def test_dtype(self):
    default_vocab = test_utils.sentencepiece_vocab()
    features = {
        "inputs": dataset_providers.Feature(  # defaults to int32
            vocabulary=default_vocab
        ),
        "targets": dataset_providers.Feature(
            dtype=tf.int64, vocabulary=default_vocab
        ),
    }

    # pylint:disable=g-long-lambda
    self.add_task(
        "task_dtypes",
        self.function_source,
        preprocessors=self.DEFAULT_PREPROCESSORS
        + (
            utils.map_over_dataset(
                lambda x: {
                    k: tf.cast(v, tf.int64) if k == "targets" else v
                    for k, v in x.items()
                }
            ),
        ),
        output_features=features,
    )
    # pylint:enable=g-long-lambda
    self.verify_task_matches_fake_datasets("task_dtypes", use_cached=False)

  def test_num_epochs(self):
    # Try repeating after preprocessing the dataset to verify the outputs are
    # the same.
    epoch1_ds = self.random_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=True,
        seed=0,
    )
    # `random_task` has 3 examples per epoch.
    epoch2_ds = (
        self.random_task.get_dataset(
            self._sequence_length,
            split="train",
            use_cached=False,
            shuffle=True,
            seed=0,
        )
        .repeat(2)
        .skip(3)
    )
    test_utils.assert_datasets_eq(epoch1_ds, epoch2_ds)

    # Try repeating before preprocessing the dataset to verify the outputs are
    # different.
    epoch1_ds = self.random_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=True,
        seed=0,
    )
    # `random_task` has 3 examples per epoch.
    epoch2_ds = self.random_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=True,
        seed=0,
        num_epochs=2,
    ).skip(3)
    test_utils.assert_datasets_neq(epoch1_ds, epoch2_ds)

  def test_same_seeds_cached_match(self):
    dataset1 = self.cached_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=True,
        shuffle=True,
        seed=0,
    )
    dataset2 = self.cached_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=True,
        shuffle=True,
        seed=0,
    )
    test_utils.assert_datasets_eq(dataset1, dataset2)

  def test_different_seeds_cached_mismatch(self):
    dataset1 = self.cached_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=True,
        shuffle=True,
        seed=0,
    )
    dataset2 = self.cached_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=True,
        shuffle=True,
        seed=42,
    )
    test_utils.assert_datasets_neq(dataset1, dataset2)

  def test_same_seeds_uncached_match(self):
    dataset1 = self.uncached_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=True,
        seed=0,
    )
    dataset2 = self.uncached_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=True,
        seed=0,
    )
    test_utils.assert_datasets_eq(dataset1, dataset2)

  def test_different_seeds_uncached_mismatch(self):
    dataset1 = self.uncached_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=True,
        seed=0,
    )
    dataset2 = self.uncached_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=True,
        seed=42,
    )
    test_utils.assert_datasets_neq(dataset1, dataset2)

  def test_same_seeds_random_tp_uncached_match(self):
    dataset1 = self.random_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=True,
        seed=0,
    ).repeat(4)
    dataset2 = self.random_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=True,
        seed=0,
    ).repeat(4)
    test_utils.assert_datasets_eq(dataset1, dataset2)

  def test_different_seeds_random_tp_uncached_mismatch(self):
    dataset1 = self.random_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=True,
        seed=0,
    )
    dataset2 = self.random_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=True,
        seed=42,
    )
    test_utils.assert_datasets_neq(dataset1, dataset2)

  def test_no_shuffle_with_seed_cached_match(self):
    dataset1 = self.cached_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=True,
        shuffle=False,
        seed=0,
    )
    dataset2 = self.cached_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=True,
        shuffle=False,
        seed=42,
    )
    test_utils.assert_datasets_eq(dataset1, dataset2)

  def test_no_shuffle_with_seed_uncached_match(self):
    dataset1 = self.uncached_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=False,
        seed=0,
    )
    dataset2 = self.uncached_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=False,
        seed=42,
    )
    test_utils.assert_datasets_eq(dataset1, dataset2)

  def test_no_shuffle_different_seeds_random_tp_uncached_mismatch(self):
    dataset1 = self.random_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=False,
        seed=0,
    )
    dataset2 = self.random_task.get_dataset(
        self._sequence_length,
        split="train",
        use_cached=False,
        shuffle=False,
        seed=42,
    )
    test_utils.assert_datasets_neq(dataset1, dataset2)

  def test_plaintext_to_pretokenized_rename(self):
    ds = self.cached_plaintext_task.get_dataset(
        self._sequence_length, split="train", use_cached=True, shuffle=False
    )
    keys = next(ds.as_numpy_iterator()).keys()
    self.assertSetEqual(
        set(keys),
        set(
            ["inputs", "inputs_pretokenized", "targets", "targets_pretokenized"]
        ),
    )

  def test_list_shards(self):
    def _get_formatted_shards_list(task_name, split):
      shards = dataset_providers.get_mixture_or_task(
          task_name
      ).source.list_shards(  # pytype: disable=attribute-error
          split
      )  # always-use-return-annotations
      shards = [s.split("/")[-1] for s in shards]
      return sorted(shards)

    self.assertListEqual(
        _get_formatted_shards_list("tfds_task", "train"),
        ["train.tfrecord-00000-of-00002", "train.tfrecord-00001-of-00002"],
    )
    self.assertListEqual(
        _get_formatted_shards_list("text_line_task", "train"),
        ["train.tsv-00000-of-00002", "train.tsv-00001-of-00002"],
    )
    self.assertListEqual(
        _get_formatted_shards_list("tf_example_task", "train"),
        ["train.tfrecord-00000-of-00002", "train.tfrecord-00001-of-00002"],
    )
    self.assertListEqual(
        _get_formatted_shards_list("proto_task", "train"),
        ["train.tfrecord-00000-of-00002", "train.tfrecord-00001-of-00002"],
    )
    self.assertListEqual(
        _get_formatted_shards_list("function_task", "train"), ["train"]
    )
    self.assertListEqual(
        _get_formatted_shards_list("fully_processed_precache", "train"),
        ["train"],
    )
    self.assertListEqual(
        _get_formatted_shards_list("tokenized_postcache", "train"), ["train"]
    )
    self.assertListEqual(
        _get_formatted_shards_list("random_task", "train"), ["train"]
    )
    self.assertListEqual(
        _get_formatted_shards_list("uncached_task", "train"),
        ["train.tfrecord-00000-of-00002", "train.tfrecord-00001-of-00002"],
    )
    self.assertListEqual(
        _get_formatted_shards_list("cached_task", "train"),
        ["train.tfrecord-00000-of-00002", "train.tfrecord-00001-of-00002"],
    )
    self.assertListEqual(
        _get_formatted_shards_list("cached_plaintext_task", "train"),
        ["train.tfrecord-00000-of-00002", "train.tfrecord-00001-of-00002"],
    )

  def test_replace_name(self):
    new_name = "new_tfds_task"
    task = TaskRegistry.get("tfds_task")
    new_task = task.replace(
        name=new_name,
    )
    # Assert that the changed attribute should be different.
    self.assertEqual(new_name, new_task.name)

    # Assert that the other attributes remain the same.
    self.assertEqual(task.source, new_task.source)
    self.assertEqual(task.preprocessors, new_task.preprocessors)
    self.assertEqual(task._postprocess_fn, new_task._postprocess_fn)
    self.assertEqual(task.metric_fns, new_task.metric_fns)
    self.assertEqual(task.metric_objs, new_task.metric_objs)
    self.assertEqual(task.shuffle_buffer_size, new_task.shuffle_buffer_size)

  def test_replace_source(self):
    new_source = dataset_providers.FunctionDataSource(
        dataset_fn=test_utils.get_fake_dataset, splits=["a_weird_split"]
    )
    task = TaskRegistry.get("tfds_task")
    new_task = task.replace(
        source=new_source,
    )
    # Assert that the changed attribute should be different.
    self.assertEqual(new_source, new_task.source)

    # Assert that the other attributes remain the same.
    self.assertEqual(task.name, new_task.name)
    self.assertEqual(task.preprocessors, new_task.preprocessors)
    self.assertEqual(task._postprocess_fn, new_task._postprocess_fn)
    self.assertEqual(task.metric_fns, new_task.metric_fns)
    self.assertEqual(task.metric_objs, new_task.metric_objs)
    self.assertEqual(task.shuffle_buffer_size, new_task.shuffle_buffer_size)

  def test_replace_output_features(self):
    new_output_features = {
        "weird_inputs": dataset_providers.Feature(
            test_utils.sentencepiece_vocab()
        ),
        "weird_targets": dataset_providers.Feature(
            test_utils.sentencepiece_vocab()
        ),
    }
    task = TaskRegistry.get("tfds_task")
    new_task = task.replace(
        output_features=new_output_features,
    )
    # Assert that the changed attribute should be different.
    self.assertEqual(new_output_features, new_task.output_features)

    # Assert that the other attributes remain the same.
    self.assertEqual(task.name, new_task.name)
    self.assertEqual(task.source, new_task.source)
    self.assertEqual(task.preprocessors, new_task.preprocessors)
    self.assertEqual(task._postprocess_fn, new_task._postprocess_fn)
    self.assertEqual(task.metric_fns, new_task.metric_fns)
    self.assertEqual(task.metric_objs, new_task.metric_objs)
    self.assertEqual(task.shuffle_buffer_size, new_task.shuffle_buffer_size)

  def test_replace_preprocessors(self):
    new_preprocessors = (dataset_providers.CacheDatasetPlaceholder(),)
    task = TaskRegistry.get("tfds_task")
    new_task = task.replace(
        preprocessors=new_preprocessors,
    )
    # Assert that the changed attribute should be different.
    self.assertEqual(new_preprocessors, new_task.preprocessors)

    # Assert that the other attributes remain the same.
    self.assertEqual(task.name, new_task.name)
    self.assertEqual(task.source, new_task.source)
    self.assertEqual(task.output_features, new_task.output_features)
    self.assertEqual(task._postprocess_fn, new_task._postprocess_fn)
    self.assertEqual(task.metric_fns, new_task.metric_fns)
    self.assertEqual(task.metric_objs, new_task.metric_objs)
    self.assertEqual(task.shuffle_buffer_size, new_task.shuffle_buffer_size)

  def test_replace_postprocess_fn(self):
    new_postprocessor = lambda x: x
    task = TaskRegistry.get("tfds_task")
    new_task = task.replace(
        postprocess_fn=new_postprocessor,
    )
    # Assert that the changed attribute should be different.
    self.assertEqual(new_postprocessor, new_task.postprocessor)

    # Assert that the other attributes remain the same.
    self.assertEqual(task.name, new_task.name)
    self.assertEqual(task.source, new_task.source)
    self.assertEqual(task.output_features, new_task.output_features)
    self.assertEqual(task.preprocessors, new_task.preprocessors)
    self.assertEqual(task.metric_fns, new_task.metric_fns)
    self.assertEqual(task.metric_objs, new_task.metric_objs)
    self.assertEqual(task.shuffle_buffer_size, new_task.shuffle_buffer_size)

  def test_replace_metric_fns(self):
    def score_metric_fn_1(targets, scores):
      del targets, scores
      return {}

    new_metric_fns = [score_metric_fn_1]
    task = TaskRegistry.get("tfds_task")
    new_task = task.replace(
        metric_fns=new_metric_fns,
    )
    # Assert that the changed attribute should be different.
    self.assertEqual(new_metric_fns, new_task.metric_fns)

    # Assert that the other attributes remain the same.
    self.assertEqual(task.name, new_task.name)
    self.assertEqual(task.source, new_task.source)
    self.assertEqual(task.output_features, new_task.output_features)
    self.assertEqual(task.preprocessors, new_task.preprocessors)
    self.assertEqual(task.postprocessor, new_task.postprocessor)
    self.assertEqual(task.shuffle_buffer_size, new_task.shuffle_buffer_size)

  def test_replace_metric_objs(self):
    def score_metric_fn_1(targets, scores):
      del targets, scores
      return {}

    new_metric_objs = [
        metrics_lib.LegacyMetric.empty(
            metric_fn=score_metric_fn_1, postprocess_fn=None
        )
    ]
    task = TaskRegistry.get("tfds_task")
    new_task = task.replace(
        metric_objs=new_metric_objs,
    )
    # Assert that the changed attribute should be different.
    self.assertEqual(new_metric_objs, new_task.metric_objs)

    # Assert that the other attributes remain the same.
    self.assertEqual(task.name, new_task.name)
    self.assertEqual(task.source, new_task.source)
    self.assertEqual(task.output_features, new_task.output_features)
    self.assertEqual(task.preprocessors, new_task.preprocessors)
    self.assertEqual(task.postprocessor, new_task.postprocessor)
    self.assertEqual(task.shuffle_buffer_size, new_task.shuffle_buffer_size)

  def test_replace_shuffle_butter_size(self):
    new_shuffle_buffer_size = 987654321
    task = TaskRegistry.get("tfds_task")
    new_task = task.replace(
        shuffle_buffer_size=new_shuffle_buffer_size,
    )
    # Assert that the changed attribute should be different.
    self.assertEqual(new_shuffle_buffer_size, new_task.shuffle_buffer_size)

    # Assert that the other attributes remain the same.
    self.assertEqual(task.name, new_task.name)
    self.assertEqual(task.source, new_task.source)
    self.assertEqual(task.output_features, new_task.output_features)
    self.assertEqual(task.preprocessors, new_task.preprocessors)
    self.assertEqual(task.postprocessor, new_task.postprocessor)
    self.assertEqual(task.metric_fns, new_task.metric_fns)
    self.assertEqual(task.metric_objs, new_task.metric_objs)

  def test_replace_fails_on_invalid_args(self):
    task = TaskRegistry.get("tfds_task")
    with self.assertRaisesRegex(ValueError, "Expected keys"):
      task.replace(this_argument_is_not_valid=616161616)





class MixturesTest(test_utils.FakeTaskTest):

  def setUp(self):
    super(MixturesTest, self).setUp()
    self._sequence_length = {"inputs": 13, "targets": 13}

  def test_tasks(self):
    self.add_task("task1", self.function_source)
    self.add_task("task2", self.function_source)
    MixtureRegistry.add("test_mix1", [("task1", 1), ("task2", 1)])
    mix = MixtureRegistry.get("test_mix1")
    self.assertEqual(len(mix.tasks), 2)

    for task in mix.tasks:
      self.verify_task_matches_fake_datasets(task.name, use_cached=False)
      self.assertEqual(mix.get_rate(task), 1)

  def test_task_objs(self):
    task1 = dataset_providers.Task(
        "task1",
        self.function_source,
        preprocessors=self.DEFAULT_PREPROCESSORS,
        output_features=self.DEFAULT_OUTPUT_FEATURES,
    )
    task2 = dataset_providers.Task(
        "task2",
        self.function_source,
        preprocessors=self.DEFAULT_PREPROCESSORS,
        output_features=self.DEFAULT_OUTPUT_FEATURES,
    )

    MixtureRegistry.add("test_mix1", [(task1, 1), (task2, 1)])
    mix = MixtureRegistry.get("test_mix1")
    self.assertEqual(len(mix.tasks), 2)

    for task in mix.tasks:
      self.verify_task_matches_fake_datasets(task=task, use_cached=False)
      self.assertEqual(mix.get_rate(task), 1)

  def test_task_objs_default_rate(self):
    task1 = dataset_providers.Task(
        "task1",
        self.function_source,
        preprocessors=self.DEFAULT_PREPROCESSORS,
        output_features=self.DEFAULT_OUTPUT_FEATURES,
    )
    task2 = dataset_providers.Task(
        "task2",
        self.function_source,
        preprocessors=self.DEFAULT_PREPROCESSORS,
        output_features=self.DEFAULT_OUTPUT_FEATURES,
    )
    MixtureRegistry.add("test_mix1", [task1, task2], default_rate=1.0)
    mix = MixtureRegistry.get("test_mix1")
    self.assertEqual(len(mix.tasks), 2)

    for task in mix.tasks:
      self.verify_task_matches_fake_datasets(task=task, use_cached=False)
      self.assertEqual(mix.get_rate(task), 1)

  def test_tasks_with_tunable_rates(self):
    self.add_task("task1", self.function_source)
    self.add_task("task2", self.function_source)
    self.add_task("task3", self.function_source)
    MixtureRegistry.add(
        "test_mix1",
        [
            ("task1", pg.oneof([1, 2])),
            ("task2", pg.floatv(0.0, 10.0, name="w2")),
        ],
        mixture_cls=dataset_providers.PyGloveTunableMixture,
    )

    MixtureRegistry.add(
        "test_mix2",
        [
            ("test_mix1", pg.oneof([10, 20], name="wmix")),
            ("task3", pg.oneof([20, 40], name="w3")),
        ],
        mixture_cls=dataset_providers.PyGloveTunableMixture,
    )

    mix = MixtureRegistry.get("test_mix2")
    self.assertEqual(len(mix.tasks), 3)

    automl_context = pg.hyper.DynamicEvaluationContext(require_hyper_name=True)
    with automl_context.collect():
      _ = [mix.get_rate(t) for t in mix.tasks]

    self.assertEqual(
        automl_context.hyper_dict,
        {
            "wmix": pg.oneof([10, 20], name="wmix"),
            "task1": pg.oneof([1, 2], name="task1"),
            "w2": pg.floatv(0.0, 10.0, name="w2"),
            "w3": pg.oneof([20, 40], name="w3"),
        },
    )

    with automl_context.apply(pg.DNA([0, 1, 5.5, 0])):
      self.assertEqual(
          [mix.get_rate(t) for t in mix.tasks], [8 / 3, 22 / 3, 20.0]
      )

    with automl_context.apply(pg.DNA([1, 0, 3.0, 1])):
      self.assertEqual([mix.get_rate(t) for t in mix.tasks], [5.0, 15.0, 40.0])

  def test_num_examples(self):
    MixtureRegistry.add("test_mix2", [(self.cached_task.name, 1)])
    mix = MixtureRegistry.get("test_mix2")
    self.assertEqual(mix.num_input_examples(split="train"), 30)

  def test_splits(self):
    MixtureRegistry.add(
        "test_mix", [(self.cached_task.name, 1), (self.uncached_task.name, 1)]
    )
    mix = MixtureRegistry.get("test_mix")
    self.assertSameElements(["train", "validation"], mix.splits, 30)

  def test_get_dataset(self):
    MixtureRegistry.add("test_mix3", [(self.cached_task.name, 1)])

    task_ds = TaskRegistry.get_dataset(
        self.cached_task.name,
        self._sequence_length,
        "validation",
        use_cached=False,
        shuffle=False,
    )

    mix_ds = MixtureRegistry.get("test_mix3").get_dataset(
        self._sequence_length, "validation", use_cached=False, shuffle=False
    )

    # mix.get_dataset strips non-output features
    task_ds = task_ds.map(lambda x: {k: x[k] for k in ["inputs", "targets"]})

    # limit size since get_dataset repeats the dataset
    test_utils.assert_datasets_eq(task_ds.repeat(2), mix_ds.take(4))

  def test_get_dataset_mix(self):
    @utils.map_over_dataset
    def _constant_preprocessor(unused_x, val):
      return {
          "targets": tf.constant([val], tf.int32),
          "inputs": tf.constant([val], tf.int32),
      }

    self.add_task(
        "two_task",
        self.function_source,
        preprocessors=(functools.partial(_constant_preprocessor, val=2),),
    )

    self.add_task(
        "three_task",
        self.function_source,
        preprocessors=(functools.partial(_constant_preprocessor, val=3),),
    )

    MixtureRegistry.add("test_mix", [("two_task", 1), ("three_task", 1)])

    sequence_length = {"inputs": 2, "targets": 2}
    mix_ds = (
        MixtureRegistry.get("test_mix")
        .get_dataset(sequence_length, "train", seed=13)
        .take(1000)
    )

    res = sum(int(item["inputs"][0]) for item in mix_ds.as_numpy_iterator())
    self.assertEqual(res, 2481)

  def test_get_dataset_passthrough_features(self):
    @utils.map_over_dataset
    def _constant_feature_preprocessor(unused_x, val):
      return {
          "targets": tf.constant([val], tf.int32),
          "inputs": tf.constant([val], tf.int32),
          "feature": tf.constant([val], tf.int32),
      }

    self.add_task(
        "two_task",
        self.function_source,
        preprocessors=(
            functools.partial(_constant_feature_preprocessor, val=2),
        ),
    )

    self.add_task(
        "three_task",
        self.function_source,
        preprocessors=(
            functools.partial(_constant_feature_preprocessor, val=3),
        ),
    )

    MixtureRegistry.add("test_mix", [("two_task", 1), ("three_task", 1)])

    sequence_length = {"inputs": 2, "targets": 2}
    passthrough_features = ["feature"]
    mix_ds = (
        MixtureRegistry.get("test_mix")
        .get_dataset(
            sequence_length,
            "train",
            seed=13,
            passthrough_features=passthrough_features,
        )
        .take(1000)
    )

    # output features are defined as "inputs" and "targets" by default.
    res = sum(int(item["feature"][0]) for item in mix_ds.as_numpy_iterator())
    self.assertEqual(res, 2481)

  def test_copy_pretokenized(self):
    @utils.map_over_dataset
    def _constant_preprocessor(unused_x, val):
      return {
          "targets": tf.constant([val], tf.int32),
          "targets_pretokenized": tf.constant(f"targets_{val}"),
          "inputs": tf.constant([val], tf.int32),
          "inputs_pretokenized": tf.constant(f"inputs_{val}"),
      }

    self.add_task(
        "two_task",
        self.function_source,
        preprocessors=(functools.partial(_constant_preprocessor, val=2),),
    )

    self.add_task(
        "three_task",
        self.function_source,
        preprocessors=(functools.partial(_constant_preprocessor, val=3),),
    )

    MixtureRegistry.add("test_mix", [("two_task", 1), ("three_task", 1)])

    sequence_length = {"inputs": 2, "targets": 2}

    mix_ds = (
        MixtureRegistry.get("test_mix")
        .get_dataset(sequence_length, "train", seed=13, copy_pretokenized=True)
        .take(1000)
    )
    inputs_pretokenized = set(
        ex["inputs_pretokenized"] for ex in mix_ds.as_numpy_iterator()
    )
    targets_pretokenized = set(
        ex["targets_pretokenized"] for ex in mix_ds.as_numpy_iterator()
    )
    self.assertCountEqual([b"inputs_2", b"inputs_3"], inputs_pretokenized)
    self.assertCountEqual([b"targets_2", b"targets_3"], targets_pretokenized)

    mix_ds = (
        MixtureRegistry.get("test_mix")
        .get_dataset(sequence_length, "train", seed=13, copy_pretokenized=False)
        .take(1000)
    )
    for ex in mix_ds.as_numpy_iterator():
      self.assertNoCommonElements(
          ["inputs_pretokenized", "targets_pretokenized"], ex.keys()
      )

  def test_get_rate_with_callable(self):
    def fn(t):
      self.assertEqual(t.name, "task4")
      return 42

    self.add_task("task4", self.function_source)
    task = TaskRegistry.get("task4")
    MixtureRegistry.add("test_mix5", [("task4", fn)])
    mix = MixtureRegistry.get("test_mix5")
    self.assertEqual(mix.get_rate(task), 42)

  def test_mixture_of_mixtures(self):
    self.add_task("task_a", self.function_source)
    self.add_task("task_b", self.function_source)
    self.add_task("task_c", self.function_source)
    MixtureRegistry.add("another_mix", [("task_a", 1), ("task_b", 1)])
    MixtureRegistry.add("supermix", [("another_mix", 1), ("task_c", 1)])
    supermix = MixtureRegistry.get("supermix")
    names = [task.name for task in supermix.tasks]
    self.assertEqual(names, ["task_a", "task_b", "task_c"])
    self.assertEqual(
        [supermix.get_rate(t) for t in supermix.tasks], [0.5, 0.5, 1]
    )

  def test_mixture_of_mixtures_dupe(self):
    self.add_task("task2_a", self.function_source)
    self.add_task("task2_b", self.function_source)
    self.add_task("task2_c", self.function_source)
    MixtureRegistry.add("yet_another_mix", [("task2_a", 1), ("task2_b", 1)])
    MixtureRegistry.add(
        "supermix_with_dupe",
        [("yet_another_mix", 1), ("task2_a", 1), ("task2_c", 1)],
    )
    supermix = MixtureRegistry.get("supermix_with_dupe")
    names = [task.name for task in supermix.tasks]
    self.assertEqual(names, ["task2_a", "task2_b", "task2_c"])
    self.assertEqual(
        [supermix.get_rate(t) for t in supermix.tasks], [1.5, 0.5, 1]
    )

  def test_mixture_with_sample_fn(self):
    def sequential_intereave(
        datasets: Sequence[tf.data.Dataset],
        rates: Sequence[float],
        sample_seed: Optional[int],
    ) -> tf.data.Dataset:
      """Sample function that simply concatenates two datasets."""
      del rates, sample_seed
      return datasets[0].concatenate(datasets[1])

    def gen_dataset(
        split, shuffle_files=False, seed=None, val: str = ""
    ) -> tf.data.Dataset:
      del split, shuffle_files, seed  # Need this to pass arg validation.
      return tf.data.Dataset.from_tensor_slices({
          "inputs": [[val]] * 3,
      })

    # Register two very simple tasks, each with 3 repeated string values.
    vocab = vocabularies.PassThroughVocabulary(0)
    tasks = []
    for task_name in ["first", "second"]:
      tasks.append(
          self.add_task(
              task_name,
              dataset_providers.FunctionDataSource(
                  dataset_fn=functools.partial(gen_dataset, val=task_name),
                  splits=["train"],
              ),
              preprocessors=[],
              output_features={
                  "inputs": dataset_providers.Feature(vocab, dtype=tf.string)
              },
          )
      )

    # Verify that by default, interleaving of datasets is random.
    MixtureRegistry.add("default_mix", [("first", 1), ("second", 1)])
    default_ds = MixtureRegistry.get("default_mix").get_dataset(
        None, "train", shuffle=False, seed=2, num_epochs=1
    )
    expected = [b"second", b"first", b"second", b"first", b"second", b"first"]
    actual = [x["inputs"] for x in default_ds.as_numpy_iterator()]
    self.assertEqual(expected, actual)

    # Verify that we can modify sampling function correctly.
    MixtureRegistry.add(
        "sequential_mix",
        [("first", 1), ("second", 1)],
        sample_fn=sequential_intereave,
    )
    sequential_ds = MixtureRegistry.get("sequential_mix").get_dataset(
        None, "train", shuffle=False, seed=2, num_epochs=1
    )
    expected = [b"first"] * 3 + [b"second"] * 3
    actual = [x["inputs"] for x in sequential_ds.as_numpy_iterator()]
    self.assertEqual(expected, actual)

  def test_mixture_with_no_tasks(self):
    with self.assertRaises(ValueError):
      MixtureRegistry.add("trivial_mixture", [])

  def test_mixture_of_tasks_with_different_features(self):
    self.add_task("task3_a", self.function_source)
    self.add_task(
        "task3_b",
        self.function_source,
        output_features={"different_feature": None},
    )
    with self.assertRaises(ValueError):
      MixtureRegistry.add(
          "mixture_with_different_features", ["task3_a", "task3_b"]
      )



class GetDatasetTest(parameterized.TestCase, tf.test.TestCase):

  def test_get_dataset_enc_dec_unpacked(self):
    mixture_or_task_name = "enc_dec_unpacked"
    x = [
        {"inputs": [7, 8, 5, 6, 9, 4, 3], "targets": [3, 9]},
        {"inputs": [8, 4], "targets": [4]},
        {"inputs": [5, 6, 7], "targets": [6, 5]},
    ]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=False)
    output_ds = dataset_providers.get_dataset(
        mixture_or_task_name=mixture_or_task_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split="train",
        shuffle=False,
        feature_converter=converter,
    )

    expected = [
        {
            "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
            "decoder_target_tokens": [3, 9, 1, 0, 0],
            "decoder_input_tokens": [0, 3, 9, 1, 0],
            "decoder_loss_weights": [1, 1, 1, 0, 0],
        },
        {
            "encoder_input_tokens": [8, 4, 1, 0, 0, 0, 0],
            "decoder_target_tokens": [4, 1, 0, 0, 0],
            "decoder_input_tokens": [0, 4, 1, 0, 0],
            "decoder_loss_weights": [1, 1, 0, 0, 0],
        },
        {
            "encoder_input_tokens": [5, 6, 7, 1, 0, 0, 0],
            "decoder_target_tokens": [6, 5, 1, 0, 0],
            "decoder_input_tokens": [0, 6, 5, 1, 0],
            "decoder_loss_weights": [1, 1, 1, 0, 0],
        },
    ]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  @parameterized.parameters(
      dict(
          task_name="enc_dec_partial_trim_both",
          task_feature_lengths={"inputs": 7, "targets": 2},
          expect_trim_inputs=True,
          expect_trim_targets=True,
      ),
      dict(
          task_name="enc_dec_partial_trim_targets",
          task_feature_lengths={"inputs": None, "targets": 2},
          expect_trim_inputs=False,
          expect_trim_targets=True,
      ),
      dict(
          task_name="enc_dec_partial_trim_inputs",
          task_feature_lengths={"inputs": 7, "targets": None},
          expect_trim_inputs=True,
          expect_trim_targets=False,
      ),
      dict(
          task_name="enc_dec_partial_trim_neither",
          task_feature_lengths={"inputs": None, "targets": None},
          expect_trim_inputs=False,
          expect_trim_targets=False,
      ),
      dict(
          task_name="enc_dec_partial_trim_nothing",
          task_feature_lengths=None,
          expect_trim_inputs=False,
          expect_trim_targets=False,
      ),
  )
  def test_partial_sequence_length(
      self,
      task_name,
      task_feature_lengths,
      expect_trim_inputs,
      expect_trim_targets,
  ):
    x = [
        {"inputs": [7, 8, 5, 6, 9, 4, 3], "targets": [3, 9]},
        {"inputs": [8, 4], "targets": [4]},
        {"inputs": [5, 6, 7], "targets": [6, 5]},
    ]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(task_name, dataset_fn=dataset_fn)
    # Unlike the other tests, don't use a feature converter. Instead, test the
    # task.get_dataset method directly, which is similar to how evaluation.py
    # infers feature lengths w/trimming.
    task = dataset_providers.get_mixture_or_task(task_name)
    output_ds = task.get_dataset(
        sequence_length=task_feature_lengths, shuffle=False
    )

    expected = [
        {
            "inputs": [7, 8, 5, 6, 9, 4, 3, 1],
            "targets": [3, 9, 1],
        },
        {
            "inputs": [8, 4, 1],
            "targets": [4, 1],
        },
        {
            "inputs": [5, 6, 7, 1],
            "targets": [6, 5, 1],
        },
    ]
    if expect_trim_inputs:
      expected[0]["inputs"] = [7, 8, 5, 6, 9, 4, 1]
    if expect_trim_targets:
      expected[0]["targets"] = [3, 1]
      expected[2]["targets"] = [6, 1]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  @parameterized.parameters(
      dict(
          task_name="enc_dec_multidim_trim_both",
          task_feature_lengths={"inputs": (2, 5), "targets": 2},
          expect_trim_inputs=True,
          expect_trim_targets=True,
      ),
      dict(
          task_name="enc_dec_multidim_trim_inputs",
          task_feature_lengths={"inputs": (2, 5), "targets": None},
          expect_trim_inputs=True,
          expect_trim_targets=False,
      ),
      dict(
          task_name="enc_dec_multidim_trim_targets",
          task_feature_lengths={"inputs": None, "targets": 2},
          expect_trim_inputs=False,
          expect_trim_targets=True,
      ),
      dict(
          task_name="enc_dec_no_multidim_trim",
          task_feature_lengths={"inputs": None, "targets": None},
          expect_trim_inputs=False,
          expect_trim_targets=False,
      ),
  )
  def test_multidimension_sequence_length(
      self,
      task_name,
      task_feature_lengths,
      expect_trim_inputs,
      expect_trim_targets,
  ):
    x = [
        {
            "inputs": [
                [7, 8, 5, 6, 9, 4, 3],
                [2, 3, 4, 5, 0, 0, 0],
                [6, 7, 1, 0, 0, 0, 0],
            ],
            "targets": [3, 9],
        },
        {"inputs": [[8, 4], [1, 0], [2, 3]], "targets": [4]},
        {"inputs": [[5, 6, 7]], "targets": [6, 5, 1]},
        {"inputs": [[7, 8, 9, 1, 2, 3, 4, 5, 6]], "targets": [10, 11, 1]},
    ]
    ds = tf.data.Dataset.from_generator(
        lambda: x,
        output_types={"inputs": tf.int32, "targets": tf.int32},
        output_shapes={"inputs": (None, None), "targets": (None,)},
    )
    dataset_fn = lambda split, shuffle_files: ds
    dataset_providers.TaskRegistry.add(
        task_name,
        source=dataset_providers.FunctionDataSource(
            dataset_fn=dataset_fn, splits=["train", "validation"]
        ),
        preprocessors=[
            dataset_providers.CacheDatasetPlaceholder(),
        ],
        output_features={
            "inputs": dataset_providers.Feature(
                test_utils.sentencepiece_vocab(), rank=2
            ),
            "targets": dataset_providers.Feature(
                test_utils.sentencepiece_vocab()
            ),
        },
        metric_fns=[],
    )
    # Unlike the other tests, don't use a feature converter. Instead, test the
    # task.get_dataset method directly, which is similar to how evaluation.py
    # infers feature lengths w/trimming.
    task = dataset_providers.get_mixture_or_task(task_name)
    output_ds = task.get_dataset(
        sequence_length=task_feature_lengths, shuffle=False
    )

    expected = copy.deepcopy(x)
    if expect_trim_inputs:
      expected[0]["inputs"] = [[7, 8, 5, 6, 9], [2, 3, 4, 5, 0]]
      expected[1]["inputs"] = [[8, 4], [1, 0]]
      expected[3]["inputs"] = [[7, 8, 9, 1, 2]]
    if expect_trim_targets:
      expected[2]["targets"] = [6, 5]
      expected[3]["targets"] = [10, 11]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  def test_get_dataset_enc_dec_packed(self):
    mixture_or_task_name = "enc_dec_packed"
    x = [
        {"inputs": [7, 8, 5, 6, 9, 4, 3], "targets": [3, 9]},
        {"inputs": [8, 4], "targets": [4]},
        {"inputs": [5, 6, 7], "targets": [6, 5]},
    ]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=True)
    output_ds = dataset_providers.get_dataset(
        mixture_or_task_name=mixture_or_task_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split="train",
        shuffle=False,
        feature_converter=converter,
    )

    expected = [
        {
            # Example 1 is trimmed
            "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
            "encoder_segment_ids": [1, 1, 1, 1, 1, 1, 1],
            "encoder_positions": [0, 1, 2, 3, 4, 5, 6],
            "decoder_target_tokens": [3, 9, 1, 0, 0],
            "decoder_input_tokens": [0, 3, 9, 0, 0],
            "decoder_loss_weights": [1, 1, 1, 0, 0],
            "decoder_segment_ids": [1, 1, 1, 0, 0],
            "decoder_positions": [0, 1, 2, 0, 0],
        },
        {
            # Example 2 and 3 are packed together
            "encoder_input_tokens": [8, 4, 1, 5, 6, 7, 1],
            "encoder_segment_ids": [1, 1, 1, 2, 2, 2, 2],
            "encoder_positions": [0, 1, 2, 0, 1, 2, 3],
            "decoder_target_tokens": [4, 1, 6, 5, 1],
            "decoder_input_tokens": [0, 4, 0, 6, 5],
            "decoder_loss_weights": [1, 1, 1, 1, 1],
            "decoder_segment_ids": [1, 1, 2, 2, 2],
            "decoder_positions": [0, 1, 0, 1, 2],
        },
    ]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  def test_get_dataset_both_train_and_validation_splits(self):
    mixture_or_task_name = "both_train_and_validation_splits"
    x_train = [{"inputs": [7, 8, 5, 6, 9, 4, 3], "targets": [3, 9]}]
    x_val = [{"inputs": [8, 4], "targets": [4]}]
    datasets = {
        "train": create_default_dataset(x_train),
        "validation": create_default_dataset(x_val),
    }
    dataset_fn = lambda split, shuffle_files: datasets[split]
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    output_ds = {}
    for split in ["train", "validation"]:
      converter = feature_converters.EncDecFeatureConverter(pack=False)
      output_ds[split] = dataset_providers.get_dataset(
          mixture_or_task_name=mixture_or_task_name,
          task_feature_lengths=task_feature_lengths,
          dataset_split=split,
          shuffle=False,
          feature_converter=converter,
      )

    expected_train = {
        "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
        "decoder_target_tokens": [3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }
    expected_val = {
        "encoder_input_tokens": [8, 4, 1, 0, 0, 0, 0],
        "decoder_target_tokens": [4, 1, 0, 0, 0],
        "decoder_input_tokens": [0, 4, 1, 0, 0],
        "decoder_loss_weights": [1, 1, 0, 0, 0],
    }
    expected_dtypes = {feat: tf.int32 for feat in expected_train.keys()}
    assert_dataset(
        output_ds["train"], expected_train, expected_dtypes=expected_dtypes
    )
    assert_dataset(
        output_ds["validation"], expected_val, expected_dtypes=expected_dtypes
    )

  def test_get_dataset_enc_dec_sharded(self):
    mixture_or_task_name = "enc_dec_sharded"
    x = [
        {"inputs": [7, 8, 5, 6, 9, 4, 3], "targets": [3, 9]},
        {"inputs": [8, 4], "targets": [4]},
        {"inputs": [5, 6, 7], "targets": [6, 5]},
    ]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=False)
    shard_info = dataset_providers.ShardInfo(index=0, num_shards=2)
    output_ds = dataset_providers.get_dataset(
        mixture_or_task_name=mixture_or_task_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split="train",
        shuffle=False,
        feature_converter=converter,
        shard_info=shard_info,
    )

    # Example index 1 should not be present in the sharded dataset.
    expected = [
        {
            "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
            "decoder_target_tokens": [3, 9, 1, 0, 0],
            "decoder_input_tokens": [0, 3, 9, 1, 0],
            "decoder_loss_weights": [1, 1, 1, 0, 0],
        },
        {
            "encoder_input_tokens": [5, 6, 7, 1, 0, 0, 0],
            "decoder_target_tokens": [6, 5, 1, 0, 0],
            "decoder_input_tokens": [0, 6, 5, 1, 0],
            "decoder_loss_weights": [1, 1, 1, 0, 0],
        },
    ]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  def test_get_dataset_enc_dec_sharded_and_packed(self):
    mixture_or_task_name = "enc_dec_sharded_and_packed"
    x = [
        {"inputs": [7, 8], "targets": [3, 9]},
        {"inputs": [8, 4], "targets": [4]},
        {"inputs": [5, 6, 7], "targets": [6]},
    ]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=True)
    shard_info = dataset_providers.ShardInfo(index=0, num_shards=2)
    output_ds = dataset_providers.get_dataset(
        mixture_or_task_name=mixture_or_task_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split="train",
        shuffle=False,
        feature_converter=converter,
        shard_info=shard_info,
    )

    # Packing should be done after the sharding.
    expected = {
        "encoder_input_tokens": [7, 8, 1, 5, 6, 7, 1],
        "encoder_segment_ids": [1, 1, 1, 2, 2, 2, 2],
        "encoder_positions": [0, 1, 2, 0, 1, 2, 3],
        "decoder_target_tokens": [3, 9, 1, 6, 1],
        "decoder_input_tokens": [0, 3, 9, 0, 6],
        "decoder_loss_weights": [1, 1, 1, 1, 1],
        "decoder_segment_ids": [1, 1, 1, 2, 2],
        "decoder_positions": [0, 1, 2, 0, 1],
    }
    expected_dtypes = {feat: tf.int32 for feat in expected.keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)


def register_dummy_task(
    task_name: str,
    dataset_fn: Callable[[str, str], tf.data.Dataset],
    output_feature_names: Sequence[str] = ("inputs", "targets"),
) -> None:
  """Register a dummy task for GetDatasetTest."""
  dataset_providers.TaskRegistry.add(
      task_name,
      source=dataset_providers.FunctionDataSource(
          dataset_fn=dataset_fn, splits=["train", "validation"]
      ),
      preprocessors=[
          dataset_providers.CacheDatasetPlaceholder(),
          preprocessors.append_eos_after_trim,
      ],
      output_features={
          feat: dataset_providers.Feature(test_utils.sentencepiece_vocab())
          for feat in output_feature_names
      },
      metric_fns=[],
  )


class TfdsDataSourceTest(test_utils.FakeTaskTest):

  def test_tfds_splits(self):
    self.assertSameElements(
        ["train", "validation"],
        dataset_providers.TfdsDataSource(tfds_name="fake:0.0.0").splits,
    )
    self.assertSameElements(
        ["validation"],
        dataset_providers.TfdsDataSource(
            tfds_name="fake:0.0.0", splits=["validation"]
        ).splits,
    )
    self.assertSameElements(
        ["validation"],
        dataset_providers.TfdsDataSource(
            tfds_name="fake:0.0.0", splits={"validation": "train"}
        ).splits,
    )
    self.assertSameElements(
        ["train"],
        dataset_providers.TfdsDataSource(
            splits={
                "train": utils.TfdsSplit(
                    dataset="fake:0.0.0", split="validation"
                )
            }
        ).splits,
    )

  def test_tfds_source_splits(self):
    default_splits_src = dataset_providers.TfdsDataSource("fake:0.0.0")
    self.assertSameElements(["train", "validation"], default_splits_src.splits)

    validation_split_src = dataset_providers.TfdsDataSource(
        "fake:0.0.0", splits=["validation"]
    )
    self.assertSameElements(["validation"], validation_split_src.splits)

    sliced_split_src = dataset_providers.TfdsDataSource(
        "fake:0.0.0", splits={"validation": "train[0:1%]"}
    )
    self.assertSameElements(["validation"], sliced_split_src.splits)



class FunctionDataSourceTest(test_utils.FakeTaskTest):

  def test_function_source_signature(self):
    # Good signatures.
    def good_fn(split, shuffle_files):
      del split
      del shuffle_files

    dataset_providers.FunctionDataSource(good_fn, splits=("train",))

    def default_good_fn(split, shuffle_files=False):
      del split
      del shuffle_files

    dataset_providers.FunctionDataSource(default_good_fn, splits=("train",))

    def seed_fn(split, shuffle_files=True, seed=0):
      del split
      del shuffle_files
      del seed

    dataset_providers.FunctionDataSource(seed_fn, splits=("train",))

    def extra_kwarg_good_fn(split, shuffle_files, unused_kwarg=True):
      del split
      del shuffle_files

    dataset_providers.FunctionDataSource(extra_kwarg_good_fn, splits=("train",))

    class GoodProtocol(dataset_providers.DatasetFnCallable):

      def __call__(self, split, shuffle_files, seed=None):
        del split, shuffle_files, seed

    dataset_providers.FunctionDataSource(GoodProtocol(), splits=("train",))

    # Bad signatures.
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "'missing_shuff' must have initial args ('split',"
            " 'shuffle_files'), got: ('split',)"
        ),
    ):

      def missing_shuff(split):
        del split

      dataset_providers.FunctionDataSource(missing_shuff, splits=("train",))

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "'missing_split' must have initial args ('split',"
            " 'shuffle_files'), got: ('shuffle_files',)"
        ),
    ):

      def missing_split(shuffle_files):
        del shuffle_files

      dataset_providers.FunctionDataSource(missing_split, splits=("train",))

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "'extra_pos_arg' may only have positional args ('split', "
            "'shuffle_files'), got: ('split', 'shuffle_files', 'unused_arg')"
        ),
    ):

      def extra_pos_arg(split, shuffle_files, unused_arg):
        del split
        del shuffle_files

      dataset_providers.FunctionDataSource(extra_pos_arg, splits=("train",))



class FileDataSourceTest(test_utils.FakeTaskTest):

  def test_str(self):
    fds = dataset_providers.FileDataSource(
        read_file_fn=lambda x: tf.data.Dataset.from_tensor_slices([x]),
        split_to_filepattern={"train": "filepattern"},
        file_shuffle_buffer_size=2,
        cycle_length=42,
    )
    self.assertEqual(str(fds), "FileDataSource({'train': 'filepattern'})")

  def test_repr(self):
    fds = dataset_providers.FileDataSource(
        read_file_fn=lambda x: tf.data.Dataset.from_tensor_slices([x]),
        split_to_filepattern={"train": "filepattern"},
        file_shuffle_buffer_size=2,
        cycle_length=42,
    )
    self.assertEqual(
        fds.__repr__(),
        "FileDataSource("
        "split_to_filepattern={'train': 'filepattern'},"
        " num_input_examples=None,"
        " caching_permitted=True,"
        " file_shuffle_buffer_size=2,"
        " cycle_length=42,"
        " block_length=16)",
    )

  @mock.patch.object(dataset_providers, "_list_files")
  def test_file_data_source_shuffle_buffer_low(self, mock_list_files):
    mock_list_files.return_value = [f"{i}" for i in range(20)]
    fds = dataset_providers.FileDataSource(
        read_file_fn=lambda x: tf.data.Dataset.from_tensor_slices([x]),
        split_to_filepattern={"train": "filepattern"},
        file_shuffle_buffer_size=2,
    )
    for _ in range(10):
      ds = [
          d.decode()
          for d in tfds.as_numpy(
              fds.get_dataset("train", shuffle=True, seed=23)
          )
      ]
      self.assertListEqual(
          ds,
          [  # Not a great shuffle.
              "0",
              "2",
              "1",
              "4",
              "5",
              "3",
              "7",
              "6",
              "9",
              "10",
              "11",
              "8",
              "13",
              "14",
              "12",
              "16",
              "15",
              "18",
              "17",
              "19",
          ],
      )

  @mock.patch.object(dataset_providers, "_list_files")
  def test_file_data_source_shuffle_buffer_full(self, mock_list_files):
    mock_list_files.return_value = [f"{i}" for i in range(20)]
    fds = dataset_providers.FileDataSource(
        read_file_fn=lambda x: tf.data.Dataset.from_tensor_slices([x]),
        split_to_filepattern={"train": "filepattern"},
        file_shuffle_buffer_size=None,
    )
    for _ in range(10):
      ds = [
          d.decode()
          for d in tfds.as_numpy(
              fds.get_dataset("train", shuffle=True, seed=23)
          )
      ]
      self.assertListEqual(
          ds,
          [  # Good shuffle.
              "2",
              "13",
              "12",
              "19",
              "15",
              "5",
              "9",
              "1",
              "6",
              "8",
              "3",
              "0",
              "10",
              "4",
              "14",
              "7",
              "16",
              "17",
              "18",
              "11",
          ],
      )



class ProtoDataSource(test_utils.FakeTaskTest):

  def test_str(self):
    self.assertEqual(
        str(self.proto_source),
        f"ProtoDataSource({{'train': '{self.test_data_dir}/train.tfrecord*'}})",
    )

  def test_repr(self):
    expected = (
        "ProtoDataSource(split_to_filepattern={'train':"
        f" '{self.test_data_dir}/train.tfrecord*'}}, num_input_examples=None,"
        " caching_permitted=True, file_shuffle_buffer_size=None,"
        " cycle_length=16, block_length=16)"
    )
    self.assertEqual(self.proto_source.__repr__(), expected)



class TFExampleDataSource(test_utils.FakeTaskTest):

  def test_str(self):
    expected = (
        "TFExampleDataSource(split_to_filepattern="
        f"{{'train': '{self.test_data_dir}/train.tfrecord*'}},"
        " feature_description={'prefix': FixedLenFeature(shape=[],"
        " dtype=tf.string, default_value=None), 'suffix':"
        " FixedLenFeature(shape=[], dtype=tf.string, default_value=None)})"
    )
    self.assertEqual(str(self.tf_example_source), expected)



if __name__ == "__main__":
  absltest.main()
