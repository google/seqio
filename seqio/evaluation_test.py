# Copyright 2022 The SeqIO Authors.
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

"""Tests for seqio.evaluation."""
# pylint:disable=g-bare-generic,g-long-lambda

import concurrent
import functools
from typing import Any, Callable, Sequence, Mapping, Optional, Tuple
from unittest import mock

import numpy as np
from seqio import dataset_providers
from seqio import evaluation
from seqio import metrics as metrics_lib
from seqio import preprocessors
from seqio import test_utils
from seqio import utils
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

Evaluator = evaluation.Evaluator

# For faster testing.
tf.compat.v1.enable_eager_execution()


def _string_label_to_class_id_postprocessor(string_label,
                                            label_classes,
                                            default=-1,
                                            **unused_kwargs):
  """Returns index of string_label in label_classes or default if not found."""
  if string_label in label_classes:
    return label_classes.index(string_label)
  else:
    return default


def _sequence_accuracy_metric(targets, predictions):
  seq_acc = 100 * np.mean([p == t for p, t in zip(predictions, targets)])
  return {"sequence_accuracy": seq_acc}


def _fake_aux_values_metric(targets, predictions, aux_values):
  del predictions, targets
  scores = aux_values["scores"]
  fake_metric = sum([int(score == 0.1) for score in scores])
  return {"fake_metric": fake_metric}


def _accuracy_metric(targets, predictions):
  acc = 100 * np.mean([np.all(p == t) for p, t in zip(predictions, targets)])
  return {"accuracy": acc}


def _sum_scores_metric(targets, scores):
  weights = [sum(ord(c) for c in t) for t in targets]
  return {"total_score": (np.array(scores) * np.array(weights)).sum()}


def _sum_scores_metric_with_intermediates(targets, scores):
  _, intermediates = scores
  scores = intermediates["score"]
  weights = [sum(ord(c) for c in t) for t in targets]
  return {
      "total_score_with_intermediates":
          (np.array(scores) * np.array(weights)).sum()
  }


def register_dummy_task(task_name: str,
                        dataset_fn: Callable[[str, bool, Optional[int]],
                                             tf.data.Dataset],
                        output_feature_names: Sequence[str] = ("inputs",
                                                               "targets"),
                        preprocessor=preprocessors.append_eos,
                        postprocess_fn=None,
                        metrics_fn=None) -> dataset_providers.Task:
  """Register a dummy task for GetDatasetTest."""
  return dataset_providers.TaskRegistry.add(
      task_name,
      source=dataset_providers.FunctionDataSource(
          dataset_fn=dataset_fn, splits=["train", "validation"]),
      preprocessors=[preprocessor],
      postprocess_fn=postprocess_fn,
      output_features={
          # Mock the sentencepiece vocabulary.
          feat: dataset_providers.Feature(mock.Mock(eos_id=True))
          for feat in output_feature_names
      },
      metric_fns=metrics_fn)


def get_mocked_task(name: str = "mocked_test",
                    predict_metric_fns: Sequence[Callable] = (
                        _sequence_accuracy_metric,),
                    predict_with_aux_metric_fns: Sequence[Callable] = (),
                    score_metric_fns: Sequence[Callable] = (),
                    target_field_name: str = "targets") -> mock.Mock:
  task = mock.Mock()
  task.name = name
  task.score_metric_fns = list(score_metric_fns)
  task.predict_metric_fns = list(predict_metric_fns)
  task.predict_with_aux_metric_fns = list(predict_with_aux_metric_fns)
  task.metric_fns = (
      list(predict_metric_fns) + list(score_metric_fns) +
      list(predict_with_aux_metric_fns))
  # Identity postprocess function
  task.postprocess_fn = lambda d, example, is_target: d
  task.metric_objs = [
      metrics_lib.LegacyMetric.empty(mf, task.postprocess_fn)
      for mf in task.metric_fns
  ]

  mock_vocab = mock.Mock()
  task.output_features = {
      target_field_name: dataset_providers.Feature(mock_vocab)
  }
  return task


def _task_from_tensor_slices(name, tensor_slices, label_classes):
  return dataset_providers.Task(
      name,
      dataset_providers.FunctionDataSource(
          lambda split, shuffle_files: tf.data.Dataset.from_tensor_slices(
              tensor_slices),
          splits=("validation")),  # pytype: disable=wrong-arg-types
      preprocessors=[
          utils.map_over_dataset(
              lambda ex: {
                  "inputs": tf.range(ex["inputs_lengths"]),
                  "targets": tf.range(ex["targets_lengths"]),
                  "targets_pretokenized": ex["targets_pretokenized"],
              })
      ],
      postprocess_fn=functools.partial(
          _string_label_to_class_id_postprocessor, label_classes=label_classes),
      output_features={
          "inputs": dataset_providers.Feature(mock.Mock()),
          "targets": dataset_providers.Feature(mock.Mock())
      })


class EvaluationTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.uncalled_fn = mock.Mock()

  def tearDown(self):
    super().tearDown()
    self.uncalled_fn.assert_not_called()

  def assertDictClose(self, a, b, delta=None, places=None):
    self.assertCountEqual(a.keys(), b.keys())
    for k in a:
      try:
        self.assertAlmostEqual(a[k], b[k], delta=delta, places=places)
      except AssertionError as e:
        raise AssertionError(str(e) + " for key '%s'" % k) from e

  def test_get_valid_eval_tasks(self):
    task_no_metrics = mock.Mock(splits=("train", "validation"), metric_fns=[],
                                metric_objs=[])
    task_no_split = mock.Mock(splits=("train"), metric_fns=[lambda x: x])
    valid_task = mock.Mock(
        splits=("train", "validation"), metric_fns=[lambda x: x])
    self.assertSequenceEqual(
        evaluation.get_valid_eval_tasks(
            [task_no_metrics, task_no_split, valid_task], "validation"),
        [valid_task])

  def test_get_targets_and_examples(self):
    task1 = _task_from_tensor_slices(
        "task1", {
            "inputs_lengths": [3, 2],
            "targets_lengths": [2, 3],
            "targets_pretokenized": ["e6", "e5"],
        }, ("e4", "e5", "e6"))
    task2 = _task_from_tensor_slices(
        "task2", {
            "inputs_lengths": [1],
            "targets_lengths": [4],
            "targets_pretokenized": ["e4"],
        }, ("e2", "e3", "e4"))
    cached_targets, cached_task_datasets, max_sequence_length = (
        evaluation.get_targets_and_examples(
            [task1, task2],
            lambda t: t.get_dataset(
                split="validation", sequence_length=None, shuffle=False),
            sequence_dims={}))

    self.assertDictEqual({"task1": [2, 1], "task2": [2]}, cached_targets)
    self.assertDictEqual({"inputs": 3, "targets": 4}, max_sequence_length)
    self.assertCountEqual(["task1", "task2"], cached_task_datasets.keys())
    self.assertLen(cached_task_datasets["task1"], 2)
    self.assertLen(cached_task_datasets["task2"], 1)
    expected_task1_examples = [{
        "inputs": [0, 1, 2],
        "targets": [0, 1],
        "targets_pretokenized": "e6"
    }, {
        "inputs": [0, 1],
        "targets": [0, 1, 2],
        "targets_pretokenized": "e5"
    }]
    expected_task2_examples = [
        {
            "inputs": [0],
            "targets": [0, 1, 2, 3],
            "targets_pretokenized": "e4"
        },
    ]
    test_utils.assert_dataset(cached_task_datasets["task1"],
                              expected_task1_examples)
    test_utils.assert_dataset(cached_task_datasets["task2"],
                              expected_task2_examples)

  def test_get_targets_and_examples_num_examples(self):

    task1 = _task_from_tensor_slices(
        "task1", {
            "inputs_lengths": [3, 2, 4],
            "targets_lengths": [2, 3, 4],
            "targets_pretokenized": ["e6", "e5", "e4"],
        }, ("e4", "e5", "e6"))
    task2 = _task_from_tensor_slices(
        "task2", {
            "inputs_lengths": [1],
            "targets_lengths": [4],
            "targets_pretokenized": ["e4"],
        }, ("e2", "e3", "e4"))
    cached_targets, cached_task_datasets, max_sequence_length = (
        evaluation.get_targets_and_examples(
            [task1, task2],
            lambda t: t.get_dataset(
                split="validation", sequence_length=None, shuffle=False),
            sequence_dims={},
            num_examples=2))

    self.assertDictEqual({"task1": [2, 1], "task2": [2]}, cached_targets)
    self.assertDictEqual({"inputs": 3, "targets": 4}, max_sequence_length)
    self.assertCountEqual(["task1", "task2"], cached_task_datasets.keys())
    self.assertLen(cached_task_datasets["task1"], 2)
    self.assertLen(cached_task_datasets["task2"], 1)
    expected_task1_examples = [{
        "inputs": [0, 1, 2],
        "targets": [0, 1],
        "targets_pretokenized": "e6"
    }, {
        "inputs": [0, 1],
        "targets": [0, 1, 2],
        "targets_pretokenized": "e5"
    }]
    expected_task2_examples = [
        {
            "inputs": [0],
            "targets": [0, 1, 2, 3],
            "targets_pretokenized": "e4"
        },
    ]
    test_utils.assert_dataset(cached_task_datasets["task1"],
                              expected_task1_examples)
    test_utils.assert_dataset(cached_task_datasets["task2"],
                              expected_task2_examples)

    cached_targets, cached_task_datasets, max_sequence_length = (
        evaluation.get_targets_and_examples(
            [task1, task2],
            lambda t: t.get_dataset(
                split="validation", sequence_length=None, shuffle=False),
            sequence_dims={},
            num_examples=3))

    self.assertDictEqual({"task1": [2, 1, 0], "task2": [2]}, cached_targets)
    self.assertDictEqual({"inputs": 4, "targets": 4}, max_sequence_length)
    self.assertCountEqual(["task1", "task2"], cached_task_datasets.keys())
    self.assertLen(cached_task_datasets["task1"], 3)
    self.assertLen(cached_task_datasets["task2"], 1)
    expected_task1_examples = [{
        "inputs": [0, 1, 2],
        "targets": [0, 1],
        "targets_pretokenized": "e6"
    }, {
        "inputs": [0, 1],
        "targets": [0, 1, 2],
        "targets_pretokenized": "e5"
    }, {
        "inputs": [0, 1, 2, 3],
        "targets": [0, 1, 2, 3],
        "targets_pretokenized": "e4"
    }]
    expected_task2_examples = [
        {
            "inputs": [0],
            "targets": [0, 1, 2, 3],
            "targets_pretokenized": "e4"
        },
    ]
    test_utils.assert_dataset(cached_task_datasets["task1"],
                              expected_task1_examples)
    test_utils.assert_dataset(cached_task_datasets["task2"],
                              expected_task2_examples)

  def test_get_targets_and_examples_nondefault_sequence_dim(self):

    def _task_from_tensor_slices_rank2(name, tensor_slices, label_classes):
      return dataset_providers.Task(
          name,
          dataset_providers.FunctionDataSource(
              lambda split, shuffle_files: tf.data.Dataset.from_tensor_slices(
                  tensor_slices),
              splits=("validation")),  # pytype: disable=wrong-arg-types
          preprocessors=[
              utils.map_over_dataset(
                  lambda ex: {
                      "inputs": [
                          tf.range(ex["inputs_lengths"]),
                          tf.range(ex["inputs_lengths"])
                      ],
                      "targets": tf.range(ex["targets_lengths"]),
                      "targets_pretokenized": ex["targets_pretokenized"],
                  })
          ],
          postprocess_fn=functools.partial(
              _string_label_to_class_id_postprocessor,
              label_classes=label_classes),
          output_features={
              "inputs": dataset_providers.Feature(mock.Mock(), rank=2),
              "targets": dataset_providers.Feature(mock.Mock())
          })

    task1 = _task_from_tensor_slices_rank2(
        "task1", {
            "inputs_lengths": [3, 2],
            "targets_lengths": [2, 3],
            "targets_pretokenized": ["e6", "e5"],
        }, ("e4", "e5", "e6"))

    task2 = _task_from_tensor_slices_rank2(
        "task2", {
            "inputs_lengths": [1],
            "targets_lengths": [4],
            "targets_pretokenized": ["e4"],
        }, ("e2", "e3", "e4"))

    cached_targets, cached_task_datasets, max_sequence_length = (
        evaluation.get_targets_and_examples(
            [task1, task2],
            lambda t: t.get_dataset(
                split="validation", sequence_length=None, shuffle=False),
            sequence_dims={"inputs": 1}))

    self.assertDictEqual({"task1": [2, 1], "task2": [2]}, cached_targets)
    self.assertDictEqual({"inputs": 3, "targets": 4}, max_sequence_length)
    self.assertCountEqual(["task1", "task2"], cached_task_datasets.keys())
    self.assertLen(cached_task_datasets["task1"], 2)
    self.assertLen(cached_task_datasets["task2"], 1)
    expected_task1_examples = [{
        "inputs": [[0, 1, 2], [0, 1, 2]],
        "targets": [0, 1],
        "targets_pretokenized": "e6"
    }, {
        "inputs": [[0, 1], [0, 1]],
        "targets": [0, 1, 2],
        "targets_pretokenized": "e5"
    }]
    expected_task2_examples = [
        {
            "inputs": [[0], [0]],
            "targets": [0, 1, 2, 3],
            "targets_pretokenized": "e4"
        },
    ]
    test_utils.assert_dataset(cached_task_datasets["task1"],
                              expected_task1_examples)
    test_utils.assert_dataset(cached_task_datasets["task2"],
                              expected_task2_examples)

  def _evaluate_single_task(self,
                            task,
                            loggers=(),
                            target_field_name="targets",
                            has_aux_values=False):
    id_to_vocab = {5: "e5", 6: "e6", 7: "e7"}
    mock_vocab = task.output_features[target_field_name].vocabulary
    # Define a dummy decoding logic.
    mock_vocab.decode = lambda ids: " ".join([id_to_vocab[i] for i in ids])

    def mock_init(self):
      mock_ds = tf.data.Dataset.from_tensor_slices(
          ["e5 e6", "e6",
           "e7"]).map(lambda x: {f"{target_field_name}_pretokenized": x})
      self._cached_model_datasets = {task.name: mock_ds}
      self._cached_task_datasets = {task.name: mock_ds}
      self._eval_tasks = [task]
      self._loggers = loggers
      self._metrics_future = None
      self._metrics_executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=1)
      self._target_field_name = target_field_name

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()  # pytype: disable=missing-parameter

      # A dummy score function that always returns the same output.
      def predict_fn(
          ds: tf.data.Dataset,
          model_feature_shapes: Optional[Mapping[str, int]] = None
      ) -> evaluation.PredictFnReturnType:
        del ds, model_feature_shapes
        return ([(0, [5, 6]), (1, [7]),
                 (2, [7])] if task.predict_metric_fns else self.uncalled_fn)

      def predict_with_aux_fn(
          ds: tf.data.Dataset,
          model_feature_shapes: Optional[Mapping[str, int]] = None
      ) -> evaluation.PredictFnReturnType:
        del ds, model_feature_shapes

        indices_and_predictions = ([
            (0, [5, 6]), (1, [7]), (2, [7])
        ] if task.predict_metric_with_aux_fns else self.uncalled_fn)

        aux_values = {
            "scores": [0.1, 0.2, 0.2]
        } if task.predict_metric_with_aux_fns else self.uncalled_fn

        return indices_and_predictions, aux_values

      def score_fn(
          ds: tf.data.Dataset,
          model_feature_shapes: Optional[Mapping[str, int]] = None
      ) -> Sequence[Tuple[int, float]]:
        del ds, model_feature_shapes
        return ([(1, 1), (0, 2),
                 (2, 3)] if task.score_metric_fns else self.uncalled_fn)

      if not has_aux_values:
        predict_with_aux_fn = None

      all_metrics, _ = evaluator.evaluate(
          compute_metrics=True,
          predict_fn=predict_fn,
          score_fn=score_fn,
          predict_with_aux_fn=predict_with_aux_fn,
          step=42)
      return all_metrics.result(), evaluator

  def test_evaluate_single_task_predict(self):
    task = get_mocked_task(
        predict_metric_fns=[_sequence_accuracy_metric], score_metric_fns=[])
    all_metrics, _ = self._evaluate_single_task(task)
    self.assertDictClose({"sequence_accuracy": 2.0 / 3 * 100},
                         all_metrics[task.name])

  def test_evaluate_single_task_score(self):
    task = get_mocked_task(
        predict_metric_fns=[], score_metric_fns=[_sum_scores_metric])
    all_metrics, _ = self._evaluate_single_task(task)
    self.assertDictClose({"total_score": 1305}, all_metrics[task.name])

  def test_evaluate_single_task_both(self):
    task = get_mocked_task(
        predict_metric_fns=[_sequence_accuracy_metric],
        score_metric_fns=[_sum_scores_metric])
    all_metrics, _ = self._evaluate_single_task(task)
    expected = {"sequence_accuracy": 2.0 / 3 * 100, "total_score": 1305}
    self.assertDictClose(expected, all_metrics[task.name])

  def test_evaluate_using_aux_score(self):
    task = get_mocked_task(
        predict_with_aux_metric_fns=[_fake_aux_values_metric])
    all_metrics, _ = self._evaluate_single_task(task, has_aux_values=True)
    self.assertEqual(1, all_metrics[task.name]["fake_metric"])

  def test_aux_scores_sorted_with_tokens(self):
    """Tests that the correct aux scores correspond with the correct tokens."""

    def _aux_metric_that_cares_about_order(targets, predictions, aux_values):
      del targets
      result = 0
      for i in range(len(predictions)):
        if (int(predictions[i]) == aux_values["scores"][i] ==
            aux_values["other_key"][i]):
          result += 1
      return {"fake_result": result}

    task = get_mocked_task(
        predict_with_aux_metric_fns=[_aux_metric_that_cares_about_order],
        predict_metric_fns=[])

    id_to_vocab = {5: "5", 6: "6", 7: "7"}
    mock_vocab = task.output_features["targets"].vocabulary
    # Define a dummy decoding logic.
    mock_vocab.decode = lambda ids: " ".join([id_to_vocab[i] for i in ids])

    def mock_init(self):
      mock_ds = tf.data.Dataset.from_tensor_slices(
          ["5", "6", "7"]).map(lambda x: {"targets_pretokenized": x})
      self._cached_model_datasets = {task.name: mock_ds}
      self._cached_task_datasets = {task.name: mock_ds}
      self._eval_tasks = [task]
      self._loggers = []
      self._metrics_future = None
      self._metrics_executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=1)
      self._target_field_name = "targets"

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()  # pytype: disable=missing-parameter

      def predict_with_aux_fn(
          ds: tf.data.Dataset,
          model_feature_shapes: Optional[Mapping[str, int]] = None
      ) -> evaluation.PredictFnReturnType:
        del ds, model_feature_shapes

        indices_and_predictions = [(2, [7]), (0, [5]), (1, [6])]
        aux_values = {
            "scores": [7, 5, 6],
            "other_key": [7, 5, 6],
        }

        return indices_and_predictions, aux_values

      all_metrics, _ = evaluator.evaluate(
          compute_metrics=True,
          predict_fn=self.uncalled_fn,
          score_fn=self.uncalled_fn,
          predict_with_aux_fn=predict_with_aux_fn,
          step=42)

      # This only passes if the scores are perfectly aligned with the
      # predictions.
      self.assertEqual(3, all_metrics.result()[task.name]["fake_result"])

  def test_evaluate_single_task_predict_target_field_name(self):
    task = get_mocked_task(
        predict_metric_fns=[_sequence_accuracy_metric],
        score_metric_fns=[],
        target_field_name="foo")
    all_metrics, _ = self._evaluate_single_task(task, target_field_name="foo")
    self.assertDictClose({"sequence_accuracy": 2.0 / 3 * 100},
                         all_metrics[task.name])

  def test_evaluate_single_task_with_loggers(self):
    loggers = (mock.Mock(), mock.Mock())

    task = get_mocked_task(
        predict_metric_fns=[_sequence_accuracy_metric],
        score_metric_fns=[_sum_scores_metric])
    task.postprocess_fn = (lambda x, example, is_target: x
                           if is_target else x.replace("e6", "e7"))
    task.metric_objs = [
        metrics_lib.LegacyMetric.empty(mf, task.postprocess_fn)
        for mf in task.metric_fns
    ]

    _, evaluator = self._evaluate_single_task(task, loggers=loggers)
    metrics = {
        "sequence_accuracy": metrics_lib.Scalar(1 / 3 * 100),
        "total_score": metrics_lib.Scalar(1305)
    }
    for logger in loggers:
      logger.assert_called_once_with(
          task_name=task.name,
          step=42,
          metrics=metrics,
          dataset=evaluator._cached_task_datasets[task.name],
          targets=["e5 e6", "e6", "e7"],
          inferences={
              "prediction": ["e5 e7", "e7", "e7"],
              "score": [2, 1, 3],
              "output": ["e5 e6", "e7", "e7"]
          })

  def test_initialize_loggers(self):
    task_name = "initialize_loggers"
    ds = tf.data.Dataset.from_tensors({
        "inputs": [7, 8],
        "targets": [3, 9],
        "targets_pretokenized": "ex 1"
    })
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        # `append_eos_after_trim` has an optional sequence_length arg
        preprocessor=preprocessors.append_eos_after_trim,
        metrics_fn=[_sequence_accuracy_metric])

    feature_converter = evaluation.EncDecFeatureConverter(pack=False)

    logger_cls = (mock.Mock(), mock.Mock())

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "'log_dir' must be provided to `Evaluator` if `logger_cls` is "
        "non-empty."):
      Evaluator(
          mixture_or_task_name=task_name,
          feature_converter=feature_converter,
          logger_cls=logger_cls)

    # Try again with `log_dir`.
    evaluator = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=feature_converter,
        log_dir="test_dir",
        logger_cls=logger_cls)

    self.assertLen(evaluator._loggers, 2)

    for logger in logger_cls:
      logger.assert_called_once_with(output_dir="test_dir")

  def test_evaluate_non_string(self):
    task = get_mocked_task()
    # Overrides postprocess fn for the task and its metric objects.
    task.postprocess_fn = lambda d, example, is_target: list(d)
    task.metric_objs = [
        metrics_lib.LegacyMetric.empty(mf, task.postprocess_fn)
        for mf in task.metric_fns
    ]
    mock_vocab = task.output_features["targets"].vocabulary
    # Identity decode function
    mock_vocab.decode = lambda ids: ids

    def mock_init(self):
      # Dummy datasets
      mock_ds = tf.data.Dataset.from_tensor_slices(
          [[5, 6], [6, 7]]).map(lambda x: {"targets_pretokenized": x})
      self._cached_model_datasets = {task.name: mock_ds}
      self._cached_task_datasets = {task.name: mock_ds}
      self._eval_tasks = [task]
      self._loggers = ()
      self._metrics_future = None
      self._metrics_executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=1)
      self._target_field_name = "targets"

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()  # pytype: disable=missing-parameter

      # A dummy prediction function that always returns the same output.
      # The first example is correct but the second is not.
      def predict_fn(
          ds: tf.data.Dataset,
          model_feature_shapes: Optional[Mapping[str, int]] = None
      ) -> evaluation.PredictFnReturnType:
        del ds, model_feature_shapes
        return [(0, [5, 6]), (1, [6, 8])]
      all_metrics, _ = evaluator.evaluate(
          compute_metrics=True,
          predict_fn=predict_fn,
          score_fn=self.uncalled_fn)
      # expected = {"accuracy": 2.0 / 3 * 100}
      expected = {"sequence_accuracy": 50}
      self.assertDictClose(expected, all_metrics.result()[task.name])

  def test_evaluate_single_task_with_postprocessor(self):
    task = get_mocked_task(predict_metric_fns=[_accuracy_metric])
    task.postprocess_fn = functools.partial(
        _string_label_to_class_id_postprocessor,
        label_classes=["e5", "e6", "e7"])

    id_to_vocab = {5: "e5", 6: "e6", 7: "e7"}
    mock_vocab = task.output_features["targets"].vocabulary
    mock_vocab.decode = lambda ids: id_to_vocab[ids[0]]

    def mock_init(self):
      # Dummy datasets
      mock_ds = tf.data.Dataset.from_tensor_slices(
          ["e5", "e6", "e7"]).map(lambda x: {"targets_pretokenized": x})
      self._cached_model_datasets = {task.name: mock_ds}
      self._cached_task_datasets = {task.name: mock_ds}
      self._eval_tasks = [task]
      self._loggers = ()
      self._metrics_future = None
      self._metrics_executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=1)
      self._target_field_name = "targets"

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()  # pytype: disable=missing-parameter

      # The output tokens will be docoded to ["e5", "e6", "e7"] and
      # postprocessed to [0, 1, 2].
      def predict_fn(
          ds: tf.data.Dataset,
          model_feature_shapes: Optional[Mapping[str, int]] = None
      ) -> evaluation.PredictFnReturnType:
        del ds, model_feature_shapes
        return [(0, [5]), (1, [6]), (2, [7])]

      all_metrics, _ = evaluator.evaluate(
          compute_metrics=True,
          predict_fn=predict_fn,
          score_fn=self.uncalled_fn)
      expected = {"accuracy": 100}
      self.assertDictClose(expected, all_metrics.result()[task.name])

  def test_evaluate_mixture(self):
    id_to_vocab = {5: "e5", 6: "e6", 7: "e7"}

    task1 = get_mocked_task(name="task1", score_metric_fns=[_sum_scores_metric])
    mock_vocab1 = task1.output_features["targets"].vocabulary
    mock_vocab1.decode = lambda ids: " ".join([id_to_vocab[i] for i in ids])

    task2 = get_mocked_task(
        name="task2",
        predict_metric_fns=[_accuracy_metric],
        score_metric_fns=[])
    task2.postprocess_fn = functools.partial(
        _string_label_to_class_id_postprocessor,
        label_classes=["e5", "e6", "e7"])
    mock_vocab2 = task2.output_features["targets"].vocabulary
    mock_vocab2.decode = lambda ids: id_to_vocab[ids[0]]

    mock_ds1 = tf.data.Dataset.from_tensor_slices(
        ["e5 e6", "e6"]).map(lambda x: {"targets_pretokenized": x})

    mock_ds2 = tf.data.Dataset.from_tensor_slices(
        ["e5", "e6", "e7"]).map(lambda x: {"targets_pretokenized": x})

    def mock_init(self):
      self._cached_model_datasets = {
          task1.name: mock_ds1,
          task2.name: mock_ds2,
      }
      self._cached_task_datasets = {
          task1.name: mock_ds1,
          task2.name: mock_ds2,
      }
      self._eval_tasks = [task1, task2]
      self._loggers = ()
      self._metrics_future = None
      self._metrics_executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=1)
      self._target_field_name = "targets"

    with mock.patch.object(Evaluator, "__init__", new=mock_init):

      def predict_fn(
          ds: tf.data.Dataset,
          model_feature_shapes: Optional[Mapping[str, int]] = None
      ) -> Optional[evaluation.PredictFnReturnType]:
        del model_feature_shapes
        if ds == mock_ds1:
          return [(0, [5, 6]), (1, [7])]
        elif ds == mock_ds2:
          return [(0, [5]), (1, [6]), (2, [7])]

      def score_fn(
          ds: tf.data.Dataset,
          model_feature_shapes: Optional[Mapping[str, int]] = None
      ) -> Sequence[Tuple[int, float]]:
        del model_feature_shapes
        self.assertEqual(ds, mock_ds1)
        return [(0, 1), (1, 2)]

      evaluator = Evaluator()  # pytype: disable=missing-parameter
      all_metrics, _ = evaluator.evaluate(
          compute_metrics=True, predict_fn=predict_fn, score_fn=score_fn)
      expected = {
          task1.name: {
              "sequence_accuracy": 50.0,
              "total_score": 651
          },
          task2.name: {
              "accuracy": 100
          }
      }
      all_metrics = all_metrics.result()
      self.assertDictClose(expected[task1.name], all_metrics[task1.name])
      self.assertDictClose(expected[task2.name], all_metrics[task2.name])

  def test_short_inputs_targets(self):
    task_name = "short_inputs_targets"
    ds = tf.data.Dataset.from_tensors({
        "inputs": [7, 8],
        "targets": [3, 9],
        "targets_pretokenized": "ex 1"
    })
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        metrics_fn=[_sequence_accuracy_metric])

    feature_converter = evaluation.EncDecFeatureConverter(pack=False)
    sequence_length = {"inputs": 10, "targets": 8}
    evaluator = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=feature_converter,
        eval_split="validation",
        sequence_length=sequence_length)
    self.assertDictEqual(
        {
            "decoder_input_tokens": (8,),
            "decoder_loss_weights": (8,),
            "decoder_target_tokens": (8,),
            "encoder_input_tokens": (10,),
        }, evaluator.model_feature_shapes)

  def test_no_sequence_length(self):
    task_name = "no_sequence_length"
    x = [{
        "inputs": [7, 8],
        "targets": [3, 9],
        "targets_pretokenized": "ex 1"
    }, {
        "inputs": [8, 4, 5, 6],
        "targets": [4],
        "targets_pretokenized": "ex 2"
    }]
    dtypes = {
        "inputs": tf.int32,
        "targets": tf.int32,
        "targets_pretokenized": tf.string
    }
    shapes = {"inputs": [None], "targets": [None], "targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=dtypes, output_shapes=shapes)
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        metrics_fn=[_sequence_accuracy_metric])

    feature_converter = evaluation.EncDecFeatureConverter(pack=False)
    evaluator = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=feature_converter,
        eval_split="validation")
    self.assertDictEqual(
        {
            "encoder_input_tokens": (5,),
            "decoder_input_tokens": (3,),
            "decoder_target_tokens": (3,),
            "decoder_loss_weights": (3,)
        }, evaluator.model_feature_shapes)

  def test_partial_sequence_length(self):
    task_name = "partial_sequence_length"
    x = [{
        "inputs": [7, 8],
        "targets": [3, 9],
        "targets_pretokenized": "ex 1"
    }, {
        "inputs": [8, 4, 5, 6],
        "targets": [4],
        "targets_pretokenized": "ex 2"
    }]
    dtypes = {
        "inputs": tf.int32,
        "targets": tf.int32,
        "targets_pretokenized": tf.string
    }
    shapes = {"inputs": [None], "targets": [None], "targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=dtypes, output_shapes=shapes)
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        metrics_fn=[_sequence_accuracy_metric])

    feature_converter = evaluation.EncDecFeatureConverter(pack=False)
    evaluator = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=feature_converter,
        eval_split="validation",
        # Set the sequence_length for inputs only, to truncate them.
        sequence_length={"inputs": 2})
    self.assertDictEqual(
        {
            "encoder_input_tokens": (2,),
            "decoder_input_tokens": (3,),
            "decoder_target_tokens": (3,),
            "decoder_loss_weights": (3,)
        }, evaluator.model_feature_shapes)

  def test_requires_sequence_length(self):
    task_name = "requires_sequence_length"
    ds = tf.data.Dataset.from_tensors({
        "inputs": [7, 8],
        "targets": [3, 9],
        "targets_pretokenized": "ex 1"
    })
    dataset_fn = lambda split, shuffle_files, seed=None: ds

    def preprocessor_with_sequence_length(dataset, sequence_length):
      del sequence_length
      return dataset

    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        # has sequence_length arg
        preprocessor=preprocessor_with_sequence_length,
        metrics_fn=[_sequence_accuracy_metric])

    feature_converter = mock.Mock(pack=False)

    with self.assertRaisesWithLiteralMatch(
        ValueError, "Preprocessor 'preprocessor_with_sequence_length' in task "
        "'requires_sequence_length' has a `sequence_length` argument, making "
        "it incompatible with automatic sequence length detection. Pass a "
        "valid `sequence_length` to `Evaluator` and try again."):
      _ = Evaluator(
          mixture_or_task_name=task_name,
          feature_converter=feature_converter,
          eval_split="validation")

  def test_preprocessor_with_optional_sequence_length(self):
    task_name = "preprocessor_with_optional_sequence_length"
    ds = tf.data.Dataset.from_tensors({
        "inputs": [7, 8],
        "targets": [3, 9],
        "targets_pretokenized": "ex 1"
    })
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        # `append_eos_after_trim` has an optional sequence_length arg
        preprocessor=preprocessors.append_eos_after_trim,
        metrics_fn=[_sequence_accuracy_metric])

    feature_converter = evaluation.EncDecFeatureConverter(pack=False)
    # Should not raise ValueError
    _ = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=feature_converter,
        eval_split="validation")

  def test_caching(self):
    task_name = "caching"
    x = [{
        "inputs": [7, 8],
        "targets": [3, 9],
        "targets_pretokenized": "ex 1"
    }, {
        "inputs": [8, 4],
        "targets": [4],
        "targets_pretokenized": "ex 2"
    }]
    dtypes = {
        "inputs": tf.int32,
        "targets": tf.int32,
        "targets_pretokenized": tf.string
    }
    shapes = {"inputs": [None], "targets": [None], "targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=dtypes, output_shapes=shapes)
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        metrics_fn=[_sequence_accuracy_metric])

    # Feature converter that just pads "inputs" and "targets".
    feature_converter = mock.Mock(
        get_model_feature_lengths=lambda x: {
            "inputs": 4,
            "targets": 4
        },
        pack=False,
        TASK_FEATURES={})
    feature_converter.side_effect = (
        lambda ds, length: utils.trim_and_pad_dataset(ds, {
            "inputs": 4,
            "targets": 4
        }))
    evaluator = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=feature_converter,
        eval_split="validation")
    expected_task_examples = [{
        "inputs": [7, 8, 1],
        "targets": [3, 9, 1],
        "targets_pretokenized": b"ex 1"
    }, {
        "inputs": [8, 4, 1],
        "targets": [4, 1],
        "targets_pretokenized": b"ex 2"
    }]
    expected_examples = [{
        "inputs": [7, 8, 1, 0],
        "targets": [3, 9, 1, 0],
        "targets_pretokenized": b"ex 1"
    }, {
        "inputs": [8, 4, 1, 0],
        "targets": [4, 1, 0, 0],
        "targets_pretokenized": b"ex 2"
    }]

    test_utils.assert_dataset(evaluator._cached_task_datasets[task_name],
                              expected_task_examples)

    # _cached_model_datasets are enumerated. Remove the index for assertion.
    eval_ds = evaluator._cached_model_datasets[task_name].map(lambda i, ds: ds)
    test_utils.assert_dataset(eval_ds, expected_examples)
    self.assertEqual(evaluator.cached_targets[task_name], ["ex 1", "ex 2"])
    self.assertDictEqual(evaluator.model_feature_shapes, {
        "inputs": (4,),
        "targets": (4,)
    })

  def test_predict_fn_called_with_cached_model_datasets(self):
    eval_ds = tf.data.Dataset.range(10)
    task = get_mocked_task()

    def mock_init(self):
      self._cached_model_datasets = {task.name: eval_ds}
      self._cached_task_datasets = {task.name: tf.data.Dataset.range(3)}
      self._eval_tasks = [task]
      self._loggers = ()
      self._metrics_future = None
      self._metrics_executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=1)
      self._target_field_name = "targets"

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()  # pytype: disable=missing-parameter
      predict_fn = mock.Mock(return_value=[(0, 1)])
      evaluator.evaluate(
          compute_metrics=False,
          predict_fn=predict_fn,
          score_fn=self.uncalled_fn)
      predict_fn.assert_called_with(eval_ds)

  def test_order_preservation(self):
    task = get_mocked_task()
    id_to_vocab = {5: "e5", 6: "e6", 7: "e7"}
    mock_vocab = task.output_features["targets"].vocabulary
    mock_vocab.decode = lambda ids: id_to_vocab[ids[0]]

    targets = tf.data.Dataset.from_tensor_slices([[5], [6], [7]])
    targets_pretokenized = tf.data.Dataset.from_tensor_slices(
        ["e5", "e6", "e7"])
    ds = tf.data.Dataset.zip((targets, targets_pretokenized))
    ds = ds.map(lambda x, y: {
        "targets": x,
        "targets_pretokenized": y
    })

    def mock_init(self):
      self._cached_model_datasets = {task.name: ds.enumerate()}
      # Dummy task datasets
      self._cached_task_datasets = {task.name: ds}
      self._eval_tasks = [task]
      self._loggers = ()
      self._metrics_future = None
      self._metrics_executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=1)
      self._target_field_name = "targets"

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()  # pytype: disable=missing-parameter

      # Dummy predict_fn where only the order is mixed.
      def mixing_order_predict_fn(
          ds: tf.data.Dataset,
          model_feature_shapes: Optional[Mapping[str, int]] = None
      ) -> evaluation.PredictFnReturnType:
        del model_feature_shapes
        exs = [(tup[0], tup[1]["targets"])
               for tup in tfds.as_numpy(ds)]
        return [exs[2], exs[0], exs[1]]

      all_metrics, all_outputs = evaluator.evaluate(
          compute_metrics=True,
          predict_fn=mixing_order_predict_fn,
          score_fn=self.uncalled_fn)
      expected_metric = {"sequence_accuracy": 100}
      expected_outputs = [
          np.array([5], dtype=np.int32),
          np.array([6], dtype=np.int32),
          np.array([7], dtype=np.int32),
      ]
      self.assertDictEqual(expected_metric, all_metrics.result()[task.name])
      self.assertEqual(
          expected_outputs,
          all_outputs[task.name][metrics_lib.ModelOutputType.PREDICTION])

  def test_duplicate_metric(self):
    task = get_mocked_task(
        predict_metric_fns=[_accuracy_metric, _accuracy_metric])
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Duplicate metric key 'accuracy' in Task 'mocked_test'."):
      self._evaluate_single_task(task)

    task = get_mocked_task(
        score_metric_fns=[_sum_scores_metric, _sum_scores_metric])
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Duplicate metric key 'total_score' in Task 'mocked_test'."):
      self._evaluate_single_task(task)

    task = get_mocked_task(
        predict_metric_fns=[_accuracy_metric],
        score_metric_fns=[lambda targets, scores: {
            "accuracy": 0
        }])
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Duplicate metric key 'accuracy' in Task 'mocked_test'."):
      self._evaluate_single_task(task)

  def test_task_with_no_metrics_fn(self):
    task_name = "no_metrics_task"
    x = [{"targets_pretokenized": "ex 1"}]
    dtypes = {"targets_pretokenized": tf.string}
    shapes = {"targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=dtypes, output_shapes=shapes)
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    register_dummy_task(task_name, dataset_fn=dataset_fn, metrics_fn=[])
    evaluator = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=evaluation.EncDecFeatureConverter())
    all_metrics, all_output = evaluator.evaluate(
        compute_metrics=True, predict_fn=mock.Mock(), score_fn=self.uncalled_fn)
    self.assertEqual({}, all_metrics.result())
    self.assertEqual({}, all_output)

  def test_task_with_no_compute_metrics(self):
    task_name = "no_compute_metrics_task"
    x = [{"targets_pretokenized": "ex 1"}]
    dtypes = {"targets_pretokenized": tf.string}
    shapes = {"targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=dtypes, output_shapes=shapes)
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    register_dummy_task(task_name, dataset_fn=dataset_fn, metrics_fn=[])
    evaluator = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=evaluation.EncDecFeatureConverter())
    all_metrics, all_output = evaluator.evaluate(
        compute_metrics=False,
        predict_fn=mock.Mock(),
        score_fn=self.uncalled_fn)
    self.assertIsNone(all_metrics.result())
    self.assertEqual({}, all_output)

  def test_task_with_no_pretokenized_targets(self):
    task_name = "no_pretokenized_task"
    ds = tf.data.Dataset.from_tensors({"targets": [42, 48], "inputs": [56]})
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    task = register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        metrics_fn=[_sum_scores_metric],
        postprocess_fn=lambda d, example, is_target: d + " 1")
    task.output_features["targets"].vocabulary.decode = mock.Mock(
        return_value="ex")
    evaluator = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=evaluation.EncDecFeatureConverter(pack=False))
    self.assertSequenceEqual(evaluator.cached_targets[task_name], ["ex 1"])
    task.output_features["targets"].vocabulary.decode.assert_called_once_with(
        [42, 48, 1])

  def test_task_with_score_fn_with_intermediates(self):

    def score_fn_with_intermediates(
        ds: tf.data.Dataset,
        model_feature_shapes: Optional[Mapping[str, int]] = None
    ) -> Tuple[Sequence[Tuple[int, float]], Mapping[str, Any]]:
      del ds, model_feature_shapes
      indices_and_scores = [(1, 1), (0, 2), (2, 3)]
      intermediates = {"score": [101, 102, 103]}
      return indices_and_scores, intermediates

    task = get_mocked_task(
        predict_metric_fns=[],
        score_metric_fns=[_sum_scores_metric_with_intermediates])

    def mock_init(self):
      mock_ds = tf.data.Dataset.from_tensor_slices(
          ["e5 e6", "e6", "e7"]).map(lambda x: {"targets_pretokenized": x})
      self._cached_model_datasets = {task.name: mock_ds}
      self._cached_task_datasets = {task.name: mock_ds}
      self._eval_tasks = [task]
      self._loggers = ()
      self._metrics_future = None
      self._metrics_executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=1)
      self._target_field_name = "targets"

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()  # pytype: disable=missing-parameter

      all_metrics, _ = evaluator.evaluate(
          compute_metrics=True,
          predict_fn=self.uncalled_fn,
          predict_with_aux_fn=self.uncalled_fn,
          score_fn=score_fn_with_intermediates,
          step=42)
      results = all_metrics.result()

    self.assertDictClose({"total_score_with_intermediates": 66505},
                         results["mocked_test"])


if __name__ == "__main__":
  tf.test.main()
