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

"""Utilities for the class-based evaluation."""

import concurrent
import functools
import inspect
import itertools
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Type

from absl import logging
from seqio import dataset_providers
from seqio import feature_converters
from seqio import loggers as loggers_lib
from seqio import metrics as metrics_lib
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import typing_extensions

Task = dataset_providers.Task
EncDecFeatureConverter = feature_converters.EncDecFeatureConverter
FeatureConverter = feature_converters.FeatureConverter

AllOutputTokensType = Mapping[str, Sequence[Sequence[int]]]
AllOutputScoresType = Mapping[str, Sequence[float]]
AllMetricsType = Mapping[str, Sequence[Mapping[str, Any]]]
MetricsAndOutputsType = Tuple[
    concurrent.futures.Future,  # metrics
    AllOutputTokensType,  # output_tokens
    AllOutputScoresType]  # output_scores


def get_valid_eval_tasks(tasks: Sequence[Task], split: str) -> Sequence[Task]:
  """Get tasks that have the specified split and a metric function."""

  valid_tasks = []

  for task in tasks:
    if split not in task.splits:
      logging.info(
          "Task %s has no '%s' split; skipping eval.", task.name, split
      )
      continue
    if not task.metric_fns:
      logging.info("Task %s has no metric_fns; skipping eval.", task.name)
      continue
    metric_types = []
    if task.predict_metric_fns:
      metric_types.append("predict")
    if task.score_metric_fns:
      metric_types.append("score")
    if task.predict_targetless_metric_fns:
      metric_types.append("targetless predict")
    if task.score_targetless_metric_fns:
      metric_types.append("targetless score")
    logging.info("Adding task '%s' with %s metric_fn(s).", task.name,
                 " and ".join(metric_types))
    valid_tasks.append(task)

  return valid_tasks


def get_targets_and_examples(
    tasks: Sequence[Task],
    dataset_fn: Callable[[Task], tf.data.Dataset],
    sequence_dims: Mapping[str, int],
    num_examples: Optional[int] = None,
    use_memory_cache: bool = True
) -> Tuple[
    Mapping[str, Any],
    Mapping[str, tf.data.Dataset],
    Mapping[str, int]]:
  """Get targets, cached datasets, and maximum sequence lengths per feature.

  Args:
    tasks: tasks objects to get targets and examples for.
    dataset_fn: function, returns the dataset from the task object.
    sequence_dims: dict of feature names to their sequence dimension.
    num_examples: an optional maximum number of examples to take from the
      beginning of each task dataset.
    use_memory_cache: whether to use tf.data.Dataset#cache. may cause
      memory issues for large datasets.
  Returns:
    cached_targets: unpreprocessed targets for each task
    cached_task_datasets: cached datasets for each task, with cardinality set
    max_sequence_length: maximum sequence lengths for inputs and targets across
      all tasks.
  """
  # Pre-load in all of the targets once before entering continuous eval loop
  cached_targets = {}
  cached_task_datasets = {}
  max_sequence_length = {k: 0 for k in tasks[0].output_features.keys()}

  for task in tasks:
    assert max_sequence_length.keys() == task.output_features.keys(), (
        "all tasks must have the same features")

  for task in tasks:
    ds = dataset_fn(task)
    if num_examples:
      ds = ds.take(num_examples)
    if use_memory_cache:
      ds = ds.cache()

    targets = []

    if "targets" in ds.element_spec or "targets_pretokenized" in ds.element_spec:
      for ex in tfds.as_numpy(ds):
        for k in max_sequence_length:
          sequence_dim = sequence_dims.get(k, 0)
          sequence_length = ex[k].shape[sequence_dim]
          max_sequence_length[k] = max(max_sequence_length[k], sequence_length)

        # Create list of postprocessed targets
        if "targets_pretokenized" in ex:
          target = ex["targets_pretokenized"]
        else:
          target = task.prediction_vocabulary.decode(
              [int(x) for x in ex["targets"]])
        if isinstance(target, bytes):
          target = target.decode("utf-8")
        targets.append(task.postprocess_fn(target, example=ex, is_target=True))

      cached_targets[task.name] = targets
      cached_task_datasets[task.name] = ds.apply(
          tf.data.experimental.assert_cardinality(len(targets)))
    else:
      cached_task_datasets[task.name] = ds.apply(
          tf.data.experimental.assert_cardinality(len(list(tfds.as_numpy(ds)))))

  return cached_targets, cached_task_datasets, max_sequence_length


class PredictFnCallable(typing_extensions.Protocol):

  def __call__(
      self,
      dataset: tf.data.Dataset,
      model_feature_lengths: Optional[Mapping[str, int]]
  ) -> Sequence[Tuple[int, Sequence[int]]]:
    ...


class ScoreFnCallable(typing_extensions.Protocol):

  def __call__(
      self,
      dataset: tf.data.Dataset,
      model_feature_lengths: Optional[Mapping[str, int]]
  ) -> Sequence[Tuple[int, float]]:
    ...


class Evaluator:
  """A class to encapsulate all eval-related information.

  Users should define `predict_fn` and then pass it to `evaluate` method.
  `predict_fn` should operate with enumerated tf.data.Dataset. See `evaluate`
  method for more detail.

  evaluation data is cached once and will be used for arbitrary number of
  evaluation runs.

  If none of the evaluation tasks has metrics functions defined, the evaluation
  will be skipped. `Evaluator.evaluate` will return ({}, {}) assuming that
  compute_metrics is True.

  Note that we cache two versions of the datasets. The first version
  (self.cached_task_datasets) has the task features (e.g., "inputs" and
  "targets"), which are returned from `seqio.Task.get_dataset`. The second
  version (self.cached_model_datasets) has model features (e.g.,
  "decoder_target_tokens"). This is returned from the feature converter. The
  former is used for postprocessing associated with the Task that requires the
  original task datasets. The latter is passed to `predict_fn` for evaluation.

  Attributes:
    eval_tasks: a mapping from a mixture or a task name to seqio.Task object(s).
    cached_model_datasets: cached evaluation datasets with model features.
    cached_task_datasets: cached evaluation datasets with task features.
    cached_targets: cached evaluation targets.
    model_feature_lengths: mapping from model feature to its length in the
      `cached_model_datasets`.
    loggers: a sequence of subclasses of `Logger`.
  """

  def __init__(self,
               mixture_or_task_name: str,
               feature_converter: FeatureConverter,
               eval_split: str = "validation",
               use_cached: bool = False,
               seed: Optional[int] = 42,
               sequence_length: Optional[Mapping[str, int]] = None,
               num_examples: Optional[int] = None,
               shuffle: bool = False,
               logger_cls: Sequence[Type[loggers_lib.Logger]] = (),
               log_dir: Optional[str] = None,
               use_memory_cache: bool = True):
    """Evaluator constructor.

    Args:
      mixture_or_task_name: a registered task or mixture name.
      feature_converter: a feature converter object to use to convert the task
        features to model features. Must be a subclass of
        seqio.FeatureConverter.
      eval_split: evaluation split. Typically "validation" or "test".
      use_cached: whether to use the cached dataset instead of processing it on
        the fly.
      seed: random seed used for dataset shuffle and preprocessing. This
        is usually not needed since eval datasets aren't shuffled and shouldn't
        use stochastic operations. It is only useful for in certain data sources
        such as `FewshotDataSource` where the training examples are randomly
        selected during evaluation.
      sequence_length: an optional length specification. If specified, these
        will be the hard-limit on the evaluation data used for prediction. If
        none of the preprocessors depend on the sequence length, it can be left
        unspecified and the maximum length for each feature will be used. These
        lengths are computed while caching the datasets.
      num_examples: an optional maximum number of examples to take from the
        beginning of each Task dataset for evaluation.
      shuffle: whether to shuffle the Task datasets. Only useful when
        `num_examples` is also set in order to get a semi-random subsample of
        the examples. Note that the shuffle will only be applied once during
        initialization (using `seed`) and the same subsample will be used on
        call to `evaluate`.
      logger_cls: a set of subclasses of `Logger` to write results with.
      log_dir: the directory to log outputs to. Required if `logger_cls` is
        non-empty.
      use_memory_cache: whether to use tf.data.Dataset#cache. may cause
        memory issues for large datasets.

    Raises:
      ValueError if `sequence_length` is None but a preprocessor depends on its
      value.
    """
    logging.info("Initializing Evaluator for '%s'", mixture_or_task_name)
    eval_tasks = dataset_providers.get_subtasks(
        dataset_providers.get_mixture_or_task(mixture_or_task_name))
    self._eval_tasks = get_valid_eval_tasks(eval_tasks, eval_split)

    self._metrics_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1)
    self._metrics_future = None

    if not self._eval_tasks:
      logging.warning(
          "No eval task with valid split and metric fn found. Skipping eval.")
      return

    # Determine if sequence_length arg is required. This occurs when any of the
    # task preprocessors have a `sequence_length` arg with no default value.
    sequence_length_required = False
    for task in self._eval_tasks:
      for prep in task.preprocessors:
        prep_params = inspect.signature(prep).parameters
        if ("sequence_length" in prep_params and
            prep_params["sequence_length"].default == inspect.Parameter.empty):
          if sequence_length is None:
            if isinstance(prep, functools.partial):
              prep_name = prep.func.__name__
            else:
              prep_name = prep.__name__
            raise ValueError(
                f"Preprocessor '{prep_name}' in task '{task.name}' has a "
                "`sequence_length` argument, making it incompatible with "
                "automatic sequence length detection. Pass a valid "
                "`sequence_length` to `Evaluator` and try again.")
          sequence_length_required = True
          break

    def dataset_fn(task: Task) -> tf.data.Dataset:
      return task.get_dataset(
          sequence_length=sequence_length,
          split=eval_split,
          shuffle=shuffle,
          num_epochs=1,
          seed=seed,
          use_cached=use_cached)

    # `task_datasets` have the output features from seqio.Task.get_dataset.
    # These features will be converted to "model features" by the feature
    # converter before being cached.
    sequence_dims = {
        k: v.sequence_dim for k, v in feature_converter.TASK_FEATURES.items()
    }
    cached_targets, cached_task_datasets, max_lengths = (
        get_targets_and_examples(
            tasks=self._eval_tasks,
            dataset_fn=dataset_fn,
            sequence_dims=sequence_dims,
            num_examples=num_examples,
            use_memory_cache=use_memory_cache))

    if sequence_length is None:
      logging.info("Setting sequence lengths to %s", max_lengths)
      sequence_length = max_lengths
    else:
      log_long_warning = False
      log_same_warning = False

      sequence_length = {
          k: sequence_length.get(k, max_lengths[k]) for k in max_lengths}

      assert set(sequence_length.keys()) == set(max_lengths.keys()), (
          "sequence_length=%s limits must match the detected max_lengths=%s" % (
              sequence_length.keys(), max_lengths.keys()))

      for k, l in sequence_length.items():
        if l is None:
          continue
        elif l > max_lengths[k]:
          log_long_warning = True
        elif not sequence_length_required and l == max_lengths[k]:
          log_same_warning = True

      if log_long_warning:
        logging.warning(
            "Given sequence lengths are longer than necessary for some "
            "evaluation inputs or targets, resulting in wasted computation. "
            "Consider passing `None` for `sequence_length` to have them be "
            "automatically computed.\n Got: %s,\n Max Lengths: %s",
            sequence_length, max_lengths)
      elif log_same_warning:
        logging.warning(
            "Given sequence lengths *may be* insufficient for some evaluation "
            "inputs or targets. Such sequences will be truncated to fit, "
            "likely leading to sub-optimal results. Consider passing `None` "
            "for `sequence_length` to have them be automatically computed.\n "
            " Got: %s,\n Max Lengths: %s", sequence_length, max_lengths)

    self._cached_model_datasets = {}

    if feature_converter.pack:
      raise ValueError("During evaluation, packing can't be used.")
    # Convert the task features to model features
    for task in self._eval_tasks:
      eval_ds = feature_converter(
          cached_task_datasets[task.name], sequence_length)

      # The eval dataset is enumerated to ensure that the order is preserved
      # throughout the entire evaluation process.
      self._cached_model_datasets[task.name] = eval_ds.enumerate()

    self._cached_targets = cached_targets
    self._cached_task_datasets = cached_task_datasets
    self._model_feature_lengths = feature_converter.get_model_feature_lengths(
        sequence_length)

    if logger_cls and not log_dir:
      raise ValueError(
          "'log_dir' must be provided to `Evaluator` if `logger_cls` is "
          "non-empty.")
    self._loggers = tuple(cls(output_dir=log_dir) for cls in logger_cls)  # pytype:disable=not-instantiable

  def evaluate(self,
               *,
               compute_metrics: bool,
               step: Optional[int] = None,
               predict_fn: PredictFnCallable,
               score_fn: ScoreFnCallable) -> MetricsAndOutputsType:
    """Predict and score self.eval_tasks.

    Evaluation must preserve the example ordering. This requirement is satisfied
    by using enumerated dataset. Each of the cached eval task datasets is an
    enumerated tf.data.Dataset where each element has (index, example) format.
    Therefore, each index serves as a unique integer id for the example.

    `predict_fn` takes as input the cached eval dataset. The output must be of
    the form Sequence[(index, token_ids)] where `token_ids` is the sequence of
    token ids output by the model with the input `example` whose index matches
    `index`. Therefore, even if `predict_fn` mixes the order of the examples
    during prediction, the order can be corrected as long as the correct index
    for each example is maintained.

    Similarly, `score_fn` takes the cached eval dataset as input and returns
    Sequence[(index, score)] where `score` is the sequence of log likelihood
    scores for the targets in the eval dataset.

    A common example is the multi-host setup where the evaluation dataset is
    split into multiple hosts that independently make predictions and combine
    the results during which the ordering can be mixed.

    There are 4 steps involved in the evaluation using predicted tokens:

    1. Model returns indices and output_tokens: Sequence[Tuple[int,
       Sequence[int]]]
    2. output tokens are decoded by `vocab.decode`
    3. Postprocessors are applied to the decoded output. These are denoted as
       predictions.
    4. Each metric function is applied to the predictions and the cached
       targets.

    There are 2 steps involved in the evaluation using scores:

    1. Model returns indices and scores: Sequence[Tuple[int, Sequence[float]]]
    2. Each metric function is applied to the scores and the cached targets.

    Args:
      compute_metrics: whether to compute metrics.
      step: an optional step number of the current evaluation. If unspecified, a
        dummy value of -1 will be used.
      predict_fn: a user-defined function, which takes in a tf.data.Dataset and
        outputs the sequence of predicted tokens. Only called if predict metrics
        exist for the tasks.
      score_fn: a user-defined function, which takes in a tf.data.Dataset and
        outputs the log likelihood score of the targets. Only called if score
        metrics exist for the task.

    Returns:
      metrics: a Future containing a mapping from task name to computed metrics,
        or None if `compute_metrics` is False.
      predicted_tokens: a mapping from task name to the output tokens
        from `predict_fn`, for tasks that have `predict_metric_fns`.
      scores: a mapping from task name to the output scores from
        `score_fn` for tasks that have `score_predict_fns`.
    """

    all_output_tokens = {}
    all_output_scores = {}

    def _infer_and_sort_outputs(infer_fn, task_name):
      indices_and_outputs = infer_fn(self.cached_model_datasets[task_name])
      if len(indices_and_outputs[0]) != 2:
        raise ValueError(
            "Expected a sequence of length-2 tuples with (index, *) format.")
      return [x[1] for x in sorted(indices_and_outputs, key=lambda x: x[0])]

    for task in self.eval_tasks:
      logging.info("Evaluating %s", task.name)
      if task.predict_metric_fns or task.predict_targetless_metric_fns:
        # output_tokens is a list of token_ids where each token_ids
        # corresponds to the model output of the input example.
        all_output_tokens[task.name] = _infer_and_sort_outputs(
            predict_fn, task.name)
      if task.score_metric_fns or task.score_targetless_metric_fns:
        all_output_scores[task.name] = _infer_and_sort_outputs(
            score_fn, task.name)

    if compute_metrics:
      if self._metrics_future:
        # Ensure previous step's metrics are finished and raise any exceptions
        # that may have occurred.
        tick = time.time()
        self._metrics_future.result()
        logging.info("Time waiting for previous metrics run: %f secs.",
                     time.time() - tick)

      def compute_metrics_fn():
        tick = time.time()
        metrics = self._compute_metrics(all_output_tokens, all_output_scores,
                                        step)
        logging.info("Time computing metrics: %f secs.", time.time() - tick)
        return metrics

      def wrap_graph(fn):
        graph = tf.compat.v1.get_default_graph()

        def wrapped_fn():
          with graph.as_default():
            return fn()

        return wrapped_fn

      if not tf.executing_eagerly():
        compute_metrics_fn = wrap_graph(compute_metrics_fn)

      self._metrics_future = self._metrics_executor.submit(compute_metrics_fn)
      all_metrics = self._metrics_future
    else:
      all_metrics = concurrent.futures.Future()
      all_metrics.set_result(None)
    return all_metrics, all_output_tokens, all_output_scores

  def _compute_metrics(
      self,
      predicted_tokens: AllOutputTokensType,
      scores: AllOutputScoresType,
      step: Optional[int] = None) -> AllMetricsType:
    """Computes and logs metrics given the predicted tokens and scores.

    Args:
      predicted_tokens: a mapping from task name to the output tokens from
        `predict_fn`, for tasks that have `predict_metric_fns`.
      scores: a mapping from task name to the output scores from
        `score_fn` for tasks that have `score_predict_fns`.
      step: an optional step number of the current evaluation. If unspecified, a
        dummy value of -1 will be used.
    Returns:
      A mapping from task name to computed metrics.
    """
    all_metrics = {}

    for task in self.eval_tasks:
      logging.info("Computing metrics for %s", task.name)
      task_dataset = self.cached_task_datasets[task.name]
      # targets may be None
      targets = self.cached_targets.get(task.name)

      task_metrics = []
      inferences = {}

      if task.predict_metric_fns or task.predict_targetless_metric_fns:
        task_predicted_tokens = predicted_tokens[task.name]

        if task.predict_metric_fns and len(targets) != len(
            task_predicted_tokens):
          raise ValueError(
              f"len(targets)({len(targets)}) != "
              f"len(predictions)({len(task_predicted_tokens)})")

        if task.prediction_vocabulary is None:
          outputs = task_predicted_tokens
        else:
          outputs = [
              task.prediction_vocabulary.decode([int(t)
                                                 for t in tokens])
              for tokens in task_predicted_tokens
          ]

        task_predictions = [
            task.postprocess_fn(d, example=ex, is_target=False)
            for d, ex in zip(outputs, tfds.as_numpy(task_dataset))
        ]
        inferences["predictions"] = task_predictions

        task_metrics.extend([
            metric_fn(targets, task_predictions) for metric_fn in
            task.predict_metric_fns
        ])
        task_metrics.extend([
            metric_fn(task_predictions)
            for metric_fn in task.predict_targetless_metric_fns
        ])

      if task.score_metric_fns or task.score_targetless_metric_fns:
        task_scores = scores[task.name]
        if task.score_metric_fns and len(targets) != len(task_scores):
          raise ValueError(f"len(targets)({len(targets)}) != "
                           f"len(task_scores)({len(task_scores)})")
        task_metrics.extend([
            metric_fn(targets, task_scores)
            for metric_fn in task.score_metric_fns
        ])
        task_metrics.extend([
            metric_fn(task_scores)
            for metric_fn in task.score_targetless_metric_fns
        ])
        inferences["scores"] = task_scores

      all_metrics[task.name] = {}
      for k, v in itertools.chain(*[m.items() for m in task_metrics]):
        if k in all_metrics[task.name]:
          raise ValueError(
              f"Duplicate metric key '{k}' in Task '{task.name}'.")
        all_metrics[task.name][k] = v

      metrics = {
          k: metrics_lib.Scalar(v)
             if not isinstance(v, metrics_lib.MetricValue) else v
          for k, v in all_metrics[task.name].items()
      }
      for logger in self.loggers:
        logger(task_name=task.name, step=step, metrics=metrics,
               dataset=task_dataset, inferences=inferences, targets=targets)

    return all_metrics

  @property
  def eval_tasks(self) -> Sequence[Task]:
    return self._eval_tasks

  @property
  def cached_model_datasets(self) -> Mapping[str, tf.data.Dataset]:
    return self._cached_model_datasets

  @property
  def cached_task_datasets(self) -> Mapping[str, tf.data.Dataset]:
    return self._cached_task_datasets

  @property
  def cached_targets(self) -> Mapping[str, Sequence[Any]]:
    return self._cached_targets

  @property
  def model_feature_lengths(self) -> Mapping[str, int]:
    return self._model_feature_lengths

  @property
  def loggers(self) -> Tuple[loggers_lib.Logger]:
    return tuple(self._loggers)
