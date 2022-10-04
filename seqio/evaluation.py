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

"""Utilities for the class-based evaluation."""

import concurrent
import functools
import inspect
import itertools
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Type, Union

from absl import logging
import jax
import numpy as np
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
AllOutputScoresType = Any  # Mapping[str, Sequence[float]]
AllOutputAuxValuesType = Mapping[str, Mapping[str, Sequence[Any]]]
AllMetricsType = Mapping[str, Mapping[str, Any]]


class AllMetricsFuture(typing_extensions.Protocol):

  def result(self) -> AllMetricsType:
    ...


MetricsAndOutputsType = Tuple[AllMetricsFuture,  # metrics
                              Any]  # outputs


def get_valid_eval_tasks(tasks: Sequence[Task], split: str) -> Sequence[Task]:
  """Get tasks that have the specified split and a metric function."""

  valid_tasks = []

  for task in tasks:
    if split not in task.splits:
      logging.info("Task %s has no '%s' split; skipping eval.", task.name,
                   split)
      continue
    if not task.metric_fns and not task.metric_objs:
      logging.info("Task %s has no metrics defined; skipping eval.", task.name)
      continue
    metric_types = []
    if task.predict_metric_fns:
      metric_types.append("predict")
    if task.predict_with_aux_metric_fns:
      metric_types.append("predict_with_aux")
    if task.score_metric_fns:
      metric_types.append("score")
    logging.info("Adding task '%s' with %s metric_fn(s).", task.name,
                 " and ".join(metric_types))
    valid_tasks.append(task)

  return valid_tasks


def get_targets_and_examples(
    tasks: Sequence[Task],
    dataset_fn: Callable[[Task], tf.data.Dataset],
    sequence_dims: Mapping[str, int],
    num_examples: Optional[int] = None,
    use_memory_cache: bool = True,
    target_field_name: str = "targets"
) -> Tuple[Mapping[str, Any], Mapping[str, tf.data.Dataset], Mapping[str, int]]:
  """Get targets, cached datasets, and maximum sequence lengths per feature.

  Args:
    tasks: tasks objects to get targets and examples for.
    dataset_fn: function, returns the dataset from the task object.
    sequence_dims: dict of feature names to their sequence dimension.
    num_examples: an optional maximum number of examples to take from the
      beginning of each task dataset.
    use_memory_cache: whether to use tf.data.Dataset#cache. may cause memory
      issues for large datasets.
    target_field_name: Field name of the target in the input dataset examples.

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

    for ex in tfds.as_numpy(ds):
      for k in max_sequence_length:
        sequence_dim = sequence_dims.get(k, 0)
        sequence_length = ex[k].shape[sequence_dim]
        max_sequence_length[k] = max(max_sequence_length[k], sequence_length)

      # Create list of postprocessed targets
      pretokenized_target_field_name = target_field_name + "_pretokenized"
      if pretokenized_target_field_name in ex:
        target = ex[pretokenized_target_field_name]
      else:
        target = task.output_features[target_field_name].vocabulary.decode(
            list(ex[target_field_name]))
      if isinstance(target, bytes):
        target = target.decode("utf-8")
      targets.append(task.postprocess_fn(target, example=ex, is_target=True))

    cached_targets[task.name] = targets
    cached_task_datasets[task.name] = ds.apply(
        tf.data.experimental.assert_cardinality(len(targets)))

  return cached_targets, cached_task_datasets, max_sequence_length


_BatchId = int
_Tokens = Sequence[int]
_AuxValues = Mapping[str, Sequence[Any]]

_IndicesAndPredictions = Sequence[Tuple[_BatchId, _Tokens]]
_IndicesAndPredictionsWithAuxValues = Tuple[_IndicesAndPredictions, _AuxValues]
PredictFnReturnType = Union[_IndicesAndPredictions,
                            _IndicesAndPredictionsWithAuxValues]

_IndicesAndScores = Sequence[Tuple[_BatchId, float]]
_IndicesAndScoresWithIntermediates = Tuple[_IndicesAndScores, Mapping[str, Any]]
ScoreFnReturnType = Union[_IndicesAndScores, _IndicesAndScoresWithIntermediates]
ModelFnReturnType = Union[
    _IndicesAndPredictions,
    _IndicesAndPredictionsWithAuxValues,
    _IndicesAndScores,
    _IndicesAndScoresWithIntermediates]


class PredictFnCallable(typing_extensions.Protocol):

  def __call__(
      self, dataset: tf.data.Dataset,
      model_feature_shapes: Optional[Mapping[str, int]]) -> PredictFnReturnType:
    ...


class ScoreFnCallable(typing_extensions.Protocol):

  def __call__(
      self, dataset: tf.data.Dataset,
      model_feature_shapes: Optional[Mapping[str, int]]) -> ScoreFnReturnType:
    ...


class ModelFnCallable(typing_extensions.Protocol):

  def __call__(
      self, dataset: tf.data.Dataset,
      model_feature_shapes: Optional[Mapping[str, int]]) -> ModelFnReturnType:
    ...


def _extract_model_output(cached_model_dataset, model_fn):
  """Extracts model output from a cached dataset."""

  def _permute(x, sorted_order):
    return [x[sorted_order[i]] for i in range(len(sorted_order))]

  model_fn_result = model_fn(cached_model_dataset)

  if isinstance(model_fn_result, tuple):
    # Some of model functions return a tuple of two outputs per example.
    # e.g., ModelOutputType.PREDICTION_WITH_AUX,
    # ModelOutputType.SCORE_WITH_INTERMEDIATES
    indices_and_outputs, all_aux_values = model_fn_result
    indices, outputs = zip(*indices_and_outputs)
    sorted_order = np.argsort(indices)

    sorted_outputs = _permute(outputs, sorted_order)
    sorted_aux = jax.tree_map(
        functools.partial(_permute, sorted_order=sorted_order),
        all_aux_values, is_leaf=lambda x: isinstance(x, list))
    return sorted_outputs, sorted_aux

  else:
    # Majority of model functions should only return one output per example.
    # e.g., ModelOutputType.SCORE, ModelOutputType.PREDICTION
    indices_and_outputs = model_fn_result
    _, sorted_outputs = zip(*sorted(indices_and_outputs, key=lambda x: x[0]))
    return list(sorted_outputs)


def _extract_tokens_and_aux_values(cached_model_dataset, predict_fn):
  """Extracts tokens and aux scores from a cached dataset."""
  predict_fn_result = predict_fn(cached_model_dataset)

  all_aux_values = {}
  if isinstance(predict_fn_result, tuple):
    indices_and_tokens, all_aux_values = predict_fn_result
    indices, tokens = zip(*indices_and_tokens)

    permutation = np.argsort(indices)

    tokens = [tokens[permutation[i]] for i in range(len(permutation))]
    for aux_keys, aux_values in all_aux_values.items():
      all_aux_values[aux_keys] = [
          aux_values[permutation[i]] for i in range(len(permutation))
      ]

  else:
    indices_and_tokens = predict_fn_result
    _, tokens = zip(*sorted(indices_and_tokens, key=lambda x: x[0]))

  return tokens, all_aux_values


def _extract_scores(cached_model_dataset, score_fn):
  """Extracts scores and intermediate values from a cached dataset."""
  indices_and_scores = score_fn(cached_model_dataset)

  if isinstance(indices_and_scores, tuple) and len(indices_and_scores) == 2:
    indices_and_scores, intermediates = indices_and_scores
    indices, scores = zip(*indices_and_scores)
    sorted_order = np.argsort(indices)

    def _permute(x):
      return [x[sorted_order[i]] for i in range(len(sorted_order))]

    sorted_scores = _permute(scores)
    sorted_intermediates = jax.tree_map(
        _permute, intermediates, is_leaf=lambda x: isinstance(x, list))
    return sorted_scores, sorted_intermediates

  if len(indices_and_scores[0]) != 2:
    raise ValueError(
        "Expected a sequence of length-2 tuples with (index, score) "
        "format.")
  return [x[1] for x in sorted(indices_and_scores, key=lambda x: x[0])]


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
    model_feature_shapes: mapping from model feature to its shape in the
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
               use_memory_cache: bool = True,
               target_field_name: str = "targets"):
    """Evaluator constructor.

    Args:
      mixture_or_task_name: a registered task or mixture name.
      feature_converter: a feature converter object to use to convert the task
        features to model features. Must be a subclass of
        seqio.FeatureConverter.
      eval_split: evaluation split. Typically "validation" or "test".
      use_cached: whether to use the cached dataset instead of processing it on
        the fly.
      seed: random seed used for dataset shuffle and preprocessing. This is
        usually not needed since eval datasets aren't shuffled and shouldn't use
        stochastic operations. It is only useful for in certain data sources
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
      use_memory_cache: whether to use tf.data.Dataset#cache. may cause memory
        issues for large datasets.
      target_field_name: Field name of the target in the input dataset examples.

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
    self._target_field_name = target_field_name

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
            use_memory_cache=use_memory_cache,
            target_field_name=self._target_field_name))

    if sequence_length is None:
      logging.info("Setting sequence lengths to %s", max_lengths)
      sequence_length = max_lengths
    else:
      log_long_warning = False
      log_same_warning = False

      sequence_length = {
          k: sequence_length.get(k, max_lengths[k]) for k in max_lengths
      }

      assert set(sequence_length.keys()) == set(max_lengths.keys()), (
          "sequence_length=%s limits must match the detected max_lengths=%s" %
          (sequence_length.keys(), max_lengths.keys()))

      for k, l in sequence_length.items():
        if l is None:
          continue
        if isinstance(l, (tuple, list)):
          logging.warning(
              "Automatic length checking is not supported when lengths are"
              "specified with a tuple for feature %s = %s. Please make "
              "sure your max lengths are not removing parts of your inputs.", k,
              l)
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
      eval_ds = feature_converter(cached_task_datasets[task.name],
                                  sequence_length)

      # The eval dataset is enumerated to ensure that the order is preserved
      # throughout the entire evaluation process.
      self._cached_model_datasets[task.name] = eval_ds.enumerate()

    self._cached_targets = cached_targets
    self._cached_task_datasets = cached_task_datasets
    self._model_feature_shapes = {
        k: tuple(spec.shape)
        for k, spec in eval_ds.element_spec.items()
        if spec.shape.rank > 0
    }

    if logger_cls and not log_dir:
      raise ValueError(
          "'log_dir' must be provided to `Evaluator` if `logger_cls` is "
          "non-empty.")
    self._loggers = tuple(cls(output_dir=log_dir) for cls in logger_cls)  # pytype:disable=not-instantiable

  def __del__(self):
    """Wait for metrics to be written before deletion."""
    self._metrics_executor.shutdown(wait=True)

  def close(self):
    """Wait for metrics to be written."""
    self._metrics_executor.shutdown(wait=True)

  def evaluate(
      self,
      *,
      compute_metrics: bool,
      step: Optional[int] = None,
      predict_fn: Optional[PredictFnCallable] = None,
      score_fn: Optional[ScoreFnCallable] = None,
      predict_with_aux_fn: Optional[PredictFnCallable] = None,
      model_fns: Optional[Mapping[metrics_lib.ModelOutputType,
                                  ModelFnCallable]] = None,
  ) -> MetricsAndOutputsType:
    """Predict and score self.eval_tasks.

    Evaluation must preserve the example ordering. This requirement is satisfied
    by using enumerated dataset. Each of the cached eval task datasets is an
    enumerated tf.data.Dataset where each element has (index, example) format.
    Therefore, each index serves as a unique integer id for the example.

    `predict_fn` takes as input the cached eval dataset. The output
    may be of the form Sequence[(index, token_ids)] where `token_ids` is the
    sequence of token ids output by the model with the input `example` whose
    index matches `index`. Therefore, even if `predict_fn` mixes the order of
    the examples during prediction, the order can be corrected as long as the
    correct index for each example is maintained. `predict_with_aux_fn` is
    almost exactly the same as `predict_fn`, except that it also returns a
    dictionary of auxiliary values along with each sequence of `token_ids`.

    Similarly, `score_fn` takes the cached eval dataset as input and returns
    Sequence[(index, score)] where `score` is the sequence of log likelihood
    scores for the targets in the eval dataset.

    A common example is the multi-host setup where the evaluation dataset is
    split into multiple hosts that independently make predictions and combine
    the results during which the ordering can be mixed.

    There are 4 steps involved in the evaluation using predicted tokens:

    1. Model returns indices and output_tokens: Sequence[Tuple[int,
       Sequence[int]]], potentially with some auxiliary values.
    2. output tokens are decoded by `vocab.decode`
    3. Postprocessors are applied to the decoded output. These are denoted as
       predictions.
    4. Each metric function is applied to the predictions and the cached
       targets.

    Using auxiliary values is exactly the same as predicted tokens, except that
    a Mapping[str, Sequence[Any]] is also returned. Where len(Sequence[Any])
    should correspond to the number of elements in the dataset.

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
      predict_with_aux_fn: a user-defined function that has exactly the same
        behaviour as predict_fn, except that it also returns a dictionary of
        auxiliary values. Only called if predict_with_aux metrics exist for the
        tasks.
      model_fns: a dict mapping model output type to the model function
        (user-defined) that can produce outputs of that model output type.

    Returns:
      metrics: a Future containing a mapping from task name to computed metrics,
        or None if `compute_metrics` is False.
      all_output: a mapping from task name to all the output that metric
        evaluation needs.
    """
    # Reorganizes score_fn, predict_fn and predict_with_aux_fn into model_fns
    # for backward compatibility.
    model_fns = dict(model_fns or {})
    if predict_fn:
      model_fns[metrics_lib.ModelOutputType.PREDICTION] = predict_fn
    if score_fn:
      model_fns[metrics_lib.ModelOutputType.SCORE] = score_fn
    if predict_with_aux_fn:
      model_fns[
          metrics_lib.ModelOutputType.PREDICTION_WITH_AUX] = predict_with_aux_fn

    # Computes all the model outputs needed by metrics, and organizes them in
    # a dictionary structure - model_outputs: Mapping[ModelOutputType, Any]
    # make sure the examples are sorted by example index.
    all_output = {}

    for task in self.eval_tasks:
      logging.info("Evaluating %s", task.name)

      all_output[task.name] = {}
      # We loop over metrics and collect all the model outputs
      # that are needed for metric computation.
      for metric_obj in task.metric_objs:
        model_output_type = metric_obj.model_output_type
        if model_output_type not in all_output[task.name]:
          model_fn = model_fns[model_output_type]
          all_output[task.name][model_output_type] = _extract_model_output(
              self._cached_model_datasets[task.name], model_fn)

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
        metrics = self._compute_clu_metrics(all_output, step)
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
    return all_metrics, all_output

  def _decode_and_postprocess_predictions(self, task, predicted_tokens,
                                          task_dataset, targets):
    """Run the model's predicted outputs through the decoder."""
    task_vocab = task.output_features[self._target_field_name].vocabulary
    task_predicted_tokens = predicted_tokens[task.name]

    if len(targets) != len(task_predicted_tokens):
      raise ValueError(f"len(targets)({len(targets)}) != "
                       f"len(predictions)({len(task_predicted_tokens)})")

    outputs = [task_vocab.decode(tokens) for tokens in task_predicted_tokens]
    postprocessed_outputs = [
        task.postprocess_fn(d, example=ex, is_target=False)
        for d, ex in zip(outputs, tfds.as_numpy(task_dataset))
    ]
    return outputs, postprocessed_outputs

  def _compute_metrics(self,
                       predicted_tokens: AllOutputTokensType,
                       scores: AllOutputScoresType,
                       all_aux_values: AllOutputAuxValuesType,
                       step: Optional[int] = None) -> AllMetricsType:
    """Computes and logs metrics given the predicted tokens and scores.

    Args:
      predicted_tokens: a mapping from task name to the output tokens from
        `predict_fn`, for tasks that have `predict_metric_fns`.
      scores: a mapping from task name to the output scores from `score_fn` for
        tasks that have `score_predict_fns`.
      all_aux_values: a mapping from task name to the output auxiliary values
        from `predict_fn` for tasks that have `predict_metric_with_aux_fns`.
      step: an optional step number of the current evaluation. If unspecified, a
        dummy value of -1 will be used.

    Returns:
      A mapping from task name to computed metrics.
    """
    all_metrics = {}

    for task in self.eval_tasks:
      logging.info("Computing metrics for %s", task.name)
      task_dataset = self.cached_task_datasets[task.name]
      targets = self.cached_targets[task.name]

      task_metrics = []
      inferences = {}

      if task.predict_metric_fns or task.predict_with_aux_metric_fns:
        (outputs,
         postprocessed_outputs) = self._decode_and_postprocess_predictions(
             task, predicted_tokens, task_dataset, targets)
        inferences["output"] = outputs
        inferences["prediction"] = postprocessed_outputs

      if task.predict_metric_fns:
        task_metrics.extend([
            metric_fn(targets, inferences["prediction"])
            for metric_fn in task.predict_metric_fns
        ])

      if task.predict_with_aux_metric_fns:
        aux_values = all_aux_values[task.name]
        task_metrics.extend([
            metric_fn(targets, inferences["prediction"], aux_values)
            for metric_fn in task.predict_with_aux_metric_fns
        ])
        inferences["aux_value"] = aux_values

      if task.score_metric_fns:
        task_scores = scores[task.name]
        is_tuple = isinstance(task_scores, tuple)
        if ((not is_tuple and len(targets) != len(task_scores)) or
            (is_tuple and len(targets) != len(task_scores[0]))):
          raise ValueError(f"len(targets)({len(targets)}) != "
                           f"len(task_scores)({len(task_scores)})")
        task_metrics.extend([
            metric_fn(targets, task_scores)
            for metric_fn in task.score_metric_fns
        ])
        inferences["score"] = task_scores

      all_metrics[task.name] = {}
      for k, v in itertools.chain(*[m.items() for m in task_metrics]):
        if k in all_metrics[task.name]:
          raise ValueError(f"Duplicate metric key '{k}' in Task '{task.name}'.")
        all_metrics[task.name][k] = v

      # pyformat: disable
      metrics = {
          k: metrics_lib.Scalar(v)
             if not isinstance(v, metrics_lib.MetricValue) else v
          for k, v in all_metrics[task.name].items()
      }
      # pyformat: enable
      for logger in self.loggers:
        logger(
            task_name=task.name,
            step=step,
            metrics=metrics,
            dataset=task_dataset,
            inferences=inferences,
            targets=targets)

    return all_metrics

  def _compute_clu_metrics(self,
                           all_output,
                           step: Optional[int] = None) -> AllMetricsType:
    """Computes and logs metrics given the predicted tokens and scores.

    Args:
      all_output: a mapping from task name to the model outputs needed by
        metrics of that task.
      step: an optional step number of the current evaluation. If unspecified, a
        dummy value of -1 will be used.

    Returns:
      A mapping from task name to computed metrics.
    """
    all_metrics = {}

    for task in self.eval_tasks:
      logging.info("Computing metrics for %s", task.name)
      task_dataset = self.cached_task_datasets[task.name]

      task_metrics = []
      inferences = {}
      for metric_obj in task.metric_objs:
        model_output = all_output[task.name][metric_obj.model_output_type]
        metric_instance = metric_obj.from_model_output(
            tfds.as_numpy(task_dataset), model_output, task.output_features,
            self._target_field_name)
        task_metrics.append(metric_instance.compute())
        # Records inferences for legacy logging compatibility.
        inferences.update({
            key: val
            for key, val in metric_instance.targets_and_inferences.items()
            if key != "targets"
        })
      # Records targets for legacy logging compatibility.
      # Each metric_instance should have identical targets.
      # Chooses the last metric_instance for this recording purpose.
      targets = metric_instance.targets_and_inferences["targets"]

      all_metrics[task.name] = {}
      for k, v in itertools.chain(*[m.items() for m in task_metrics]):
        if k in all_metrics[task.name]:
          raise ValueError(f"Duplicate metric key '{k}' in Task '{task.name}'.")
        all_metrics[task.name][k] = v

      # pyformat: disable
      metrics = {
          k: metrics_lib.Scalar(v)
             if not isinstance(v, metrics_lib.MetricValue) else v
          for k, v in all_metrics[task.name].items()
      }
      # pyformat: enable
      for logger in self.loggers:
        logger(
            task_name=task.name,
            step=step,
            metrics=metrics,
            dataset=task_dataset,
            targets=targets,
            inferences=inferences)

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
  def cached_targets(self) -> Mapping[str, Sequence[str]]:
    return self._cached_targets

  @property
  def model_feature_shapes(self) -> Mapping[str, Tuple[int, ...]]:
    return self._model_feature_shapes

  @property
  def loggers(self) -> Tuple[loggers_lib.Logger]:
    return tuple(self._loggers)
