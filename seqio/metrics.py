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

"""MetricValue objects to wrap results being returned by metric funcitons."""

import dataclasses
import enum
import inspect
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union, List

import clu.metrics
import flax
import jax
import jax.numpy as jnp
import numpy as np
from seqio import utils
import tensorflow.compat.v2 as tf


@dataclasses.dataclass
class MetricValue:
  """A base method for the dataclasses that represent tensorboard values.

  Task `metric_fn`s should output `Mapping[str, MetricValue]` which will be
  written by a `Logger`.
  """


@dataclasses.dataclass
class Scalar(MetricValue):
  """The default tensorflow value, used for creating time series graphs."""

  value: Union[int, float]


@dataclasses.dataclass
class Text(MetricValue):
  """Text to output to tensorboard, markdown is rendered by tensorboard."""

  textdata: Union[str, bytes]


@dataclasses.dataclass
class Image(MetricValue):
  """An image to output to tensorboard.

  The format for the image array should match the format expected for the data
  parameter described
  [here](https://www.tensorflow.org/api_docs/python/tf/summary/image).
  """

  image: np.ndarray
  max_outputs: int = 3


@dataclasses.dataclass
class Audio(MetricValue):
  """An audio example to output to tensorboard.

  The format for the audio array should match the format expected for the data
  parameter described
  [here](https://www.tensorflow.org/api_docs/python/tf/summary/audio).
  """

  audiodata: np.ndarray
  sample_rate: int = 44100
  max_outputs: int = 3


@dataclasses.dataclass
class Histogram(MetricValue):
  """A histogram to output to tensorboard."""

  values: np.ndarray
  bins: Optional[int] = None


@dataclasses.dataclass
class Generic(MetricValue):
  """A raw tensor to output to tensorboard."""

  tensor: np.ndarray
  metadata: tf.compat.v1.SummaryMetadata


class ModelOutputType(enum.IntEnum):
  """Model output types."""

  PREDICTION = 1
  SCORE = 2
  PREDICTION_WITH_AUX = 3
  SCORE_WITH_INTERMEDIATES = 4

  @classmethod
  def to_str(cls, enm):
    return {
        cls.PREDICTION: "prediction",
        cls.SCORE: "score",
        cls.PREDICTION_WITH_AUX: "prediction_with_aux",
        cls.SCORE_WITH_INTERMEDIATES: "score_with_intermediates",
    }[enm]


MetricFnCallable = Callable[..., Mapping[str, Union[MetricValue, float]]]


@flax.struct.dataclass
class Metric(clu.metrics.Metric):
  """Base Metric class for seqio evaluation."""

  model_output_type: ModelOutputType

  @classmethod
  def from_model_output(
      cls,
      inputs: Sequence[Mapping[str, Any]],
      model_output: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
      features: Mapping[str, utils.Feature],
      target_field_name: str = "targets",
      mask: Optional[np.ndarray] = None,
      indices_2d: Optional[np.ndarray] = None) -> "Metric":
    """Creates a `seqio.Metric` from model outputs.

    Args:
      inputs: Examples in dataset.
      model_output: Model output computed by model functions.
      features: Output features defined in seqio.Task.
      target_field_name: Field name of the target sequence.
      mask: A boolean array to indicate which examples in the inputs are
        included for metric evaluation.
      indices_2d: 2d-indices of examples in the inputs/model_output. First
        dimension is shard id, the second is the example id within that shard.

    Returns:
      An instance of Metric.
    Raises:
      NotImplementedError: Must override from_model_output()
    """

    raise NotImplementedError("Must override from_model_output()")


class CollectingMetric(clu.metrics.CollectingMetric):
  """CollectingMetric interface for seqio evaluation."""

  @classmethod
  def from_model_output(  # pylint:disable=missing-function-docstring
      cls,
      inputs: Sequence[Mapping[str, Any]],
      model_output: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
      features: Mapping[str, utils.Feature],
      target_field_name: str = "targets",
      mask: Optional[np.ndarray] = None,
      indices_2d: Optional[np.ndarray] = None):

    del inputs, features, target_field_name
    num_examples = len(model_output[0]) if isinstance(
        model_output, tuple) else len(model_output)

    if mask is None:
      mask = jnp.ones((num_examples,), jnp.int32)

    if indices_2d is None:
      indices_2d = jnp.transpose(
          jnp.stack([
              jnp.zeros((num_examples,), jnp.int32),
              jnp.arange(num_examples, dtype=jnp.int32)
          ]))
    return cls(values={
        "model_output": model_output,
        "indices_2d": indices_2d,
        "mask": mask
    })

  def actual_compute(self, task_dataset_as_numpy, task_output_features,
                     target_field_name: str = "targets",
                     cached_targets: Optional[List[str]] = None):
    """Implements the metric computation logics for CollectingMetric.

    Args:
      task_dataset_as_numpy: Examples in dataset.
      task_output_features: Output features defined in the seqio.Task.
      target_field_name: Field name of the target sequence.
      cached_targets: targets that have been cached by Evaluator and can be
        supplied here to save time of post-processing targets.

    Returns:
      A tuple of two items, first item is a dict of metric results, the second
        item is targets_and_inferences.
    Raises:
      NotImplementedError: Must override from_model_output()
    """

    raise NotImplementedError("Must override from_model_output()")


# TODO(kehanghan): consider using CollectingMetric for LegacyMetric.
@flax.struct.dataclass
class LegacyMetric(Metric):
  """Metric class for legacy use-case where metric fn is supplied."""

  _metric_fn: MetricFnCallable
  _postprocess_fn: Callable[..., Any]
  metric_fn_kwargs: Dict[str, Any]
  targets_and_inferences: Dict[str, Any]

  @classmethod
  def empty(cls, metric_fn, postprocess_fn) -> "LegacyMetric":
    pos_args = tuple(
        key
        for key, param in inspect.signature(metric_fn).parameters.items()
        if param.default == inspect.Parameter.empty
        and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if pos_args == ("targets", "scores"):
      model_output_type = ModelOutputType.SCORE
    elif pos_args == ("targets", "predictions"):
      model_output_type = ModelOutputType.PREDICTION
    elif pos_args == ("targets", "predictions", "aux_values"):
      model_output_type = ModelOutputType.PREDICTION_WITH_AUX
    else:
      raise ValueError(
          "Metric functions must have positional arguments matching either "
          "('targets', 'scores'), ('targets', 'predictions') or "
          "('targets', 'predictions', 'aux_values'). "
          f"Got: {pos_args}")

    return cls(
        _metric_fn=metric_fn,
        _postprocess_fn=postprocess_fn,
        model_output_type=model_output_type,
        metric_fn_kwargs={},
        targets_and_inferences={},
    )

  def postprocess_fn(
      self, targets_or_predictions: Any, **postprocess_kwargs
  ) -> Any:
    """Applies the postprocessing to targets or predictions."""
    if self._postprocess_fn:
      return self._postprocess_fn(targets_or_predictions, **postprocess_kwargs)
    return targets_or_predictions

  def from_model_output(  # pylint:disable=arguments-renamed
      self,
      inputs: Sequence[Mapping[str, Any]],
      model_output: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
      features: Mapping[str, utils.Feature],
      target_field_name: str = "targets",
      mask: Optional[np.ndarray] = None,
  ) -> "LegacyMetric":
    if not self.metric_fn_kwargs.get("targets"):
      # Postprocesses the targets here.
      postprocessed_targets = []
      for ex in inputs:
        pretokenized_target_field_name = target_field_name + "_pretokenized"
        if pretokenized_target_field_name in ex:
          target = ex[pretokenized_target_field_name]
        else:
          target = features[target_field_name].vocabulary.decode(
              list(ex[target_field_name])
          )
        if isinstance(target, bytes):
          target = target.decode("utf-8")

        postprocessed_targets.append(
            self.postprocess_fn(target, example=ex, is_target=True)
        )
      self.metric_fn_kwargs["targets"] = postprocessed_targets
      self.targets_and_inferences["targets"] = postprocessed_targets

    if self.model_output_type == ModelOutputType.SCORE:
      self.metric_fn_kwargs["scores"] = model_output
      self.targets_and_inferences["score"] = model_output
    else:
      vocab = features[target_field_name].vocabulary
      if self.model_output_type == ModelOutputType.PREDICTION_WITH_AUX:
        self.metric_fn_kwargs["aux_values"] = model_output[1]
        self.targets_and_inferences["aux_value"] = model_output[1]
        predictions = [vocab.decode(tokens) for tokens in model_output[0]]
      elif self.model_output_type == ModelOutputType.PREDICTION:
        # Default behavior for top-1 decoding, model_output is a 2d array.
        # first dim is for batch, second is for sequence length.
        if isinstance(model_output, np.ndarray) and model_output.ndim == 2:
          predictions = [vocab.decode(tokens) for tokens in model_output]
        else:
          # In case of top-k decoding, model_output will be a 3d array
          # first dim is for batch, second is for num_decodes, third is for
          # sequence length.
          predictions = []
          for sequences in model_output:
            predictions_for_one_example = []
            for sequence in sequences:
              predictions_for_one_example.append(vocab.decode(sequence))
            predictions.append(predictions_for_one_example)
        self.targets_and_inferences["output"] = predictions

      # Postprocesses the predictions here.
      postprocessed_predictions = [
          self.postprocess_fn(p, example=ex, is_target=False)
          for ex, p in zip(inputs, predictions)
      ]

      self.metric_fn_kwargs["predictions"] = postprocessed_predictions
      self.targets_and_inferences["prediction"] = postprocessed_predictions

    return self

  def compute(self):
    return self._metric_fn(**self.metric_fn_kwargs)


def remove_padding_examples(model_output, indices_2d, mask):
  """Removes padding examples indicated by the mask array.

  Args:
    model_output: model outputs of all the examples (including the padding
      ones). The padding examples are used to make sure during inference, the
      inference function receives full batch if the last batch does not enough
      examples.
    indices_2d: 2d indices of all the examples.
    mask: an array of booleans. 1 indicates valid example, 0 indicates padded
      example that needs to be removed.

  Returns:
    2d-indices and model outputs of all the non-padding examples.
  """
  indices_2d = indices_2d[mask == 1]
  model_output = jax.tree_map(lambda x: x[mask == 1], model_output)

  return indices_2d, model_output


def globally_sort_model_output(model_output, indices_2d):
  """Globally sorts model ouputs by the 2d indices of the examples.

  The sorting is done first by shard id (first index of the 2d-index) and then
  by example id (second index of the 2d-index).

  Args:
    model_output: model outputs of all the examples.
    indices_2d: 2d indices of all the examples.

  Returns:
    sorted model outputs.
  """
  permutation = np.lexsort((indices_2d[:, 1], indices_2d[:, 0]))

  def _sort_by_permutation(x):
    return np.array([x[permutation[i]]for i in range(len(permutation))])

  model_output = jax.tree_map(_sort_by_permutation, model_output)

  return model_output


@flax.struct.dataclass
class PassthroughLegacyMetric(CollectingMetric):
  """Makes PassthroughLegacyMetric from metric functions."""

  @classmethod
  def from_metric_fn(cls,
                     metric_fn: MetricFnCallable,
                     postprocess_fn: Optional[Callable[..., Any]] = None):
    """Creates `PassthroughLegacyMetric` from `metric_fn` and `postprocess_fn`.

    Example:

    ```
    squad_cls = PassthroughLegacyMetric.from_metric_fn(
        metric_fn=t5_metrics.squd, postprocess_fn=t5_postprocessors.qa)
    ```

    Args:
      metric_fn: Function used to compute metric.
      postprocess_fn: Function used to process targets (vocab decoded) and
        predictions (vocab decoded) before feeding into metric_fn.

    Returns:
      A `Metric` that calls `metric_fn` and `postprocess_fn` in its
      `.from_model_output()`.
    """

    def _get_model_output_type() -> ModelOutputType:
      pos_args = tuple(
          key
          for key, param in inspect.signature(metric_fn).parameters.items()
          if param.default == inspect.Parameter.empty
          and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD)
      if pos_args == ("targets", "scores"):
        model_output_type = ModelOutputType.SCORE
      elif pos_args == ("targets", "predictions"):
        model_output_type = ModelOutputType.PREDICTION
      elif pos_args == ("targets", "predictions", "aux_values"):
        model_output_type = ModelOutputType.PREDICTION_WITH_AUX
      else:
        raise ValueError(
            "Metric functions must have positional arguments matching either "
            "('targets', 'scores'), ('targets', 'predictions') or "
            "('targets', 'predictions', 'aux_values'). "
            f"Got: {pos_args}")

      return model_output_type

    @flax.struct.dataclass
    class FromMetricFun(cls):
      """Wrapper PassthroughLegacyMetric class that runs metric_fn."""

      model_output_type: ModelOutputType = _get_model_output_type()

      @classmethod
      def postprocess(cls, targets_or_predictions: Any,
                      **postprocess_kwargs) -> Any:
        """Applies the postprocessing to targets or predictions."""
        if postprocess_fn:
          return postprocess_fn(targets_or_predictions, **postprocess_kwargs)
        return targets_or_predictions

      def postprocess_targets(self, task_dataset_as_numpy, task_output_features,
                              target_field_name: str = "targets"):
        """Applies the postprocessing to targets."""
        # Postprocesses the targets here.
        postprocessed_targets = []
        for ex in task_dataset_as_numpy:
          pretokenized_target_field_name = target_field_name + "_pretokenized"
          if pretokenized_target_field_name in ex:
            target = ex[pretokenized_target_field_name]
          else:
            target = task_output_features[
                target_field_name
            ].vocabulary.decode(list(ex[target_field_name]))
          if isinstance(target, bytes):
            target = target.decode("utf-8")

          postprocessed_targets.append(
              type(self).postprocess(target, example=ex, is_target=True)
          )
        return postprocessed_targets

      def actual_compute(self, task_dataset_as_numpy, task_output_features,
                         target_field_name: str = "targets",
                         cached_targets: Optional[List[str]] = None):
        # Postprocesses the targets here.
        if not cached_targets:
          postprocessed_targets = self.postprocess_targets(
              task_dataset_as_numpy, task_output_features, target_field_name)
        else:
          postprocessed_targets = cached_targets

        metric_fn_kwargs, targets_and_inferences = {}, {}
        metric_fn_kwargs["targets"] = postprocessed_targets
        targets_and_inferences["targets"] = postprocessed_targets

        # We process the model outputs here by the steps below.
        # Step 1: removes padded examples using mask.
        indices_2d, model_output = remove_padding_examples(
            self.values["model_output"],
            self.values["indices_2d"],
            self.values["mask"],
        )

        assert len(postprocessed_targets) == len(indices_2d)

        # Step 2: sorts the model outputs by 2d-indices, namely (shard_id,
        # index_within_shard) to align with targets.
        model_output = globally_sort_model_output(model_output, indices_2d)

        if type(self).model_output_type == ModelOutputType.SCORE:
          metric_fn_kwargs["scores"] = model_output
          targets_and_inferences["score"] = model_output
        else:
          vocab = task_output_features[target_field_name].vocabulary
          if type(
              self).model_output_type == ModelOutputType.PREDICTION_WITH_AUX:
            metric_fn_kwargs["aux_values"] = model_output[1]
            targets_and_inferences["aux_value"] = model_output[1]
            predictions = [vocab.decode(tokens) for tokens in model_output[0]]
          elif type(self).model_output_type == ModelOutputType.PREDICTION:
            # Default behavior for top-1 decoding, model_output is a 2d array.
            # first dim is for batch, second is for sequence length.
            if isinstance(model_output, np.ndarray) and model_output.ndim == 2:
              predictions = [vocab.decode(tokens) for tokens in model_output]
            elif (
                isinstance(model_output, np.ndarray) and model_output.ndim == 3
            ):
              # In case of top-k decoding, model_output will be a 3d array
              # first dim is for batch, second is for num_decodes, third is for
              # sequence length.
              predictions = []
              for sequences in model_output:
                predictions_for_one_example = []
                for sequence in sequences:
                  predictions_for_one_example.append(vocab.decode(sequence))
                predictions.append(predictions_for_one_example)
            else:
              # If neither 2d or 3d, assume that model_output is already
              # decoded.
              predictions = model_output
          targets_and_inferences["output"] = predictions

          # Postprocesses the predictions here.
          postprocessed_predictions = [
              type(self).postprocess(p, example=ex, is_target=False)
              for ex, p in zip(task_dataset_as_numpy, predictions)
          ]

          metric_fn_kwargs["predictions"] = postprocessed_predictions
          targets_and_inferences["prediction"] = postprocessed_predictions

        return metric_fn(**metric_fn_kwargs), targets_and_inferences

    return FromMetricFun
