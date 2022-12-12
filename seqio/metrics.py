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

"""MetricValue objects to wrap results being returned by metric funcitons."""

import dataclasses
import enum
import inspect
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import clu.metrics
import flax
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


class ModelOutputType(enum.Enum):
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

  @staticmethod
  def model_output_type() -> ModelOutputType:
    """Determines the type of model output this metric needs."""

    raise NotImplementedError("Must override model_output_type()")

  @classmethod
  def from_model_output(
      cls,
      inputs: Sequence[Mapping[str, Any]],
      model_output: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
      features: Mapping[str, utils.Feature],
      target_field_name: str = "targets",
      mask: Optional[np.ndarray] = None) -> "Metric":
    """Creates a `seqio.Metric` from model outputs.

    Args:
      inputs: Examples in dataset.
      model_output: Model output computed by model functions.
      features: Output features defined in seqio.Task.
      target_field_name: Field name of the target sequence.
      mask: A boolean array to indicate which example in the inputs are included
        for metric evaluation.

    Returns:
      An instance of Metric.
    Raises:
      NotImplementedError: Must override from_model_output()
    """

    raise NotImplementedError("Must override from_model_output()")


# TODO(kehanghan): consider using CollectingMetric for LegacyMetric.
@flax.struct.dataclass
class LegacyMetric(Metric):
  """Makes LegacyMetric from metric functions."""

  metric_fn_kwargs: Dict[str, Any]
  targets_and_inferences: Dict[str, Any]

  @classmethod
  def empty(cls) -> "LegacyMetric":
    return cls(metric_fn_kwargs={}, targets_and_inferences={})

  @classmethod
  def from_metric_fn(cls,
                     metric_fn: MetricFnCallable,
                     postprocess_fn: Optional[Callable[..., Any]] = None):
    """Constructs `LegacyMetric` from `metric_fn` and `postprocess_fn`.

    Example:

    ```
    squad_cls = LegacyMetric.from_metric_fn(
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

    @flax.struct.dataclass
    class FromMetricFun(cls):
      """Wrapper LegacyMetric class that runs metric_fn."""

      metric_fn_kwargs: Dict[str, Any]
      targets_and_inferences: Dict[str, Any]

      @classmethod
      def postprocess(cls, targets_or_predictions: Any,
                      **postprocess_kwargs) -> Any:
        """Applies the postprocessing to targets or predictions."""
        if postprocess_fn:
          return postprocess_fn(targets_or_predictions, **postprocess_kwargs)
        return targets_or_predictions

      @staticmethod
      def model_output_type() -> ModelOutputType:  # pylint:disable=missing-function-docstring
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

      @classmethod
      def from_model_output(  # pylint:disable=missing-function-docstring
          cls,
          inputs: Sequence[Mapping[str, Any]],
          model_output: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
          features: Mapping[str, utils.Feature],
          target_field_name: str = "targets",
          mask: Optional[np.ndarray] = None):

        del mask
        # Postprocesses the targets here.
        postprocessed_targets = []
        for ex in inputs:
          pretokenized_target_field_name = target_field_name + "_pretokenized"
          if pretokenized_target_field_name in ex:
            target = ex[pretokenized_target_field_name]
          else:
            target = features[target_field_name].vocabulary.decode(
                list(ex[target_field_name]))
          if isinstance(target, bytes):
            target = target.decode("utf-8")

          postprocessed_targets.append(
              cls.postprocess(target, example=ex, is_target=True))

        metric_fn_kwargs, targets_and_inferences = {}, {}
        metric_fn_kwargs["targets"] = postprocessed_targets
        targets_and_inferences["targets"] = postprocessed_targets

        if cls.model_output_type() == ModelOutputType.SCORE:
          metric_fn_kwargs["scores"] = model_output
          targets_and_inferences["score"] = model_output
        else:
          vocab = features[target_field_name].vocabulary
          if cls.model_output_type() == ModelOutputType.PREDICTION_WITH_AUX:
            metric_fn_kwargs["aux_values"] = model_output[1]
            targets_and_inferences["aux_value"] = model_output[1]
            predictions = [vocab.decode(tokens) for tokens in model_output[0]]
          elif cls.model_output_type() == ModelOutputType.PREDICTION:
            # Default behavior for top-1 decoding, model_output is a list of
            # lists. Also check empty outputs so that we don't attempt to access
            # non-existent elements.
            if (
                not model_output
                or not isinstance(model_output[0], list)
                or not model_output[0]
                or not isinstance(model_output[0][0], list)
            ):
              predictions = [vocab.decode(tokens) for tokens in model_output]
            else:
              # In case of top-k decoding, model_output will be a list of list
              # of lists. For instance, a top-2 output looks like:
              # [[t11, t12, t13], [t21, t22]], with tij the j-th token of the
              # i-th output.
              predictions = []
              for sequences in model_output:
                predictions_for_one_example = []
                for sequence in sequences:
                  predictions_for_one_example.append(vocab.decode(sequence))
                predictions.append(predictions_for_one_example)
          targets_and_inferences["output"] = predictions

          # Postprocesses the predictions here.
          postprocessed_predictions = [
              cls.postprocess(p, example=ex, is_target=False)
              for ex, p in zip(inputs, predictions)
          ]

          metric_fn_kwargs["predictions"] = postprocessed_predictions
          targets_and_inferences["prediction"] = postprocessed_predictions

        return cls(metric_fn_kwargs=metric_fn_kwargs,
                   targets_and_inferences=targets_and_inferences)

      def compute(self):
        return metric_fn(**self.metric_fn_kwargs)

    return FromMetricFun
