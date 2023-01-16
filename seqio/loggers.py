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

"""Classes for logging evaluation metrics and inference results."""

import abc
import base64
import dataclasses
import itertools
import json
import os
import time
from typing import Any, Mapping, Optional, Sequence, Type, Dict, Tuple

from absl import logging
import numpy as np
from seqio import metrics as metrics_lib
import tensorflow as tf
import tensorflow_datasets as tfds


def skip_none_value_dict_factory(
    data: Sequence[Tuple[str, Any]]
) -> Dict[str, Any]:
  """Dictionnary factory which skip None value."""
  return {k: v for k, v in data if v is not None}


class Logger(abc.ABC):
  """Abstract base class for logging.

  Attributes:
    output_dir: a directory to save the logging results (e.g., TensorBoard
      summary) as well as the evaluation results (e.g., "inputs_pretokenized",
      "target_pretokenize" and "prediction").
  """

  def __init__(self, output_dir):
    self.output_dir = output_dir

  @abc.abstractmethod
  def __call__(
      self,
      task_name: str,
      step: Optional[int],
      metrics: Mapping[str, metrics_lib.MetricValue],
      dataset: Optional[tf.data.Dataset],
      inferences: Optional[Mapping[str, Sequence[Any]]],
      targets: Optional[Sequence[Any]],
  ) -> None:
    """Logs the metrics and inferences for each task.

    Args:
      task_name: The name of the task these datapoints are relevant to.
      step: The timestep to place this datapoint at.
      metrics: A mapping from series names to numeric datapoints to be added to
        that series.
      dataset: The Task dataset.
      inferences: Mapping from inference type ("predictions", "scores",
        "aux_value") to the model outputs, aligned with the dataset.
      targets: The postprocessed targets, aligned with the dataset.
    """
    ...


class PyLoggingLogger(Logger):
  """A logger that writes metrics using the standard Python log."""

  def __init__(self, output_dir: str, level: int = logging.INFO):
    self._level = level
    super().__init__(output_dir)

  def __call__(
      self,
      task_name: str,
      step: Optional[int],
      metrics: Mapping[str, metrics_lib.MetricValue],
      dataset: Optional[tf.data.Dataset],
      inferences: Optional[Mapping[str, Sequence[Any]]],
      targets: Optional[Sequence[Any]],
  ) -> None:
    del dataset
    del inferences
    del targets
    step = step or -1
    for metric_name, metric_value in metrics.items():
      if isinstance(metric_value, metrics_lib.Scalar):
        float_value = float(np.array(metric_value.value))
        strvalue = f"{float_value:.3f}"
      elif isinstance(metric_value, metrics_lib.Text):
        strvalue = metric_value.textdata
      else:
        strvalue = f"unloggable type {type(metric_value)}"
      logging.info(
          "%s/%s at step %d: %s", task_name, metric_name, step, strvalue
      )


class TensorBoardLogger(Logger):
  """A logger that writes metrics to TensorBoard summaries."""

  def __init__(self, output_dir: str):
    """TensorBoardLogger initializer.

    Args:
      output_dir: The base directory where all logs will be written.
    """
    super().__init__(output_dir)
    self._summary_writers = {}

  def _get_summary_writer(self, summary_dir: str) -> tf.summary.SummaryWriter:
    """Get or create a summary writer for a specific task.

    Args:
      summary_dir: The task we are getting the writer for.

    Returns:
      The summary writer associated with the directory.
    """
    if summary_dir not in self._summary_writers:
      self._summary_writers[summary_dir] = tf.summary.create_file_writer(
          summary_dir, flush_millis=120
      )
    return self._summary_writers[summary_dir]

  def _write_metric(
      self,
      tag: str,
      value: metrics_lib.MetricValue,
      step: int,
      writer: tf.summary.SummaryWriter,
  ):
    """Log a metric value to tensorboard, dispatched on value type."""
    if isinstance(value, metrics_lib.Scalar):
      value: metrics_lib.Scalar = value
      value = float(np.array(value.value))
      with writer.as_default():
        tf.summary.scalar(name=tag, data=value, step=step)
    elif isinstance(value, metrics_lib.Image):
      value: metrics_lib.Image = value
      image = tf.convert_to_tensor(value.image)
      with writer.as_default():
        tf.summary.image(
            name=tag, data=image, step=step, max_outputs=value.max_outputs
        )
    elif isinstance(value, metrics_lib.Audio):
      value: metrics_lib.Audio = value
      audio = tf.convert_to_tensor(value.audiodata, dtype=tf.float32)
      with writer.as_default():
        tf.summary.audio(
            name=tag,
            data=audio,
            sample_rate=value.sample_rate,
            step=step,
            max_outputs=value.max_outputs,
            encoding="wav",
        )
    elif isinstance(value, metrics_lib.Histogram):
      value: metrics_lib.Histogram = value
      values = np.array(value.values)
      with writer.as_default():
        tf.summary.histogram(
            name=tag, data=values, step=step, buckets=value.bins
        )
    elif isinstance(value, metrics_lib.Text):
      value: metrics_lib.Text = value
      if not isinstance(value.textdata, (str, bytes)):
        raise ValueError("`textdata` should be of the type `str` or `bytes`.")
      with writer.as_default():
        tf.summary.text(name=tag, data=tf.constant(value.textdata), step=step)
    elif isinstance(value, metrics_lib.Generic):
      with writer.as_default():
        tf.summary.write(
            tag=tag, tensor=value.tensor, metadata=value.metadata, step=step
        )
    else:
      raise TypeError(
          f"Value type not understood, got '{type(value).__name__}'."
      )

  def __call__(
      self,
      task_name: str,
      step: Optional[int],
      metrics: Mapping[str, metrics_lib.MetricValue],
      dataset: Optional[tf.data.Dataset],
      inferences: Optional[Mapping[str, Sequence[Any]]],
      targets: Optional[Sequence[Any]],
  ) -> None:
    """Log metrics to tensorboard.

    Args:
      task_name: The name of the task these datapoints are relevant to.
      step: The timestep to place this datapoint at.
      metrics: A mapping from series names to numeric datapoints to be added to
        that series.
      dataset: The Task dataset, which is unused by this logger.
      inferences: The model outputs, which are unused by this logger.
      targets: The postprocessed targets, which are unused by this logger.
    """
    del dataset
    del inferences
    del targets
    if step is None:
      logging.warning(
          "Step number for the logging session is not provided. "
          "A dummy value of -1 will be used."
      )
      step = -1

    writer = self._get_summary_writer(os.path.join(self.output_dir, task_name))
    for metric_name, metric_value in metrics.items():
      # We prefix the tag with "eval/" for backward compatibility.
      # TODO(adarob): Find a way to remove this or make it an option.
      self._write_metric(
          tag=f"eval/{metric_name}",
          value=metric_value,
          step=step,
          writer=writer,
      )
    writer.flush()


class TensorBoardLoggerV1(Logger):
  """A logger that writes metrics to TensorBoard summaries in TF1."""

  def __init__(self, output_dir: str):
    """TensorBoardLoggerV1 initializer.

    Args:
      output_dir: The base directory where all logs will be written.
    """
    super().__init__(output_dir)
    self._summary_writers = {}

  def _get_summary_writer(self, task_name: str) -> tf.summary.SummaryWriter:
    """Create (if needed) and return a SummaryWriter for a given task."""
    if task_name not in self._summary_writers:
      with tf.compat.v1.Graph().as_default():
        self._summary_writers[task_name] = tf.compat.v1.summary.FileWriter(
            os.path.join(self.output_dir, task_name)
        )
    return self._summary_writers[task_name]

  def __call__(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self,
      task_name: str,
      step: Optional[int],
      metrics: Mapping[str, metrics_lib.Scalar],
      dataset: Optional[tf.data.Dataset],
      inferences: Optional[Mapping[str, Sequence[Any]]],
      targets: Optional[Sequence[Any]],
  ) -> None:
    """Log the eval results and optionally write summaries for TensorBoard.

    Note:
      This is the default implementation using tensorflow v1 operations. This
      only supports logging metrics of the Scalar type.

    Args:
      task_name: The name of the task these datapoints are relevant to.
      step: The timestep to place this datapoint at.
      metrics: A mapping from series names to numeric datapoints to be added to
        that series.
      dataset: The Task dataset, which is unused by this logger.
      inferences: The model outputs, which are unused by this logger.
      targets: The postprocessed targets, which are unused by this logger.
    """
    del dataset
    del inferences
    del targets
    if step is None:
      logging.warning(
          "Step number for the logging session is not provided. "
          "A dummy value of -1 will be used."
      )
      step = -1

    summary_writer = self._get_summary_writer(task_name)

    for metric_name, metric_value in metrics.items():
      if not isinstance(metric_value, metrics_lib.Scalar):
        raise ValueError(
            f"Value for metric '{metric_name}' should be of "
            f"type 'Scalar, got '{type(metric_value).__name__}'."
        )
      summary = tf.compat.v1.Summary()

      tag = f"eval/{metric_name}"
      logging.info("%s at step %d: %.3f", tag, step, metric_value.value)

      summary.value.add(tag=tag, simple_value=metric_value.value)
      summary_writer.add_summary(summary, step)

    summary_writer.flush()


class TensorAndNumpyEncoder(json.JSONEncoder):
  """JSON Encoder to use when encoding dicts with tensors and numpy arrays."""

  def __init__(self, *args, max_ndarray_size=32, **kwargs):
    self.max_ndarray_size = max_ndarray_size
    super().__init__(*args, **kwargs)

  def default(self, obj):
    if isinstance(obj, tf.Tensor):
      if obj.dtype == tf.bfloat16:
        # bfloat16 not supported, convert to float32.
        obj = tf.cast(obj, tf.float32)
      obj = obj.numpy()

    if isinstance(obj, np.ndarray):
      obj_dtype = obj.dtype
      if str(obj.dtype) == "bfloat16":
        # bfloat16 not supported, convert to float32.
        obj = obj.astype(np.float32)
      if obj.size <= self.max_ndarray_size:
        return obj.tolist()  # Convert arrays to lists of py-native types.
      else:
        # If the ndarray is larger than allowed, return a summary string
        # instead of the entire array.
        first_five_str = str(obj.reshape([-1])[:5].tolist())[1:-1]
        last_five_str = str(obj.reshape([-1])[-5:].tolist())[1:-1]
        return (
            f"{type(obj).__name__}(shape={obj.shape}, dtype={obj_dtype}); "
            f"summary: {first_five_str} ... {last_five_str}"
        )
    elif np.issubdtype(type(obj), np.number) or np.issubdtype(
        type(obj), np.bool_
    ):
      return obj.item()  # Convert most primitive np types to py-native types.
    elif hasattr(obj, "dtype") and obj.dtype == tf.bfloat16.as_numpy_dtype:
      return float(obj)
    elif isinstance(obj, bytes):
      # JSON doesn't support bytes. First, try to decode using utf-8 in case
      # it's text. Otherwise, just base64 encode the bytes.
      try:
        return obj.decode("utf-8")
      except UnicodeDecodeError:
        return base64.b64encode(obj)

    if dataclasses.is_dataclass(obj):
      return dataclasses.asdict(obj, dict_factory=skip_none_value_dict_factory)

    return json.JSONEncoder.default(self, obj)


def _check_json_serializable(
    field_name: str, value: Any, json_encoder_cls: Type[json.JSONEncoder]
) -> bool:
  try:
    json.dumps(value, cls=json_encoder_cls)
    return True
  except TypeError:
    logging.warning("`%s` is not JSON serializable", field_name, exc_info=True)
    return False


class JSONLogger(Logger):
  """A logger that writes metrics and model outputs to JSONL files."""

  def __init__(
      self,
      output_dir: str,
      write_n_results: Optional[int] = None,
      json_encoder_cls: Type[json.JSONEncoder] = TensorAndNumpyEncoder,
  ):
    """JSONLogger constructor.

    Args:
      output_dir: The base directory where all logs will be written.
      write_n_results: number of scores/predictions to be written to the file at
        each step. If None, scores and predictions from all examples are
        written.
      json_encoder_cls: Class to use for serializing JSON to file.
    """
    super().__init__(output_dir)
    self._write_n_results = write_n_results
    self._json_encoder_cls = json_encoder_cls

  def __call__(
      self,
      task_name: str,
      step: Optional[int],
      metrics: Mapping[str, metrics_lib.MetricValue],
      dataset: Optional[tf.data.Dataset],
      inferences: Optional[Mapping[str, Sequence[Any]]],
      targets: Optional[Sequence[Any]],
  ) -> None:
    if step is None:
      logging.warning(
          "Step number for the logging session is not provided. "
          "A dummy value of -1 will be used."
      )
      step = -1

    metrics_fname = os.path.join(self.output_dir, f"{task_name}-metrics.jsonl")

    serializable_metrics = {}
    for metric_name, metric_value in metrics.items():
      if isinstance(metric_value, metrics_lib.Scalar):
        serializable_metrics[metric_name] = metric_value.value
      elif isinstance(metric_value, metrics_lib.Text):
        serializable_metrics[metric_name] = metric_value.textdata
      elif isinstance(metric_value, metrics_lib.Generic):
        serializable_metrics[metric_name] = metric_value.tensor.tolist()
      else:
        logging.warning(
            "Skipping JSON logging of non-serializable metric '%s' of type %s.",
            metric_name,
            type(metric_value),
        )

    if metrics:
      logging.info("Appending metrics to %s", metrics_fname)
      # We simulate an atomic append for filesystems that do not suppport
      # mode="a".
      file_contents = ""
      if tf.io.gfile.exists(metrics_fname):
        with tf.io.gfile.GFile(metrics_fname, "r") as f:
          file_contents = f.read()
      with tf.io.gfile.GFile(metrics_fname + ".tmp", "w") as f:
        f.write(file_contents)
        f.write(
            json.dumps(
                {"step": step, **serializable_metrics},
                cls=self._json_encoder_cls,
            )
        )
        f.write("\n")
      tf.io.gfile.rename(metrics_fname + ".tmp", metrics_fname, overwrite=True)

    if self._write_n_results == 0:
      return

    if not inferences or not targets or not dataset:
      logging.info(
          "Skipping inference logging as one or more of inferences, "
          "targets or dataset is unset"
      )
      return

    write_tick = time.time()
    inferences_fname = os.path.join(
        self.output_dir, f"{task_name}-{step:06}.jsonl"
    )
    logging.info("Writing inferences to %s", inferences_fname)
    with tf.io.gfile.GFile(inferences_fname, "w") as f:
      inference_types = list(inferences.keys())

      # The auxiliary values have a different shape than the others to conserve
      # memory. They are handled separately below.
      all_aux_values = {}
      if "aux_value" in inference_types:
        inference_types.remove("aux_value")
        all_aux_values = inferences["aux_value"]
      to_zip = [tfds.as_numpy(dataset), targets] + [
          inferences.get(t) for t in inference_types
      ]
      examples_with_results = itertools.zip_longest(*to_zip)
      if self._write_n_results:
        examples_with_results = itertools.islice(
            examples_with_results, 0, self._write_n_results
        )
      field_names = ["target"] + inference_types

      for example_index, (inp, *results) in enumerate(examples_with_results):
        # tfds.as_numpy does not convert ragged tensors
        for k in inp:
          if isinstance(inp[k], tf.RaggedTensor):
            inp[k] = inp[k].numpy()

        json_dict = {"input": inp}

        for field_name, res in zip(field_names, results):
          if _check_json_serializable(field_name, res, self._json_encoder_cls):
            json_dict[field_name] = res

        for aux_value_name in all_aux_values:
          aux_value = inferences["aux_value"][aux_value_name][example_index]
          if _check_json_serializable(
              aux_value_name, aux_value, self._json_encoder_cls
          ):
            json_dict[f"aux_{aux_value_name}"] = aux_value

        json_str = json.dumps(json_dict, cls=self._json_encoder_cls)
        f.write(json_str + "\n")
    write_time = time.time() - write_tick
    logging.info(
        "Writing completed in %02f seconds (%02f examples/sec).",
        write_time,
        len(inferences) / write_time,
    )
