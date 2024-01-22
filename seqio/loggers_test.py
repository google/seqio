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

"""Tests for seqio.loggers."""

# pylint:disable=g-bare-generic,g-long-lambda

import collections
import dataclasses
import itertools
import json
import os
from typing import Optional

import numpy as np
from seqio import loggers
from seqio import metrics as metrics_lib
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# For faster testing.
tf.compat.v1.enable_eager_execution()


class TensorBoardLoggerTestV1(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.logger = loggers.TensorBoardLoggerV1(self.create_tempdir().full_path)

  def test_logging(self):
    task_metrics = {
        "rouge1": metrics_lib.Scalar(50),
        "rouge2": metrics_lib.Scalar(100),
    }
    self.logger(
        task_name="log_eval_task",
        step=1,
        metrics=task_metrics,
        dataset=tf.data.Dataset.range(0),
        inferences={},
        targets=[],
    )
    task_output_dir = os.path.join(self.logger.output_dir, "log_eval_task")
    event_file = os.path.join(
        task_output_dir, tf.io.gfile.listdir(task_output_dir)[0]
    )
    # First event is boilerplate
    serialized_events = list(
        tfds.as_numpy(tf.data.TFRecordDataset(event_file))
    )[1:]
    event1 = tf.compat.v1.Event.FromString(serialized_events[0]).summary.value[
        0
    ]
    rouge1 = event1.simple_value
    tag_rouge1 = event1.tag
    event2 = tf.compat.v1.Event.FromString(serialized_events[1]).summary.value[
        0
    ]
    rouge2 = event2.simple_value
    tag_rouge2 = event2.tag

    self.assertEqual(tag_rouge1, "eval/rouge1")
    self.assertEqual(tag_rouge2, "eval/rouge2")
    self.assertAlmostEqual(rouge1, 50, places=4)
    self.assertAlmostEqual(rouge2, 100, places=4)


class TensorBoardLoggerTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.logger = loggers.TensorBoardLogger(self.create_tempdir().full_path)

  def _log_and_read(self, metrics):
    self.logger(
        task_name="log_eval_task",
        step=1,
        metrics=metrics,
        dataset=tf.data.Dataset.range(0),
        inferences={},
        targets=[],
    )
    task_summary_dir = os.path.join(self.logger.output_dir, "log_eval_task")
    event_file = os.path.join(
        task_summary_dir, tf.io.gfile.listdir(task_summary_dir)[0]
    )
    logged_metrics = {}
    plugin = {}
    for event_str in tfds.as_numpy(tf.data.TFRecordDataset(event_file).skip(1)):
      tf.compat.v1.logging.info(tf.compat.v1.Event.FromString(event_str))
      value = tf.compat.v1.Event.FromString(event_str).summary.value[0]
      logged_metrics[value.tag] = tf.make_ndarray(value.tensor)
      plugin[value.tag] = value.metadata.plugin_data.plugin_name
    return logged_metrics, plugin

  def test_log_scalar(self):
    task_metrics = {
        "rouge1": metrics_lib.Scalar(50),
        "rouge2": metrics_lib.Scalar(np.float32(100)),  # pytype: disable=wrong-arg-types  # numpy-scalars
    }
    logged_metrics, plugins = self._log_and_read(task_metrics)
    self.assertDictEqual(
        plugins, {"eval/rouge1": "scalars", "eval/rouge2": "scalars"}
    )
    self.assertDictEqual(
        logged_metrics, {"eval/rouge1": 50.0, "eval/rouge2": 100.0}
    )

  def test_log_text(self):
    task_metrics = {
        "str_sample": metrics_lib.Text("test1"),
        "bytes_sample": metrics_lib.Text(b"test2"),
    }
    logged_metrics, plugins = self._log_and_read(task_metrics)
    self.assertDictEqual(
        plugins, {"eval/str_sample": "text", "eval/bytes_sample": "text"}
    )
    self.assertDictEqual(
        logged_metrics,
        {
            "eval/str_sample": np.array(b"test1"),
            "eval/bytes_sample": np.array(b"test2"),
        },
    )

  def test_log_image(self):
    task_metrics = {
        "1image": metrics_lib.Image(np.arange(36).reshape((1, 3, 4, 3))),
        "3image": metrics_lib.Image(
            np.arange(108).reshape((3, 3, 4, 3)), max_outputs=2
        ),
    }
    logged_metrics, plugins = self._log_and_read(task_metrics)
    self.assertDictEqual(
        plugins, {"eval/1image": "images", "eval/3image": "images"}
    )
    self.assertLen(logged_metrics, 2)
    # All images have shape (3, 4):
    for res in logged_metrics.values():
      self.assertAllEqual(res[0:2], np.array([b"4", b"3"]))
    # 1 input, 1 output.
    self.assertLen(logged_metrics["eval/1image"][2:], 1)
    # 3 inputs, 2 outputs.
    self.assertLen(logged_metrics["eval/3image"][2:], 2)

  def test_log_histogram(self):
    task_metrics = {
        "1d": metrics_lib.Histogram(np.arange(10), bins=2),
    }
    logged_metrics, plugins = self._log_and_read(task_metrics)
    self.assertDictEqual(
        plugins,
        {
            "eval/1d": "histograms",
        },
    )
    self.assertSameElements(logged_metrics, ["eval/1d"])
    self.assertAllEqual(
        logged_metrics["eval/1d"], np.array([[0.0, 4.5, 5.0], [4.5, 9.0, 5.0]])
    )

  def test_log_audio(self):
    task_metrics = {
        "1mono": metrics_lib.Audio(
            np.linspace(-1, 1, 10).reshape((1, 10, 1)), max_outputs=2
        ),
        "3mono": metrics_lib.Audio(
            np.linspace(-1, 1, 30).reshape((3, 10, 1)), max_outputs=2
        ),
        "1stereo": metrics_lib.Audio(
            np.linspace(-1, 1, 20).reshape((1, 10, 2)), max_outputs=2
        ),
    }
    logged_metrics, plugins = self._log_and_read(task_metrics)
    self.assertDictEqual(
        plugins,
        {"eval/1mono": "audio", "eval/3mono": "audio", "eval/1stereo": "audio"},
    )
    self.assertSameElements(
        logged_metrics, ["eval/1mono", "eval/3mono", "eval/1stereo"]
    )
    # 1 input, 1 output.
    self.assertLen(logged_metrics["eval/1mono"], 1)
    # 3 inputs, 2 outputs.
    self.assertLen(logged_metrics["eval/3mono"], 2)
    # 1 input, 1 output.
    self.assertLen(logged_metrics["eval/1stereo"], 1)

  def test_log_generic(self):
    task_metrics = {
        "foo": metrics_lib.Generic(
            tensor=np.array([1, 2, 3, 4, 5]),
            metadata=tf.compat.v1.SummaryMetadata(
                plugin_data=tf.compat.v1.SummaryMetadata.PluginData(
                    plugin_name="frobber"
                )
            ),
        )
    }
    logged_metrics, plugins = self._log_and_read(task_metrics)
    self.assertDictEqual(plugins, {"eval/foo": "frobber"})
    self.assertSameElements(logged_metrics, ["eval/foo"])
    self.assertAllEqual(logged_metrics["eval/foo"], [1, 2, 3, 4, 5])


class JSONLoggerTest(tf.test.TestCase):

  def _get_task_dataset_for_write_to_file_tests(self):
    x = [
        {"inputs_pretokenized": "i0", "targets_pretokenized": "t0"},
        {"inputs_pretokenized": "i1", "targets_pretokenized": "t1"},
    ]
    output_types = {
        "inputs_pretokenized": tf.string,
        "targets_pretokenized": tf.string,
    }
    output_shapes = {"targets_pretokenized": [], "inputs_pretokenized": []}
    task_dataset = tf.data.Dataset.from_generator(
        lambda: x, output_types=output_types, output_shapes=output_shapes
    )
    return task_dataset

  def test_logging(self):
    inferences = {"prediction": ["pred0", "pred1"], "score": [0.2, 0.3]}
    targets = ["target0", "target1"]
    tmp_dir = self.create_tempdir().full_path
    task_dataset = self._get_task_dataset_for_write_to_file_tests()

    logger = loggers.JSONLogger(tmp_dir)
    logger(
        task_name="test",
        step=42,
        metrics={"accuracy": metrics_lib.Scalar(100)},
        dataset=task_dataset,
        inferences=inferences,
        targets=targets,
    )

    # Validate the metrics file.
    with open(os.path.join(tmp_dir, "test-metrics.jsonl")) as f:
      self.assertDictEqual(json.load(f), {"step": 42, "accuracy": 100.0})

    # Read the written jsonl file.
    with open(os.path.join(tmp_dir, "test-000042.jsonl")) as f:
      actual = [json.loads(line.strip()) for line in f]

    expected = [
        {
            "input": {
                "inputs_pretokenized": "i0",
                "targets_pretokenized": "t0",
            },
            "prediction": "pred0",
            "target": "target0",
            "score": 0.2,
        },
        {
            "input": {
                "inputs_pretokenized": "i1",
                "targets_pretokenized": "t1",
            },
            "prediction": "pred1",
            "target": "target1",
            "score": 0.3,
        },
    ]
    self.assertEqual(actual, expected)

  def test_n_prediction_and_scores(self):
    inferences = {"prediction": ["pred0", "pred1"], "score": [0.2, 0.3]}
    targets = ["target0", "target1"]
    tmp_dir = self.create_tempdir().full_path
    task_dataset = self._get_task_dataset_for_write_to_file_tests()

    logger = loggers.JSONLogger(tmp_dir, write_n_results=1)
    logger(
        task_name="test",
        step=42,
        metrics={"accuracy": metrics_lib.Scalar(100)},
        dataset=task_dataset,
        inferences=inferences,
        targets=targets,
    )

    # Validate the metrics file.
    with open(os.path.join(tmp_dir, "test-metrics.jsonl")) as f:
      self.assertDictEqual(json.load(f), {"step": 42, "accuracy": 100.0})

    # Read the written jsonl file.
    with open(os.path.join(tmp_dir, "test-000042.jsonl")) as f:
      actual = [json.loads(line.strip()) for line in f]

    expected = [{
        "input": {"inputs_pretokenized": "i0", "targets_pretokenized": "t0"},
        "prediction": "pred0",
        "target": "target0",
        "score": 0.2,
    }]
    self.assertEqual(actual, expected)

  def test_predictions_only(self):
    inferences = {"prediction": ["pred0", "pred1"]}
    targets = ["target0", "target1"]
    tmp_dir = self.create_tempdir().full_path
    task_dataset = self._get_task_dataset_for_write_to_file_tests()

    logger = loggers.JSONLogger(tmp_dir)
    logger(
        task_name="test",
        step=42,
        metrics={"accuracy": metrics_lib.Scalar(100)},
        dataset=task_dataset,
        inferences=inferences,
        targets=targets,
    )

    # Validate the metrics file.
    with open(os.path.join(tmp_dir, "test-metrics.jsonl")) as f:
      self.assertDictEqual(json.load(f), {"step": 42, "accuracy": 100.0})

    # Read the written jsonl file.
    with open(os.path.join(tmp_dir, "test-000042.jsonl")) as f:
      actual = [json.loads(line.strip()) for line in f]

    expected = [
        {
            "input": {
                "inputs_pretokenized": "i0",
                "targets_pretokenized": "t0",
            },
            "prediction": "pred0",
            "target": "target0",
        },
        {
            "input": {
                "inputs_pretokenized": "i1",
                "targets_pretokenized": "t1",
            },
            "prediction": "pred1",
            "target": "target1",
        },
    ]
    self.assertEqual(actual, expected)

  def test_predictions_and_aux_values(self):
    inferences = {
        "prediction": ["pred0", "pred1"],
        "aux_value": {
            "scores": [0.2, 0.3],
            "other_aux_values": [10.0, 20.0],
        },
    }
    targets = ["target0", "target1"]
    tmp_dir = self.create_tempdir().full_path
    task_dataset = self._get_task_dataset_for_write_to_file_tests()

    logger = loggers.JSONLogger(tmp_dir)
    logger(
        task_name="test",
        step=42,
        metrics={"accuracy": metrics_lib.Scalar(100)},
        dataset=task_dataset,
        inferences=inferences,
        targets=targets,
    )

    # Validate the metrics file.
    with open(os.path.join(tmp_dir, "test-metrics.jsonl")) as f:
      self.assertDictEqual(json.load(f), {"step": 42, "accuracy": 100.0})

    # Read the written jsonl file.
    with open(os.path.join(tmp_dir, "test-000042.jsonl")) as f:
      actual = [json.loads(line.strip()) for line in f]

    expected = [
        {
            "input": {
                "inputs_pretokenized": "i0",
                "targets_pretokenized": "t0",
            },
            "prediction": "pred0",
            "target": "target0",
            "aux_scores": 0.2,
            "aux_other_aux_values": 10.0,
        },
        {
            "input": {
                "inputs_pretokenized": "i1",
                "targets_pretokenized": "t1",
            },
            "prediction": "pred1",
            "target": "target1",
            "aux_scores": 0.3,
            "aux_other_aux_values": 20.0,
        },
    ]
    self.assertEqual(actual, expected)

  def test_numpy_data(self):
    inferences = {
        "prediction": [np.zeros((2, 2)), np.ones((2, 2))],
        "score": [0.2, 0.3],
    }
    targets = ["target0", "target1"]
    tmp_dir = self.create_tempdir().full_path
    task_dataset = self._get_task_dataset_for_write_to_file_tests()

    logger = loggers.JSONLogger(tmp_dir)
    logger(
        task_name="test",
        step=42,
        metrics={"accuracy": metrics_lib.Scalar(np.float32(100))},  # pytype: disable=wrong-arg-types  # numpy-scalars
        dataset=task_dataset,
        inferences=inferences,
        targets=targets,
    )

    # Validate the metrics file.
    with open(os.path.join(tmp_dir, "test-metrics.jsonl")) as f:
      self.assertDictEqual(json.load(f), {"step": 42, "accuracy": 100.0})

    # Read the written jsonl file.
    with open(os.path.join(tmp_dir, "test-000042.jsonl")) as f:
      actual = [json.loads(line.strip()) for line in f]

    expected = [
        {
            "input": {
                "inputs_pretokenized": "i0",
                "targets_pretokenized": "t0",
            },
            "prediction": [[0.0, 0.0], [0.0, 0.0]],
            "score": 0.2,
            "target": "target0",
        },
        {
            "input": {
                "inputs_pretokenized": "i1",
                "targets_pretokenized": "t1",
            },
            "prediction": [[1.0, 1.0], [1.0, 1.0]],
            "score": 0.3,
            "target": "target1",
        },
    ]
    self.assertEqual(actual, expected)

  def test_non_serializable_prediction(self):
    inferences = {"prediction": [object(), object()], "score": [0.2, 0.3]}
    targets = ["target0", "target1"]
    tmp_dir = self.create_tempdir().full_path
    task_dataset = self._get_task_dataset_for_write_to_file_tests()

    logger = loggers.JSONLogger(tmp_dir)
    logger(
        task_name="test",
        step=42,
        metrics={"accuracy": metrics_lib.Scalar(100)},
        dataset=task_dataset,
        inferences=inferences,
        targets=targets,
    )

    # Validate the metrics file.
    with open(os.path.join(tmp_dir, "test-metrics.jsonl")) as f:
      self.assertDictEqual(json.load(f), {"step": 42, "accuracy": 100.0})

    # Read the written jsonl file.
    with open(os.path.join(tmp_dir, "test-000042.jsonl")) as f:
      actual = [json.loads(line.strip()) for line in f]

    expected = [
        {
            "input": {
                "inputs_pretokenized": "i0",
                "targets_pretokenized": "t0",
            },
            "score": 0.2,
            "target": "target0",
        },
        {
            "input": {
                "inputs_pretokenized": "i1",
                "targets_pretokenized": "t1",
            },
            "score": 0.3,
            "target": "target1",
        },
    ]
    self.assertEqual(actual, expected)

  def test_non_serializable_target(self):
    inferences = {"prediction": ["pred0", "pred1"], "score": [0.2, 0.3]}
    targets = [object(), object()]
    tmp_dir = self.create_tempdir().full_path
    task_dataset = self._get_task_dataset_for_write_to_file_tests()

    logger = loggers.JSONLogger(tmp_dir)
    logger(
        task_name="test",
        step=42,
        metrics={"accuracy": metrics_lib.Scalar(100)},
        dataset=task_dataset,
        inferences=inferences,
        targets=targets,
    )

    # Validate the metrics file.
    with open(os.path.join(tmp_dir, "test-metrics.jsonl")) as f:
      self.assertDictEqual(json.load(f), {"step": 42, "accuracy": 100.0})

    # Read the written jsonl file.
    with open(os.path.join(tmp_dir, "test-000042.jsonl")) as f:
      actual = [json.loads(line.strip()) for line in f]

    expected = [
        {
            "input": {
                "inputs_pretokenized": "i0",
                "targets_pretokenized": "t0",
            },
            "prediction": "pred0",
            "score": 0.2,
        },
        {
            "input": {
                "inputs_pretokenized": "i1",
                "targets_pretokenized": "t1",
            },
            "prediction": "pred1",
            "score": 0.3,
        },
    ]
    self.assertEqual(actual, expected)

  def test_prediction_bytes(self):
    inferences = {
        "prediction": [b"\x99", b"\x88"],
    }
    targets = ["target0", "target1"]
    tmp_dir = self.create_tempdir().full_path
    task_dataset = self._get_task_dataset_for_write_to_file_tests()

    logger = loggers.JSONLogger(tmp_dir)
    logger(
        task_name="test",
        step=42,
        metrics={"accuracy": metrics_lib.Scalar(100)},
        dataset=task_dataset,
        inferences=inferences,
        targets=targets,
    )

    # Validate the metrics file.
    with open(os.path.join(tmp_dir, "test-metrics.jsonl")) as f:
      self.assertDictEqual(json.load(f), {"step": 42, "accuracy": 100.0})

    # Read the written jsonl file.
    with open(os.path.join(tmp_dir, "test-000042.jsonl")) as f:
      actual = [json.loads(line.strip()) for line in f]

    expected = [
        {
            "input": {
                "inputs_pretokenized": "i0",
                "targets_pretokenized": "t0",
            },
            "prediction": "mQ==",
            "target": "target0",
        },
        {
            "input": {
                "inputs_pretokenized": "i1",
                "targets_pretokenized": "t1",
            },
            "prediction": "iA==",
            "target": "target1",
        },
    ]
    self.assertEqual(actual, expected)

  def test_2d_ragged_input(self):
    x = [
        {
            "inputs": tf.ragged.constant([[9, 4, 1], [8, 1]]),
            "inputs_pretokenized": ["i0_0", "i0_1"],
        },
        {
            "inputs": tf.ragged.constant([[9, 1], [7, 2, 3, 1]]),
            "inputs_pretokenized": ["i1_0", "i1_1"],
        },
    ]
    task_dataset = tf.data.Dataset.from_generator(
        lambda: x,
        output_signature={
            "inputs": tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32),
            "inputs_pretokenized": tf.TensorSpec(shape=[None], dtype=tf.string),
        },
    )
    inferences = {"prediction": ["pred0", "pred1"], "score": [0.2, 0.3]}
    targets = ["target0", "target1"]
    tmp_dir = self.create_tempdir().full_path

    logger = loggers.JSONLogger(tmp_dir)
    logger(
        task_name="test",
        step=42,
        metrics={"accuracy": metrics_lib.Scalar(100)},
        dataset=task_dataset,
        inferences=inferences,
        targets=targets,
    )

    # Validate the metrics file.
    with open(os.path.join(tmp_dir, "test-metrics.jsonl")) as f:
      self.assertDictEqual(json.load(f), {"step": 42, "accuracy": 100.0})

    # Read the written jsonl file.
    with open(os.path.join(tmp_dir, "test-000042.jsonl")) as f:
      actual = [json.loads(line.strip()) for line in f]

    expected = [
        {
            "input": {
                "inputs": [[9, 4, 1], [8, 1]],
                "inputs_pretokenized": ["i0_0", "i0_1"],
            },
            "prediction": "pred0",
            "target": "target0",
            "score": 0.2,
        },
        {
            "input": {
                "inputs": [[9, 1], [7, 2, 3, 1]],
                "inputs_pretokenized": ["i1_0", "i1_1"],
            },
            "prediction": "pred1",
            "target": "target1",
            "score": 0.3,
        },
    ]
    self.assertEqual(actual, expected)

  def test_metrics_multiple_steps(self):
    tmp_dir = self.create_tempdir().full_path

    logger = loggers.JSONLogger(tmp_dir, write_n_results=0)
    logger(
        task_name="test",
        step=42,
        metrics={"accuracy": metrics_lib.Scalar(100)},
        dataset=tf.data.Dataset.range(0),
        inferences={},
        targets=[],
    )

    logger(
        task_name="test",
        step=48,
        metrics={"accuracy": metrics_lib.Scalar(50)},
        dataset=tf.data.Dataset.range(0),
        inferences={},
        targets=[],
    )

    # Read the written jsonl file.
    with open(os.path.join(tmp_dir, "test-metrics.jsonl")) as f:
      actual = [json.loads(line.strip()) for line in f]

    expected = [{"step": 42, "accuracy": 100}, {"step": 48, "accuracy": 50}]

    self.assertEqual(actual, expected)

  def test_metrics_non_serializable(self):
    tmp_dir = self.create_tempdir().full_path

    logger = loggers.JSONLogger(tmp_dir, write_n_results=0)
    logger(
        task_name="test",
        step=42,
        metrics={
            "scalar": metrics_lib.Scalar(100),
            "text": metrics_lib.Text("foo"),
            "image": metrics_lib.Image(np.ones(10)),
            "1d_array": metrics_lib.Generic(
                np.array([1, 2, 3]), metadata="1d_array"
            ),
            "2d_array": metrics_lib.Generic(
                np.array([[1, 2, 3], [4, 5, 6]]), metadata="2d_array"
            ),
        },
        dataset=tf.data.Dataset.range(0),
        inferences={},
        targets=[],
    )

    with open(os.path.join(tmp_dir, "test-metrics.jsonl")) as f:
      self.assertDictEqual(
          json.load(f),
          {
              "step": 42,
              "scalar": 100.0,
              "text": "foo",
              "1d_array": [1, 2, 3],
              "2d_array": [[1, 2, 3], [4, 5, 6]],
          },
      )

  def test_logging_metrics_only(self):
    tmp_dir = self.create_tempdir().full_path

    logger = loggers.JSONLogger(tmp_dir)
    logger(
        task_name="test",
        step=42,
        metrics={"accuracy": metrics_lib.Scalar(100)},
        dataset=None,
        inferences=None,
        targets=None,
    )

    # Validate the metrics file.
    with open(os.path.join(tmp_dir, "test-metrics.jsonl")) as f:
      self.assertDictEqual(json.load(f), {"step": 42, "accuracy": 100.0})

    # Verify inferences not written.
    self.assertFalse(
        tf.io.gfile.exists(os.path.join(tmp_dir, "test-000042.jsonl"))
    )


class TensorAndNumpyEncoderLoggerTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.logger = loggers.TensorAndNumpyEncoder()

  def test_tensor(self):
    obj = tf.constant([1, 2, 3])
    self.assertEqual(self.logger.encode(obj), "[1, 2, 3]")

  def test_numpy(self):
    obj = np.array([1, 2, 3])
    self.assertEqual(self.logger.encode(obj), "[1, 2, 3]")

  def test_long_numpy(self):
    obj = np.arange(100, dtype=np.int32)
    self.assertEqual(
        self.logger.encode(obj),
        (
            '"ndarray(shape=(100,), dtype=int32); summary: 0, 1, 2, '
            '3, 4 ... 95, 96, 97, 98, 99"'
        ),
    )

  def test_dataclass(self):
    @dataclasses.dataclass
    class Foo:
      bar: int
      baz: str

    obj = Foo(1, "bazbaz")
    self.assertEqual(self.logger.encode(obj), '{"bar": 1, "baz": "bazbaz"}')

  def test_dataclass_with_none_value(self):
    @dataclasses.dataclass
    class Foo:
      bar: int
      baz: Optional[str] = None

    obj = Foo(1)
    self.assertEqual(self.logger.encode(obj), '{"bar": 1}')




if __name__ == "__main__":
  tf.test.main()
