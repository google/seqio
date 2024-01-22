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

"""Tests for MetricManager and ShardedEvaluator in seqio.evaluation."""

from typing import Any, Mapping, Optional, Sequence
from unittest import mock

from absl.testing import absltest
import flax
import numpy as np
from seqio import dataset_providers
from seqio import evaluation
from seqio import metrics as metrics_lib
from seqio import utils
from seqio import vocabularies

ModelOutputType = metrics_lib.ModelOutputType


@flax.struct.dataclass
class Count(metrics_lib.Metric):
  """Implements a simple counting metric, which inherits seqio.Metric."""

  count: int = 0
  model_output_type: ModelOutputType = ModelOutputType.PREDICTION

  @classmethod
  def empty(cls) -> "Count":
    return cls(count=0)

  @classmethod
  def from_model_output(
      cls,
      inputs: Sequence[Mapping[str, Any]],
      model_output: np.ndarray,
      features: Mapping[str, utils.Feature],
      target_field_name: str = "targets",
      mask: Optional[np.ndarray] = None,
      indices_2d: Optional[np.ndarray] = None,
  ) -> "Count":

    if mask is None:
      mask = np.ones((len(inputs),))

    return cls(count=mask.sum())

  def merge(self, other: "Count") -> "Count":
    """Returns `Count` that is the accumulation of `self` and `other`.

    Args:
      other: A `Count` whose inermediate values should be accumulated onto the
        values of `self`. Note that in a distributed setting, `other` will
        typically be the output of a `jax.lax` parallel operator and thus have a
        dimension added to the dataclass returned by `.from_model_output()`.

    Returns:
      A new `Count` instance that accumulates the value from both `self` and
        `other`.
    """
    return type(self)(count=self.count + other.count)

  def compute(self):
    return {"count": self.count}


class MetricManagerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.uncalled_fn = mock.Mock()

    self.task = mock.Mock()
    self.task.name = "dummy_task"
    mf0 = lambda targets, predictions: {"accuracy": 1.0}
    mf1 = lambda targets, predictions: {"loss": 0.3}
    self.task.metric_objs = [
        metrics_lib.PassthroughLegacyMetric.from_metric_fn(mf0, None).empty(),
        metrics_lib.PassthroughLegacyMetric.from_metric_fn(mf1, None).empty(),
        Count.empty(),
    ]

    mock_vocab = mock.create_autospec(vocabularies.Vocabulary)
    self.task.output_features = {
        "targets": dataset_providers.Feature(mock_vocab)
    }

    self.metric_manager = evaluation.MetricManager([self.task])

  def tearDown(self):
    super().tearDown()
    self.uncalled_fn.assert_not_called()

  def test_initialize_metrics(self):
    model_output_type = metrics_lib.ModelOutputType.PREDICTION
    self.metric_manager.initialize_metrics(self.task.name, model_output_type)

    self.assertSequenceEqual(
        list(self.metric_manager.output_metrics_collections.keys()),
        ["dummy_task"],
    )
    self.assertSequenceEqual(
        list(
            self.metric_manager.output_metrics_collections["dummy_task"].keys()
        ),
        [model_output_type],
    )

    metrics = self.metric_manager.output_metrics_collections["dummy_task"][
        model_output_type
    ]
    self.assertEqual(metrics.FromMetricFun_0.values, {})
    self.assertEqual(metrics.FromMetricFun_1.values, {})
    self.assertEqual(metrics.Count_2.count, 0)

  def test_from_model_output(self):
    task_batch = [
        {"targets": [0, 1, 2, 3]},
        {"targets": [1, 2, 3, 4]},
    ]

    batch_result = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])

    batch_indices_2d = np.array([
        [0, 0],  # 1st example is from shard 0 and index 0
        [0, 1],  # 1st example is from shard 0 and index 1
    ])
    model_output_type = metrics_lib.ModelOutputType.PREDICTION
    metrics_batch = self.metric_manager.from_model_output(
        inputs=task_batch,
        model_output=batch_result,
        features=self.task.output_features,
        mask=(batch_indices_2d[:, 1] >= 0),
        indices_2d=batch_indices_2d,
        task_name=self.task.name,
        model_output_type=model_output_type,
    )

    self.assertEqual(metrics_batch.Count_2.count, 2)

  def test_merge(self):
    model_output_type = metrics_lib.ModelOutputType.PREDICTION
    self.metric_manager.initialize_metrics(self.task.name, model_output_type)

    task_batch = [
        {"targets": [0, 1, 2, 3]},
        {"targets": [1, 2, 3, 4]},
    ]
    batch_result = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    batch_indices_2d = np.array([
        [0, 0],  # 1st example is from shard 0 and index 0
        [0, 1],  # 1st example is from shard 0 and index 1
    ])
    metrics_batch = self.metric_manager.from_model_output(
        inputs=task_batch,
        model_output=batch_result,
        features=self.task.output_features,
        mask=(batch_indices_2d[:, 1] >= 0),
        indices_2d=batch_indices_2d,
        task_name=self.task.name,
        model_output_type=model_output_type,
    )

    self.metric_manager.merge(
        metrics_batch,
        task_name=self.task.name,
        model_output_type=model_output_type,
    )
    metrics = self.metric_manager.output_metrics_collections["dummy_task"][
        model_output_type
    ]
    # The first two metrics are CollectingMetric's, so they are just passing
    # through model outputs.
    np.testing.assert_array_equal(
        metrics.FromMetricFun_0.values["indices_2d"], batch_indices_2d
    )
    np.testing.assert_array_equal(
        metrics.FromMetricFun_0.values["mask"], np.array([True, True])
    )
    np.testing.assert_array_equal(
        metrics.FromMetricFun_0.values["model_output"], batch_result
    )
    np.testing.assert_array_equal(
        metrics.FromMetricFun_1.values["indices_2d"], batch_indices_2d
    )
    np.testing.assert_array_equal(
        metrics.FromMetricFun_1.values["mask"], np.array([True, True])
    )
    np.testing.assert_array_equal(
        metrics.FromMetricFun_1.values["model_output"], batch_result
    )
    # The last metric is batch-update enabled metric.
    self.assertEqual(metrics.Count_2.count, 2)

    # Merge a second time, so that everything in the metric objects should be
    # doubled in terms of the number of items for CollectingMetric or scalar
    # size for Count metric.
    self.metric_manager.merge(
        metrics_batch,
        task_name=self.task.name,
        model_output_type=model_output_type,
    )
    metrics = self.metric_manager.output_metrics_collections["dummy_task"][
        model_output_type
    ]

    # The first two metrics are CollectingMetric's, so they are just passing
    # through model outputs.
    np.testing.assert_array_equal(
        metrics.FromMetricFun_0.values["indices_2d"],
        np.concatenate([batch_indices_2d, batch_indices_2d], axis=0),
    )
    np.testing.assert_array_equal(
        metrics.FromMetricFun_0.values["mask"],
        np.array([True, True, True, True]),
    )
    np.testing.assert_array_equal(
        metrics.FromMetricFun_0.values["model_output"],
        np.concatenate([batch_result, batch_result], axis=0),
    )
    np.testing.assert_array_equal(
        metrics.FromMetricFun_1.values["indices_2d"],
        np.concatenate([batch_indices_2d, batch_indices_2d], axis=0),
    )
    np.testing.assert_array_equal(
        metrics.FromMetricFun_1.values["mask"],
        np.array([True, True, True, True]),
    )
    np.testing.assert_array_equal(
        metrics.FromMetricFun_1.values["model_output"],
        np.concatenate([batch_result, batch_result], axis=0),
    )
    # The last metric is batch-update enabled metric.
    self.assertEqual(metrics.Count_2.count, 4)


if __name__ == "__main__":
  absltest.main()
