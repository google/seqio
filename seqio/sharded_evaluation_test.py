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
    self.task.metric_objs = [
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
        ["dummy_task"]
    )
    self.assertSequenceEqual(
        list(self.metric_manager.output_metrics_collections[
            "dummy_task"].keys()),
        [model_output_type],
    )

    metrics = self.metric_manager.output_metrics_collections["dummy_task"][
        model_output_type
    ]
    self.assertEqual(metrics.Count_0.count, 0)

  def test_from_model_output(self):
    task_batch = [
        {"targets": [0, 1, 2, 3]},
        {"targets": [1, 2, 3, 4]},
    ]

    batch_result = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    model_output_type = metrics_lib.ModelOutputType.PREDICTION
    metrics_batch = self.metric_manager.from_model_output(
        inputs=task_batch,
        model_output=batch_result,
        features=self.task.output_features,
        task_name=self.task.name,
        model_output_type=model_output_type,
    )

    self.assertEqual(metrics_batch.Count_0.count, 2)

  def test_merge(self):
    model_output_type = metrics_lib.ModelOutputType.PREDICTION
    self.metric_manager.initialize_metrics(self.task.name, model_output_type)

    task_batch = [
        {"targets": [0, 1, 2, 3]},
        {"targets": [1, 2, 3, 4]},
    ]
    batch_result = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    metrics_batch = self.metric_manager.from_model_output(
        inputs=task_batch,
        model_output=batch_result,
        features=self.task.output_features,
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
    self.assertEqual(metrics.Count_0.count, 2)

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

    self.assertEqual(metrics.Count_0.count, 4)


if __name__ == "__main__":
  absltest.main()
