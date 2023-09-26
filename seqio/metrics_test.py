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

"""Tests for seqio.metrics."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from seqio import dataset_providers
from seqio import metrics
from seqio import vocabularies


class MetricsTest(absltest.TestCase):

  def test_remove_padding_examples(self):
    mask = np.array([1, 1, 0])
    indices_2d = np.array(
        [[0, 1], [0, 2], [0, 0]]
    )
    model_output = np.array(
        [[1, 2, 3],
         [1, 3, 4],
         [1, 2, 3]]
    )
    indices_2d_cleaned, model_output_cleaned = metrics.remove_padding_examples(
        model_output, indices_2d, mask)

    np.testing.assert_array_equal(indices_2d_cleaned, np.array(
        [[0, 1], [0, 2]]
    ))
    np.testing.assert_array_equal(model_output_cleaned, np.array(
        [[1, 2, 3],
         [1, 3, 4]]
    ))

  def test_globally_sort_model_output(self):
    indices_2d = np.array(
        [[0, 1], [0, 2], [0, 0]]
    )
    model_output = np.array(
        [[1, 2, 3],
         [1, 3, 4],
         [1, 2, 3]]
    )
    model_output_sorted = metrics.globally_sort_model_output(
        model_output, indices_2d)

    np.testing.assert_array_equal(model_output_sorted, np.array(
        [[1, 2, 3],
         [1, 2, 3],
         [1, 3, 4]]
    ))


class MetricsComputeTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_passthrough_legacy_metric(self, do_external_decode: bool):
    mf = lambda targets, predictions: {"accuracy": 1.0}
    metric_obj = metrics.PassthroughLegacyMetric.from_metric_fn(
        mf, None
    ).empty()
    mock_vocab = mock.create_autospec(vocabularies.Vocabulary)

    task_batch = [
        {"targets": [0, 1, 2, 3]},
        {"targets": [1, 2, 3, 4]},
    ]
    model_output = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    output_features = {"targets": dataset_providers.Feature(mock_vocab)}

    if do_external_decode:
      model_output = [mock_vocab.decode(tokens) for tokens in model_output]

    metric_instance = metric_obj.from_model_output(
        inputs=task_batch,
        model_output=model_output,
        features=output_features,
    )
    metric_value, _ = metric_instance.actual_compute(
        task_batch, output_features
    )
    self.assertEqual(metric_value["accuracy"], 1.0)

if __name__ == "__main__":
  absltest.main()
