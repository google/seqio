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

"""Tests for task_registry_provenance_tracking."""

import inspect

from absl.testing import absltest
from seqio import dataset_providers
from seqio import task_registry_provenance_tracking


class TestRegistryProvenanceTrackingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    task_registry_provenance_tracking.turn_on_tracking()

  def tearDown(self):
    super().tearDown()
    task_registry_provenance_tracking._PROVIDER_PROVENANCE_LOOKUP = None

  def test_track_registration_provenance(self):
    input_task_name = "test_task_track_creation_provenance"

    input_data_source = dataset_providers.TextLineDataSource({})
    input_output_features = {}

    # Use a closure just to test the "more difficult" case for tracking
    # provenance of having extra closed-over state.
    def _closure_containing_task_init():
      task = dataset_providers.Task(
          input_task_name,
          input_data_source,
          input_output_features,
      )

      # These two statements are on the same line because the registration
      # provenance tracking looks at filename and line number; in order to get a
      # correct expected value, both things need to happen on the same line.
      # pyformat: disable
      expected_registration_provenance = task_registry_provenance_tracking.informative_upstream_callsites_from_frame(inspect.currentframe()); dataset_providers.TaskRegistry.add_provider(input_task_name, task)  # pylint: disable=multiple-statements
      # pyformat: enable
      return expected_registration_provenance

    expected_provider_type = dataset_providers.Task.__name__
    expected_registration_provenance = _closure_containing_task_init()

    actual_provider_type, actual_registration_provenance = (
        task_registry_provenance_tracking.view_provenances()[input_task_name]
    )

    self.assertEqual(actual_provider_type, expected_provider_type)

    self.assertEqual(
        actual_registration_provenance,
        expected_registration_provenance,
    )


if __name__ == "__main__":
  absltest.main()
