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

"""Tests for seqio.scripts.registration_provenance_main."""

import tempfile

from absl.testing import absltest
from seqio import dataset_providers
from seqio import task_registry_provenance_tracking
from seqio.scripts import registration_provenance_main
import tensorflow.compat.v2 as tf


_TEST_OUTPUT_FEATURES = {}


class RegistrationProvenanceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.assertEmpty(task_registry_provenance_tracking.view_provenances())
    task_registry_provenance_tracking.turn_on_tracking()

  def tearDown(self):
    super().tearDown()

    task_registry_provenance_tracking._PROVIDER_PROVENANCE_LOOKUP = None
    self.assertEmpty(task_registry_provenance_tracking.view_provenances())

  def test_get_tsv_lines_empty(self):
    self.assertEmpty(task_registry_provenance_tracking.view_provenances())
    task_registry_provenance_tracking.turn_on_tracking()

    with self.assertRaisesRegex(ValueError, "No task/mixture"):
      registration_provenance_main._get_lines_for_tsv()

  def test_get_tsv_lines(self):
    ### Test setup.
    dataset_providers.TaskRegistry.add(
        "registration_provenance_main_test_task",
        source=tf.data.Dataset.from_generator(
            lambda: iter([]), {"unused": (1.0,)}
        ),
        output_features=_TEST_OUTPUT_FEATURES,
    )

    ### Run function.
    actual_lines = registration_provenance_main._get_lines_for_tsv()

    ### Make assertions.
    expected_header = [
        "provider_name",
        "provider_type",
        "call_site_0",
        "call_site_1",
        "call_site_2",
        "call_site_3",
        "call_site_4",
    ]
    self.assertEqual(actual_lines[0], expected_header)
    self.assertLen(actual_lines, 2)  # One for header, one for task.

    actual_provider_type = actual_lines[1][1]
    expected_provider_type = "Task"
    self.assertEqual(actual_provider_type, expected_provider_type)

  def test_write_lines(self):
    lines = [["a", "b"], ["c", "d"]]
    with tempfile.NamedTemporaryFile(mode="rt+") as f:
      registration_provenance_main._write_lines(lines, f.name)
      f.seek(0)
      expected = "a\tb\nc\td\n"
      actual = f.read()
      self.assertEqual(actual, expected)


if __name__ == "__main__":
  absltest.main()
