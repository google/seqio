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

"""Tests for seqio_lineage_log_helpers utility function."""

import os
from typing import Optional

import mock
from seqio import dataset_providers
from seqio import seqio_lineage_log_helpers
from seqio import vocabularies

from sentencepiece import sentencepiece_model_pb2

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_data"
)


def _sentencepiece_vocab(
    extra_ids=0,
    normalizer_spec_overrides: Optional[
        sentencepiece_model_pb2.NormalizerSpec
    ] = None,
    reverse_extra_ids: bool = True,
):
  return vocabularies.SentencePieceVocabulary(
      os.path.join(TEST_DATA_DIR, "sentencepiece", "sentencepiece.model"),
      extra_ids=extra_ids,
      normalizer_spec_overrides=normalizer_spec_overrides,
      reverse_extra_ids=reverse_extra_ids,
  )


DEFAULT_OUTPUT_FEATURES = {
    "inputs": dataset_providers.Feature(_sentencepiece_vocab()),
    "targets": dataset_providers.Feature(_sentencepiece_vocab()),
}


class SeqioLineageLogHelpersTest(googletest.TestCase):

  def setUp(self):
    super().setUp()

    self.test_tfds_source = dataset_providers.TfdsDataSource(
        tfds_name="fake:1.1.1", splits=("train", "validation")
    )
    self.test_cns_source = dataset_providers.TfdsDataSource(
        splits=("train", "validation"), tfds_data_dir="cns/fake/path"
    )

    self.task_with_tfds_source = dataset_providers.Task(
        name="test_tfds_task",
        source=self.test_tfds_source,
        output_features=DEFAULT_OUTPUT_FEATURES,
    )

    self.task_with_cns_source = dataset_providers.Task(
        name="test_cns_task",
        source=self.test_cns_source,
        output_features=DEFAULT_OUTPUT_FEATURES,
    )

  @mock.patch.object(lineage_log, "input")
  def test_seqio_task_entry_creation(
      self,
      mock_lineage_log_input,
  ):
    sample_task = self.task_with_tfds_source

    seqio_lineage_log_helpers.create_task_entry(sample_task)

    expected_handle = guri.ToString(
        guri_pb2.GUri(
            experimental=guri_pb2.Experimental(
                experimental_type="seqio_task",
                part1=(
                    # hash generated using sample task with Fingerprint2011,
                    # deterministic should pass every time.
                    "test_tfds_task__2431744215085621481"
                ),
            )
        )
    )

    mock_lineage_log_input.assert_called_with(
        handle=expected_handle, intent="SEQIO_TASK"
    )

  @mock.patch.object(lineage_log, "input")
  def test_seqio_task_entry_create_relationship_with_tfds(
      self, mock_lineage_log_input
  ):
    sample_tfds_task = self.task_with_tfds_source

    # expected datahub lib response:
    # DatahubEntryId(project='mldataset', dataset='tfds',
    # entry=None, variant=None)
    seqio_lineage_log_helpers.create_relationship_between_task_and_tfds_or_cns(
        sample_tfds_task
    )

    expected_handle = lineage_log._SeqIOTaskHandle(
        uri=guri.ToString(
            guri_pb2.GUri(
                experimental=guri_pb2.Experimental(
                    experimental_type="seqio_task",
                    part1=(
                        # hash generated using sample task with Fingerprint2011,
                        # deterministic should pass every time.
                        "test_tfds_task__2431744215085621481"
                    ),
                )
            )
        ),
        data_source_path=guri.ToString(
            guri_pb2.GUri(
                ml_data=guri_pb2.MlData(
                    dataset_namespace="tfds",
                    path="fake/1.1.1",
                )
            )
        ),
    )

    mock_lineage_log_input.assert_called_with(
        handle=expected_handle, intent="SEQIO_TASK_AND_TFDS_OR_CNS"
    )

  @mock.patch.object(lineage_log, "input")
  def test_seqio_task_entry_create_relationship_with_cns(
      self, mock_lineage_log_input
  ):
    sample_cns_task = self.task_with_cns_source

    seqio_lineage_log_helpers.create_relationship_between_task_and_tfds_or_cns(
        sample_cns_task
    )

    expected_handle = lineage_log._SeqIOTaskHandle(
        uri=guri.ToString(
            guri_pb2.GUri(
                experimental=guri_pb2.Experimental(
                    experimental_type="seqio_task",
                    part1=(
                        # hash generated using sample task with Fingerprint2011,
                        # deterministic should pass every time.
                        "test_cns_task__4932933169896005997"
                    ),
                )
            )
        ),
        data_source_path=guri.ToString(
            guri_pb2.GUri(cns=guri_pb2.Cns(path="cns/fake/path"))
        ),
    )

    mock_lineage_log_input.assert_called_with(
        handle=expected_handle, intent="SEQIO_TASK_AND_TFDS_OR_CNS"
    )


if __name__ == "__main__":
  googletest.main()
