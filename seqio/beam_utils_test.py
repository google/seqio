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

"""Tests for beam_utils."""

import json
import os
from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
from apache_beam.testing.test_pipeline import TestPipeline
import numpy as np
import seqio
from seqio import beam_utils
import tensorflow.compat.v2 as tf


class BeamUtilsTest(seqio.test_utils.FakeTaskTest):

  def test_preprocess_task(self):
    def _np_to_list(ex):
      return {
          k: v.tolist() if isinstance(v, np.ndarray) else v
          for k, v in ex.items()
      }

    with TestPipeline() as p:
      pcoll = (
          p | beam_utils.PreprocessTask(
              task=seqio.get_mixture_or_task("tfds_task"),
              split="train",
              preprocessors_seed=42,
              add_provenance=True)
          | beam.Map(_np_to_list))
      util.assert_that(
          pcoll,
          util.equal_to([{
              "inputs_pretokenized": b"complete: this",
              "inputs": [3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 8, 6],
              "targets_pretokenized": b"is a test",
              "targets": [3, 8, 6, 3, 5, 10],
              "provenance/task": "tfds_task",
              "provenance/source_shard": "train.tfrecord-00000-of-00002",
              "provenance/source_shard_index": 0,
              "provenance/index_within_shard": 0,
              "provenance/preprocessors_seed": 42,
          }, {
              "inputs_pretokenized": b"complete: those",
              "inputs": [3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 7, 6, 4],
              "targets_pretokenized": b"were tests",
              "targets": [17, 4, 23, 4, 10, 6],
              "provenance/task": "tfds_task",
              "provenance/source_shard": "train.tfrecord-00000-of-00002",
              "provenance/source_shard_index": 0,
              "provenance/index_within_shard": 1,
              "provenance/preprocessors_seed": 42,
          }, {
              "inputs_pretokenized": b"complete: that",
              "inputs": [3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 18],
              "targets_pretokenized": b"was a test",
              "targets": [17, 5, 6, 3, 5, 10],
              "provenance/task": "tfds_task",
              "provenance/source_shard": "train.tfrecord-00001-of-00002",
              "provenance/source_shard_index": 1,
              "provenance/index_within_shard": 0,
              "provenance/preprocessors_seed": 42,
          }]))

  def test_write_example_tf_record(self):
    output_path = os.path.join(self.test_data_dir, "output.tfrecord")
    example = {
        "inputs": np.arange(5),
        "targets": np.arange(10),
    }
    with TestPipeline() as p:
      _ = (
          p
          | beam.Create([example])
          | beam_utils.WriteExampleTfRecord(
              output_path=output_path, num_shards=1))
    ds = tf.data.TFRecordDataset(output_path + "-00000-of-00001")
    parsed_example = tf.train.Example.FromString(next(iter(ds)).numpy())
    self.assertEqual(parsed_example, seqio.dict_to_tfexample(example))

  def test_write_json(self):
    output_path = os.path.join(self.test_data_dir, "output.json")
    data = {
        "key_1": 12,
        "key_2": "value",
    }
    with TestPipeline() as p:
      _ = (
          p
          | beam.Create([data])
          | beam_utils.WriteJson(output_path=output_path))
    with open(output_path) as f:
      actual_json = json.load(f)
    self.assertEqual(json.dumps(actual_json), json.dumps(data))

  def test_get_info(self):
    input_examples = [{
        "targets": range(10),
        "inputs": "test",
        "2d_shape": np.ones((1, 3), np.int32),
        "3d_shape": np.ones((1, 2, 3), np.int32),
    }]
    with TestPipeline() as p:
      pcoll = (p
               | beam.Create(input_examples)
               | beam_utils.GetInfo(num_shards=3))

      util.assert_that(
          pcoll,
          util.equal_to([{
              "num_shards": 3,
              "features": {
                  "targets": {
                      "shape": [None],
                      "dtype": "int32"
                  },
                  "inputs": {
                      "shape": [],
                      "dtype": "string"
                  },
                  "2d_shape": {
                      "shape": [None, 3],
                      "dtype": "int32"
                  },
                  "3d_shape": {
                      "shape": [None, 2, 3],
                      "dtype": "int32",
                  },
              },
              "seqio_version": seqio.__version__
          }]))

  def test_get_stats(self):
    input_examples = [{
        "inputs": np.arange(1, 6),
        "targets": np.arange(1, 11)
    }, {
        "inputs": np.arange(3, 6),
        "targets": np.arange(5, 16)
    }]

    with TestPipeline() as p:
      pcoll = (
          p
          | beam.Create(input_examples)
          | beam_utils.GetStats(output_features=seqio.test_utils.FakeTaskTest
                                .DEFAULT_OUTPUT_FEATURES))

      util.assert_that(
          pcoll,
          util.equal_to([{
              "inputs_tokens": 7,
              "targets_tokens": 20,
              "inputs_max_tokens": 4,
              "targets_max_tokens": 11,
              "examples": 2
          }]))


if __name__ == "__main__":
  absltest.main()
