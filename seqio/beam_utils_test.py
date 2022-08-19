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

  def test_count_characters_str_dataset(self):
    input_examples = [{
        "text": b"this is a string of length 29"
    }, {
        "text": b"this is another string of length 35"
    }]
    output_features = {
        "text":
            seqio.Feature(
                seqio.PassThroughVocabulary(1), dtype=tf.string, rank=0)
    }

    with TestPipeline() as p:
      pcoll = (
          p
          | beam.Create(input_examples)
          | beam.ParDo(
              beam_utils._CountCharacters(output_features=output_features)))

      util.assert_that(
          pcoll, util.equal_to([("text_chars", 29), ("text_chars", 35)]))

  def test_count_characters_str_dataset_in_get_stats(self):
    input_examples = [{
        "text": b"this is a string of length 29"
    }, {
        "text": b"this is another string of length 35"
    }]
    output_features = {
        "text":
            seqio.Feature(
                seqio.PassThroughVocabulary(1), dtype=tf.string, rank=0)
    }

    with TestPipeline() as p:
      pcoll = (
          p
          | beam.Create(input_examples)
          | beam_utils.GetStats(output_features=output_features))

      util.assert_that(
          pcoll, util.equal_to([{"text_chars": 64, "examples": 2}]))

  def test_get_stats_tokenized_dataset(self):
    # These examples are assumed to be decoded by
    # `seqio.test_utils.sentencepiece_vocab()`.
    input_examples = [{
        # Decoded as "ea", i.e., length 2 string
        "inputs": np.array([4, 5]),
        # Decoded as "ea test", i.e., length 7 string
        "targets": np.array([4, 5, 10]),
    }, {
        # Decoded as "e", i.e., length 1 string
        "inputs": np.array([4]),
        # Decoded as "asoil", i.e., length 5 string. "1" is an EOS id.
        "targets": np.array([5, 6, 7, 8, 9, 1])
    }]

    output_features = seqio.test_utils.FakeTaskTest.DEFAULT_OUTPUT_FEATURES
    with TestPipeline() as p:
      pcoll = (
          p
          | beam.Create(input_examples)
          | beam_utils.GetStats(output_features=output_features))

      util.assert_that(
          pcoll,
          util.equal_to([{
              "inputs_tokens": 3,  # 4 and 3 from the first and second exmaples.
              "targets_tokens": 8,
              "inputs_max_tokens": 2,
              "targets_max_tokens": 5,
              "examples": 2,
              "inputs_chars": 3,
              "targets_chars": 12,
          }]))

  def test_count_characters_tokenized_dataset(self):
    # These examples are assumed to be decoded by
    # `seqio.test_utils.sentencepiece_vocab()`.
    input_examples = [{
        # Decoded as "ea", i.e., length 2 string
        "inputs": np.array([4, 5]),
        # Decoded as "ea test", i.e., length 7 string
        "targets": np.array([4, 5, 10]),
    }, {
        # Decoded as "e", i.e., length 1 string
        "inputs": np.array([4]),
        # Decoded as "asoil", i.e., length 5 string. "1" is an EOS id.
        "targets": np.array([5, 6, 7, 8, 9, 1])
    }]

    output_features = seqio.test_utils.FakeTaskTest.DEFAULT_OUTPUT_FEATURES
    with TestPipeline() as p:
      pcoll = (
          p
          | beam.Create(input_examples)
          | beam.ParDo(
              beam_utils._CountCharacters(output_features=output_features)))

      util.assert_that(
          pcoll,
          util.equal_to([("inputs_chars", 2), ("targets_chars", 7),
                         ("inputs_chars", 1), ("targets_chars", 5)]))

  def test_count_characters_tokenized_dataset_with_non_spm_vocab(self):
    input_examples = [{
        "feature": np.array([4, 5]),
    }]
    output_features = {
        "feature":
            seqio.Feature(
                seqio.PassThroughVocabulary(1), dtype=tf.int32, rank=1)
    }

    with TestPipeline() as p:
      pcoll = (
          p
          | beam.Create(input_examples)
          | beam.ParDo(
              beam_utils._CountCharacters(output_features=output_features)))

      util.assert_that(
          pcoll, util.equal_to([]))


if __name__ == "__main__":
  absltest.main()
