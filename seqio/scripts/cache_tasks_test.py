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

"""Tests for seqio.scripts.cache_tasks_main."""

import functools
import os

from absl import flags
from absl.testing import absltest
from apache_beam.testing.test_pipeline import TestPipeline
import seqio
from seqio.scripts import cache_tasks_main
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

flags.FLAGS.min_shards = 0
TaskRegistry = seqio.TaskRegistry
test_utils = seqio.test_utils


class ProcessTaskBeamTest(test_utils.FakeTaskTest):

  def validate_pipeline(self,
                        task_name,
                        expected_task_dir="cached_task",
                        token_preprocessed=False,
                        ndfeatures=False,
                        num_shards=2):
    self.assertTrue(TaskRegistry.get("cached_task").cache_dir)
    task = TaskRegistry.get(task_name)
    self.assertFalse(task.cache_dir)

    with TestPipeline() as p:
      output_dirs = cache_tasks_main.run_pipeline(
          p, ["cached_task", task_name], cache_dir=self.test_data_dir)

    actual_task_dir = os.path.join(
        self.test_data_dir, seqio.get_task_dir_from_name(task_name))
    expected_task_dir = os.path.join(test_utils.TEST_DATA_DIR,
                                     expected_task_dir)
    expected_tfrecord_files = [
        "train.tfrecord-%05d-of-%05d" % (i, num_shards)
        for i in range(num_shards)
    ]
    expected_auxiliary_files = [
        "stats.train.json", "info.train.json", "COMPLETED"
    ]

    if "validation" in task.splits:
      expected_tfrecord_files.append("validation.tfrecord-00000-of-00001")
      expected_auxiliary_files.extend(
          ["stats.validation.json", "info.validation.json"])
    self.assertEqual([actual_task_dir], output_dirs)
    self.assertCountEqual(
        expected_tfrecord_files + expected_auxiliary_files,
        tf.io.gfile.listdir(actual_task_dir))

    for fname in expected_auxiliary_files:
      actual_content = tf.io.gfile.GFile(
          os.path.join(actual_task_dir, fname)).read()
      expected_content = tf.io.gfile.GFile(
          os.path.join(expected_task_dir, fname)).read()

      # Accept minor formatting difference.
      actual_content = actual_content.replace(", ", ",")
      # Replace with actual number of shards.
      expected_content = expected_content.replace(
          '"num_shards": 2', f'"num_shards": {num_shards}')
      # Replace with actual version.
      version = seqio.__version__
      expected_content = expected_content.replace(
          '"seqio_version": "0.0.0"', f'"seqio_version": "{version}"')
      self.assertEqual(expected_content, actual_content,
                       f"Contents of {fname} mismath")

    # Check datasets.
    self.verify_task_matches_fake_datasets(
        task_name,
        use_cached=True,
        splits=task.splits,
        token_preprocessed=token_preprocessed,
        ndfeatures=ndfeatures)

  def test_tfds_pipeline(self):
    self.validate_pipeline("tfds_task")

  def test_new_tfds_pipeline(self):
    self.validate_pipeline("t5:tfds_task")

  def test_text_line_pipeline(self):
    self.validate_pipeline("text_line_task")

  def test_function_pipeline(self):
    self.validate_pipeline("function_task", num_shards=1)

  def test_tf_example_pipeline(self):
    self.validate_pipeline("tf_example_task")

  def test_cache_before_tokenization_ndfeatures_pipeline(self):
    self.add_task(
        "task_tokenized_postcache_ndfeatures",
        seqio.dataset_providers.FunctionDataSource(
            dataset_fn=functools.partial(
                test_utils.get_fake_dataset, ndfeatures=True),
            splits=["train", "validation"]),
        output_features={
            "inputs":
                seqio.Feature(test_utils.sentencepiece_vocab()),
            "targets":
                seqio.Feature(test_utils.sentencepiece_vocab()),
            "2d_feature":
                seqio.Feature(
                    seqio.PassThroughVocabulary(1000, eos_id=0),
                    add_eos=False,
                    rank=2),
            "3d_feature":
                seqio.Feature(
                    seqio.PassThroughVocabulary(1000, eos_id=0),
                    add_eos=False,
                    rank=3),
        },
        preprocessors=[
            test_utils.test_text_preprocessor,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.tokenize,
            test_utils.token_preprocessor_no_sequence_length,
            seqio.preprocessors.append_eos_after_trim,
        ])
    self.validate_pipeline(
        "task_tokenized_postcache_ndfeatures",
        expected_task_dir="cached_untokenized_ndfeatures_task",
        num_shards=1,
        token_preprocessed=True,
        ndfeatures=True)

  def test_cache_before_tokenization_pipeline(self):
    self.add_task(
        "task_tokenized_postcache",
        self.function_source,
        preprocessors=[
            test_utils.test_text_preprocessor,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.tokenize,
            test_utils.token_preprocessor_no_sequence_length,
            seqio.preprocessors.append_eos_after_trim
        ])
    self.validate_pipeline(
        "task_tokenized_postcache",
        expected_task_dir="cached_untokenized_task",
        num_shards=1,
        token_preprocessed=True)

  def test_overwrite(self):
    with TestPipeline() as p:
      _ = cache_tasks_main.run_pipeline(
          p, ["uncached_task"], cache_dir=self.test_data_dir, overwrite=True)

    actual_task_dir = os.path.join(self.test_data_dir, "uncached_task")
    stat_old = tf.io.gfile.stat(
        os.path.join(actual_task_dir, "train.tfrecord-00000-of-00002"))

    with TestPipeline() as p:
      _ = cache_tasks_main.run_pipeline(
          p, ["uncached_task"], cache_dir=self.test_data_dir, overwrite=True)

    stat_new = tf.io.gfile.stat(
        os.path.join(actual_task_dir, "train.tfrecord-00000-of-00002"))

    self.assertGreater(stat_new.mtime_nsec, stat_old.mtime_nsec)


if __name__ == "__main__":
  absltest.main()
