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

"""Tests for seqio.dataset_providers."""

import copy
from typing import Callable, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from seqio import dataset_providers
from seqio import dataset_providers_helpers
from seqio import feature_converters
from seqio import preprocessors
from seqio import test_utils
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

TaskRegistry = dataset_providers.TaskRegistry
MixtureRegistry = dataset_providers.MixtureRegistry
mock = absltest.mock
assert_dataset = test_utils.assert_dataset
create_default_dataset = test_utils.create_default_dataset


class GetDatasetTest(parameterized.TestCase, tf.test.TestCase):

  def test_get_dataset_enc_dec_unpacked(self):
    mixture_or_task_name = "enc_dec_unpacked"
    x = [{
        "inputs": [7, 8, 5, 6, 9, 4, 3],
        "targets": [3, 9]
    }, {
        "inputs": [8, 4],
        "targets": [4]
    }, {
        "inputs": [5, 6, 7],
        "targets": [6, 5]
    }]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=False)
    output_ds = dataset_providers_helpers.get_dataset(
        mixture_or_task_name=mixture_or_task_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split="train",
        shuffle=False,
        feature_converter=converter)

    expected = [{
        "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
        "decoder_target_tokens": [3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }, {
        "encoder_input_tokens": [8, 4, 1, 0, 0, 0, 0],
        "decoder_target_tokens": [4, 1, 0, 0, 0],
        "decoder_input_tokens": [0, 4, 1, 0, 0],
        "decoder_loss_weights": [1, 1, 0, 0, 0],
    }, {
        "encoder_input_tokens": [5, 6, 7, 1, 0, 0, 0],
        "decoder_target_tokens": [6, 5, 1, 0, 0],
        "decoder_input_tokens": [0, 6, 5, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  @parameterized.parameters(
      dict(
          task_name="enc_dec_partial_trim_both",
          task_feature_lengths={
              "inputs": 7,
              "targets": 2
          },
          expect_trim_inputs=True,
          expect_trim_targets=True),
      dict(
          task_name="enc_dec_partial_trim_targets",
          task_feature_lengths={
              "inputs": None,
              "targets": 2
          },
          expect_trim_inputs=False,
          expect_trim_targets=True),
      dict(
          task_name="enc_dec_partial_trim_inputs",
          task_feature_lengths={
              "inputs": 7,
              "targets": None
          },
          expect_trim_inputs=True,
          expect_trim_targets=False),
      dict(
          task_name="enc_dec_partial_trim_neither",
          task_feature_lengths={
              "inputs": None,
              "targets": None
          },
          expect_trim_inputs=False,
          expect_trim_targets=False),
      dict(
          task_name="enc_dec_partial_trim_nothing",
          task_feature_lengths=None,
          expect_trim_inputs=False,
          expect_trim_targets=False))
  def test_partial_sequence_length(self, task_name, task_feature_lengths,
                                   expect_trim_inputs, expect_trim_targets):
    x = [{
        "inputs": [7, 8, 5, 6, 9, 4, 3],
        "targets": [3, 9]
    }, {
        "inputs": [8, 4],
        "targets": [4]
    }, {
        "inputs": [5, 6, 7],
        "targets": [6, 5]
    }]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(task_name, dataset_fn=dataset_fn)
    # Unlike the other tests, don't use a feature converter. Instead, test the
    # task.get_dataset method directly, which is similar to how evaluation.py
    # infers feature lengths w/trimming.
    task = dataset_providers_helpers.get_mixture_or_task(task_name)
    output_ds = task.get_dataset(
        sequence_length=task_feature_lengths, shuffle=False)

    expected = [{
        "inputs": [7, 8, 5, 6, 9, 4, 3, 1],
        "targets": [3, 9, 1],
    }, {
        "inputs": [8, 4, 1],
        "targets": [4, 1],
    }, {
        "inputs": [5, 6, 7, 1],
        "targets": [6, 5, 1],
    }]
    if expect_trim_inputs:
      expected[0]["inputs"] = [7, 8, 5, 6, 9, 4, 1]
    if expect_trim_targets:
      expected[0]["targets"] = [3, 1]
      expected[2]["targets"] = [6, 1]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  @parameterized.parameters(
      dict(
          task_name="enc_dec_multidim_trim_both",
          task_feature_lengths={
              "inputs": (2, 5),
              "targets": 2
          },
          expect_trim_inputs=True,
          expect_trim_targets=True,
      ),
      dict(
          task_name="enc_dec_multidim_trim_inputs",
          task_feature_lengths={
              "inputs": (2, 5),
              "targets": None
          },
          expect_trim_inputs=True,
          expect_trim_targets=False,
      ),
      dict(
          task_name="enc_dec_multidim_trim_targets",
          task_feature_lengths={
              "inputs": None,
              "targets": 2
          },
          expect_trim_inputs=False,
          expect_trim_targets=True,
      ),
      dict(
          task_name="enc_dec_no_multidim_trim",
          task_feature_lengths={
              "inputs": None,
              "targets": None
          },
          expect_trim_inputs=False,
          expect_trim_targets=False))
  def test_multidimension_sequence_length(self, task_name, task_feature_lengths,
                                          expect_trim_inputs,
                                          expect_trim_targets):
    x = [{
        "inputs": [[7, 8, 5, 6, 9, 4, 3], [2, 3, 4, 5, 0, 0, 0],
                   [6, 7, 1, 0, 0, 0, 0]],
        "targets": [3, 9]
    }, {
        "inputs": [[8, 4], [1, 0], [2, 3]],
        "targets": [4]
    }, {
        "inputs": [[5, 6, 7]],
        "targets": [6, 5, 1]
    }, {
        "inputs": [[7, 8, 9, 1, 2, 3, 4, 5, 6]],
        "targets": [10, 11, 1]
    }]
    ds = tf.data.Dataset.from_generator(
        lambda: x,
        output_types={
            "inputs": tf.int32,
            "targets": tf.int32
        },
        output_shapes={
            "inputs": (None, None),
            "targets": (None,)
        })
    dataset_fn = lambda split, shuffle_files: ds
    dataset_providers.TaskRegistry.add(
        task_name,
        source=dataset_providers.FunctionDataSource(
            dataset_fn=dataset_fn, splits=["train", "validation"]),
        preprocessors=[
            dataset_providers.CacheDatasetPlaceholder(),
        ],
        output_features={
            "inputs":
                dataset_providers.Feature(
                    test_utils.sentencepiece_vocab(), rank=2),
            "targets":
                dataset_providers.Feature(test_utils.sentencepiece_vocab())
        },
        metric_fns=[])
    # Unlike the other tests, don't use a feature converter. Instead, test the
    # task.get_dataset method directly, which is similar to how evaluation.py
    # infers feature lengths w/trimming.
    task = dataset_providers_helpers.get_mixture_or_task(task_name)
    output_ds = task.get_dataset(
        sequence_length=task_feature_lengths, shuffle=False)

    expected = copy.deepcopy(x)
    if expect_trim_inputs:
      expected[0]["inputs"] = [[7, 8, 5, 6, 9], [2, 3, 4, 5, 0]]
      expected[1]["inputs"] = [[8, 4], [1, 0]]
      expected[3]["inputs"] = [[7, 8, 9, 1, 2]]
    if expect_trim_targets:
      expected[2]["targets"] = [6, 5]
      expected[3]["targets"] = [10, 11]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  def test_get_dataset_enc_dec_packed(self):
    mixture_or_task_name = "enc_dec_packed"
    x = [{
        "inputs": [7, 8, 5, 6, 9, 4, 3],
        "targets": [3, 9]
    }, {
        "inputs": [8, 4],
        "targets": [4]
    }, {
        "inputs": [5, 6, 7],
        "targets": [6, 5]
    }]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=True)
    output_ds = dataset_providers_helpers.get_dataset(
        mixture_or_task_name=mixture_or_task_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split="train",
        shuffle=False,
        feature_converter=converter)

    expected = [
        {
            # Example 1 is trimmed
            "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
            "encoder_segment_ids": [1, 1, 1, 1, 1, 1, 1],
            "encoder_positions": [0, 1, 2, 3, 4, 5, 6],
            "decoder_target_tokens": [3, 9, 1, 0, 0],
            "decoder_input_tokens": [0, 3, 9, 0, 0],
            "decoder_loss_weights": [1, 1, 1, 0, 0],
            "decoder_segment_ids": [1, 1, 1, 0, 0],
            "decoder_positions": [0, 1, 2, 0, 0],
        },
        {
            # Example 2 and 3 are packed together
            "encoder_input_tokens": [8, 4, 1, 5, 6, 7, 1],
            "encoder_segment_ids": [1, 1, 1, 2, 2, 2, 2],
            "encoder_positions": [0, 1, 2, 0, 1, 2, 3],
            "decoder_target_tokens": [4, 1, 6, 5, 1],
            "decoder_input_tokens": [0, 4, 0, 6, 5],
            "decoder_loss_weights": [1, 1, 1, 1, 1],
            "decoder_segment_ids": [1, 1, 2, 2, 2],
            "decoder_positions": [0, 1, 0, 1, 2],
        }
    ]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  def test_get_dataset_both_train_and_validation_splits(self):
    mixture_or_task_name = "both_train_and_validation_splits"
    x_train = [{"inputs": [7, 8, 5, 6, 9, 4, 3], "targets": [3, 9]}]
    x_val = [{"inputs": [8, 4], "targets": [4]}]
    datasets = {
        "train": create_default_dataset(x_train),
        "validation": create_default_dataset(x_val)
    }
    dataset_fn = lambda split, shuffle_files: datasets[split]
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    output_ds = {}
    for split in ["train", "validation"]:
      converter = feature_converters.EncDecFeatureConverter(pack=False)
      output_ds[split] = dataset_providers_helpers.get_dataset(
          mixture_or_task_name=mixture_or_task_name,
          task_feature_lengths=task_feature_lengths,
          dataset_split=split,
          shuffle=False,
          feature_converter=converter)

    expected_train = {
        "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
        "decoder_target_tokens": [3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }
    expected_val = {
        "encoder_input_tokens": [8, 4, 1, 0, 0, 0, 0],
        "decoder_target_tokens": [4, 1, 0, 0, 0],
        "decoder_input_tokens": [0, 4, 1, 0, 0],
        "decoder_loss_weights": [1, 1, 0, 0, 0],
    }
    expected_dtypes = {feat: tf.int32 for feat in expected_train.keys()}
    assert_dataset(
        output_ds["train"], expected_train, expected_dtypes=expected_dtypes)
    assert_dataset(
        output_ds["validation"], expected_val, expected_dtypes=expected_dtypes)

  def test_get_dataset_enc_dec_sharded(self):
    mixture_or_task_name = "enc_dec_sharded"
    x = [{
        "inputs": [7, 8, 5, 6, 9, 4, 3],
        "targets": [3, 9]
    }, {
        "inputs": [8, 4],
        "targets": [4]
    }, {
        "inputs": [5, 6, 7],
        "targets": [6, 5]
    }]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=False)
    shard_info = dataset_providers.ShardInfo(index=0, num_shards=2)
    output_ds = dataset_providers_helpers.get_dataset(
        mixture_or_task_name=mixture_or_task_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split="train",
        shuffle=False,
        feature_converter=converter,
        shard_info=shard_info)

    # Example index 1 should not be present in the sharded dataset.
    expected = [{
        "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
        "decoder_target_tokens": [3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }, {
        "encoder_input_tokens": [5, 6, 7, 1, 0, 0, 0],
        "decoder_target_tokens": [6, 5, 1, 0, 0],
        "decoder_input_tokens": [0, 6, 5, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  def test_get_dataset_enc_dec_sharded_and_packed(self):
    mixture_or_task_name = "enc_dec_sharded_and_packed"
    x = [{
        "inputs": [7, 8],
        "targets": [3, 9]
    }, {
        "inputs": [8, 4],
        "targets": [4]
    }, {
        "inputs": [5, 6, 7],
        "targets": [6]
    }]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=True)
    shard_info = dataset_providers.ShardInfo(index=0, num_shards=2)
    output_ds = dataset_providers_helpers.get_dataset(
        mixture_or_task_name=mixture_or_task_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split="train",
        shuffle=False,
        feature_converter=converter,
        shard_info=shard_info)

    # Packing should be done after the sharding.
    expected = {
        "encoder_input_tokens": [7, 8, 1, 5, 6, 7, 1],
        "encoder_segment_ids": [1, 1, 1, 2, 2, 2, 2],
        "encoder_positions": [0, 1, 2, 0, 1, 2, 3],
        "decoder_target_tokens": [3, 9, 1, 6, 1],
        "decoder_input_tokens": [0, 3, 9, 0, 6],
        "decoder_loss_weights": [1, 1, 1, 1, 1],
        "decoder_segment_ids": [1, 1, 1, 2, 2],
        "decoder_positions": [0, 1, 2, 0, 1],
    }
    expected_dtypes = {feat: tf.int32 for feat in expected.keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)


def register_dummy_task(
    task_name: str,
    dataset_fn: Callable[[str, str], tf.data.Dataset],
    output_feature_names: Sequence[str] = ("inputs", "targets")
) -> None:
  """Register a dummy task for GetDatasetTest."""
  dataset_providers.TaskRegistry.add(
      task_name,
      source=dataset_providers.FunctionDataSource(
          dataset_fn=dataset_fn, splits=["train", "validation"]),
      preprocessors=[
          dataset_providers.CacheDatasetPlaceholder(),
          preprocessors.append_eos_after_trim,
      ],
      output_features={
          feat: dataset_providers.Feature(test_utils.sentencepiece_vocab())
          for feat in output_feature_names
      },
      metric_fns=[])


if __name__ == "__main__":
  absltest.main()
