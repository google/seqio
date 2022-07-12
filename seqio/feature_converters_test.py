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

"""Tests for seqio.feature_converters."""

import re
from unittest import mock
from seqio import feature_converters
from seqio import test_utils
import tensorflow.compat.v2 as tf

FeatureSpec = feature_converters.FeatureConverter.FeatureSpec

tf.compat.v1.enable_eager_execution()

assert_dataset = test_utils.assert_dataset
create_default_dataset = test_utils.create_default_dataset


class HelperFunctionsTest(tf.test.TestCase):

  def test_non_padding_position(self):
    x = tf.constant([3, 8, 5, 0, 0, 2, 0])
    non_padding_position = feature_converters.non_padding_position(x)
    expected = [1, 1, 1, 0, 0, 1, 0]
    actual = self.evaluate(non_padding_position)
    self.assertAllEqual(actual, expected)

  def test_check_lengths_strict_no_exception(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 5, "targets": 4}
    sequence_axis_mapping = {"inputs": 0, "targets": 0}
    ds = feature_converters._check_lengths(
        ds,
        task_feature_lengths,
        sequence_axis_mapping,
        strict=True,
        error_label="initial")
    list(ds.as_numpy_iterator())

  def test_check_lengths_strict_exception(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 7, "targets": 4}
    sequence_axis_mapping = {"inputs": 0, "targets": 0}
    expected_msg = (
        r".*Feature \\'inputs\\' has length not equal to the expected length of"
        r" 7 during initial validation.*")
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, expected_msg):
      ds = feature_converters._check_lengths(
          ds,
          task_feature_lengths,
          sequence_axis_mapping,
          strict=True,
          error_label="initial")
      list(ds.as_numpy_iterator())

  def test_check_lengths_not_strict_no_exception(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 7, "targets": 4}
    sequence_axis_mapping = {"inputs": 0, "targets": 0}
    ds = feature_converters._check_lengths(
        ds,
        task_feature_lengths,
        sequence_axis_mapping,
        strict=False,
        error_label="initial")
    list(ds.as_numpy_iterator())

  def test_check_lengths_not_strict_exception(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 4, "targets": 4}
    sequence_axis_mapping = {"inputs": 0, "targets": 0}
    expected_msg = (
        r".*Feature \\'inputs\\' has length not less than or equal to the "
        r"expected length of 4 during initial validation.*")
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, expected_msg):
      ds = feature_converters._check_lengths(
          ds,
          task_feature_lengths,
          sequence_axis_mapping,
          strict=False,
          error_label="initial")
      list(ds.as_numpy_iterator())

  def test_check_lengths_extra_features(self):
    x = [{"targets": [3, 9, 4, 5], "targets_pretokenized": "some text"}]
    output_types = {"targets": tf.int64, "targets_pretokenized": tf.string}
    output_shapes = {"targets": [4], "targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=output_types, output_shapes=output_shapes)
    task_feature_lengths = {"targets": 4}
    sequence_axis_mapping = {"targets": 0}
    ds = feature_converters._check_lengths(
        ds,
        task_feature_lengths,
        sequence_axis_mapping,
        strict=True,
        error_label="initial")
    list(ds.as_numpy_iterator())

  def test_check_lengths_seq_axis_1(self):
    x = [{
        "targets": [[1, 2, 3], [4, 5, 6]],
        "targets_pretokenized": "some text"
    }]
    output_types = {"targets": tf.int64, "targets_pretokenized": tf.string}
    output_shapes = {"targets": [2, 3], "targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=output_types, output_shapes=output_shapes)
    task_feature_lengths = {"targets": 3}
    sequence_axis_mapping = {"targets": 1}
    ds = feature_converters._check_lengths(
        ds,
        task_feature_lengths,
        sequence_axis_mapping,
        strict=True,
        error_label="initial")
    list(ds.as_numpy_iterator())

  def test_check_exact_match_redundant_features(self):
    expected_msg = (
        "The input_dataset contains extra features not specified in the "
        "task_features: ({'random', 'inputs'}|{'inputs', 'random'})")
    expected_msg = re.compile(expected_msg)
    with self.assertRaisesRegex(ValueError, expected_msg):
      feature_converters._check_exact_match(
          expected_features=["targets"],
          actual_features=["inputs", "targets", "random"],
          expected_feature_source="task_features",
          actual_feature_source="input_dataset")

  def test_check_exact_match_missing_features(self):
    expected_msg = (
        "The input_dataset is missing features specified in the "
        "task_features: ({'random', 'inputs'}|{'inputs', 'random'})")
    expected_msg = re.compile(expected_msg)
    with self.assertRaisesRegex(ValueError, expected_msg):
      feature_converters._check_exact_match(
          expected_features=["inputs", "targets", "random"],
          actual_features=["targets"],
          expected_feature_source="task_features",
          actual_feature_source="input_dataset")


class FeatureConvertersTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    feature_converters.FeatureConverter.TASK_FEATURES = {}
    feature_converters.FeatureConverter.MODEL_FEATURES = {}
    feature_converters.FeatureConverter.PACKING_FEATURE_DTYPES = {}

  def tearDown(self):
    del feature_converters.FeatureConverter.TASK_FEATURES
    del feature_converters.FeatureConverter.MODEL_FEATURES
    del feature_converters.FeatureConverter.PACKING_FEATURE_DTYPES
    super().tearDown()

  def test_validate_dataset_missing_feature(self):
    x = [{"targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x, feature_names=["targets"])
    task_feature_lengths = {"inputs": 4, "targets": 4}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()  # pytype: disable=not-instantiable
      expected_msg = ("Dataset is missing an expected feature during "
                      "initial validation: 'inputs'")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter._validate_dataset(
            ds,
            expected_features={
                "inputs": FeatureSpec(dtype=tf.int32),
                "targets": FeatureSpec(dtype=tf.int32)
            },
            expected_lengths=task_feature_lengths,
            strict=False,
            error_label="initial")

  def test_validate_dataset_incorrect_dtype(self):
    x = [{"inputs": [9, 4, 3, 8, 6], "targets": [3, 9, 4, 5]}]
    task_feature_dtypes = {"inputs": tf.int32, "targets": tf.int64}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=task_feature_dtypes,
        output_shapes={"inputs": [None], "targets": [None]})
    task_feature_lengths = {"inputs": 5, "targets": 4}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      feature_converters.FeatureConverter.TASK_FEATURES = {
          k: FeatureSpec(v) for k, v in task_feature_dtypes.items()}
      converter = feature_converters.FeatureConverter()  # pytype: disable=not-instantiable
      expected_msg = ("Dataset has incorrect type for feature 'inputs' during "
                      "initial validation: Got int32, expected int64")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter._validate_dataset(
            ds,
            expected_features={
                "inputs": FeatureSpec(dtype=tf.int64),
                "targets": FeatureSpec(dtype=tf.int64)
            },
            expected_lengths=task_feature_lengths,
            strict=False,
            error_label="initial")

  def test_validate_dataset_incorrect_rank(self):
    x = [{"inputs": [[9, 4, 3, 8, 6]], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [None, 1], "targets": [None]})
    task_feature_lengths = {"inputs": 5, "targets": 4}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()  # pytype: disable=not-instantiable
      expected_msg = ("Dataset has incorrect rank for feature 'inputs' during "
                      "initial validation: Got 2, expected 1")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter._validate_dataset(
            ds,
            expected_features={
                "inputs": FeatureSpec(dtype=tf.int64),
                "targets": FeatureSpec(dtype=tf.int64)
            },
            expected_lengths=task_feature_lengths,
            strict=False,
            error_label="initial")

  def test_validate_dataset_rank_2(self):
    x = [{"inputs": [[9, 4, 3, 8, 6]], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [None, 1], "targets": [None]})
    task_feature_lengths = {"inputs": 5, "targets": 4}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()  # pytype: disable=not-instantiable
      converter._validate_dataset(
          ds,
          expected_features={
              "inputs": FeatureSpec(dtype=tf.int64, rank=2),
              "targets": FeatureSpec(dtype=tf.int64)
          },
          expected_lengths=task_feature_lengths,
          strict=False,
          error_label="initial")

  def test_validate_dataset_rank_2_with_pack(self):
    x = [{"inputs": [[9, 4, 3, 8, 6]], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [None, 1], "targets": [None]})
    task_feature_lengths = {"inputs": 5, "targets": 4}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()),\
        mock.patch.object(feature_converters.FeatureConverter,
                          "_convert_features", return_value=ds):
      converter = feature_converters.FeatureConverter(pack=True)  # pytype: disable=not-instantiable
      feature_converters.FeatureConverter.TASK_FEATURES = {
          "inputs": FeatureSpec(tf.int64, rank=2),
          "targets": FeatureSpec(tf.int64)
      }
      feature_converters.FeatureConverter.MODEL_FEATURES = {
          "inputs": FeatureSpec(tf.int64, rank=2),
          "targets": FeatureSpec(tf.int64)
      }
      expected_msg = ("When packing is enabled, expected ranks must be 1 or "
                      "use_custom_packing_ops must be set. Got expected rank 2 "
                      "for feature inputs.")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter(ds, task_feature_lengths)

  def test_call_missing_input_lengths(self):
    x = [{"inputs": [9, 4, 3, 8, 6], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [5], "targets": [5]})
    task_feature_lengths = {"inputs": 5}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()  # pytype: disable=not-instantiable
      feature_converters.FeatureConverter.TASK_FEATURES = {
          "inputs": FeatureSpec(tf.int64),
          "targets": FeatureSpec(tf.int64)
      }
      expected_msg = ("The task_feature_lengths is missing features specified "
                      "in the TASK_FEATURES: {'targets'}")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter(ds, task_feature_lengths)

  def test_validate_dataset_pretokenized_field(self):
    x = [{"targets": [3, 9, 4, 5], "targets_pretokenized": "some text"}]
    output_types = {"targets": tf.int64, "targets_pretokenized": tf.string}
    output_shapes = {"targets": [4], "targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=output_types, output_shapes=output_shapes)

    task_feature_lengths = {"targets": 4}
    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()  # pytype: disable=not-instantiable
      # _validate_dataset works even if ds has targets and targets_pretokenized
      ds = converter._validate_dataset(
          ds,
          expected_features={"targets": FeatureSpec(dtype=tf.int64)},
          expected_lengths=task_feature_lengths,
          strict=True,
          error_label="initial")


class EncDecFeatureConverterTest(tf.test.TestCase):

  def test_encoder_decoder_unpacked(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 7, "targets": 5}

    converter = feature_converters.EncDecFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "encoder_input_tokens": [9, 4, 3, 8, 1, 0, 0],
        "decoder_target_tokens": [3, 9, 4, 1, 0],
        # mtf.transformer.autoregressive_inputs does not zero out the last eos
        # when the data is not packed. This test mimic the behavior.
        "decoder_input_tokens": [0, 3, 9, 4, 1],
        "decoder_loss_weights": [1, 1, 1, 1, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_targets_max_length(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 5, "targets": 5}

    converter = feature_converters.EncDecFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "encoder_input_tokens": [9, 4, 3, 8, 1],
        "decoder_target_tokens": [3, 9, 4, 5, 1],
        "decoder_input_tokens": [0, 3, 9, 4, 5],
        "decoder_loss_weights": [1, 1, 1, 1, 1],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_extra_long_inputs(self):
    x = [{"inputs": [9, 4, 3, 8, 4, 5, 1], "targets": [3, 9, 4, 7, 8, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 5, "targets": 8}
    expected_msg = (
        r".*Feature \\'inputs\\' has length not less than or equal to the "
        r"expected length of 5 during input_validation.*")
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, expected_msg):
      converter = feature_converters.EncDecFeatureConverter(pack=False)
      converted_ds = converter(ds, task_feature_lengths)
      list(converted_ds.as_numpy_iterator())

  def test_encoder_decoder_packed(self):
    x = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
         {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 10, "targets": 7}

    converter = feature_converters.EncDecFeatureConverter(pack=True)
    converted_ds = converter(ds, task_feature_lengths)
    expected = {
        "encoder_input_tokens": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        "encoder_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 0, 4, 0, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0, 0],
        "decoder_positions": [0, 1, 2, 0, 1, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_packed_long_sequences(self):
    x = [{"inputs": [7, 8, 5, 6, 9, 4, 1], "targets": [3, 9, 1]},
         {"inputs": [8, 4, 9, 3, 5, 1], "targets": [4, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 7, "targets": 3}

    converter = feature_converters.EncDecFeatureConverter(pack=True)
    converted_ds = converter(ds, task_feature_lengths)

    # Corner case: packing is true but task_feature_lengths are too long for
    # packing to happen. We should still get the *_segment_id, *_position
    # fields.
    expected = [{
        "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
        "encoder_segment_ids": [1, 1, 1, 1, 1, 1, 1],
        "encoder_positions": [0, 1, 2, 3, 4, 5, 6],
        "decoder_target_tokens": [3, 9, 1],
        "decoder_input_tokens": [0, 3, 9],
        "decoder_loss_weights": [1, 1, 1],
        "decoder_segment_ids": [1, 1, 1],
        "decoder_positions": [0, 1, 2],
    }, {
        "encoder_input_tokens": [8, 4, 9, 3, 5, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 1, 1, 0],
        "encoder_positions": [0, 1, 2, 3, 4, 5, 0],
        "decoder_target_tokens": [4, 1, 0],
        "decoder_input_tokens": [0, 4, 0],
        "decoder_loss_weights": [1, 1, 0],
        "decoder_segment_ids": [1, 1, 0],
        "decoder_positions": [0, 1, 0],
    }]
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_pretokenized_field(self):
    x = [{
        "inputs": [7, 8, 5, 1],
        "targets": [3, 9, 1],
        "targets_pretokenized": "abc"
    }, {
        "inputs": [8, 4, 9, 3, 1],
        "targets": [4, 1],
        "targets_pretokenized": "def"
    }]
    types = {
        "inputs": tf.int32,
        "targets": tf.int32,
        "targets_pretokenized": tf.string
    }
    shapes = {"inputs": [None], "targets": [None], "targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=types, output_shapes=shapes)

    task_feature_lengths = {"inputs": 10, "targets": 7}
    converter = feature_converters.EncDecFeatureConverter(pack=True)
    # Check whether convert_features raise error because targets_pretokenized is
    # present in the ds but not in the task_feature_lengths
    converter(ds, task_feature_lengths)


class LMFeatureConverter(tf.test.TestCase):

  def test_lm_unpacked(self):
    x = [{"targets": [3, 9, 1]}]
    ds = create_default_dataset(x, feature_names=["targets"])
    task_feature_lengths = {"targets": 5}

    converter = feature_converters.LMFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "decoder_target_tokens": [3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_lm_only_packed(self):
    x = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]
    ds = create_default_dataset(x, feature_names=["targets"])
    task_feature_lengths = {"targets": 6}

    converter = feature_converters.LMFeatureConverter(pack=True)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0],
        "decoder_input_tokens": [0, 3, 9, 0, 4, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 0],
        "decoder_positions": [0, 1, 2, 0, 1, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0]
    }
    assert_dataset(converted_ds, expected)

  def test_lm_pack_long_sequences(self):
    x = [{"targets": [3, 9, 4, 5, 1]}, {"targets": [4, 3, 2, 1]}]
    ds = create_default_dataset(x, feature_names=["targets"])
    task_feature_lengths = {"targets": 5}

    converter = feature_converters.LMFeatureConverter(pack=True)
    converted_ds = converter(ds, task_feature_lengths)

    expected = [{
        "decoder_target_tokens": [3, 9, 4, 5, 1],
        "decoder_input_tokens": [0, 3, 9, 4, 5],
        "decoder_loss_weights": [1, 1, 1, 1, 1],
        "decoder_positions": [0, 1, 2, 3, 4],
        "decoder_segment_ids": [1, 1, 1, 1, 1]
    }, {
        "decoder_target_tokens": [4, 3, 2, 1, 0],
        "decoder_input_tokens": [0, 4, 3, 2, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 0],
        "decoder_positions": [0, 1, 2, 3, 0],
        "decoder_segment_ids": [1, 1, 1, 1, 0]
    }]
    assert_dataset(converted_ds, expected)

  def test_lm_plaintext_field(self):
    x = [{"targets": [3, 9, 1], "targets_plaintext": "abc"},
         {"targets": [4, 1], "targets_plaintext": "abc"}]
    types = {"targets": tf.int32, "targets_plaintext": tf.string}
    shapes = {"targets": [None], "targets_plaintext": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=types, output_shapes=shapes)
    task_feature_lengths = {"targets": 6}

    converter = feature_converters.LMFeatureConverter(pack=True)
    converter(ds, task_feature_lengths)


class PrefixLMFeatureConverter(tf.test.TestCase):

  def test_prefix_lm_unpacked(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]
    ds = create_default_dataset(x)

    task_feature_lengths = {"inputs": 5, "targets": 4}
    converter = feature_converters.PrefixLMFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "decoder_target_tokens": [9, 4, 6, 1, 3, 9, 1, 0, 0],
        # The last EOS token is kept if unpacked.
        "decoder_input_tokens": [0, 9, 4, 6, 1, 3, 9, 1, 0],
        "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0]
    }
    assert_dataset(converted_ds, expected)

  def test_prefix_lm_unpacked_trivial_targets(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": []}]
    ds = create_default_dataset(x)

    task_feature_lengths = {"inputs": 5, "targets": 4}
    converter = feature_converters.PrefixLMFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "decoder_target_tokens": [9, 4, 6, 1, 0, 0, 0, 0, 0],
        # The last EOS token is kept if unpacked.
        "decoder_input_tokens": [0, 9, 4, 6, 1, 0, 0, 0, 0],
        "decoder_loss_weights": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0]
    }

    assert_dataset(converted_ds, expected)

  def test_prefix_lm_long_inputs_feature_length(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]
    ds = create_default_dataset(x)

    task_feature_lengths = {"inputs": 10, "targets": 4}
    converter = feature_converters.PrefixLMFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "decoder_target_tokens": [9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0, 0],
        # The last EOS token is kept if unpacked.
        "decoder_input_tokens": [0, 9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0],
        "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    assert_dataset(converted_ds, expected)

  def test_prefix_lm_packed(self):
    x = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
         {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]
    ds = create_default_dataset(x)

    task_feature_lengths = {"inputs": 8, "targets": 7}
    converter = feature_converters.PrefixLMFeatureConverter(pack=True)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "decoder_input_tokens": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
        "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        "decoder_positions": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
        "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
        "decoder_causal_attention": [
            1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0
        ]
    }
    assert_dataset(converted_ds, expected)

  def test_prefix_lm_unpacked_loss_on_inputs_and_targets(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]
    ds = create_default_dataset(x)

    task_feature_lengths = {"inputs": 5, "targets": 4}
    converter = feature_converters.PrefixLMFeatureConverter(
        pack=False, loss_on_targets_only=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "decoder_target_tokens": [9, 4, 6, 1, 3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 9, 4, 6, 1, 3, 9, 1, 0],
        # Loss weights on the inputs portion and padding should be zeroed out.
        "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 0, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0]
    }
    assert_dataset(converted_ds, expected)

  def test_prefix_lm_packed_loss_on_inputs_and_targets(self):
    x = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
         {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]
    ds = create_default_dataset(x)

    task_feature_lengths = {"inputs": 8, "targets": 7}
    converter = feature_converters.PrefixLMFeatureConverter(
        pack=True, loss_on_targets_only=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "decoder_input_tokens": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "decoder_positions": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
        "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
        "decoder_causal_attention": [
            1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0
        ]
    }
    assert_dataset(converted_ds, expected)

  def test_prefix_lm_long_inputs(self):
    x = [{"inputs": [7, 8, 5, 6, 1], "targets": [3, 9, 7, 1]},
         {"inputs": [8, 4, 9, 3, 8, 1], "targets": [4, 1]}]
    ds = create_default_dataset(x)

    task_feature_lengths = {"inputs": 4, "targets": 3}
    expected_msg = (
        r".*Feature \\'inputs\\' has length not less than or equal to the "
        r"expected length of 4 during input_validation.*")
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, expected_msg):
      converter = feature_converters.PrefixLMFeatureConverter(pack=True)
      converted_ds = converter(ds, task_feature_lengths)
      list(converted_ds.as_numpy_iterator())

  def test_prefix_lm_pack_long_sequences(self):
    x = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
         {"inputs": [8, 4, 1], "targets": [5, 1]}]
    ds = create_default_dataset(x)

    task_feature_lengths = {"inputs": 4, "targets": 3}
    converter = feature_converters.PrefixLMFeatureConverter(pack=True)
    converted_ds = converter(ds, task_feature_lengths)

    # The examples should not be packed because examples are not short enough.
    expected = [{
        "decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1],
        "decoder_input_tokens": [0, 7, 8, 5, 1, 3, 9],
        "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1],
        "decoder_positions": [0, 1, 2, 3, 4, 5, 6],
        "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0]
    }, {
        "decoder_target_tokens": [8, 4, 1, 5, 1, 0, 0],
        "decoder_input_tokens": [0, 8, 4, 1, 5, 0, 0],
        "decoder_loss_weights": [0, 0, 0, 1, 1, 0, 0],
        "decoder_positions": [0, 1, 2, 3, 4, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 1, 1, 0, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 0, 0, 0]
    }]
    assert_dataset(converted_ds, expected)

  def test_convert_example(self):
    features = {
        "targets": tf.constant([7, 8, 5, 1, 3, 9, 1, 0]),
        "inputs_width": tf.constant([4, 4, 4, 4, 4, 4, 4, 0]),
        "inputs_width_add_pos": tf.constant([5, 5, 5, 5, 5, 5, 5, 0])
    }
    converter = feature_converters.PrefixLMFeatureConverter(pack=False)
    expected = {"decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1, 0],
                "decoder_input_tokens": [0, 7, 8, 5, 1, 3, 9, 1],
                "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0],
                "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0]}
    actual = converter._convert_example(features)
    for feat, tensor in actual.items():
      self.assertAllEqual(expected[feat], self.evaluate(tensor))


class DecoderFeatureConverterTest(FeatureConvertersTest):

  def test_prefixlm(self):
    x = [{
        "inputs": [7, 8, 5, 1],
        "targets": [3, 9, 1]
    }, {
        "inputs": [8, 4, 9, 3, 1],
        "targets": [4, 1]
    }]
    ds = create_default_dataset(x)

    task_feature_lengths = {"inputs": 8, "targets": 7}
    converter = feature_converters.DecoderFeatureConverter(
        pack=True, loss_on_targets_only=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "decoder_input_tokens": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "decoder_positions": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
        "decoder_segment_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
        "decoder_causal_attention": [
            1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0
        ]
    }
    assert_dataset(converted_ds, expected)

  def test_lm(self):
    x = [{"targets": [3, 9, 4, 5, 1]}, {"targets": [4, 3, 2, 1]}]
    ds = create_default_dataset(x, feature_names=["targets"])
    task_feature_lengths = {"targets": 5}

    converter = feature_converters.DecoderFeatureConverter(
        pack=True, loss_on_targets_only=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = [{
        "decoder_target_tokens": [3, 9, 4, 5, 1],
        "decoder_input_tokens": [0, 3, 9, 4, 5],
        "decoder_loss_weights": [1, 1, 1, 1, 1],
        "decoder_positions": [0, 1, 2, 3, 4],
        "decoder_segment_ids": [1, 1, 1, 1, 1]
    }, {
        "decoder_target_tokens": [4, 3, 2, 1, 0],
        "decoder_input_tokens": [0, 4, 3, 2, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 0],
        "decoder_positions": [0, 1, 2, 3, 0],
        "decoder_segment_ids": [1, 1, 1, 1, 0]
    }]
    assert_dataset(converted_ds, expected)


class EncoderFeatureConverterTest(FeatureConvertersTest):

  def test_encoder_unpacked(self):
    x = [{
        # Assume 9 is the sentinel used to indicate prediction-tokens (e.g., for
        # MLM this would be [MASK] token).
        "inputs": [8, 9, 4, 9, 1],
        "targets": [8, 7, 4, 6, 1]
    }]

    ds = create_default_dataset(x)
    input_lengths = {"inputs": 6, "targets": 6}
    converter = feature_converters.EncoderFeatureConverter(
        mask_id=9, pack=False)
    converted_ds = converter(ds, input_lengths)

    # Determine the loss weight by tf.equal(inputs == mask_sentinel)
    # Let 8 be the index of the sentinel used for classification. For BERT this
    # corresponds to [CLS] token.
    expected = {
        "encoder_input_tokens": [8, 9, 4, 9, 1, 0],
        "encoder_target_tokens": [8, 7, 4, 6, 1, 0],
        "encoder_loss_weights": [0, 1, 0, 1, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_packed(self):
    x = [{"inputs": [8, 9, 9, 3, 4, 1], "targets": [8, 7, 4, 3, 4, 1]},
         {"inputs": [8, 3, 9, 1], "targets": [8, 3, 6, 1]}]

    ds = create_default_dataset(x)
    input_lengths = {"inputs": 11, "targets": 11}
    converter = feature_converters.EncoderFeatureConverter(mask_id=9)
    converted_ds = converter(ds, input_lengths)

    expected = {
        "encoder_input_tokens": [8, 9, 9, 3, 4, 1, 8, 3, 9, 1, 0],
        "encoder_target_tokens": [8, 7, 4, 3, 4, 1, 8, 3, 6, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0],
        "encoder_positions": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0],
        "encoder_loss_weights": [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_pack_long_sequences(self):
    x = [{"inputs": [8, 9, 9, 3, 1], "targets": [8, 7, 4, 3, 1]},
         {"inputs": [8, 3, 9, 1], "targets": [8, 3, 6, 1]}]

    ds = create_default_dataset(x)
    input_lengths = {"inputs": 5, "targets": 5}
    converter = feature_converters.EncoderFeatureConverter(mask_id=9)
    converted_ds = converter(ds, input_lengths)

    expected = [{
        "encoder_input_tokens": [8, 9, 9, 3, 1],
        "encoder_target_tokens": [8, 7, 4, 3, 1],
        "encoder_segment_ids": [1, 1, 1, 1, 1],
        "encoder_positions": [0, 1, 2, 3, 4],
        "encoder_loss_weights": [0, 1, 1, 0, 0],
    }, {
        "encoder_input_tokens": [8, 3, 9, 1, 0],
        "encoder_target_tokens": [8, 3, 6, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 0],
        "encoder_positions": [0, 1, 2, 3, 0],
        "encoder_loss_weights": [0, 0, 1, 0, 0],
    }]
    assert_dataset(converted_ds, expected)

  def test_encoder_plaintext_field(self):
    x = [{
        "inputs": [8, 9, 9, 3, 4, 1],
        "targets": [8, 7, 4, 3, 4, 1],
        "targets_plaintext": "abc"
    }, {
        "inputs": [8, 3, 9, 1],
        "targets": [8, 3, 6, 1],
        "targets_plaintext": "def"
    }]
    types = {
        "inputs": tf.int32,
        "targets": tf.int32,
        "targets_plaintext": tf.string
    }
    shapes = {"inputs": [None], "targets": [None], "targets_plaintext": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=types, output_shapes=shapes)

    input_lengths = {"inputs": 7, "targets": 7}
    converter = feature_converters.EncoderFeatureConverter(mask_id=9)
    # Check whether convert_features raise error because targets_plaintext is
    # present in the ds but not in the output_features
    converter(ds, input_lengths)


class PassThroughFeatureConverterTest(tf.test.TestCase):

  def test_equivalence(self):
    x = [{
        "decoder_target_tokens": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "decoder_input_tokens": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    }]
    ds = create_default_dataset(
        x,
        feature_names=[
            "decoder_target_tokens", "decoder_input_tokens",
            "decoder_loss_weights"
        ])
    converter = feature_converters.PassThroughFeatureConverter()
    converted_ds = converter(ds, task_feature_lengths={})
    test_utils.assert_datasets_eq(converted_ds, ds)


class PrePackedEncDecFeatureConverterTest(tf.test.TestCase):

  def test_encoder_decoder_packed(self):
    x = [{"inputs": [7, 8, 5, 1, 8, 4, 9, 3, 1],
          "inputs_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2],
          "inputs_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4],
          "targets": [3, 9, 1, 4, 1],
          "targets_segment_ids": [1, 1, 1, 2, 2],
          "targets_positions": [0, 1, 2, 0, 1]}]
    ds = create_default_dataset(x, feature_names=("inputs",
                                                  "inputs_segment_ids",
                                                  "inputs_positions",
                                                  "targets",
                                                  "targets_segment_ids",
                                                  "targets_positions"))
    task_feature_lengths = {"inputs": 10,
                            "inputs_segment_ids": 10,
                            "inputs_positions": 10,
                            "targets": 7,
                            "targets_segment_ids": 7,
                            "targets_positions": 7}
    converter = feature_converters.PrePackedEncDecFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)
    expected = {
        "encoder_input_tokens": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        "encoder_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 0, 4, 0, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0, 0],
        "decoder_positions": [0, 1, 2, 0, 1, 0, 0],
    }
    assert_dataset(converted_ds, expected)

if __name__ == "__main__":
  tf.test.main()
