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

"""Tests for seqio.utils."""
from typing import Mapping, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from seqio import test_utils
from seqio import utils
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()

mock = absltest.mock
assert_dataset = test_utils.assert_dataset


class AnyArg(object):

  def __eq__(self, var):
    return True


class LazyTfdsLoaderTest(absltest.TestCase):

  def setUp(self):
    utils.LazyTfdsLoader._MEMOIZED_BUILDERS = {}
    super().setUp()

  @mock.patch("tensorflow_datasets.builder")
  def test_builder_memoization(self, mock_tfds_builder):
    mock_tfds_builder.side_effect = (
        lambda name, data_dir: ",".join([name, data_dir or ""]))

    ds1 = utils.LazyTfdsLoader("ds1")
    self.assertEqual("ds1,", ds1.builder)
    self.assertEqual(1, tfds.builder.call_count)

    # Builder should be cached with same name.
    self.assertEqual("ds1,", ds1.builder)
    self.assertEqual(1, tfds.builder.call_count)

    # Same name but different data dir is a cache miss.
    ds1_dir1 = utils.LazyTfdsLoader("ds1", "dir1")
    self.assertEqual("ds1,dir1", ds1_dir1.builder)
    self.assertEqual(2, tfds.builder.call_count)
    # Same name and data dir is a cache hit.
    self.assertEqual("ds1,dir1", ds1_dir1.builder)
    self.assertEqual(2, tfds.builder.call_count)

    # Different name is a cache miss.
    ds2 = utils.LazyTfdsLoader("ds2")
    self.assertEqual("ds2,", ds2.builder)
    self.assertEqual(3, tfds.builder.call_count)

    # Different split map name is a cache hit.
    ds2 = utils.LazyTfdsLoader("ds2", split_map={"train": "validation"})
    self.assertEqual("ds2,", ds2.builder)
    self.assertEqual(3, tfds.builder.call_count)

    # Try calling everything again, order shouldn't matter.
    self.assertEqual("ds1,", ds1.builder)
    self.assertEqual("ds1,dir1", ds1_dir1.builder)
    self.assertEqual("ds2,", ds2.builder)
    self.assertEqual(3, tfds.builder.call_count)

  @mock.patch("tensorflow_datasets.load")
  def test_split_map(self, mock_tfds_load):
    seed = 0
    utils.LazyTfdsLoader._MEMOIZED_BUILDERS[("ds/c1", None)] = mock.Mock(
        info=mock.Mock(
            splits={
                "validation":
                    mock.Mock(num_examples=420, file_instructions=["f1", "f2"]),
                "test":
                    mock.Mock(num_examples=42, file_instructions=["f3"]),
            }))

    ds = utils.LazyTfdsLoader(
        "ds/c1", split_map={
            "train": "validation",
            "validation": "test"
        })

    # test .load()
    ds.load("train", shuffle_files=False, seed=seed)
    mock_tfds_load.assert_called_once_with(
        "ds/c1",
        split="validation",
        data_dir=None,
        shuffle_files=False,
        download=True,
        try_gcs=True,
        read_config=AnyArg(),
        decoders=None)

    # test .size()
    self.assertEqual(420, ds.size(split="train"))
    self.assertEqual(42, ds.size(split="validation"))
    with self.assertRaises(KeyError):
      ds.size(split="test")

    # test .files()
    self.assertListEqual(["f1", "f2"], ds.files(split="train"))
    self.assertListEqual(["f3"], ds.files(split="validation"))
    with self.assertRaises(KeyError):
      ds.files(split="test")


class UtilsTest(parameterized.TestCase, tf.test.TestCase):
  _tfdict = {
      "bool":
          tf.constant([True, False], dtype=tf.bool),
      "int32":
          tf.constant([1], dtype=tf.int32),
      "int64":
          tf.constant([1], dtype=tf.int64),
      "float":
          tf.constant([1.]),
      "string":
          tf.constant(["a"]),
      "2d_tensor":
          tf.reshape(tf.range(4), [2, 2]),
      "3d_tensor":
          tf.reshape(tf.range(6), [2, 1, 3]),
      "2d_ragged":
          tf.ragged.constant([[1], [2, 3]]),
      "3d_ragged":
          tf.ragged.constant([[[1]], [[2, 3], [4, 5, 6]]]),
      "2d_sparse":
          tf.sparse.SparseTensor(
              indices=[[0, 1], [2, 3]],
              values=[10, 20],
              dense_shape=[3, 4],
          ),
      "3d_sparse":
          tf.sparse.SparseTensor(
              indices=[[0, 1, 2], [3, 4, 5]],
              values=[10, 20],
              dense_shape=[4, 5, 6],
          ),
  }
  _tfexample = tf.train.Example(
      features=tf.train.Features(
          feature={
              "bool":
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 0])),
              # NOTE: TFExamples only stores int64s, so we can't avoid the
              # up-casting here.
              "int32":
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
              "int64":
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
              "float":
                  tf.train.Feature(float_list=tf.train.FloatList(value=[1.])),
              "string":
                  tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"a"])),
              "2d_tensor":
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[0, 1, 2, 3])),
              "_sh:2d_tensor":
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[2, 2])),
              "3d_tensor":
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[0, 1, 2, 3, 4, 5])),
              "_sh:3d_tensor":
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[2, 1, 3])),
              "2d_ragged":
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[1, 2, 3])),
              "_rl:0:2d_ragged":
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2])),
              "3d_ragged":
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[1, 2, 3, 4, 5, 6])),
              "_rl:0:3d_ragged":
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2])),
              "_rl:1:3d_ragged":
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[1, 2, 3])),
              "2d_sparse":
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[10, 20])),
              "_sp:0:2d_sparse":
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 2])),
              "_sp:1:2d_sparse":
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 3])),
              "_sh:2d_sparse":
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[3, 4])),
              "3d_sparse":
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[10, 20])),
              "_sp:0:3d_sparse":
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 3])),
              "_sp:1:3d_sparse":
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 4])),
              "_sp:2:3d_sparse":
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[2, 5])),
              "_sh:3d_sparse":
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[4, 5, 6])),
          }))

  def assertTensorDictEqual(
      self,
      expected: utils.NestedTFDict,
      actual: utils.NestedTFDict,
  ):
    self.assertEqual(expected.keys(), actual.keys())
    for key, expected_value in expected.items():
      if isinstance(expected_value, dict):
        # Default assertEqual implementation does not check equality of nested
        # dictionaries.
        self.assertTensorDictEqual(expected_value, actual[key])
      elif isinstance(expected_value, tf.SparseTensor):
        # Default assertEqual implementation does not check equality of sparse
        # tensors.
        self.assertAllEqual(
            tf.sparse.to_dense(expected_value), tf.sparse.to_dense(actual[key]))
      else:
        self.assertAllEqual(expected_value, actual[key])

  def test_dict_to_tfexample(self):
    expected = self._tfexample
    actual = utils.dict_to_tfexample(self._tfdict, store_shapes=True)
    self.assertProtoEquals(expected, actual)

  def test_parse_dict_to_tfexample(self):
    expected = self._tfdict
    # NOTE: TFExamples only store int64s so we must upcast bools and int32s.
    expected["bool"] = tf.cast(expected["bool"], tf.int64)
    expected["int32"] = tf.cast(expected["int32"], tf.int64)

    actual = tf.io.parse_single_example(
        serialized=tf.constant(self._tfexample.SerializeToString(), tf.string),
        features={
            "bool":
                tf.io.FixedLenFeature([2], dtype=tf.int64),
            "int32":
                tf.io.FixedLenFeature([1], dtype=tf.int64),
            "int64":
                tf.io.FixedLenFeature([1], dtype=tf.int64),
            "float":
                tf.io.FixedLenFeature([1], dtype=tf.float32),
            "string":
                tf.io.FixedLenFeature([1], dtype=tf.string),
            "2d_tensor":
                tf.io.FixedLenFeature([2, 2], dtype=tf.int64),
            "3d_tensor":
                tf.io.FixedLenFeature([2, 1, 3], dtype=tf.int64),
            "2d_ragged":
                tf.io.RaggedFeature(
                    dtype=tf.int64,
                    value_key="2d_ragged",
                    partitions=(tf.io.RaggedFeature.RowLengths(
                        utils.tfexample_ragged_length_key("2d_ragged", 0)),)),
            "3d_ragged":
                tf.io.RaggedFeature(
                    dtype=tf.int64,
                    value_key="3d_ragged",
                    partitions=(
                        tf.io.RaggedFeature.RowLengths(
                            utils.tfexample_ragged_length_key("3d_ragged", 0)),
                        tf.io.RaggedFeature.RowLengths(
                            utils.tfexample_ragged_length_key("3d_ragged", 1)),
                    )),
            "2d_sparse":
                tf.io.SparseFeature(
                    value_key="2d_sparse",
                    index_key=[
                        utils.tfexample_sparse_indices_key("2d_sparse", 0),
                        utils.tfexample_sparse_indices_key("2d_sparse", 1),
                    ],
                    size=[3, 4],
                    dtype=tf.int64,
                ),
            "3d_sparse":
                tf.io.SparseFeature(
                    value_key="3d_sparse",
                    index_key=[
                        utils.tfexample_sparse_indices_key("3d_sparse", 0),
                        utils.tfexample_sparse_indices_key("3d_sparse", 1),
                        utils.tfexample_sparse_indices_key("3d_sparse", 2),
                    ],
                    size=[4, 5, 6],
                    dtype=tf.int64,
                ),
        })
    self.assertTensorDictEqual(expected, actual)

  def test_tfexample_to_dict(self):
    expected = dict(self._tfdict)
    # NOTE: TFExamples only store int64s so we must upcast bools and int32s.
    expected["bool"] = tf.cast(expected["bool"], tf.int64)
    expected["int32"] = tf.cast(expected["int32"], tf.int64)

    actual = utils.tfexample_to_dict(self._tfexample)
    self.assertTensorDictEqual(expected, actual)

  def test_dict_to_tfexample_legacy(self):
    features = {
        "inputs": "this is an input",
        "targets": "this is a target",
        "weight": 5.0,
        "idx1": np.array([1, 2], np.int32),
        "idx2": np.array([3, 4], np.int64),
        "is_correct": False,
        "2d_shape": np.arange(3).reshape((1, 3)),
        "3d_shape": np.arange(6).reshape((1, 2, 3)),
    }
    tfe = utils.dict_to_tfexample(features)

    self.assertLen(tfe.features.feature, len(features))
    self.assertEqual(tfe.features.feature["inputs"].bytes_list.value,
                     [b"this is an input"])
    self.assertEqual(tfe.features.feature["targets"].bytes_list.value,
                     [b"this is a target"])
    self.assertEqual(tfe.features.feature["weight"].float_list.value, [5.0])
    np.testing.assert_array_equal(tfe.features.feature["idx1"].int64_list.value,
                                  np.array([1, 2], np.int64))
    np.testing.assert_array_equal(tfe.features.feature["idx2"].int64_list.value,
                                  np.array([3, 4], np.int64))
    np.testing.assert_array_equal(
        tfe.features.feature["is_correct"].int64_list.value,
        np.array([0], np.int64))
    np.testing.assert_array_equal(
        tfe.features.feature["2d_shape"].int64_list.value,
        np.arange(3).reshape((1, 3)).flatten())
    np.testing.assert_array_equal(
        tfe.features.feature["3d_shape"].int64_list.value,
        np.arange(6).reshape((1, 2, 3)).flatten())

  def test_flatten_dict(self):
    expected = {
        "key1/subkey1": tf.constant([1]),
        "key1/subkey2": tf.constant([2]),
        "key2/subkey3": tf.constant([3]),
        "key3": tf.constant([4]),
    }

    actual = utils.flatten_dict({
        "key1": {
            "subkey1": tf.constant([1]),
            "subkey2": tf.constant([2]),
        },
        "key2": {
            "subkey3": tf.constant([3]),
        },
        "key3": tf.constant([4]),
    })

    self.assertDictEqual(expected, actual)

  def test_unflatten_dict(self):
    expected = {
        "key1": {
            "subkey1": tf.constant([1]),
            "subkey2": tf.constant([2]),
        },
        "key2": {
            "subkey3": tf.constant([3]),
        },
        "key3": tf.constant([4]),
    }

    actual = utils.unflatten_dict({
        "key1/subkey1": tf.constant([1]),
        "key1/subkey2": tf.constant([2]),
        "key2/subkey3": tf.constant([3]),
        "key3": tf.constant([4]),
    })
    self.assertTensorDictEqual(expected, actual)

  def test_stateless_shuffle(self):
    value = np.arange(6)
    expected_output_1 = np.array([0, 3, 4, 2, 1, 5])
    expected_output_2 = np.array([3, 4, 0, 2, 5, 1])
    np.testing.assert_array_equal(
        utils.stateless_shuffle(value, (0, 1)), expected_output_1)
    np.testing.assert_array_equal(
        utils.stateless_shuffle(value.reshape((2, 3)), (0, 1)),
        expected_output_1.reshape((2, 3)))
    np.testing.assert_array_equal(
        utils.stateless_shuffle(value, (2, 3)), expected_output_2)

  def test_map_over_dataset(self):
    inputs = tf.data.Dataset.range(5)

    @utils.map_over_dataset
    def test_fn(x):
      return x + 1

    self.assertEqual(list(test_fn(inputs).as_numpy_iterator()), [1, 2, 3, 4, 5])

  # We disable no-value-for-parameter since the utils.map_over_dataset leads to
  # a false positive when seeds are provided.
  # pylint:disable=no-value-for-parameter

  def test_map_over_dataset_with_one_seed(self):
    inputs = tf.data.Dataset.range(2)

    tf.random.set_seed(None)
    utils._NEXT_MAP_SEED = 42

    @utils.map_over_dataset(num_seeds=1)
    def test_fn(x, seed):
      return x + seed

    expected = [
        np.array([2985944072, 3810604164]),
        np.array([4132877645, 4228622226])
    ]
    for exp, act in zip(expected, test_fn(inputs).as_numpy_iterator()):
      np.testing.assert_array_equal(exp, act)

  def test_map_over_dataset_with_seeds(self):
    inputs = tf.data.Dataset.range(2)

    tf.random.set_seed(None)
    utils._NEXT_MAP_SEED = 42

    @utils.map_over_dataset(num_seeds=2)
    def test_fn(x, seeds):
      return x + seeds

    expected = [
        np.array([[2985944072, 3810604164], [64669036, 3548694723]]),
        np.array([[4132877645, 4228622226], [2495033825, 798765318]])
    ]
    for exp, act in zip(expected, test_fn(inputs).as_numpy_iterator()):
      np.testing.assert_array_equal(exp, act)

  # pylint:enable=no-value-for-parameter

  def test_map_seed_manager(self):
    utils._NEXT_MAP_SEED = None
    self.assertIsNone(utils._NEXT_MAP_SEED)
    with utils.map_seed_manager(42):
      self.assertEqual(utils._NEXT_MAP_SEED, 42)
      with utils.map_seed_manager(410):
        self.assertEqual(utils._NEXT_MAP_SEED, 410)
        utils._NEXT_MAP_SEED += 10
        self.assertEqual(utils._NEXT_MAP_SEED, 420)
      utils._NEXT_MAP_SEED += 10
      self.assertEqual(utils._NEXT_MAP_SEED, 52)
    self.assertIsNone(utils._NEXT_MAP_SEED)

  def test_trim_and_pad_dataset(self):
    x = [{
        "inputs": [7, 8, 5, 6, 1],
        "targets": [[3, 0.5], [9, 0], [1, 2]],
        "idx": [0]
    }, {
        "inputs": [8, 4, 9, 3, 5, 7, 9, 5],
        "targets": [[4, 1.2], [1, 1]],
        "idx": [1, 2]
    }]
    ds = ds = tf.data.Dataset.from_generator(
        lambda: x,
        output_signature={
            "inputs": tf.TensorSpec([None], tf.int32),
            "targets": tf.TensorSpec([None, None], tf.float32),
            "idx": tf.TensorSpec([None], tf.int32)
        })
    padded_ds = utils.trim_and_pad_dataset(
        ds, feature_lengths={
            "inputs": 7,
            "targets": 3
        })
    expected = [{
        "inputs": [7, 8, 5, 6, 1, 0, 0],
        "targets": [[3, 0.5], [9, 0], [1, 2]],
        "idx": [0],
    }, {
        "inputs": [8, 4, 9, 3, 5, 7, 9],
        "targets": [[4, 1.2], [1, 1], [0, 0]],
        "idx": [1, 2],
    }]
    assert_dataset(padded_ds, expected, {
        "inputs": tf.int32,
        "targets": tf.float32
    })

  def test_trim_and_pad_dataset_with_multirank_features(self):
    x = [{
        "inputs": [[[7, 8, 5, 6, 1]], [[1, 2, 3, 4, 5]]],
        "targets": [[3, 0.5], [9, 0], [1, 2]],
    }, {
        "inputs": [[[8, 4, 9, 3, 5, 7, 9, 5]]],
        "targets": [[4, 1.2], [1, 1]],
    }]
    ds = tf.data.Dataset.from_generator(
        lambda: x,
        output_signature={
            "inputs": tf.TensorSpec([None, None, None], tf.int32),
            "targets": tf.TensorSpec([None, None], tf.float32),
        })
    padded_ds = utils.trim_and_pad_dataset(
        ds, feature_lengths={
            "inputs": [2, 1, 5],
            "targets": [3, 3]
        })
    expected = [{
        "inputs": [[[7, 8, 5, 6, 1]], [[1, 2, 3, 4, 5]]],
        "targets": [[3, 0.5, 0], [9, 0, 0], [1, 2, 0]],
    }, {
        "inputs": [[[8, 4, 9, 3, 5]], [[0, 0, 0, 0, 0]]],
        "targets": [[4, 1.2, 0], [1, 1, 0], [0, 0, 0]],
    }]
    assert_dataset(padded_ds, expected, {
        "inputs": tf.int32,
        "targets": tf.float32
    })

  _PACK_PARAMETERS = ({"use_custom_ops": False},)

  @parameterized.parameters(*_PACK_PARAMETERS)
  def test_trim_and_pack_dataset(self, use_custom_ops):
    x = [{
        "inputs": [7, 8, 5, 1],
        "targets": [3, 9, 1],
        "idx": [0]
    }, {
        "inputs": [8, 4, 9, 3, 1],
        "targets": [4, 1],
        "idx": [1]
    }]
    ds = create_default_dataset(x, feature_names=("inputs", "targets", "idx"))
    packed_ds = utils.trim_and_pack_dataset(
        ds,
        feature_lengths={
            "inputs": 10,
            "targets": 7
        },
        use_custom_ops=use_custom_ops)

    expected = {
        "inputs": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "inputs_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        "inputs_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
        "targets": [3, 9, 1, 4, 1, 0, 0],
        "targets_positions": [0, 1, 2, 0, 1, 0, 0],
        "targets_segment_ids": [1, 1, 1, 2, 2, 0, 0],
    }
    assert_dataset(packed_ds, expected, {
        "inputs": tf.int32,
        "targets": tf.int32
    })


  @parameterized.parameters(*_PACK_PARAMETERS)
  def test_trim_and_pack_dataset_no_eos(self, use_custom_ops):
    x = [{
        "inputs": [7, 8, 5],
        "targets": [3, 9]
    }, {
        "inputs": [8, 4, 9, 3],
        "targets": [4]
    }]
    ds = create_default_dataset(x)
    packed_ds = utils.trim_and_pack_dataset(
        ds,
        feature_lengths={
            "inputs": 8,
            "targets": 5
        },
        use_custom_ops=use_custom_ops)

    # Packing still works without the eos.
    expected = {
        "inputs": [7, 8, 5, 8, 4, 9, 3, 0],
        "inputs_segment_ids": [1, 1, 1, 2, 2, 2, 2, 0],
        "inputs_positions": [0, 1, 2, 0, 1, 2, 3, 0],
        "targets": [3, 9, 4, 0, 0],
        "targets_positions": [0, 1, 0, 0, 0],
        "targets_segment_ids": [1, 1, 2, 0, 0],
    }
    assert_dataset(packed_ds, expected, {
        "inputs": tf.int32,
        "targets": tf.int32
    })

  @parameterized.parameters(*_PACK_PARAMETERS)
  def test_trim_and_pack_dataset_long_seq(self, use_custom_ops):
    x = [{
        "inputs": [7, 8, 5, 6, 9, 4, 1],
        "targets": [3, 9, 1]
    }, {
        "inputs": [8, 4, 9, 3, 5, 7, 9, 1],
        "targets": [4, 1]
    }]
    ds = create_default_dataset(x)
    packed_ds = utils.trim_and_pack_dataset(
        ds,
        feature_lengths={
            "inputs": 7,
            "targets": 3
        },
        use_custom_ops=use_custom_ops)
    expected = [
        {
            "inputs": [7, 8, 5, 6, 9, 4, 1],
            "inputs_segment_ids": [1, 1, 1, 1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3, 4, 5, 6],
            "targets": [3, 9, 1],
            "targets_positions": [0, 1, 2],
            "targets_segment_ids": [1, 1, 1],
        },
        {
            # EOS is trimmed
            "inputs": [8, 4, 9, 3, 5, 7, 9],
            "inputs_segment_ids": [1, 1, 1, 1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3, 4, 5, 6],
            "targets": [4, 1, 0],
            "targets_positions": [0, 1, 0],
            "targets_segment_ids": [1, 1, 0],
        }
    ]
    assert_dataset(packed_ds, expected, {
        "inputs": tf.int32,
        "targets": tf.int32
    })

  def test_autoregressive_inputs_unpacked(self):
    x = tf.constant([3, 8, 9, 5, 1, 0, 0])
    autoreg_inputs = utils.make_autoregressive_inputs(x)
    actual = self.evaluate(autoreg_inputs)
    expected = [0, 3, 8, 9, 5, 1, 0]
    self.assertAllEqual(actual, expected)
    self.assertEqual(actual.dtype, np.int32)

  def test_autoregressive_inputs_unpacked_2d(self):
    x = tf.constant([[3, 8, 1, 0, 0], [9, 5, 2, 0, 6]])
    autoreg_inputs = utils.make_autoregressive_inputs(x)
    actual = self.evaluate(autoreg_inputs)
    expected = [[0, 0, 0, 0, 0], [3, 8, 1, 0, 0]]
    self.assertAllEqual(actual, expected)
    self.assertEqual(actual.dtype, np.int32)

  def test_autoregressive_inputs_packed(self):
    x = tf.constant([3, 8, 1, 9, 1, 5, 4, 1, 0, 0])
    sequence_id = tf.constant([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    autoreg_inputs = utils.make_autoregressive_inputs(
        x, sequence_id=sequence_id)
    actual = self.evaluate(autoreg_inputs)
    expected = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
    self.assertAllEqual(actual, expected)
    self.assertEqual(actual.dtype, np.int32)

  def test_autoregressive_inputs_packed_2d(self):
    x = tf.constant([[3, 8, 1, 0, 0], [9, 5, 2, 0, 6]])
    sequence_id = tf.constant([1, 2])
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Only 1-D sequences are supported with packing. "
        "Got a packed 2-D sequence."):
      utils.make_autoregressive_inputs(x, sequence_id=sequence_id)

  def test_autoregressive_inputs_packed_non_eos(self):
    # In the correct input format, x[4] should have been 1 (EOS).
    x = tf.constant([3, 8, 1, 9, 6, 5, 4, 1, 0, 0])
    # sequence_id is correctly formatted.
    sequence_id = tf.constant([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    autoreg_inputs = utils.make_autoregressive_inputs(
        x, sequence_id=sequence_id)
    actual = self.evaluate(autoreg_inputs)
    # The incorrect x[4] should not affect the output as long as the sequence_id
    # is correct.
    expected = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
    self.assertAllEqual(actual, expected)
    self.assertEqual(actual.dtype, np.int32)

  def test_autoregressive_inputs_different_dtypes(self):
    x = tf.constant([3, 8, 1, 9.9, 1, 5, 4, 1, 0, 0])
    sequence_id = tf.constant([1, 1, 1, 2, 2, 3, 3, 3, 0, 0], tf.int32)
    autoreg_inputs = utils.make_autoregressive_inputs(
        x, sequence_id=sequence_id, output_dtype=tf.float32)
    actual = self.evaluate(autoreg_inputs)
    # The incorrect x[4] should not affect the output as long as the sequence_id
    # is correct.
    expected = [0, 3, 8, 0, 9.9, 0, 5, 4, 0, 0]
    self.assertAllClose(actual, expected)
    self.assertEqual(actual.dtype, np.float32)


class MixtureRateTest(test_utils.FakeTaskTest):

  def test_mixing_rate_num_examples(self):
    self.assertEqual(3.0, utils.mixing_rate_num_examples(self.cached_task))

    self.assertEqual(81.0,
                     utils.mixing_rate_num_examples(self.cached_task, scale=27))

    self.assertEqual(
        9.0,
        utils.mixing_rate_num_examples(
            self.cached_task, scale=27, temperature=2))

    self.assertEqual(
        2550.25,
        utils.mixing_rate_num_examples(
            self.cached_task, scale=27, temperature=0.5, maximum=50.5))

    # Test fallback.
    self.assertEqual(
        3.0,
        utils.mixing_rate_num_examples(
            self.cached_task, fallback_to_num_input_examples=False))
    with self.assertRaises(AssertionError):
      utils.mixing_rate_num_examples(
          self.uncached_task, fallback_to_num_input_examples=False)
    self.assertEqual(
        30.0,
        utils.mixing_rate_num_examples(
            self.uncached_task, fallback_to_num_input_examples=True))

  def test_mixing_rate_num_characters(self):
    task = mock.Mock(
        cache_dir="", get_cached_stats=lambda split: {"text_chars": 10})
    rate = utils.mixing_rate_num_characters(task, temperature=0.7)
    self.assertAlmostEqual(rate, 26.8269579528)  # 10**(1 / 0.7)


def create_default_dataset(
    x: Sequence[Mapping[str, int]],
    feature_names: Sequence[str] = ("inputs", "targets")
) -> tf.data.Dataset:
  output_types = {feature_name: tf.int32 for feature_name in feature_names}
  output_shapes = {feature_name: [None] for feature_name in feature_names}

  ds = tf.data.Dataset.from_generator(
      lambda: x, output_types=output_types, output_shapes=output_shapes)
  return ds


if __name__ == "__main__":
  absltest.main()
