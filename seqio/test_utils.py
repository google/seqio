# Copyright 2024 The SeqIO Authors.
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

"""SeqIO test utilities."""

import collections
import copy
import functools
import os
import shutil
import sys
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union

from absl import flags
from absl import logging
from absl.testing import absltest
import numpy as np
from seqio import dataset_providers
from seqio import evaluation
from seqio import feature_converters
from seqio import preprocessors
from seqio import utils as dataset_utils
from seqio import vocabularies
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from sentencepiece import sentencepiece_model_pb2

TaskRegistry = dataset_providers.TaskRegistry
MixtureRegistry = dataset_providers.MixtureRegistry

mock = absltest.mock

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_data"
)


# _ProxyTest is required because py2 does not allow instantiating
# absltest.TestCase directly.
class _ProxyTest(absltest.TestCase):
  """Instance of TestCase to reuse methods for testing."""

  maxDiff = None

  def runTest(self):
    pass


_pyunit_proxy = _ProxyTest()


_DEFAULT_SEQUENCE_LENGTH = {"inputs": 13, "targets": 13}

_FAKE_DATASET: Optional[Mapping[str, Any]] = None
_FAKE_TOKENIZED_DATASET: Optional[Mapping[str, Any]] = None
_FAKE_PLAINTEXT_TOKENIZED_DATASET: Optional[Mapping[str, Any]] = None
_FAKE_TOKEN_PREPROCESSED_DATASET: Optional[Mapping[str, Any]] = None
_FAKE_TOKEN_PREPROCESSED_NDFEATURES_DATASET: Optional[Mapping[str, Any]] = None
_FAKE_TOKEN_PREPROCESSED_RAGGED_FEATURES_DATASET: Optional[
    Mapping[str, Any]
] = None
_FAKE_DATASETS = None


def _make_fake_datasets():
  """Generate the fake data used in several tests.

  This uses TF operations so it needs to be differed until after the
  enable_eager_execution calls some tests use or those calls will error.
  """
  global _FAKE_DATASET
  global _FAKE_TOKENIZED_DATASET
  global _FAKE_PLAINTEXT_TOKENIZED_DATASET
  global _FAKE_TOKEN_PREPROCESSED_DATASET
  global _FAKE_TOKEN_PREPROCESSED_NDFEATURES_DATASET
  global _FAKE_TOKEN_PREPROCESSED_RAGGED_FEATURES_DATASET
  global _FAKE_DATASETS

  if _FAKE_DATASET:
    return

  empty_ragged_tensor = tf.ragged.constant([[]], ragged_rank=1, dtype=tf.int32)
  empty_ragged_tensor = tf.expand_dims(
      tf.concat([empty_ragged_tensor, empty_ragged_tensor], 0), 2
  )
  empty_ragged_tensor = tf.concat([empty_ragged_tensor, empty_ragged_tensor], 2)
  _FAKE_DATASET = {
      "train": [
          {
              "prefix": "this",
              "suffix": "is a test",
              "2d_feature": ((1, 2, 3),),
              "3d_feature": (((1, 2, 3), (4, 5, 6)),),
              "ragged_feature": tf.RaggedTensor.from_row_splits(
                  tf.constant([[3, 1], [4, 1], [5, 9], [2, 6]]),
                  row_splits=[0, 2, 4],
                  validate=True,
              ),
          },
          {
              "prefix": "that",
              "suffix": "was a test",
              "2d_feature": ((1, 2, 3),),
              "3d_feature": (((1, 2, 3), (4, 5, 6)),),
              "ragged_feature": tf.RaggedTensor.from_row_splits(
                  tf.constant([[3, 1], [4, 1], [5, 9]]),
                  row_splits=[0, 1, 3],
                  validate=True,
              ),
          },
          {
              "prefix": "those",
              "suffix": "were tests",
              "2d_feature": ((1, 2, 3),),
              "3d_feature": (((1, 2, 3), (4, 5, 6)),),
              "ragged_feature": empty_ragged_tensor,
          },
      ],
      "validation": [
          {
              "idx": 0,
              "idxs": (100,),
              "id": "a",
              "ids": ("a1", "a2"),
              "prefix": "this",
              "suffix": "is a validation",
              "2d_feature": ((3, 2, 1),),
              "3d_feature": (((6, 5, 4), (3, 2, 1)),),
              "ragged_feature": tf.RaggedTensor.from_row_splits(
                  tf.constant([[6, 2], [9, 0], [3, 3]]),
                  row_splits=[0, 1, 3],
                  validate=True,
              ),
          },
          {
              "idx": 1,
              "idxs": (200, 201),
              "id": "b",
              "ids": ("b1",),
              "prefix": "that",
              "suffix": "was another validation",
              "2d_feature": ((3, 2, 1),),
              "3d_feature": (((6, 5, 4), (3, 2, 1)),),
              "ragged_feature": tf.RaggedTensor.from_row_splits(
                  tf.constant([[0, 9], [8, 7], [2, 4], [6, 5]]),
                  row_splits=[0, 2, 4],
                  validate=True,
              ),
          },
      ],
  }

  # Text preprocessed and tokenized.
  _FAKE_TOKENIZED_DATASET = {
      "train": [
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 8, 6),
              "inputs_pretokenized": "complete: this",
              "targets": (3, 8, 6, 3, 5, 10),
              "targets_pretokenized": "is a test",
          },
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 18),
              "inputs_pretokenized": "complete: that",
              "targets": (17, 5, 6, 3, 5, 10),
              "targets_pretokenized": "was a test",
          },
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 7, 6, 4),
              "inputs_pretokenized": "complete: those",
              "targets": (17, 4, 23, 4, 10, 6),
              "targets_pretokenized": "were tests",
          },
      ],
      "validation": [
          {
              "idx": 0,
              "idxs": (100,),
              "id": "a",
              "ids": ("a1", "a2"),
              "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 8, 6),
              "inputs_pretokenized": "complete: this",
              "targets": (3, 8, 6, 3, 5, 3, 25, 5, 9, 8, 21, 18, 8, 7, 22),
              "targets_pretokenized": "is a validation",
          },
          {
              "idx": 1,
              "idxs": (200, 201),
              "id": "b",
              "ids": ("b1",),
              "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 18),
              "inputs_pretokenized": "complete: that",
              "targets": (
                  17,
                  5,
                  6,
                  3,
                  5,
                  22,
                  7,
                  24,
                  20,
                  4,
                  23,
                  3,
                  25,
                  5,
                  9,
                  8,
                  21,
                  18,
                  8,
                  7,
                  22,
              ),
              "targets_pretokenized": "was another validation",
          },
      ],
  }

  # Text preprocessed and tokenized.
  # Simulates legacy cached dataset that used '_plaintext' suffix instead of
  # '_pretokenized'.
  _FAKE_PLAINTEXT_TOKENIZED_DATASET = {
      "train": [
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 8, 6),
              "inputs_plaintext": "complete: this",
              "targets": (3, 8, 6, 3, 5, 10),
              "targets_plaintext": "is a test",
          },
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 18),
              "inputs_plaintext": "complete: that",
              "targets": (17, 5, 6, 3, 5, 10),
              "targets_plaintext": "was a test",
          },
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 7, 6, 4),
              "inputs_plaintext": "complete: those",
              "targets": (17, 4, 23, 4, 10, 6),
              "targets_plaintext": "were tests",
          },
      ],
  }

  # Text preprocessed and tokenized.
  _FAKE_TOKEN_PREPROCESSED_DATASET = {
      "train": [
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 8, 6),
              "inputs_pretokenized": "complete: this",
              "targets": (3, 8, 6, 3, 5, 10),
              "targets_pretokenized": "is a test",
          },
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 50),
              "inputs_pretokenized": "complete: that",
              "targets": (17, 5, 6, 3, 5, 10),
              "targets_pretokenized": "was a test",
          },
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 7, 6, 4),
              "inputs_pretokenized": "complete: those",
              "targets": (17, 4, 23, 4, 10, 6),
              "targets_pretokenized": "were tests",
          },
      ],
      "validation": [
          {
              "idx": 0,
              "idxs": (100,),
              "id": "a",
              "ids": ("a1", "a2"),
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 8, 6),
              "inputs_pretokenized": "complete: this",
              "targets": (3, 8, 6, 3, 5, 3, 25, 5, 9, 8, 21, 18, 8, 7, 22),
              "targets_pretokenized": "is a validation",
          },
          {
              "idx": 1,
              "idxs": (200, 201),
              "id": "b",
              "ids": ("b1",),
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 50),
              "inputs_pretokenized": "complete: that",
              "targets": (
                  17,
                  5,
                  6,
                  3,
                  5,
                  22,
                  7,
                  24,
                  20,
                  4,
                  23,
                  3,
                  25,
                  5,
                  9,
                  8,
                  21,
                  18,
                  8,
                  7,
                  22,
              ),
              "targets_pretokenized": "was another validation",
          },
      ],
  }

  _FAKE_TOKEN_PREPROCESSED_NDFEATURES_DATASET = {
      "train": [
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 8, 6),
              "inputs_pretokenized": "complete: this",
              "targets": (3, 8, 6, 3, 5, 10),
              "targets_pretokenized": "is a test",
              "2d_feature": ((1, 2, 3),),
              "3d_feature": (((1, 2, 3), (4, 5, 6)),),
              "2d_feature_pretokenized": ((1, 2, 3),),
              "3d_feature_pretokenized": (((1, 2, 3), (4, 5, 6)),),
          },
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 50),
              "inputs_pretokenized": "complete: that",
              "targets": (17, 5, 6, 3, 5, 10),
              "targets_pretokenized": "was a test",
              "2d_feature": ((1, 2, 3),),
              "3d_feature": (((1, 2, 3), (4, 5, 6)),),
              "2d_feature_pretokenized": ((1, 2, 3),),
              "3d_feature_pretokenized": (((1, 2, 3), (4, 5, 6)),),
          },
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 7, 6, 4),
              "inputs_pretokenized": "complete: those",
              "targets": (17, 4, 23, 4, 10, 6),
              "targets_pretokenized": "were tests",
              "2d_feature": ((1, 2, 3),),
              "3d_feature": (((1, 2, 3), (4, 5, 6)),),
              "2d_feature_pretokenized": ((1, 2, 3),),
              "3d_feature_pretokenized": (((1, 2, 3), (4, 5, 6)),),
          },
      ],
      "validation": [
          {
              "idx": 0,
              "idxs": (100,),
              "id": "a",
              "ids": ("a1", "a2"),
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 8, 6),
              "inputs_pretokenized": "complete: this",
              "targets": (3, 8, 6, 3, 5, 3, 25, 5, 9, 8, 21, 18, 8, 7, 22),
              "targets_pretokenized": "is a validation",
              "2d_feature": ((3, 2, 1),),
              "3d_feature": (((6, 5, 4), (3, 2, 1)),),
              "2d_feature_pretokenized": ((3, 2, 1),),
              "3d_feature_pretokenized": (((6, 5, 4), (3, 2, 1)),),
          },
          {
              "idx": 1,
              "idxs": (200, 201),
              "id": "b",
              "ids": ("b1",),
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 50),
              "inputs_pretokenized": "complete: that",
              "targets": (
                  17,
                  5,
                  6,
                  3,
                  5,
                  22,
                  7,
                  24,
                  20,
                  4,
                  23,
                  3,
                  25,
                  5,
                  9,
                  8,
                  21,
                  18,
                  8,
                  7,
                  22,
              ),
              "targets_pretokenized": "was another validation",
              "2d_feature": ((3, 2, 1),),
              "3d_feature": (((6, 5, 4), (3, 2, 1)),),
              "2d_feature_pretokenized": ((3, 2, 1),),
              "3d_feature_pretokenized": (((6, 5, 4), (3, 2, 1)),),
          },
      ],
  }

  _FAKE_TOKEN_PREPROCESSED_RAGGED_FEATURES_DATASET = {
      "train": [
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 8, 6),
              "inputs_pretokenized": "complete: this",
              "targets": (3, 8, 6, 3, 5, 10),
              "targets_pretokenized": "is a test",
              "ragged_feature": tf.RaggedTensor.from_row_splits(
                  tf.constant([[3, 1], [4, 1], [5, 9], [2, 6]]),
                  row_splits=[0, 2, 4],
                  validate=True,
              ),
              "ragged_feature_pretokenized": tf.RaggedTensor.from_row_splits(
                  tf.constant([[3, 1], [4, 1], [5, 9], [2, 6]]),
                  row_splits=[0, 2, 4],
                  validate=True,
              ),
          },
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 50),
              "inputs_pretokenized": "complete: that",
              "targets": (17, 5, 6, 3, 5, 10),
              "targets_pretokenized": "was a test",
              "ragged_feature": tf.RaggedTensor.from_row_splits(
                  tf.constant([[3, 1], [4, 1], [5, 9]]),
                  row_splits=[0, 1, 3],
                  validate=True,
              ),
              "ragged_feature_pretokenized": tf.RaggedTensor.from_row_splits(
                  tf.constant([[3, 1], [4, 1], [5, 9]]),
                  row_splits=[0, 1, 3],
                  validate=True,
              ),
          },
          {
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 7, 6, 4),
              "inputs_pretokenized": "complete: those",
              "targets": (17, 4, 23, 4, 10, 6),
              "targets_pretokenized": "were tests",
              "ragged_feature": empty_ragged_tensor,
              "ragged_feature_pretokenized": empty_ragged_tensor,
          },
      ],
      "validation": [
          {
              "idx": 0,
              "idxs": (100,),
              "id": "a",
              "ids": ("a1", "a2"),
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 8, 6),
              "inputs_pretokenized": "complete: this",
              "targets": (3, 8, 6, 3, 5, 3, 25, 5, 9, 8, 21, 18, 8, 7, 22),
              "targets_pretokenized": "is a validation",
              "ragged_feature": tf.RaggedTensor.from_row_splits(
                  tf.constant([[6, 2], [9, 0], [3, 3]]),
                  row_splits=[0, 1, 3],
                  validate=True,
              ),
              "ragged_feature_pretokenized": tf.RaggedTensor.from_row_splits(
                  tf.constant([[6, 2], [9, 0], [3, 3]]),
                  row_splits=[0, 1, 3],
                  validate=True,
              ),
          },
          {
              "idx": 1,
              "idxs": (200, 201),
              "id": "b",
              "ids": ("b1",),
              "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 50),
              "inputs_pretokenized": "complete: that",
              "targets": (
                  17,
                  5,
                  6,
                  3,
                  5,
                  22,
                  7,
                  24,
                  20,
                  4,
                  23,
                  3,
                  25,
                  5,
                  9,
                  8,
                  21,
                  18,
                  8,
                  7,
                  22,
              ),
              "targets_pretokenized": "was another validation",
              "ragged_feature": tf.RaggedTensor.from_row_splits(
                  tf.constant([[0, 9], [8, 7], [2, 4], [6, 5]]),
                  row_splits=[0, 2, 4],
                  validate=True,
              ),
              "ragged_feature_pretokenized": tf.RaggedTensor.from_row_splits(
                  tf.constant([[0, 9], [8, 7], [2, 4], [6, 5]]),
                  row_splits=[0, 2, 4],
                  validate=True,
              ),
          },
      ],
  }

  _FAKE_DATASETS = {
      "input": _FAKE_DATASET,
      "tokenized": _FAKE_TOKENIZED_DATASET,
      "token_preprocessed": _FAKE_TOKEN_PREPROCESSED_DATASET,
      "token_preprocessed_ndfeatures": (
          _FAKE_TOKEN_PREPROCESSED_NDFEATURES_DATASET
      ),
      "token_preprocessed_ragged_features": (
          _FAKE_TOKEN_PREPROCESSED_RAGGED_FEATURES_DATASET
      ),
  }


def get_fake_dataset(
    split,
    shuffle_files=False,
    seed=None,
    shard_info=None,
    ndfeatures=False,
    ragged_features=False,
):
  """Returns a tf.data.Dataset with fake data."""
  del shuffle_files  # Unused, to be compatible with TFDS API.
  del seed

  _make_fake_datasets()

  output_signature = {
      "prefix": tf.TensorSpec(shape=(), dtype=tf.string),
      "suffix": tf.TensorSpec(shape=(), dtype=tf.string),
  }
  if split == "validation":
    output_signature.update({
        "idx": tf.TensorSpec(shape=(), dtype=tf.int64),
        "idxs": tf.TensorSpec(shape=[None], dtype=tf.int32),
        "id": tf.TensorSpec(shape=(), dtype=tf.string),
        "ids": tf.TensorSpec(shape=[None], dtype=tf.string),
    })

  if ndfeatures:
    # If we are using ndfeatures fake dataset add the info.
    output_signature.update({
        "2d_feature": tf.TensorSpec(shape=(None, 3), dtype=tf.int32),
        "3d_feature": tf.TensorSpec(shape=(None, 2, 3), dtype=tf.int32),
    })

  if ragged_features:
    output_signature.update({
        "ragged_feature": tf.RaggedTensorSpec(
            shape=(2, None, 2),
            dtype=tf.int32,
            ragged_rank=1,
        ),
    })

  # Keep only defined features.
  examples = list(
      map(lambda ex: {k: ex[k] for k in output_signature}, _FAKE_DATASET[split])
  )

  ds = tf.data.Dataset.from_generator(
      lambda: examples,
      output_signature=output_signature,
  )
  if shard_info:
    ds = ds.shard(num_shards=shard_info.num_shards, index=shard_info.index)
  return ds


def _get_comparable_examples_from_ds(ds):
  """Puts dataset into format that allows examples to be compared in Py2/3."""
  examples = []

  def _to_tuple(v):
    if isinstance(v, list):
      return tuple(_to_tuple(i) for i in v)
    else:
      return v

  def _clean_value(v):
    if isinstance(v, bytes):
      return tf.compat.as_text(v)
    if isinstance(v, np.ndarray):
      if isinstance(v[0], bytes):
        return tuple(tf.compat.as_text(s) for s in v)
      return _to_tuple(v.tolist())
    return v

  for ex in tfds.as_numpy(ds):
    examples.append(tuple((k, _clean_value(v)) for k, v in sorted(ex.items())))
  return examples


def _dump_examples_to_tfrecord(path, examples):
  """Writes list of example dicts to a TFRecord file of tf.Example protos."""
  logging.info("Writing examples to TFRecord: %s", path)
  with tf.io.TFRecordWriter(path) as writer:
    for ex in examples:
      writer.write(dataset_utils.dict_to_tfexample(ex).SerializeToString())


def _dump_examples_to_tsv(path, examples, field_names=("prefix", "suffix")):
  """Writes list of example dicts to a TSV."""
  logging.info("Writing examples to TSV: %s", path)
  with tf.io.gfile.GFile(path, "w") as writer:
    writer.write("\t".join(field_names) + "\n")
    for ex in examples:
      writer.write("\t".join([ex[field] for field in field_names]) + "\n")


def _dump_fake_dataset(path, fake_examples, shard_sizes, dump_fn):
  """Dumps the fake dataset split to sharded TFRecord file."""
  offsets = np.cumsum([0] + shard_sizes)
  for i in range(len(offsets) - 1):
    start, end = offsets[i : i + 2]
    shard_path = "%s-%05d-of-%05d" % (path, i, len(shard_sizes))
    dump_fn(shard_path, fake_examples[start:end])


def _maybe_as_bytes(v):
  if isinstance(v, list):
    return [_maybe_as_bytes(x) for x in v]
  if isinstance(v, str):
    return tf.compat.as_bytes(v)
  return v


def _maybe_as_text(v):
  if isinstance(v, list):
    return [_maybe_as_text(x) for x in v]
  if isinstance(v, bytes):
    return tf.compat.as_text(v)
  return v


def dataset_as_text(ds):
  for ex in tfds.as_numpy(ds):
    yield {k: _maybe_as_text(v) for k, v in ex.items()}


def assert_dataset(
    dataset: tf.data.Dataset,
    expected: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    expected_dtypes: Optional[Mapping[str, tf.DType]] = None,
    rtol=1e-7,
    atol=0,
):
  """Tests whether the entire dataset == expected or [expected].

  Args:
    dataset: a tf.data dataset
    expected: either a single example, or a list of examples. Each example is a
      dictionary.
    expected_dtypes: an optional mapping from feature key to expected dtype.
    rtol: the relative tolerance.
    atol: the absolute tolerance.
  """

  if not isinstance(expected, list):
    expected = [expected]
  actual = list(tfds.as_numpy(dataset))
  _pyunit_proxy.assertEqual(len(actual), len(expected))

  def _compare_dict(actual_dict, expected_dict):
    _pyunit_proxy.assertEqual(
        set(actual_dict.keys()), set(expected_dict.keys())
    )
    for key, actual_value in actual_dict.items():
      if isinstance(actual_value, dict):
        _compare_dict(actual_value, expected_dict[key])
      elif isinstance(actual_value, tf.RaggedTensor) or isinstance(
          actual_value, tf.compat.v1.ragged.RaggedTensorValue
      ):
        actual_value = actual_value.to_list()
        np.testing.assert_array_equal(
            np.array(actual_value, dtype=object),
            np.array(_maybe_as_bytes(expected_dict[key]), dtype=object),
            key,
        )
      elif (
          isinstance(actual_value, np.floating)
          or isinstance(actual_value, np.ndarray)
          and np.issubdtype(actual_value.dtype, np.floating)
      ):
        np.testing.assert_allclose(
            actual_value, expected_dict[key], err_msg=key, rtol=rtol, atol=atol
        )
      else:
        np.testing.assert_array_equal(
            actual_value, _maybe_as_bytes(expected_dict[key]), key
        )

  for actual_ex, expected_ex in zip(actual, expected):
    _compare_dict(actual_ex, expected_ex)

  if expected_dtypes:
    actual_dtypes = {k: dataset.element_spec[k].dtype for k in expected_dtypes}
    _pyunit_proxy.assertDictEqual(expected_dtypes, actual_dtypes)


def assert_datasets_eq(dataset1: tf.data.Dataset, dataset2: tf.data.Dataset):
  """Assert that two tfds datasets are equal."""

  dataset1 = list(tfds.as_numpy(dataset1))
  dataset2 = list(tfds.as_numpy(dataset2))
  _pyunit_proxy.assertEqual(len(dataset1), len(dataset2))

  def _compare_dict(dataset1, dataset2):
    _pyunit_proxy.assertEqual(set(dataset1.keys()), set(dataset2.keys()))
    for key, value1 in dataset1.items():
      if isinstance(value1, dict):
        _compare_dict(value1, dataset2[key])
        continue
      if isinstance(value1, tf.RaggedTensor):
        value1 = value1.to_list()
      np.testing.assert_array_equal(value1, _maybe_as_bytes(dataset2[key]), key)

  for ex1, ex2 in zip(dataset1, dataset2):
    _compare_dict(ex1, ex2)


def assert_datasets_neq(dataset1, dataset2):
  """Assert that two tfds datasets are unequal."""

  _pyunit_proxy.assertRaises(
      AssertionError, assert_datasets_eq, dataset1, dataset2
  )


def _assert_compare_to_fake_dataset(
    ds: tf.data.Dataset,
    split: str,
    features,
    sequence_length: Optional[Mapping[str, int]],
    token_preprocessed: bool = False,
    ndfeatures: bool = False,
    ragged_features: bool = False,
):
  """Calls assertion to compare fake examples to actual dataset."""
  if ndfeatures and ragged_features:
    raise ValueError(
        "At most one of ndfeatures and ragged_features can be True at once."
    )
  dataset = "token_preprocessed" if token_preprocessed else "tokenized"
  dataset = dataset if not ndfeatures else "token_preprocessed_ndfeatures"
  dataset = (
      dataset if not ragged_features else "token_preprocessed_ragged_features"
  )
  _make_fake_datasets()
  fake_examples = copy.deepcopy(_FAKE_DATASETS[dataset][split])

  for key, feat in features.items():
    for n, ex in enumerate(fake_examples):
      if sequence_length and key in sequence_length:
        fake_examples[n][key] = ex[key][
            : sequence_length[key] - int(feat.add_eos)
        ]
      if feat.add_eos:
        fake_examples[n][key] = fake_examples[n][key] + (
            feat.vocabulary.eos_id,
        )

  expected_output_shapes = {
      "inputs": [None],
      "targets": [None],
      "inputs_pretokenized": [],
      "targets_pretokenized": [],
  }
  expected_output_dtypes = {
      "inputs": tf.int32,
      "targets": tf.int32,
      "inputs_pretokenized": tf.string,
      "targets_pretokenized": tf.string,
  }
  if split == "validation":
    expected_output_shapes.update(
        {"id": [], "ids": [None], "idx": [], "idxs": [None]}
    )
    expected_output_dtypes.update(
        {"id": tf.string, "ids": tf.string, "idx": tf.int64, "idxs": tf.int32}
    )
  if ndfeatures:
    # If we are using ndfeatures fake dataset add the info.
    expected_output_dtypes.update({
        "2d_feature": tf.int32,
        "3d_feature": tf.int32,
        "2d_feature_pretokenized": tf.int32,
        "3d_feature_pretokenized": tf.int32,
    })
    expected_output_shapes.update({
        "2d_feature": [None, 3],
        "3d_feature": [None, 2, 3],
        "2d_feature_pretokenized": [None, 3],
        "3d_feature_pretokenized": [None, 2, 3],
    })
  if ragged_features:
    expected_output_dtypes.update({
        "ragged_feature": tf.int32,
        "ragged_feature_pretokenized": tf.int32,
    })
    expected_output_shapes.update({
        "ragged_feature": [None, None, 2],
        "ragged_feature_pretokenized": [None, None, 2],
    })
  # Override with Feature dtypes.
  for k, f in features.items():
    expected_output_dtypes[k] = f.dtype
  _pyunit_proxy.assertDictEqual(
      expected_output_shapes,
      {k: v.shape.as_list() for k, v in ds.element_spec.items()},
  )
  _pyunit_proxy.assertDictEqual(
      expected_output_dtypes, {k: v.dtype for k, v in ds.element_spec.items()}
  )

  actual_examples = _get_comparable_examples_from_ds(ds)
  expected_examples = [tuple(sorted(ex.items())) for ex in fake_examples]

  # Replace RaggedTensors in a nested way.
  def recursive_ragged_tensor_to_list(x: Any):
    if isinstance(x, tf.RaggedTensor):
      x: tf.RaggedTensor
      return x.to_list()
    if isinstance(x, dict) or isinstance(x, list) or isinstance(x, tuple):
      return tf.nest.map_structure(recursive_ragged_tensor_to_list, x)
    return x

  actual_examples = recursive_ragged_tensor_to_list(actual_examples)
  expected_examples = recursive_ragged_tensor_to_list(expected_examples)
  _pyunit_proxy.assertCountEqual(expected_examples, actual_examples)


def create_default_dataset(
    x: Sequence[Mapping[str, Sequence[int]]],
    feature_names: Sequence[str] = ("inputs", "targets"),
    output_types: Optional[Mapping[str, tf.dtypes.DType]] = None,
    output_shapes: Optional[Mapping[str, Tuple[None]]] = None,
) -> tf.data.Dataset:
  """Creates a dataset from the given sequence."""
  if output_types is None:
    output_types = {feature_name: tf.int32 for feature_name in feature_names}
  if output_shapes is None:
    output_shapes = {feature_name: [None] for feature_name in feature_names}

  ds = tf.data.Dataset.from_generator(
      lambda: x, output_types=output_types, output_shapes=output_shapes
  )
  return ds


def test_text_preprocessor(dataset):
  """Performs preprocessing on the text dataset."""

  def my_fn(ex):
    res = dict(ex)
    del res["prefix"]
    del res["suffix"]
    res.update({
        "inputs": tf.strings.join(["complete: ", ex["prefix"]]),
        "targets": ex["suffix"],
    })
    return res

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def split_tsv_preprocessor(dataset, field_names=("prefix", "suffix")):
  """Splits TSV into dictionary."""

  def parse_line(line):
    return dict(
        zip(
            field_names,
            tf.io.decode_csv(
                line,
                record_defaults=[""] * len(field_names),
                field_delim="\t",
                use_quote_delim=False,
            ),
        )
    )

  return dataset.map(
      parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )


def test_token_preprocessor(dataset, output_features, sequence_length):
  """Change all occurrences of non-zero even numbered tokens in inputs to 50."""
  del output_features
  del sequence_length

  def my_fn(ex):
    inputs = ex["inputs"]
    res = ex.copy()
    res["inputs"] = tf.where(
        tf.greater(inputs, 15), tf.constant(50, inputs.dtype), inputs
    )
    return res

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@dataset_utils.map_over_dataset(num_seeds=1)
def random_token_preprocessor(ex, seed, sequence_length):
  """Selects a random shift to roll the tokens by for each feature."""
  for feat in sequence_length:
    tokens = ex[feat]
    res = ex.copy()
    n_tokens = tf.size(tokens)
    random_shift = tf.random.stateless_uniform(
        [], maxval=n_tokens, dtype=tf.int32, seed=seed
    )
    res[feat] = tf.roll(tokens, shift=random_shift, axis=0)
  return res


def token_preprocessor_no_sequence_length(dataset, output_features):
  return test_token_preprocessor(dataset, output_features, sequence_length=None)


class DataInjector:
  """Inject `per_split_data` into `task` while within the scope of this object.

  This context takes `per_split_data`, wraps it in a FunctionDataSource,
  and replaces the data source in `task` with it. After calling this function,
  `task`'s `get_dataset(split)` function will return `per_split_data[split]`.

  Attributes:
    task_name: A SeqIO task name.
    per_split_data: A string-keyed dict of string-keyed dicts. The top-level
      dict should be keyed by dataset splits, and the second-level dict should
      hold the dataset data.
  """

  def __init__(self, task_name, per_split_data):
    self._task = dataset_providers.get_mixture_or_task(task_name)

    self.per_split_data = per_split_data
    self._saved_source = self._task._source  # pytype: disable=attribute-error  # always-use-return-annotations

  def __enter__(self):
    def ds_fn(split, shuffle_files, seed=None):
      del shuffle_files, seed
      data = self.per_split_data[split]
      ds = tf.data.Dataset.from_tensors(data)
      return ds

    mock_source = dataset_providers.FunctionDataSource(
        ds_fn, splits=self.per_split_data.keys()
    )
    self._task._source = mock_source
    self._mock_source = mock_source

  def __exit__(self, exc_type, exc_value, exc_traceback):
    if self._task._source == self._mock_source:
      self._task._source = self._saved_source
    else:
      raise RuntimeError(
          "The task source was changed and not restored within the DataInjector"
          " scope."
      )


def assert_dict_values_equal(a, b):
  """Assert that a and b contain equivalent numpy arrays."""
  tf.nest.map_structure(np.testing.assert_equal, a, b)


def assert_dict_contains(expected, actual):
  """Assert that 'expected' is a subset of the data in 'actual'."""
  for k, v in expected.items():
    np.testing.assert_equal(actual[k], v)


def encode_str(task_name, s, output_feature_name="targets"):
  task = dataset_providers.get_mixture_or_task(task_name)
  return task.output_features[output_feature_name].vocabulary.encode(s)


def create_prediction(task_name, s, output_feature_name="targets"):
  task = dataset_providers.get_mixture_or_task(task_name)
  return [(0, task.output_features[output_feature_name].vocabulary.encode(s))]


def test_task(
    task_name: str,
    raw_data: Mapping[str, Any],
    output_feature_name: str = "targets",
    feature_encoder: feature_converters.FeatureConverter = (
        feature_converters.EncDecFeatureConverter(pack=False)
    ),
    seed: Optional[int] = None,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
  """Test the preprocessing and metrics functionality for a given task.

  This function injects `raw_data` into the task, then creates an Evaluator
  based on that task. It runs the task preprocessing on that raw data and
  extracts the expected value based on `output_feature_name`. Then, it
  creates an `Evaluator` object based on the `task_name` and runs `evaluate`
  using the expected value, returning both the result of the preprocessing
  and the metrics from the `evaluate` call.

  The expected format for `raw_data` is a nested dict of the form
  {'split_name': {'data_key': data}}.

  Note that testing metrics that use score_outputs from this API is currently
  unsupported.

  Args:
    task_name: A SeqIO task name.
    raw_data: A string-keyed dict of string-keyed dicts. The top-level dict
      should be keyed by dataset splits, and the second-level dict should hold
      the dataset data.
    output_feature_name: A string key for the output feature. Used to extract
      the expected target from the preprocessing output.
    feature_encoder: An optional feature encoder object. Defaults to
      EncDecFeatureEncoder.
    seed: optional seeed used for deterministic Task preprocessing.
      Specifically, this seed is passed to the Task to be used in
      map_seed_manager() wrappers around preprocessor functions.

  Returns:
    A tuple (preprocessing_output, metrics), where `preprocessing_output`
    is the result of running the tasks' preprocessing code on `raw_data` and
    `metrics` is a mapping from task name to computed metrics.
  """
  output = test_preprocessing_single(task_name, raw_data, seed=seed)

  eval_output = test_postprocessing(
      task_name,
      raw_data,
      predict_output=output[output_feature_name],
      feature_encoder=feature_encoder,
  )
  return output, eval_output


def test_preprocessing(
    task_name: str,
    raw_data: Mapping[str, Any],
    seed: Optional[int] = None,
    sequence_length: Optional[Mapping[str, int]] = None,
) -> Iterator[Mapping[str, Any]]:
  """Test task preprocessing, returning iterator of the generated dataset.

  This function injects `raw_data` into `task` and runs the preprocessing
  routines from `task`, returning the output of
  `task.get_dataset().as_numpy_iterator()`.

  Args:
    task_name: A SeqIO task name.
    raw_data: A string-keyed dict of string-keyed dicts. The top-level dict
      should be keyed by dataset splits, and the second-level dict should hold
      the dataset data.
    seed: optional seed used for deterministic Task preprocessing. Specifically,
      this seed is passed to the Task to be used in map_seed_manager() wrappers
      around preprocessor functions.
    sequence_length: optional mapping of feature names to their token lengths
      used in the model.

  Returns:
    Iterator with the result of running the tasks' preprocessing code on
    `raw_data`.
  """
  if len(raw_data) > 1:
    raise ValueError("test_preprocessing supports a single split in raw_data.")

  with DataInjector(task_name, raw_data):
    split = list(raw_data.keys())[0]
    task = dataset_providers.get_mixture_or_task(task_name)
    iterator = task.get_dataset(
        sequence_length=sequence_length, split=split, shuffle=False, seed=seed
    ).as_numpy_iterator()
    return iterator


def test_preprocessing_single(
    task_name: str,
    raw_data: Mapping[str, Any],
    seed: Optional[int] = None,
    sequence_length: Optional[Mapping[str, int]] = None,
) -> Mapping[str, Any]:
  """Test task preprocessing, where a single item is expected to be generated.

  This is similar to test_preprocessing, but returns a single generated item.
  This also asserts that no more than a single item is generated during
  preprocessing.

  This function injects `raw_data` into `task` and runs the preprocessing
  routines from `task`, returning the output of
  `next(task.get_dataset().as_numpy_iterator())`.

  Args:
    task_name: A SeqIO task name.
    raw_data: A string-keyed dict of string-keyed dicts. The top-level dict
      should be keyed by dataset splits, and the second-level dict should hold
      the dataset data.
    seed: optional seed used for deterministic Task preprocessing. Specifically,
      this seed is passed to the Task to be used in map_seed_manager() wrappers
      around preprocessor functions.
    sequence_length: optional mapping of feature names to their token lengths
      used in the model.

  Returns:
    The result of running the tasks' preprocessing code on `raw_data`.
  """
  iterator = test_preprocessing(
      task_name, raw_data, seed=seed, sequence_length=sequence_length
  )
  item = next(iterator)
  # Verify that we've reached the end of the generator.
  _pyunit_proxy.assertIsNone(
      next(iterator, None),
      msg="Expected dataset with a single item, but more were generated.",
  )
  return item


def test_postprocessing(
    task_name: str,
    raw_data: Mapping[str, Any],
    target_feature_name: str = "targets",
    predict_output: Optional[Sequence[str]] = None,
    score_output: Optional[Sequence[float]] = None,
    feature_encoder: feature_converters.FeatureConverter = feature_converters.EncDecFeatureConverter(
        pack=False
    ),
    sequence_length: Optional[Mapping[str, int]] = None,
) -> Mapping[str, Any]:
  """Test the postprocessing and metrics for a given task.

  This function injects `raw_data` into `task`, then creates an Evaluator
  based on that task. It then calls `Evaluator.evaluate()` using predict_fn and
  score_fn args that return `predict_output` and `score_output`, returning the
  output of the `evaluate()` call. (Note that, due to the fact that `evaluate`
  uses the task data, this test will also actuate the task preprocessing code.)

  Usually, this function will be invoked `metrics, _, _ = test_postprocessing()`
  since the second and third returned data should be the same as the passed
  predict_output and score_output.

  Args:
    task_name: A SeqIO task name.
    raw_data: A string-keyed dict of string-keyed dicts. The top-level dict
      should be keyed by dataset splits, and the second-level dict should hold
      the dataset data.
    target_feature_name: Feature whose vocabulary will be used to encode
      predict_output. Defaults to 'targets'.
    predict_output: A list of strings representing model predictions for the
      raw_data. Optional, only used when the task specifies metric_fns.
    score_output: A list of floats representing the score of the raw_data.
      Optional, only used when the task specifies score_metric_fns.
    feature_encoder: An optional feature encoder object. Defaults to None.
    sequence_length: An optional length specification.

  Returns:
    metrics: a mapping from metric name to values.
  """

  class PredictCallable(evaluation.PredictFnCallable):

    def __call__(
        self,
        dataset: Optional[tf.data.Dataset] = None,
        model_feature_lengths: Optional[Mapping[str, int]] = None,
    ):
      if predict_output is None:
        return []
      task = dataset_providers.get_mixture_or_task(task_name)
      return list(
          enumerate(
              task.output_features[target_feature_name].vocabulary.encode(s)
              for s in predict_output
          )
      )

  class ScoreCallable(evaluation.PredictFnCallable):

    def __call__(
        self,
        dataset: Optional[tf.data.Dataset] = None,
        model_feature_lengths: Optional[Mapping[str, int]] = None,
    ):
      if score_output is None:
        return []
      return list(enumerate(score_output))

  with DataInjector(task_name, raw_data):
    evaluator = evaluation.Evaluator(
        task_name,
        feature_converter=feature_encoder,
        sequence_length=sequence_length,
    )

    return evaluator.evaluate(
        compute_metrics=True,
        predict_fn=PredictCallable(),
        score_fn=ScoreCallable(),
    )[0].result()[task_name]


class MockVocabulary(vocabularies.Vocabulary):
  """Mocks a vocabulary object for testing."""

  def __init__(self, encode_dict, vocab_size=None):
    self._encode_dict = encode_dict
    self._vocab_size = vocab_size

  def unk_id(self) -> Optional[int]:
    raise NotImplementedError

  def encode(self, s):
    return self._encode_dict[s]

  def encode_tf(self, s):
    res = tf.constant([-1], tf.int32)
    for k, v in self._encode_dict.items():
      if tf.equal(s, k):
        res = tf.constant(v, tf.int32)
      else:
        pass
    return res

  def _encode(self, s: str) -> Sequence[int]:
    raise NotImplementedError

  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def _decode(self, ids):
    raise NotImplementedError

  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def _base_vocab_size(self) -> int:
    raise NotImplementedError

  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def eos_id(self):
    return 1


def sentencepiece_vocab(
    extra_ids=0,
    normalizer_spec_overrides: Optional[
        sentencepiece_model_pb2.NormalizerSpec
    ] = None,
    reverse_extra_ids: bool = True,
    use_fast_tokenizer: bool = False,
):
  return vocabularies.SentencePieceVocabulary(
      os.path.join(TEST_DATA_DIR, "sentencepiece", "sentencepiece.model"),
      extra_ids=extra_ids,
      normalizer_spec_overrides=normalizer_spec_overrides,
      reverse_extra_ids=reverse_extra_ids,
      use_fast_tokenizer=use_fast_tokenizer,
  )


def bertwordpiece_vocab(start_of_sequence_id=101):
  return vocabularies.BertWordPieceVocabulary(
      os.path.join(TEST_DATA_DIR, "bertwordpiece", "vocab.txt"),
      start_of_sequence_id=start_of_sequence_id,
  )


def clear_tasks():
  TaskRegistry._REGISTRY = {}  # pylint:disable=protected-access


def clear_mixtures():
  MixtureRegistry._REGISTRY = {}  # pylint:disable=protected-access


# pylint:disable=invalid-name
FakeLazyTfds = collections.namedtuple(
    "FakeLazyTfds",
    [
        "name",
        "tfds_splits",
        "resolved_tfds_name",
        "data_dir",
        "load",
        "load_shard",
        "info",
        "files",
        "size",
    ],
)
FakeTfdsInfo = collections.namedtuple(
    "FakeTfdsInfo",
    [
        "splits",
        "features",
        "description",
        "version",
        "homepage",
        "file_format",
        "config_name",
    ],
)
# pylint:enable=invalid-name


class FakeTaskTest(absltest.TestCase):
  """TestCase that sets up fake cached and uncached tasks."""

  DEFAULT_PREPROCESSORS = (
      test_text_preprocessor,
      preprocessors.tokenize,
      dataset_providers.CacheDatasetPlaceholder(),
      preprocessors.append_eos_after_trim,
  )

  DEFAULT_OUTPUT_FEATURES = {
      "inputs": dataset_providers.Feature(sentencepiece_vocab()),
      "targets": dataset_providers.Feature(sentencepiece_vocab()),
  }

  def add_task(
      self,
      name,
      source,
      preprocessors=DEFAULT_PREPROCESSORS,  # pylint:disable=redefined-outer-name
      output_features=None,
      **kwargs,
  ):
    output_features = output_features or self.DEFAULT_OUTPUT_FEATURES

    return TaskRegistry.add(
        name,
        source=source,
        preprocessors=preprocessors,
        output_features=output_features,
        **kwargs,
    )

  def get_tempdir(self):
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      # Need to initialize flags when running `pytest`.
      flags.FLAGS(sys.argv)
    return self.create_tempdir().full_path

  def setUp(self):
    super().setUp()
    self.maxDiff = None  # pylint:disable=invalid-name

    # Set up data directory.
    self.test_tmpdir = self.get_tempdir()
    self.test_data_dir = os.path.join(self.test_tmpdir, "test_data")
    shutil.copytree(TEST_DATA_DIR, self.test_data_dir)
    for root, dirs, _ in os.walk(self.test_data_dir):
      for d in dirs + [""]:
        os.chmod(os.path.join(root, d), 0o777)

    self._prepare_sources_and_tasks()

  def _prepare_sources_and_tasks(self):
    """Prepares data sources and tasks."""
    clear_tasks()
    clear_mixtures()
    _make_fake_datasets()
    # Prepare TfdsSource
    # Note we don't use mock.Mock since they fail to pickle.
    fake_tfds_paths = {
        "train": [
            {  # pylint:disable=g-complex-comprehension
                "filename": "train.tfrecord-%05d-of-00002" % i,
                "skip": 0,
                "take": -1,
            }
            for i in range(2)
        ],
        "validation": [{
            "filename": "validation.tfrecord-00000-of-00001",
            "skip": 0,
            "take": -1,
        }],
    }

    def _load_shard(shard_instruction, shuffle_files, seed):
      del shuffle_files
      del seed
      fname = shard_instruction["filename"]
      if "train" in fname:
        ds = get_fake_dataset("train")
        if fname.endswith("00000-of-00002"):
          return ds.take(2)
        else:
          return ds.skip(2)
      else:
        return get_fake_dataset("validation")

    fake_tfds = FakeLazyTfds(
        name="fake:0.0.0",
        tfds_splits=None,
        resolved_tfds_name=lambda: "fake:0.0.0",
        data_dir="/tfds",
        load=get_fake_dataset,
        load_shard=_load_shard,
        info=FakeTfdsInfo(
            splits={"train": None, "validation": None},
            description="This is a fake TFDS dataset.",
            version="0.0.0",
            config_name=None,
            homepage="http://data.org/fake",
            file_format=tfds.core.file_adapters.FileFormat.TFRECORD,
            features={},
        ),
        files=fake_tfds_paths.get,
        size=lambda x: 30 if x == "train" else 10,
    )
    self._tfds_patcher = mock.patch(
        "seqio.utils.LazyTfdsLoader", new=mock.Mock(return_value=fake_tfds)
    )
    self._tfds_patcher.start()

    # Set up data directory.
    self.test_tmpdir = self.get_tempdir()
    self.test_data_dir = os.path.join(self.test_tmpdir, "test_data")
    shutil.copytree(TEST_DATA_DIR, self.test_data_dir)
    for root, dirs, _ in os.walk(self.test_data_dir):
      for d in dirs + [""]:
        os.chmod(os.path.join(root, d), 0o777)

    # Prepare uncached TFDS task.
    self.tfds_source = dataset_providers.TfdsDataSource(
        tfds_name="fake:0.0.0", splits=("train", "validation")
    )
    self.add_task("tfds_task", source=self.tfds_source)

    # Add task with prefix
    self.add_task("t5:tfds_task", source=self.tfds_source)

    # Prepare TextLineSource.
    _dump_fake_dataset(
        os.path.join(self.test_data_dir, "train.tsv"),
        _FAKE_DATASET["train"],
        [2, 1],
        _dump_examples_to_tsv,
    )
    self.text_line_source = dataset_providers.TextLineDataSource(
        split_to_filepattern={
            "train": os.path.join(self.test_data_dir, "train.tsv*"),
        },
        skip_header_lines=1,
    )
    self.add_task(
        "text_line_task",
        source=self.text_line_source,
        preprocessors=(split_tsv_preprocessor,) + self.DEFAULT_PREPROCESSORS,
    )

    # Prepare TFExampleSource.
    _dump_fake_dataset(
        os.path.join(self.test_data_dir, "train.tfrecord"),
        _FAKE_DATASET["train"],
        [2, 1],
        _dump_examples_to_tfrecord,
    )
    self.tf_example_source = dataset_providers.TFExampleDataSource(
        split_to_filepattern={
            "train": os.path.join(self.test_data_dir, "train.tfrecord*"),
        },
        feature_description={
            "prefix": tf.io.FixedLenFeature([], tf.string),
            "suffix": tf.io.FixedLenFeature([], tf.string),
        },
    )
    self.add_task("tf_example_task", source=self.tf_example_source)

    # Prepare ProtoDataSource.
    def decode_tf_example_fn(example):
      feature_description = {
          "prefix": tf.io.FixedLenFeature([], tf.string),
          "suffix": tf.io.FixedLenFeature([], tf.string),
      }
      return tf.io.parse_single_example(example, feature_description)

    self.proto_source = dataset_providers.ProtoDataSource(
        split_to_filepattern={
            "train": os.path.join(self.test_data_dir, "train.tfrecord*"),
        },
        decode_proto_fn=decode_tf_example_fn,
    )
    self.add_task("proto_task", source=self.proto_source)

    # Prepare FunctionDataSource
    self.function_source = dataset_providers.FunctionDataSource(
        dataset_fn=get_fake_dataset, splits=["train", "validation"]
    )
    self.add_task("function_task", source=self.function_source)

    # Prepare Task that is tokenized and preprocessed before caching.
    self.add_task(
        "fully_processed_precache",
        source=self.function_source,
        preprocessors=(
            test_text_preprocessor,
            preprocessors.tokenize,
            token_preprocessor_no_sequence_length,
            dataset_providers.CacheDatasetPlaceholder(),
        ),
    )

    # Prepare Task that is tokenized after caching.
    self.add_task(
        "tokenized_postcache",
        source=self.function_source,
        preprocessors=(
            test_text_preprocessor,
            dataset_providers.CacheDatasetPlaceholder(),
            preprocessors.tokenize,
            token_preprocessor_no_sequence_length,
        ),
    )

    # Prepare Task with randomization.
    self.random_task = self.add_task(
        "random_task",
        source=self.function_source,
        preprocessors=(
            test_text_preprocessor,
            dataset_providers.CacheDatasetPlaceholder(),
            preprocessors.tokenize,
            random_token_preprocessor,
        ),
    )

    self.uncached_task = self.add_task("uncached_task", source=self.tfds_source)

    # Prepare cached task.
    dataset_utils.set_global_cache_dirs([self.test_data_dir])
    self.cached_task_dir = os.path.join(self.test_data_dir, "cached_task")
    _dump_fake_dataset(
        os.path.join(self.cached_task_dir, "train.tfrecord"),
        _FAKE_TOKENIZED_DATASET["train"],
        [2, 1],
        _dump_examples_to_tfrecord,
    )
    _dump_fake_dataset(
        os.path.join(self.cached_task_dir, "validation.tfrecord"),
        _FAKE_TOKENIZED_DATASET["validation"],
        [2],
        _dump_examples_to_tfrecord,
    )
    self.cached_task = self.add_task("cached_task", source=self.tfds_source)

    # Prepare cached plaintext task.
    _dump_fake_dataset(
        os.path.join(
            self.test_data_dir, "cached_plaintext_task", "train.tfrecord"
        ),
        _FAKE_PLAINTEXT_TOKENIZED_DATASET["train"],
        [2, 1],
        _dump_examples_to_tfrecord,
    )
    self.cached_plaintext_task = self.add_task(
        "cached_plaintext_task",
        source=self.tfds_source,
        preprocessors=self.DEFAULT_PREPROCESSORS + (test_token_preprocessor,),
    )

  def tearDown(self):
    super().tearDown()
    self._tfds_patcher.stop()
    tf.random.set_seed(None)

  def verify_task_matches_fake_datasets(  # pylint:disable=dangerous-default-value
      self,
      task_name="",
      use_cached=False,
      token_preprocessed=False,
      ndfeatures=False,
      ragged_features=False,
      splits=("train", "validation"),
      sequence_length=_DEFAULT_SEQUENCE_LENGTH,
      num_shards=None,
      task=None,
  ):
    """Assert all splits for both tokenized datasets are correct."""
    task = TaskRegistry.get(task_name) if not task else task
    for split in splits:
      get_dataset = functools.partial(
          task.get_dataset,
          sequence_length,
          split,
          use_cached=use_cached,
          shuffle=False,
      )
      if num_shards:
        ds = get_dataset(shard_info=dataset_providers.ShardInfo(0, num_shards))
        for i in range(1, num_shards):
          ds = ds.concatenate(
              get_dataset(shard_info=dataset_providers.ShardInfo(i, num_shards))
          )
      else:
        ds = get_dataset()
      _assert_compare_to_fake_dataset(
          ds,
          split,
          task.output_features,
          sequence_length,
          token_preprocessed=token_preprocessed,
          ndfeatures=ndfeatures,
          ragged_features=ragged_features,
      )


class FakeMixtureTest(FakeTaskTest):
  """TestCase that sets up fake cached and uncached tasks."""

  def setUp(self):
    super().setUp()
    clear_mixtures()
    MixtureRegistry.add(
        "uncached_mixture",
        [("uncached_task", 1.0)],
    )
    self.uncached_mixture = MixtureRegistry.get("uncached_mixture")
    MixtureRegistry.add(
        "cached_mixture",
        [("cached_task", 1.0)],
    )
    self.cached_mixture = MixtureRegistry.get("cached_mixture")
    MixtureRegistry.add(
        "uncached_random_mixture",
        [("random_task", 1.0)],
    )
    self.uncached_mixture = MixtureRegistry.get("uncached_random_mixture")
