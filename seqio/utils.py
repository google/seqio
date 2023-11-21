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

"""Utilities for data loading and processing."""

import collections
import contextlib
import dataclasses
import functools
import inspect
import os
import re
import types
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import numpy as np
from seqio.vocabularies import Vocabulary
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

_INFO_FILENAME = "info.{split}.json"
_STATS_FILENAME = "stats.{split}.json"
_TFRECORD_PREFIX = "{split}.tfrecord"

_TFDS_DATA_DIR_OVERRIDE = None
_TFDS_DATA_READ_CONFIG_OVERRIDE = None
_GLOBAL_CACHE_DIRECTORIES = []
_MapTransform = object
_RandomMapTransform = object


@dataclasses.dataclass(frozen=True)
class Feature:
  """A container for attributes of output features of data providers."""

  vocabulary: Vocabulary
  add_eos: bool = True
  required: bool = True
  dtype: tf.DType = tf.int32
  rank: int = 1



def set_tfds_data_dir_override(tfds_data_dir):
  global _TFDS_DATA_DIR_OVERRIDE
  _TFDS_DATA_DIR_OVERRIDE = tfds_data_dir




def set_tfds_read_config_override(tfds_read_config):
  global _TFDS_DATA_READ_CONFIG_OVERRIDE
  _TFDS_DATA_READ_CONFIG_OVERRIDE = tfds_read_config




def get_global_cache_dirs():
  return _GLOBAL_CACHE_DIRECTORIES


def set_global_cache_dirs(global_cache_dirs):
  global _GLOBAL_CACHE_DIRECTORIES
  _GLOBAL_CACHE_DIRECTORIES = global_cache_dirs


def add_global_cache_dirs(global_cache_dirs):
  global _GLOBAL_CACHE_DIRECTORIES
  _GLOBAL_CACHE_DIRECTORIES += global_cache_dirs


def _validate_tfds_name(name: str) -> None:
  """Validates TFDS dataset name."""
  if (
      name
      and ":" not in name
  ):
    raise ValueError(f"TFDS name must contain a version number, got: {name}")


@dataclasses.dataclass(frozen=True)
class TfdsSplit:
  """Points to a specific TFDS split.

  Attributes:
    dataset: dataset name.
    split: TFDS split (e.g. 'train'), or slice (e.g. 'train[":1%"]').
    data_dir: directory to read/write TFDS data.
  """

  dataset: str
  split: Optional[str]
  data_dir: Optional[str] = None

  def __post_init__(self):
    _validate_tfds_name(self.dataset)


class LazyTfdsLoader(object):
  """Wrapper for TFDS datasets with memoization and additional functionality.

  Lazily loads info from TFDS and provides memoization to avoid expensive hidden
  file operations. Also provides additional utility methods.
  """

  _MEMOIZED_BUILDERS = {}

  def __init__(
      self,
      name: Optional[str] = None,
      data_dir: Optional[str] = None,
      split_map: Union[Mapping[str, str], Mapping[str, TfdsSplit], None] = None,
      decoders=None,
  ):
    """LazyTfdsLoader constructor.

    Args:
      name: str (optional), the name of the TFDS dataset. If `name` is not
        specified then `split_map` values must be instances of `TfdsSplit`.
      data_dir: str (optional), directory to read/write TFDS data.
      split_map: dict (optional), mapping from canonical splits (e.g.,
        'validation') to TFDS splits (e.g. 'train'), or slices (e.g.,
        'train[':1%']), or `TfdsSplit` (e.g. `TfdsSplit(dataset='mnist',
        split='train')`). If `TfdsSplit` are used then `name` must be empty.
      decoders: dict (optional), mapping from features to tfds.decode.Decoders,
        such as tfds.decode.SkipDecoding() for skipping image byte decoding.
    """
    _validate_tfds_name(name)
    self._name = name
    self._data_dir = data_dir
    self._split_map = split_map
    self._decoders = decoders

    self._is_custom_split_map = False
    if split_map:
      random_split_value = next(iter(split_map.values()))
      if isinstance(random_split_value, TfdsSplit):
        self._is_custom_split_map = True
        if self._name or self._data_dir:
          raise ValueError(
              "If split values are instances of `TfdsSplit`, `name` and"
              " `data_dir` must be `None`."
          )

  @property
  def name(self) -> Optional[str]:
    return self._name

  @property
  def tfds_splits(self) -> Optional[Mapping[str, TfdsSplit]]:
    return self._split_map if self._is_custom_split_map else None

  def resolved_tfds_name(self, split: Optional[str] = None) -> Optional[str]:
    """Returns the resolved TFDS dataset name.

    When the specified TFDS name doesn't specify everything, e.g. the version
    has a wildcard or the config is not specified, then this function returns
    the complete TFDS name if the dataset has already been loaded.

    Args:
      split: optional split name.

    Returns:
      complete TFDS name.
    """
    if self.is_memoized(split):
      return (
          self._get_builder(split)
          .get_reference()
          .tfds_name(include_version=True)
      )
    else:
      dataset, _ = self.get_split_params(split)
      return dataset

  def __str__(self):
    return (
        f"{self.__class__.__name__}(name={self.name}, data_dir={self.data_dir})"
    )

  def __repr__(self):
    return (
        f"{self.__class__.__name__}("
        f"name={self.name},"
        f" data_dir={self.data_dir},"
        f" split_map={self._split_map},"
        f" decoders={self._decoders})"
    )

  def get_split_params(
      self, split: Optional[str] = None
  ) -> Tuple[Optional[str], Optional[str]]:
    """Returns a tuple of (dataset, data_dir) for the given canonical split."""
    if self._is_custom_split_map:
      if mapped_split := self._split_map.get(split):
        dataset = mapped_split.dataset
        data_dir = mapped_split.data_dir
      else:
        raise ValueError(
            "`LazyTfdsLoader` refers to multiple datasets, pass `split` value "
            "corresponding to one of them to `get_split_params()`."
        )
    else:
      dataset = self.name
      data_dir = self.data_dir

    return dataset, data_dir


  @property
  def data_dir(self) -> Optional[str]:
    """Returns the data directory for this TFDS dataset."""

    if self._is_custom_split_map:
      logging.warning(
          "`LazyTfdsLoader` refers to multiple datasets, `data_dir` is unknown."
      )
      return None


    if (
        _TFDS_DATA_DIR_OVERRIDE
    ):
      if self._data_dir:
        logging.warning(
            "Overriding TFDS data directory '%s' with '%s' for dataset '%s'.",
            self._data_dir,
            _TFDS_DATA_DIR_OVERRIDE,
            self.name,
        )
      return _TFDS_DATA_DIR_OVERRIDE
    return self._data_dir

  @property
  def read_config(self):
    if _TFDS_DATA_READ_CONFIG_OVERRIDE:
      return _TFDS_DATA_READ_CONFIG_OVERRIDE
    return tfds.ReadConfig()

  def _get_builder_key(
      self, dataset: Optional[str], data_dir: Optional[str]
  ) -> Tuple[Optional[str], Optional[str]]:
    return (dataset, data_dir)

  def is_memoized(self, split: Optional[str] = None) -> bool:
    """Returns true if the dataset is memoized."""
    dataset, data_dir = self.get_split_params(split)

    return (
        self._get_builder_key(dataset, data_dir)
        in LazyTfdsLoader._MEMOIZED_BUILDERS
    )

  @property
  def builder(self):
    return self._get_builder()

  def _get_builder(self, split: Optional[str] = None):
    """Returns the DatasetBuilder for this TFDS dataset."""
    dataset, data_dir = self.get_split_params(split)
    builder_key = self._get_builder_key(dataset, data_dir)
    if builder_key not in LazyTfdsLoader._MEMOIZED_BUILDERS:
      if dataset:
        builder = tfds.builder(dataset, data_dir=data_dir)
      else:
        builder = tfds.builder_from_directory(data_dir)
      LazyTfdsLoader._MEMOIZED_BUILDERS[builder_key] = builder
    return LazyTfdsLoader._MEMOIZED_BUILDERS[builder_key]

  @property
  def info(self):
    return self.builder.info

  def _map_split(self, split: str) -> Optional[str]:
    """Maps the given split to a dataset split."""
    if self._is_custom_split_map:
      self._split_map: Mapping[str, TfdsSplit]
      return self._split_map[split].split
    elif self._split_map:
      self._split_map: Mapping[str, str]
      return self._split_map[split]
    else:
      return split

  def files(self, split: str):
    """Returns set of instructions for reading TFDS files for the dataset."""
    dataset_split = self._map_split(split)
    builder = self._get_builder(split)

    if (
        self.name is not None
        and "/" not in self.name
        and builder.BUILDER_CONFIGS
    ):
      # If builder has multiple configs, and no particular config was
      # requested, raise an error.
      raise ValueError("Dataset '%s' has multiple configs." % self.name)

    split_info = builder.info.splits[dataset_split]
    files = split_info.file_instructions

    if not files:
      logging.fatal("No TFRecord files found for dataset: %s", self.name)
    return files

  def load(
      self,
      split: Optional[str],
      shuffle_files: bool,
      seed: Optional[int] = None,
      shard_info=None,
  ):
    """Returns a tf.data.Dataset for the given split."""
    dataset_split = self._map_split(split)
    dataset, data_dir = self.get_split_params(split)
    read_config = self.read_config
    read_config.input_context = (
        tf.distribute.InputContext(  # pylint: disable=g-long-ternary
            num_input_pipelines=shard_info.num_shards,
            input_pipeline_id=shard_info.index,
        )
        if shard_info
        else None
    )
    read_config.shuffle_seed = seed
    read_config.skip_prefetch = True
    return tfds.load(
        dataset,
        split=dataset_split,
        data_dir=data_dir,
        shuffle_files=shuffle_files,
        download=True,
        try_gcs=True,
        read_config=read_config,
        decoders=self._decoders,
    )

  def load_shard(
      self,
      file_instruction,
      shuffle_files: bool = False,
      seed: Optional[int] = None,
  ):
    """Returns a dataset for a single shard of the TFDS TFRecord files."""
    # pytype:disable=attribute-error
    ds = self.builder._tfrecords_reader.read_files(  # pylint:disable=protected-access
        [file_instruction],
        read_config=tfds.ReadConfig(shuffle_seed=seed),
        shuffle_files=shuffle_files,
    )
    # pytype:enable=attribute-error
    return ds

  def size(self, split: str) -> Optional[int]:
    """Returns the number of examples in the split."""
    dataset_split = self._map_split(split)
    ds_splits = self._get_builder(split).info.splits
    dataset_size = ds_splits[dataset_split].num_examples
    # Very large datasets have num_examples = 0; default instead to np.inf
    dataset_size = dataset_size if dataset_size > 0 else np.inf
    return dataset_size


# ============================== TFExamples ====================================

# Type alias for "dictionary of tensors"
TFDict = Dict[
    str, Union[np.ndarray, tf.Tensor, tf.RaggedTensor, tf.SparseTensor]
]
# NOTE: We use short prefixes to minimize feature key overhead.
# Used to demarcate features used to store shapes of other tensor
_TFEXAMPLE_SHAPE_PREFIX = "_sh:"
# See `tfexample_ragged_prefix`
_TFEXAMPLE_RAGGED_PREFIX = "_rl:"
# See `tfexample_sparse_indices_prefix`
_TFEXAMPLE_SPARSE_PREFIX = "_sp:"


def tfexample_ragged_length_key(key: str, dim: int) -> str:
  """Demarcates feature used to store `dim`-th ragged lengths for `key`.

  This function can be used when parsing data generated by `dict_to_tfexample`,
  by specifying the following ragged feature:
  >>> KEY = "ragged_tensor"
  >>> tf.io.parse_single_example(example, {
  >>>   KEY: tf.io.RaggedFeature(
  >>>     dtype,
  >>>     value_key=KEY,
  >>>     partitions=(
  >>>       tf.io.RaggedFeature.RowLengths(tfexample_ragged_length_key(KEY, 0)),
  >>>       tf.io.RaggedFeature.RowLengths(tfexample_ragged_length_key(KEY, 1)),
  >>>       ... (repeat for however many ragged dimensions the data has)
  >>>   ),)
  >>> })

  Args:
    key: The key storing the values of the ragged tensor.
    dim: The ragged dimension to generate a prefix for.

  Returns:
    The key of the feature storing the `dim`-th ragged lengths.
  """
  return f"{_TFEXAMPLE_RAGGED_PREFIX}{dim}:{key}"


def tfexample_sparse_indices_key(key: str, dim: int) -> str:
  """Demarcates feature used to store `dim`-th sparse feature indices for `key`.

  This function can be used when parsing data generated by `dict_to_tfexample`,
  by specifying the following sparse feature:
  >>> KEY = "sparse_tensor"
  >>> tf.io.parse_single_example(example, {
  >>>   KEY: tf.io.SparseFeature(
  >>>     value_key=KEY,
  >>>     index_key=[
  >>>       tfexample_sparse_indices_key(KEY, 0),
  >>>       tfexample_sparse_indices_key(KEY, 1),
  >>>       ... (repeat for however many sparse dimensions the data has)
  >>>     ],
  >>>     size=[shape0, shape1, ...],
  >>>     dtype=dtype,
  >>>   ),
  >>> })

  Args:
    key: The key storing the values of the sparse tensor.
    dim: The sparse dimension to generate a prefix for.

  Returns:
    The key of the feature storing the `dim`-th sparse indices.
  """
  return f"{_TFEXAMPLE_SPARSE_PREFIX}{dim}:{key}"


def _to_tffeature(tensor: tf.Tensor) -> tf.train.Feature:
  """Creates an appropriately typed `tf.train.Feature` for `tensor`.

  This method only stores the flattened values of `tensor`; make sure you have
  stored the shape before use.

  Args:
    tensor: A tensor to convert.

  Returns:
    A `tf.train.Feature` for `tensor`
  """
  # Flatten the values of the  tensor.
  values = tf.reshape(tensor, [-1]).numpy().tolist()

  if tensor.dtype.is_bool or tensor.dtype.is_integer:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
  elif tensor.dtype.is_floating:
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))
  elif tensor.dtype is tf.string:
    values = [tf.compat.as_bytes(v) for v in values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
  else:
    raise ValueError(f"Unsupported type {tensor.dtype}")


def dict_to_tfexample(
    dct: TFDict, store_shapes: bool = False
) -> tf.train.Example:
  """Convert dictionary of tensors to a `tf.train.Example` proto.

  NOTE: Unfortunately, tensorflow.Example is a very simple proto that can only
  store keyed lists of int64, float and bytes, and doesn't have a marker to
  store metadata needed for multi-dimensional or ragged tensors. This function
  stores using additional features; using `tfexample_to_dict` will allow you to
  recover the original dictionary of tensors modulo the following caveat.

  tensorflow.train.Example only stores int64s and float32s: this function
  casts bools, ints and floats respectively, and this type information is lost.

  Args:
    dct: A dictionary mapping feature keys to 1-D tensors.
    store_shapes: If true, add an additional feature to store the shapes of any
      2+d or sparse tensors. This feature is only required when using
      `tfexample_to_dict`, as `tf.io.parse_single_example` requires the shape be
      provided as an argument.

  Returns:
    A `tf.train.Example` for `dct`
  """
  features = {}
  for key, value in dct.items():
    if isinstance(value, tf.RaggedTensor):
      value: tf.RaggedTensor
      features[key] = _to_tffeature(value.flat_values)
      for dim, length in enumerate(value.nested_row_lengths()):
        features[tfexample_ragged_length_key(key, dim)] = _to_tffeature(length)
    elif isinstance(value, tf.SparseTensor):
      value: tf.SparseTensor
      features[key] = _to_tffeature(value.values)
      for dim, indices in enumerate(tf.transpose(value.indices)):
        features[tfexample_sparse_indices_key(key, dim)] = _to_tffeature(
            indices
        )
      if store_shapes:
        features[_TFEXAMPLE_SHAPE_PREFIX + key] = _to_tffeature(
            tf.constant(value.shape)
        )
    else:
      # Cast to a dense tensor.
      value = tf.constant(value)
      if store_shapes and len(value.shape) > 1:
        features[_TFEXAMPLE_SHAPE_PREFIX + key] = _to_tffeature(
            tf.constant(value.shape)
        )
      features[key] = _to_tffeature(value)

  return tf.train.Example(features=tf.train.Features(feature=features))


def tfexample_to_dict(example: tf.train.Example) -> TFDict:
  """Helper function to create a dictionary of tensors from a TFExample.

  NOTE: this function is less efficient than `tf.io.parse_single_example`, but
  allows parsing of TFExample without knowing its features ahead of time. See
  the documentation of `tfexample_ragged_length_key` for an example of how to
  parse ragged tensors efficiently using `tf.io.parse_single_example`.

  Args:
    example: An instance of tf.train.Example we will convert into a dict.

  Returns:
    A dict with the keys of `example` and tensors as values.
  """
  dct = {}
  shapes = {}
  ragged_lengths = collections.defaultdict(dict)
  sparse_indices = collections.defaultdict(dict)

  for key, feature in example.features.feature.items():
    if feature.int64_list.value:
      value = tf.constant(list(feature.int64_list.value), dtype=tf.int64)
    elif feature.float_list.value:
      value = tf.constant(list(feature.float_list.value), dtype=tf.float32)
    elif feature.bytes_list.value:
      value = tf.constant(list(feature.bytes_list.value), dtype=tf.string)
    else:
      # This is an empty list. We don't know what the type is, so we default
      # to tf.string
      value = tf.constant(list(feature.bytes_list.value), dtype=tf.string)

    if key.startswith(_TFEXAMPLE_RAGGED_PREFIX):
      _, dim, key = key.split(":", 2)
      ragged_lengths[key][int(dim)] = value
    elif key.startswith(_TFEXAMPLE_SPARSE_PREFIX):
      _, dim, key = key.split(":", 2)
      sparse_indices[key][int(dim)] = value
    elif key.startswith(_TFEXAMPLE_SHAPE_PREFIX):
      key = key[len(_TFEXAMPLE_SHAPE_PREFIX) :]
      shapes[key] = value
    else:
      dct[key] = value

  # Assemble RaggedTensors
  for key, length_map in ragged_lengths.items():
    flat_values = dct[key]
    nested_row_lengths = []
    for dim in range(len(length_map)):
      if dim not in length_map:
        raise ValueError(f"Couldn't find {dim}-th ragged length for {key}")
      nested_row_lengths.append(length_map[dim])
    dct[key] = tf.RaggedTensor.from_nested_row_lengths(
        flat_values, nested_row_lengths
    )

  # Assemble SparseTensors
  for key, indices_map in sparse_indices.items():
    if key not in shapes:
      raise ValueError(f"Couldn't find dense shape for sparse tensor {key}")
    dense_shape = shapes.pop(key)

    values = dct[key]
    indices_per_dim = []
    for dim in range(len(indices_map)):
      if dim not in indices_map:
        raise ValueError(f"Couldn't find {dim}-th sparse indices for {key}")
      indices_per_dim.append(indices_map[dim])
    indices = tf.transpose(tf.stack(indices_per_dim))
    dct[key] = tf.SparseTensor(indices, values, dense_shape)

  # Reshape multi-dimensional tensors
  for key, shape in shapes.items():
    dct[key] = tf.reshape(dct[key], shape)
  return dct


# Type alias that supports nested TFDicts.
NestedTFDict = Dict[
    str, Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor, "NestedTFDict"]
]
# Used to linearize keys in nested TFDicts
_TFEXAMPLE_NESTED_DELIMITER = "/"


def unflatten_dict(
    dct: TFDict, delimiter=_TFEXAMPLE_NESTED_DELIMITER
) -> NestedTFDict:
  """Create a nested dictionary from one with nested keys.

  This method converts a "flat" TFDict with nested keys like:
  >>> {
  >>>     "key1/subkey1": ...,
  >>>     "key1/subkey2": ...,
  >>>     "key2/subkey1": ...,
  >>> }
  into a nested dictionary:
  >>> {
  >>>     "key1": {
  >>>       "subkey1": ...,
  >>>       "subkey2": ...,
  >>>     },
  >>>     "key2": {
  >>>       "subkey3": ...
  >>>     },
  >>> }

  Args:
    dct: An dictionary of tensors.
    delimiter: A delimiter used to separate keys from subkeys.

  Returns:
    A nested-version of `dct`.
  """
  nested_dct: NestedTFDict = {}

  for key_path, value in dct.items():
    keys = key_path.split(delimiter)
    # We'll index the value at the last key.
    last_key = keys.pop()

    sub_dct = nested_dct
    for key in keys:
      sub_dct = sub_dct.setdefault(key, {})
    sub_dct[last_key] = value
  return nested_dct


def flatten_dict(
    nested_dct: NestedTFDict, delimiter=_TFEXAMPLE_NESTED_DELIMITER
) -> TFDict:
  """Create a "flattened" dictionary from one with nested keys.

  This method converts a nested TFDict like:
  >>> {
  >>>     "key1": {
  >>>       "subkey1": ...,
  >>>       "subkey2": ...,
  >>>     },
  >>>     "key2": {
  >>>       "subkey1": ...
  >>>     },
  >>> }
  into a flattened dictionary:
  >>> {
  >>>     "key1/subkey1": ...,
  >>>     "key1/subkey2": ...,
  >>>     "key2/subkey1": ...,
  >>> }

  Args:
    nested_dct: A nested dictionary of tensors.
    delimiter: A delimiter used to separate keys from subkeys.

  Returns:
    A flattened version of `nested_dct`.
  """
  unnested_dct = {}

  def _unnest_dct(dct: NestedTFDict, prefix_key: str = ""):
    for key, value in dct.items():
      key_ = prefix_key + key
      if isinstance(value, dict):
        _unnest_dct(value, prefix_key + key + delimiter)
      else:
        unnested_dct[key_] = value

  _unnest_dct(nested_dct)
  return unnested_dct


# ================================ Tasks =======================================


def add_kwargs_to_transform(transform, **kwargs):
  """Returns the partial function or dataclass with the kwargs.

  We use this function to add common arguments (sequence_length,
  output_features) on all transformations that require those.

  Args:
    transform: A dataclass or function.
    **kwargs: Arguments to be passed to the transform.

  Returns:
    If `transform` is a dataclasses attributes matching the kwargs keys will be
    set to the kwargs values.
    Otherwise if `transform` is a function and takes any of the provided kwargs
    these will be passed (by running a partial function).
  """
  is_dataclass = dataclasses.is_dataclass(transform)
  # Filter kwargs by attributes of the dataclass/arguments of the function.
  if is_dataclass:
    avaialabe_arg_names = [f.name for f in dataclasses.fields(transform)]
  else:
    avaialabe_arg_names = set(inspect.signature(transform).parameters.keys())
  kwargs = {k: v for k, v in kwargs.items() if k in avaialabe_arg_names}
  if not kwargs:
    return transform
  # Add attributes/arguments.
  if is_dataclass:
    return dataclasses.replace(transform, **kwargs)
  return functools.partial(transform, **kwargs)


def get_cached_info_path(data_dir, split):
  return os.path.join(data_dir, _INFO_FILENAME.format(split=split))


def get_cached_tfrecord_prefix(data_dir, split):
  return os.path.join(data_dir, _TFRECORD_PREFIX.format(split=split))


def get_cached_stats_path(data_dir, split):
  return os.path.join(data_dir, _STATS_FILENAME.format(split=split))


def get_task_dir_from_name(task_name):
  return os.path.join(*(task_name.split(":")))


def stateless_shuffle(value, seed):
  """Randomly shuffles a tensor, statelessly."""
  flat_value = tf.reshape(value, [-1])
  indices = tf.argsort(
      tf.random.stateless_uniform(tf.shape(flat_value), seed=seed)
  )
  flat_shuffle = tf.gather(flat_value, indices)
  return tf.reshape(flat_shuffle, tf.shape(value))


def trim_and_pad_dataset(
    dataset: tf.data.Dataset, feature_lengths: Mapping[str, int]
) -> tf.data.Dataset:
  """Trim and pad first dimension of features to `feature_lengths`.

  Args:
    dataset: tf.data.Dataset, the dataset to trim/pad examples in.
    feature_lengths: map from feature key to final length. Other features will
      be returned unchanged.

  Returns:
    Trimmed/padded tf.data.Dataset.
  """

  def _trim_and_pad(k: str, t: tf.Tensor) -> tf.Tensor:
    """Trim/pad to the first axis of `t` to be of size `length`."""
    if k not in feature_lengths:
      return t
    if isinstance(t, tf.RaggedTensor):
      t = t.to_tensor()

    length_k = feature_lengths[k]
    if isinstance(length_k, int):
      t = t[:length_k]
      pad_amt = length_k - tf.shape(t)[0]
      padded_t = tf.pad(t, [(0, pad_amt)] + [(0, 0)] * (len(t.shape) - 1))
      padded_t.set_shape([length_k] + t.shape.as_list()[1:])
      return padded_t

    slices = tuple((slice(0, limit) for limit in length_k))
    t = t[slices]
    pad_amt = tf.pad((length_k - tf.shape(t))[..., None], ((0, 0), (1, 0)))
    padded_t = tf.pad(t, pad_amt)
    padded_t.set_shape(length_k)
    return padded_t

  return dataset.map(
      lambda x: {k: _trim_and_pad(k, t) for k, t in x.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )


def trim_dataset(
    dataset: tf.data.Dataset, sequence_length, output_features
) -> tf.data.Dataset:
  """Trim output features to sequence length."""

  def _trim(k: str, v: tf.Tensor) -> tf.Tensor:
    if (
        k not in output_features
        or not sequence_length
        or k not in sequence_length
        or sequence_length[k] is None
    ):
      return v
    # Unify lengths into an iterable so we can create a slice for each
    # dimension, even if the length is a single int.
    lengths = sequence_length[k]
    if isinstance(lengths, int):
      lengths = [lengths]
    slices = tuple((slice(0, limit) for limit in lengths))
    return v[slices]

  return dataset.map(
      lambda ex: {k: _trim(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )


def _strip_packed_feature_key(key: str) -> str:
  strip_suffix = lambda k, s: k[: -len(s)] if k.endswith(s) else k
  return strip_suffix(strip_suffix(key, "_positions"), "_segment_ids")


def trim_and_pack_dataset(
    dataset: tf.data.Dataset,
    feature_lengths: Mapping[str, int],
    use_custom_ops: bool = False,
) -> tf.data.Dataset:
  """Creates a 'packed' version of a dataset on-the-fly.

  Modified from the tensor2tensor library.

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.

  Each example in the output dataset represents several examples in the
  input dataset.

  For each key in the input dataset that also exists in `feature_lengths`, two
  additional keys are created:
    <key>_segment_ids: an int32 tensor identifying the parts
       representing the original example.
    <key>_positions: an int32 tensor identifying the position within the
       original example.

  Features that are not in `feature_lengths` will be removed.

  Example:
    Two input examples get combined to form an output example.
    The input examples are:
    {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0], "idx": 0}
    {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1], "idx": 1}
    The output example is:
    {
                   "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
       "inputs_segment_ids": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
         "inputs_positions": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                  "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
      "targets_segment_ids": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
        "targets_positions": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
    }

    0 represents padding in both the inputs and the outputs.

    Sequences in the incoming examples are truncated to length in
    `feature_lengths`, and the sequences in the output examples all have this
    fixed (padded) length. Features not in `features_length` (i.e, "idx") are
    removed.

  Args:
    dataset: a tf.data.Dataset
    feature_lengths: map from feature key to final length. Other features will
      be discarded.
    use_custom_ops: a boolean - custom ops are faster but require a custom-built
      binary, which is not currently possible on cloud-tpu.

  Returns:
    a tf.data.Dataset
  """
  element_spec = dataset.element_spec
  # Make sure that the dataset contains all keys in `feature_lengths`.
  for k in feature_lengths:
    if k not in element_spec:
      raise ValueError(
          f"Feature '{k}' not found in dataset. Available keys are "
          f"{list(element_spec.keys())}"
      )
    if (
        not element_spec[k].shape.is_compatible_with(tf.TensorShape([None]))
        and not use_custom_ops
    ):
      raise ValueError(
          f"Features to be packed must be one-dimensional. '{k}' is not.' "
          "Consider setting use_custom_ops if you have higher-rank features."
      )

  # Warn if there are any additional keys that will be removed.
  additional_keys = set(element_spec) - set(feature_lengths)
  if additional_keys:
    logging.warning(
        "Features not in `features_length` will be removed during packing: %s",
        additional_keys,
    )

  ds = dataset.map(
      lambda x: {k: x[k][:l, ...] for k, l in feature_lengths.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )

  # Setting batch_size=length ensures that the concatenated sequences (if they
  # have length >=1) are sufficient to fill at least one packed example.
  batch_size = max(feature_lengths.values())
  padded_shapes = {k: [-1] for k in feature_lengths}
  for k in feature_lengths:
    padded_shapes[k].extend(dataset.element_spec[k].shape[1:])
  ds = ds.padded_batch(batch_size, padded_shapes=padded_shapes)

  if use_custom_ops:
    ds = _pack_with_custom_ops(ds, feature_lengths)
  else:
    ds = _pack_with_tf_ops(ds, feature_lengths)

  # Set the Tensor shapes correctly since they get lost in the process.
  def _set_shape(x):
    for k, v in x.items():
      new_shape = [feature_lengths[_strip_packed_feature_key(k)]]
      new_shape.extend(v.get_shape()[1:])
      v.set_shape(new_shape)
    return x

  return ds.map(_set_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def _pack_with_tf_ops(
    dataset: tf.data.Dataset, feature_lengths: Mapping[str, int]
) -> tf.data.Dataset:
  """Helper-function for packing a dataset which has already been batched.

  See trim_and_pack_dataset()

  Uses tf.while_loop. Slow.

  Args:
    dataset: a dataset containing padded batches of examples.
    feature_lengths: mapping from feature key to packed length.

  Returns:
    a dataset.
  """
  empty_example = {}
  for k in feature_lengths:
    for suff in ("", "_positions"):
      empty_example[k + suff] = tf.zeros([0], dtype=tf.int32)
      empty_example[k + suff].set_shape([None])
  keys_etc = empty_example.keys()

  def _write_packed_example(partial, outputs):
    new_partial = empty_example.copy()
    new_outputs = {}
    for k in keys_etc:
      new_outputs[k] = outputs[k].write(
          outputs[k].size(),
          tf.pad(
              partial[k],
              [[
                  0,
                  feature_lengths[_strip_packed_feature_key(k)]
                  - tf.size(partial[k]),
              ]],
          ),
      )
    return new_partial, new_outputs

  def pack_batch(x: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Internal function to map over.

    Consumes a batch of input examples and produces a variable number of output
    examples.

    Args:
      x: a single example

    Returns:
      a tf.data.Dataset
    """
    keys = list(feature_lengths)
    partial = empty_example.copy()
    first_key, *_ = keys
    dynamic_batch_size = tf.shape(x[first_key])[0]
    outputs = {}
    for k in keys:
      outputs[k] = tf.TensorArray(
          tf.int32,
          size=0,
          dynamic_size=True,
          element_shape=[feature_lengths[k]],
      )
      outputs[k + "_positions"] = tf.TensorArray(
          tf.int32,
          size=0,
          dynamic_size=True,
          element_shape=[feature_lengths[k]],
      )

    for i in tf.range(0, dynamic_batch_size):
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[
              (partial, {k: tf.TensorShape([None]) for k in keys_etc}),
              (outputs, {k: tf.TensorShape(None) for k in keys_etc}),
          ]
      )

      can_append = True
      one_example = {}
      for k in keys:
        val = tf.cast(x[k][i], tf.int32)
        val = val[: tf.reduce_sum(tf.cast(tf.not_equal(val, 0), tf.int32))]
        one_example[k] = val
      for k in keys:
        can_append = tf.logical_and(
            can_append,
            tf.less_equal(
                tf.size(partial[k]) + tf.size(one_example[k]),
                feature_lengths[k],
            ),
        )

      if not can_append:
        partial, outputs = _write_packed_example(partial, outputs)

      new_partial = {}
      for k in keys:
        new_seq = one_example[k][: feature_lengths[k]]
        new_seq_len = tf.size(new_seq)
        new_partial[k] = tf.concat([partial[k], new_seq], 0)
        new_partial[k + "_positions"] = tf.concat(
            [partial[k + "_positions"], tf.range(new_seq_len, dtype=tf.int32)],
            0,
        )
      partial = new_partial

    _, outputs = _write_packed_example(partial, outputs)
    packed = {k: outputs[k].stack() for k in keys_etc}
    for k in keys:
      packed[k + "_segment_ids"] = tf.cumsum(
          tf.cast(tf.equal(packed[k + "_positions"], 0), tf.int32), axis=1
      ) * tf.cast(tf.not_equal(packed[k], 0), tf.int32)
    return packed

  dataset = dataset.map(
      pack_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  return dataset.unbatch()


def _pack_with_custom_ops(
    dataset: tf.data.Dataset, feature_lengths: Mapping[str, int]
) -> tf.data.Dataset:
  """Helper-function for packing a dataset which has already been batched.

  See trim_and_pack_dataset()

  Relies on custom ops which require a custom compiled binary.
  Faster than _pack_with_tf_ops(), and denser packing.

  Args:
    dataset: a dataset containing padded batches of examples.
    feature_lengths: mapping from feature key to packed length.

  Returns:
    a dataset.
  """
  # TODO(adarob): Move ops into this library and fix int64 issue.
  from tensor2tensor.data_generators.ops import pack_sequences_ops  # pylint: disable=g-import-not-at-top

  keys = list(feature_lengths)
  use_generic_custom_ops = False
  if len(keys) == 1:
    (k1,) = keys
    k2 = k1
  elif len(keys) == 2:
    k1, k2 = keys
  else:
    use_generic_custom_ops = True
    logging.info(
        "`pack_sequences_2` cannot pack more than 2 features. "
        "Using `pack_sequences_k` instead."
    )

  element_spec = dataset.element_spec
  for k in feature_lengths:
    if not element_spec[k].dtype.is_integer:
      use_generic_custom_ops = True
      logging.info(
          (
              "`pack_sequences_2` cannot pack non-integer feature '%s'. "
              "Using `pack_sequences_k` instead."
          ),
          k,
      )
    if not element_spec[k].shape.is_compatible_with(
        tf.TensorShape([None, None])
    ):
      use_generic_custom_ops = True
      logging.info(
          (
              "`pack_sequences_2` cannot pack higher rank feature '%s'. "
              "Using `pack_sequences_k` instead."
          ),
          k,
      )

  def custom_pack_batch(x):
    """Map-function."""
    if use_generic_custom_ops:
      xs = []
      max_lengths = []
      for k in sorted(feature_lengths.keys()):
        xs.append(x[k])
        max_lengths.append(feature_lengths[k])
      (packed, segment_ids, positions) = pack_sequences_ops.pack_sequences_k(
          inputs=xs, max_lengths=max_lengths
      )
      y = {}
      for i, k in enumerate(sorted(feature_lengths.keys())):
        y[k] = packed[i]
        y[f"{k}_segment_ids"] = segment_ids[i]
        y[f"{k}_positions"] = positions[i]
      return y
    logging.info(
        "Features are compatible with `pack_sequences_2`. "
        "Not using `pack_sequences_k`."
    )
    (
        k1_packed,
        k1_segment_ids,
        k1_positions,
        k2_packed,
        k2_segment_ids,
        k2_positions,
    ) = pack_sequences_ops.pack_sequences2(
        # cast to int64 for compatibility with custom ops
        tf.cast(x[k1], tf.int64),
        tf.cast(x[k2], tf.int64),
        feature_lengths[k1],
        feature_lengths[k2],
    )
    packed = {
        k1: k1_packed,
        k1 + "_segment_ids": k1_segment_ids,
        k1 + "_positions": k1_positions,
    }
    if len(keys) == 2:
      packed.update({
          k2: k2_packed,
          k2 + "_segment_ids": k2_segment_ids,
          k2 + "_positions": k2_positions,
      })

    # cast back to int32
    for k, v in packed.items():
      packed[k] = tf.cast(v, tf.int32)

    return packed

  dataset = dataset.map(
      custom_pack_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  dataset = dataset.unbatch()
  return dataset


def _shift_right_by_one(tensor: tf.Tensor, bos_id: int = 0) -> tf.Tensor:
  """Shift the input tensor to the right by one position without wrapping."""

  if not (tensor.dtype.is_integer or tensor.dtype.is_floating):
    raise ValueError(f"Only numeric types are supported. Got: {tensor.dtype}")
  # tf.roll wraps around the axis.
  rolled = tf.roll(tensor, shift=1, axis=0)

  # Zero out the first position by multiplying with [0, 1, 1, ..., 1].
  depth = tf.shape(tensor)[0]
  mask = tf.one_hot(0, depth=depth, on_value=0, off_value=1, dtype=tensor.dtype)

  # Expand dims of mask to broadcast to rolled.
  dim_expansion = [slice(None, None)] + [None] * (len(rolled.shape) - 1)
  mask = mask[dim_expansion]
  return rolled * mask + (1 - mask) * bos_id


def make_autoregressive_inputs(
    targets: tf.Tensor,
    sequence_id: tf.Tensor = None,
    output_dtype: Optional[tf.dtypes.DType] = None,
    bos_id: int = 0,
) -> tf.Tensor:
  """Generate inputs for an autoregressive model, by shifting the targets.

  Modified from mesh_tensorflow.transformer.transformer.autoregressive_inputs.

  For the first element of each sequence, the returned input id is 0.

  For a "packed" dataset, also pass the sequence_id tensor, which aligns
  with the targets tensor and contains different values for different
  concatenated examples.

  Example for a packed dataset:

  ```
        targets = [3, 8, 1, 9, 1, 5, 4, 1, 0, 0]
    sequence_id = [1, 1, 1, 2, 2, 3, 3, 3, 0, 0]
         inputs = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
                            |     |        |
                            These positions are set to 0 if sequence_id is not
                            None.
  ```

  Args:
    targets: a tf.int32 tensor with shape [length].
    sequence_id: an optional tensor with the same shape as targets.
    output_dtype: an optional output data type.
    bos_id: bos id.

  Returns:
    a tensor with dtype tf.int32 and the same shape as targets.
  """
  output_dtype = output_dtype or targets.dtype
  if sequence_id is not None and not sequence_id.dtype.is_integer:
    raise ValueError(
        "The sequence_id should be integer-valued tensors for a packed dataset."
    )
  if sequence_id is not None and len(targets.shape) > 1:
    raise ValueError(
        "Only 1-D sequences are supported with packing. Got a "
        f"packed {len(targets.shape)}-D sequence."
    )

  inputs = _shift_right_by_one(targets, bos_id)
  if inputs.dtype != output_dtype:
    inputs = tf.cast(inputs, output_dtype)

  # We should have a 0 at the beginning of each sequence rather than the
  # shifted EOS (e.g. 1) from the previous sequence.
  if sequence_id is not None:
    not_first_in_sequence = tf.equal(
        sequence_id, _shift_right_by_one(sequence_id)
    )
    not_first_in_sequence = tf.cast(not_first_in_sequence, output_dtype)
    first_ids = tf.cast((1 - not_first_in_sequence) * bos_id, output_dtype)
    inputs = inputs * not_first_in_sequence + first_ids
  return inputs


# ========================= Mixing Rate Functions ==============================


def mixing_rate_num_examples(
    task,
    maximum: Optional[int] = None,
    scale: float = 1.0,
    temperature: float = 1.0,
    fallback_to_num_input_examples: bool = True,
    split: str = "train",
) -> float:
  """Mixing rate based on the number of examples for the task's split.

  It should be noted that SeqIO only injects the task, and all other parameters
  must be provided by the user when initializing the Mixture.

  Args:
    task: the seqio.Task to compute a rate for.
    maximum: an optional maximum value to clip at after constant scaling but
      before temperature scaling.
    scale: a multiplicative scaling factor applied before temperature.
    temperature: a temperature (T) to scale rate (r) by as r^(1/T).
    fallback_to_num_input_examples: whether to fallback to using the number of
      input examples when the Task is not cached. Otherwise, an error will be
      raised.
    split: the split to look at for cached stats.

  Returns:
    The mixing rate for this task.
  """

  if task.cache_dir or not fallback_to_num_input_examples:
    ret = task.get_cached_stats(split)["examples"]
  else:
    logging.warning(
        (
            "Task '%s' not cached so using number of input examples instead of "
            "preprocessed examples to compute rate."
        ),
        task.name,
    )
    ret = task.num_input_examples(split)

  ret *= scale
  if maximum:
    ret = min(ret, maximum)
  if temperature != 1.0:
    ret = ret ** (1.0 / temperature)
  return ret


def mixing_rate_num_characters(
    task, temperature: float = 1.0, char_count_name: str = "text_chars"
) -> float:
  """Mixing rate based on the number of characters for the task's 'train' split.

  Args:
    task: the seqio.Task to compute a rate for.
    temperature: a temperature (T) to scale rate (r) by as r^(1/T).
    char_count_name: feature name of the character counts in the cached stats
      file.

  Returns:
    The mixing rate for this task.
  """
  if task.cache_dir is None:
    raise ValueError(
        "`mixing_rate_num_characters` requires that each task has is cached "
        "with the character count stats."
    )
  ret = task.get_cached_stats("train")[char_count_name]
  if temperature != 1.0:
    ret = ret ** (1.0 / temperature)
  return ret


# ======================== Decorators =========================================

_NEXT_MAP_SEED = None




@contextlib.contextmanager
def map_seed_manager(initial_seed=None):
  """Contextmanager to control the initial seed used by `map_over_dataset`."""
  global _NEXT_MAP_SEED
  old_map_seed = _NEXT_MAP_SEED
  _NEXT_MAP_SEED = initial_seed
  yield
  _NEXT_MAP_SEED = old_map_seed


def set_preprocessor_seed(preprocessor_fn, seed=None):
  """Sets the internal map seed for the provided preprocessor."""
  return add_kwargs_to_transform(preprocessor_fn, seqio_internal_map_seed=seed)


_SPECIAL_KWARGS = ("sequence_length", "output_features")


@dataclasses.dataclass
class _GrainRandomMapFn(_RandomMapTransform):
  """Grain Transform to represent existing SeqIO random map preprocessors."""

  map_fn: Callable[..., Any]
  num_seeds: int
  num_parallel_calls: int = tf.data.AUTOTUNE

  # These are set by SeqIO before applying the preprocessor or before passing
  # it to Grain.
  sequence_length: Optional[Mapping[str, int]] = None
  output_features: Optional[Mapping[str, Any]] = None

  def _map_fn_with_special_kwargs(self, *args, **kwargs):
    """Handle _SPECIAL_KWARGS before calling self.map_fn()."""
    special_kwargs = {
        k: getattr(self, k)
        for k in _SPECIAL_KWARGS
        if getattr(self, k) is not None
    }
    map_fn = add_kwargs_to_transform(self.map_fn, **special_kwargs)
    return map_fn(*args, **kwargs)

  # Path for Grain. Uses seed provided by Grain.
  def random_map(self, element, rng: tf.Tensor):
    if self.num_seeds == 1:
      return self._map_fn_with_special_kwargs(element, seed=rng)
    rngs = tf.random.experimental.stateless_split(rng, self.num_seeds)
    rngs = tf.unstack(rngs)
    return self._map_fn_with_special_kwargs(element, seeds=rngs)

  # Path for SeqIO; preserves legacy logic to manage seeds and differs in
  # seed-management behavior in Grain.
  def __call__(self, dataset: tf.data.Dataset, *args, **kwargs):
    global _NEXT_MAP_SEED
    if _NEXT_MAP_SEED is None:
      random_ds_seeds = ((None, None),) * self.num_seeds
    else:
      random_ds_seeds = np.arange(
          _NEXT_MAP_SEED, _NEXT_MAP_SEED + 2 * self.num_seeds
      ).reshape(-1, 2)
      random_ds_seeds = tuple(tuple(s) for s in random_ds_seeds)
      _NEXT_MAP_SEED += 2 * self.num_seeds
    seed_datasets = tf.nest.map_structure(
        tf.data.experimental.RandomDataset, random_ds_seeds
    )

    def map_fn(element, seeds):
      if self.num_seeds == 1:
        return self._map_fn_with_special_kwargs(
            element, seed=seeds[0], *args, **kwargs
        )
      return self._map_fn_with_special_kwargs(
          element, seeds=seeds, *args, **kwargs
      )

    return tf.data.Dataset.zip((dataset, seed_datasets)).map(
        map_fn, num_parallel_calls=self.num_parallel_calls
    )


@dataclasses.dataclass
class _GrainMapFn(_MapTransform):
  """Grain Transform to represent existing SeqIO map preprocessors."""

  map_fn: Callable[..., Any]
  num_parallel_calls: int

  # These are set by SeqIO before applying the preprocessor or before passing
  # it to Grain.
  sequence_length: Optional[Mapping[str, int]] = None
  output_features: Optional[Mapping[str, Any]] = None

  def _map_fn_with_special_kwargs(self, *args, **kwargs):
    """Handle _SPECIAL_KWARGS before calling self.map_fn()."""
    special_kwargs = {
        k: getattr(self, k)
        for k in _SPECIAL_KWARGS
        if getattr(self, k) is not None
    }
    map_fn = add_kwargs_to_transform(self.map_fn, **special_kwargs)
    return map_fn(*args, **kwargs)

  # Path for Grain
  def map(self, element):
    return self._map_fn_with_special_kwargs(element)

  # Path for SeqIO
  def __call__(
      self, dataset: tf.data.Dataset, *args, **kwargs
  ) -> tf.data.Dataset:
    return dataset.map(
        lambda x: self._map_fn_with_special_kwargs(x, *args, **kwargs),
        num_parallel_calls=self.num_parallel_calls,
    )


def map_over_dataset(
    fn=None, *, num_seeds=None, num_parallel_calls=tf.data.experimental.AUTOTUNE
):
  """Decorator to map decorated function over dataset.

  Many preprocessors map a function over a dataset. This decorator helps reduce
  boilerplate for this common pattern.

  If `num_seeds` is set to 1, a unique random seed (pair of int32) will be
  passed to the mapping function with keyword 'seed'.
  If `num_seeds` is greater than 1, unique random seeds (pairs of int32) will be
  passed to the mapping function with keyword 'seeds'.
  These seeds can be generated deterministically by using the `map_seed_manager`
  to set the seed for the process that generates the individual seeds for each
  mapping function. These seeds will be set sequentially from the initial seed
  for each call to `map_over_dataset` where `num_seeds > 0`.

  Args:
    fn: map function
    num_seeds: optional number of random seeds (pairs of int32) to pass to the
      mapping function.
    num_parallel_calls: num_parallel_calls value to pass to Dataset.map

  Returns:
    Callable transform which takes dataset as first argument.
  """

  def map_without_seeds(fn):
    @functools.wraps(fn)
    def wrapped_fn(ds, *args, **kwargs):
      return _GrainMapFn(fn, num_parallel_calls)(ds, *args, **kwargs)

    return wrapped_fn

  def map_with_seeds(fn):
    @functools.wraps(fn)
    def wrapped_fn(ds, *args, **kwargs):
      return _GrainRandomMapFn(fn, num_seeds, num_parallel_calls)(
          ds, *args, **kwargs
      )

    return wrapped_fn

  wrapper = map_without_seeds if num_seeds is None else map_with_seeds
  return wrapper if fn is None else wrapper(fn)


def fully_qualified_class_name(instance: Any) -> str:
  """Returns the fully qualified class name of the given instance."""
  return f"{type(instance).__module__}.{type(instance).__name__}"


def function_name(function) -> str:
  """Returns the name of a (possibly partially applied) function."""
  if inspect.isclass(function):
    # function can be a protocol.
    return function.__class__.__name__
  elif isinstance(function, functools.partial):
    # functools.partial can be applied multiple times.
    return function_name(function.func)
  else:
    return function.__name__


