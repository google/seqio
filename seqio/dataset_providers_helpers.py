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

"""Classes for data loading and processing.

Defines Tasks, TaskRegistry, Mixture, and MixtureRegistry
"""

import re
from typing import Mapping, Optional, Union

from absl import logging
from seqio import dataset_providers
from seqio import feature_converters
from seqio import grain_dataset_providers
from seqio import utils
import tensorflow.compat.v2 as tf

_DEFAULT_FEATURE_KEYS = ["inputs", "targets"]

_VALID_TASK_NAME_REGEX = re.compile(r"^[\w\d\.\:_]+$")
_MAX_EXAMPLES_TO_MEM_CACHE = 10000
SHUFFLE_BUFFER_SIZE = 1000

Feature = utils.Feature


def get_mixture_or_task(
    task_or_mixture_name
) -> Union[dataset_providers.Task, dataset_providers.Mixture]:
  """Return the Task or Mixture from the appropriate registry."""
  mixtures = dataset_providers.MixtureRegistry.names()
  tasks = dataset_providers.TaskRegistry.names()
  if task_or_mixture_name in mixtures:
    if task_or_mixture_name in tasks:
      logging.warning("%s is both a Task and a Mixture, returning Mixture",
                      task_or_mixture_name)
    return dataset_providers.MixtureRegistry.get(task_or_mixture_name)
  if task_or_mixture_name in tasks:
    return dataset_providers.TaskRegistry.get(task_or_mixture_name)
  else:
    for available_task in sorted(tasks):
      logging.info("Available task: %s", available_task)
    for available_mixture in sorted(mixtures):
      logging.info("Available mixture: %s", available_mixture)
    raise ValueError(
        "No Task or Mixture found with name '%s'." % task_or_mixture_name)


def get_subtasks(task_or_mixture):
  """Returns all the Tasks in a Mixture as a list or the Task itself."""
  if isinstance(task_or_mixture, dataset_providers.Task):
    return [task_or_mixture]
  else:
    return task_or_mixture.tasks


def get_dataset(mixture_or_task_name: str,
                task_feature_lengths: Mapping[str, int],
                feature_converter: feature_converters.FeatureConverter,
                dataset_split: str = "train",
                use_cached: bool = False,
                shuffle: bool = False,
                num_epochs: Optional[int] = 1,
                shard_info: Optional[dataset_providers.ShardInfo] = None,
                verbose: bool = True,
                seed: Optional[int] = None,
                batch_size: Optional[int] = None,
                trim_output_features: bool = True) -> tf.data.Dataset:
  """Get processed dataset with the model features.

  In order to use options specific to a feature converter, e.g., packing,
  `feature_converter` instance should be instantiated with those options before
  being pased to this function.

  Getting sharded datasets is supported. To use this feature, pass in
  `shard_info`, with shard_index and num_shards information. Sharding is done
  before the feature converter stage. Therefore, if packing is used it will be
  done on the sharded dataset.

  Args:
    mixture_or_task_name: mixture or task name for the Task API.
    task_feature_lengths: dict mapping task feature key to its sequence length.
      This specifies the sequence length of the dataset from the Task API.
    feature_converter: a feature converter object to use to convert the task
      features to model features. Must be a subclass of FeatureConverter.
    dataset_split: the split to use.
    use_cached: whether to use the cached dataset instead of processing it on
      the fly.
    shuffle: whether to shuffle the dataset.
    num_epochs: the number of times to iterate through the dataset, or `None` to
      repeat indefinitely. Note that the repeat occurs in the pipeline after
      offline caching, but before applying potentially stochastic post-cache
      preprocessors and is therefore typically preferred to calling `repeat()`
      on the returned dataset. Defaults to `1`.
    shard_info: number of shards and shard index information.
    verbose: if true, log the feature shapes.
    seed: a random seed to for shuffling tf.data.
    batch_size: Optional batch size.
    trim_output_features: If True, it trims output features to be less than
        the length given by `sequence_length`.

  Returns:
    ds: the processed dataset.
  """
  if not isinstance(feature_converter, feature_converters.FeatureConverter):
    raise TypeError(
        "feature_converter should be an instance of FeatureConverter.")

  mixture_or_task = get_mixture_or_task(mixture_or_task_name)
  is_grain_task = False
  if is_grain_task:
    ds = mixture_or_task.get_dataset(
        sequence_length=task_feature_lengths,
        split=dataset_split,
        use_cached=use_cached,
        shuffle=shuffle,
        seed=seed,
        shard_info=shard_info,
        num_epochs=num_epochs,
        batch_size=batch_size,
        feature_converter=feature_converter,
        trim_output_features=trim_output_features)
  else:
    ds = mixture_or_task.get_dataset(
        task_feature_lengths,
        split=dataset_split,
        use_cached=use_cached,
        shuffle=shuffle,
        seed=seed,
        shard_info=shard_info,
        num_epochs=num_epochs,
        trim_output_features=trim_output_features)
    ds = feature_converter(ds, task_feature_lengths=task_feature_lengths)
    if batch_size is not None:
      ds = ds.batch(batch_size, drop_remainder=True)

  if verbose:
    logging.info(
        "The output dataset from seqio.get_dataset has the following features")
    element_spec = utils.flatten_dict(ds.element_spec, delimiter=".")
    for feature_name, tensor_spec in element_spec.items():
      if isinstance(tensor_spec, tf.TensorSpec):
        logging.info("feature: %s \t shape: %s \t dtype: %s", feature_name,
                     tensor_spec.shape.as_list(), tensor_spec.dtype.name)
      else:
        logging.error("Unknown tensor_spec type %s for feature %s.",
                      type(tensor_spec), feature_name)
  return ds
