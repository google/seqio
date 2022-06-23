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

"""A collection of helper methods."""

# pylint:disable=protected-access

import dataclasses
from typing import Mapping, Optional, Sequence, Union

from seqio import dataset_providers as dp
from seqio import vocabularies as vc
import tensorflow.compat.v2 as tf


def mixture_or_task_with_new_vocab(
    mixture_or_task_name: str,
    new_mixture_or_task_name: str,
    *,
    new_vocab: Optional[vc.Vocabulary] = None,
    new_output_features: Optional[Mapping[str, dp.Feature]] = None,
    add_to_seqio_registry: bool = True) -> Union[dp.Task, dp.Mixture]:
  """Creates a new Task/Mixture from a given Task/Mixture with a new vocabulary.

  Args:
    mixture_or_task_name: The name of the original Task or Mixture.
    new_mixture_or_task_name: The name of the new Task or Mixture. For Mixtures,
      this is also used as a prefix for subtasks, e.g. "subtask_1" is registered
      with the new vocabulary as "new_mixture_or_task_name.subtask_1".
    new_vocab: The new vocabulary to be used. This is used for all features. If
      configuring different vocabularies for different features, pass the
      `new_output_features` arg instead. Note that only one of `new_vocab` or
      `new_output_features` must be provided.
    new_output_features: A dict of feature name to `seqio.Feature` to be used
      for the new Mixture and its subtasks. This dict must (1) have the same
      keys as the original `mix_or_task.output_features` and (2) for each key,
      only the `vocabulary` and `add_eos` fields may differ in the new
      `seqio.Feature`. This can be created from the original
      `mix_or_task.output_features` as follows: ``` new_output_features = {}
      new_output_features["f1"] = dataclasses.replace(
      mix_or_task.output_features["f1"], vocaulary=f1_vocab, add_eos=True)
      new_output_features["f2"] = dataclasses.replace(
      mix_or_task.output_features["f2"], vocaulary=f2_vocab) ```
    add_to_seqio_registry: If True, adds the new Task/Mixture to the SeqIO
      Registry. For Mixtures, sub-Tasks/Mixtures are always registered so that
      the new Mixture can refer to these.

  Returns:
    The new `Task` or `Mixture` object.
  """
  if (new_vocab, new_output_features).count(None) != 1:
    raise ValueError("exactly one of `new_vocab` and `new_output_features` "
                     "must be specified.")

  def _validate_output_features(og_output_features, new_output_features):
    if set(og_output_features) != set(new_output_features):
      raise ValueError(f"new_output_features: {new_output_features} doesn't "
                       f"match original output_features: {og_output_features}")
    # Only `vocabulary`, `add_eos` and `required` fields may differ.
    all_fields = [f.name for f in dataclasses.fields(dp.Feature)]
    ignored_fields = ["vocabulary", "add_eos", "required"]
    fields_to_check = [f for f in all_fields if f not in ignored_fields]
    for feature_name in og_output_features:
      og_feature = og_output_features[feature_name]
      new_feature = new_output_features[feature_name]
      for field in fields_to_check:
        if getattr(og_feature, field) != getattr(new_feature, field):
          raise ValueError(
              f"new_output_features: {new_output_features} incompatible with "
              f"original output_features: {og_output_features}")

  if mixture_or_task_name in dp.TaskRegistry.names():
    # This is a Task. Create a new Task with the provided vocab/output_features.
    og_task: dp.Task = dp.get_mixture_or_task(mixture_or_task_name)

    if new_vocab:
      new_output_features = {
          f_name: dataclasses.replace(f, vocabulary=new_vocab)
          for f_name, f in og_task.output_features.items()
      }
    else:
      _validate_output_features(og_task.output_features, new_output_features)

    new_task = dp.Task(
        new_mixture_or_task_name,
        source=og_task.source,
        output_features=new_output_features,
        preprocessors=og_task.preprocessors,
        postprocess_fn=og_task.postprocessor,
        metric_fns=og_task.metric_fns,
        shuffle_buffer_size=og_task._shuffle_buffer_size)
    if add_to_seqio_registry:
      dp.TaskRegistry.add_provider(new_mixture_or_task_name, new_task)
    return new_task

  # This is a Mixture. Create and register new sub-Tasks/Mixtures with the
  # provided vocab/output_features, then create a new Mixture.
  og_mix: dp.Mixture = dp.get_mixture_or_task(mixture_or_task_name)

  new_tasks_and_rates = []
  for task_name, rate in og_mix._task_to_rate.items():
    new_task_name = f"{new_mixture_or_task_name}.{task_name}"
    _ = mixture_or_task_with_new_vocab(
        task_name,
        new_task_name,
        new_vocab=new_vocab,
        new_output_features=new_output_features,
        add_to_seqio_registry=True)
    new_tasks_and_rates.append((new_task_name, rate))

  new_mix = dp.Mixture(
      new_mixture_or_task_name,
      new_tasks_and_rates,
      default_rate=None,
      sample_fn=og_mix._sample_fn)
  if add_to_seqio_registry:
    dp.MixtureRegistry.add_provider(new_mixture_or_task_name, new_mix)
  return new_mix


class TruncatedDatasetProvider(dp.DataSource):
  """Wraps a dataset provider, truncating its data using `ds.take(N)`."""

  def __init__(self,
               child: dp.DataSource,
               split_sizes: Mapping[str, int],
               shuffle_buffer_size: Optional[int] = None):
    self.child = child
    self.split_sizes = split_sizes
    self.shuffle_buffer_size = (
        shuffle_buffer_size if shuffle_buffer_size else dp.SHUFFLE_BUFFER_SIZE)

  @property
  def caching_permitted(self) -> bool:
    """See base class for documentation."""
    return self.child.caching_permitted

  @property
  def splits(self) -> Sequence[str]:
    """See base class for documentation."""
    return self.child.splits

  @property
  def supports_arbitrary_sharding(self) -> bool:
    """See base class for documentation."""
    return self.child.supports_arbitrary_sharding

  @property
  def output_features(self) -> Mapping[str, dp.Feature]:
    """See base class for documentation."""
    return self.child.output_features

  def list_shards(self, split: str) -> Sequence[str]:
    """See base class for documentation."""
    return self.child.list_shards(split)

  def get_dataset(self,
                  split: str,
                  shuffle: bool = True,
                  seed: Optional[int] = None,
                  shard_info: Optional[dp.ShardInfo] = None) -> tf.data.Dataset:
    """See base class for documentation."""
    if split not in self.split_sizes:
      return self.child.get_dataset(
          split=split, shuffle=shuffle, seed=seed, shard_info=shard_info)

    max_items: int = self.split_sizes[split]
    ds = self.child.get_dataset(
        split=split,
        # Never shuffle since we can't guarantee reproducibility without it
        # (unless we further require deterministic seqio).
        shuffle=False,
        seed=seed,
        shard_info=shard_info)
    ds = ds.take(max_items)
    if shuffle:
      ds = ds.shuffle(self.shuffle_buffer_size)
    return ds

  def num_input_examples(self, split: str) -> Optional[int]:
    """See base class for documentation."""
    if split not in self.split_sizes:
      return self.child.num_input_examples(split)

    child_num_inputs: Optional[int] = self.child.num_input_examples(split)
    if child_num_inputs is None:
      return None
    max_items: int = self.split_sizes[split]
    return min(max_items, child_num_inputs)


def mixture_or_task_with_truncated_data(
    mixture_or_task_name: str,
    new_mixture_or_task_name: str,
    *,
    split_sizes: Mapping[str, int],
    add_to_seqio_registry: bool = True) -> Union[dp.Task, dp.Mixture]:
  """Creates a new Task/Mixture from a given Task/Mixture with less data.

  This can be used for creating smaller subsets of datasets for quick evaluation
  and few-shot fine-tuning datasets.

  Args:
    mixture_or_task_name: The name of the original Task or Mixture.
    new_mixture_or_task_name: The name of the new Task or Mixture. For Mixtures,
      this is also used as a prefix for subtasks, e.g. "subtask_1" is registered
      with the new vocabulary as "new_mixture_or_task_name.subtask_1".
    split_sizes: Dict-like of maximum number of examples to keep in each split.
      For mixtures, this is the maximum number of examples *for each task*. e.g.
      `split_sizes={'train': 1000, 'validation': 500, 'test': 500}`
    add_to_seqio_registry: If True, adds the new Task/Mixture to the SeqIO
      Registry. For Mixtures, sub-Tasks/Mixtures are always registered so that
      the new Mixture can refer to these.

  Returns:
    The new `Task` or `Mixture` object.
  """

  if mixture_or_task_name in dp.TaskRegistry.names():
    # This is a `Task`.
    og_task: dp.Task = dp.get_mixture_or_task(mixture_or_task_name)

    new_task = dp.Task(
        new_mixture_or_task_name,
        source=TruncatedDatasetProvider(
            og_task.source,
            split_sizes=split_sizes,
            shuffle_buffer_size=og_task._shuffle_buffer_size),
        output_features=og_task.output_features,
        preprocessors=og_task.preprocessors,
        postprocess_fn=og_task.postprocessor,
        metric_fns=og_task.metric_fns,
        shuffle_buffer_size=og_task._shuffle_buffer_size)
    if add_to_seqio_registry:
      dp.TaskRegistry.add_provider(new_mixture_or_task_name, new_task)
    return new_task
  else:
    # This is a Mixture. Create and register new sub-Tasks/Mixtures with the
    # provided vocab/output_features, then create a new Mixture.
    og_mix: dp.Mixture = dp.get_mixture_or_task(mixture_or_task_name)

    new_tasks_and_rates = []
    for task_name, rate in og_mix._task_to_rate.items():
      new_task_name = f"{new_mixture_or_task_name}.{task_name}"
      _ = mixture_or_task_with_truncated_data(
          task_name,
          new_task_name,
          split_sizes=split_sizes,
          add_to_seqio_registry=True)
      new_tasks_and_rates.append((new_task_name, rate))

    new_mix = dp.Mixture(
        new_mixture_or_task_name,
        new_tasks_and_rates,
        default_rate=None,
        sample_fn=og_mix._sample_fn)
    if add_to_seqio_registry:
      dp.MixtureRegistry.add_provider(new_mixture_or_task_name, new_mix)
    return new_mix
