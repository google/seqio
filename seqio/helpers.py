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

"""A collection of helper methods."""

# pylint:disable=protected-access

import dataclasses
import functools
import inspect
from typing import Mapping, Optional, Sequence, Union

from seqio import dataset_providers as dp
from seqio import vocabularies as vc
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def mixture_or_task_with_new_vocab(
    mixture_or_task: Union[dp.Task, dp.Mixture, str],
    new_mixture_or_task_name: str,
    *,
    new_vocab: Optional[vc.Vocabulary] = None,
    new_output_features: Optional[Mapping[str, dp.Feature]] = None,
    add_to_seqio_registry: bool = True,
    add_cache_placeholder: bool = False,
    validate_features=True,
) -> Union[dp.Task, dp.Mixture]:
  """Creates a new Task/Mixture from a given Task/Mixture with a new vocabulary.

  Args:
    mixture_or_task: The original Task or Mixture, or the name of a registered
      Task or Mixture.
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
    add_cache_placeholder: If True, adds CacheDatasetPlaceholder in new tasks if
      their old tasks do not have it.
    validate_features: Whether to validate the new feature set is compatible
      with the source task's output features.

  Returns:
    The new `Task` or `Mixture` object.
  """
  if (new_vocab, new_output_features).count(None) != 1:
    raise ValueError(
        "exactly one of `new_vocab` and `new_output_features` "
        "must be specified."
    )

  def _validate_output_features(og_output_features, new_output_features):
    if set(og_output_features) != set(new_output_features):
      raise ValueError(
          f"new_output_features: {new_output_features} doesn't "
          f"match original output_features: {og_output_features}"
      )
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
              f"original output_features: {og_output_features}"
          )

  if isinstance(mixture_or_task, str):
    mixture_or_task = dp.get_mixture_or_task(mixture_or_task)

  if isinstance(mixture_or_task, dp.Task):
    # This is a Task. Create a new Task with the provided vocab/output_features.

    if new_vocab:
      new_output_features = {
          f_name: dataclasses.replace(f, vocabulary=new_vocab)
          for f_name, f in mixture_or_task.output_features.items()
      }

    if validate_features:
      _validate_output_features(
          mixture_or_task.output_features, new_output_features
      )

    preprocessors = mixture_or_task.preprocessors
    if add_cache_placeholder:
      no_cache_placeholder = True
      for prep in preprocessors:
        if isinstance(prep, dp.CacheDatasetPlaceholder):
          no_cache_placeholder = False
          break
      if no_cache_placeholder:
        #  check the first preprocessor requiring "sequence_length" arg
        #  and insert the cache placeholder before it
        preprocessors = list(preprocessors)
        insert_pos = len(preprocessors)
        for pos, prep in enumerate(preprocessors):
          if "sequence_length" in inspect.signature(prep).parameters.keys():
            insert_pos = pos
            break
        preprocessors.insert(insert_pos, dp.CacheDatasetPlaceholder())

    new_task = dp.Task(
        new_mixture_or_task_name,
        source=mixture_or_task.source,
        output_features=new_output_features,
        preprocessors=preprocessors,
        postprocess_fn=mixture_or_task.postprocessor,
        metric_fns=mixture_or_task.metric_fns,
        shuffle_buffer_size=mixture_or_task._shuffle_buffer_size,
    )
    if add_to_seqio_registry:
      dp.TaskRegistry.add_provider(new_mixture_or_task_name, new_task)
    return new_task

  # This is a Mixture. Create and register new sub-Tasks/Mixtures with the
  # provided vocab/output_features, then create a new Mixture.
  new_tasks_and_rates = []
  for task_name, rate in mixture_or_task._task_to_rate.items():
    new_task_name = f"{new_mixture_or_task_name}.{task_name}"
    new_task = mixture_or_task_with_new_vocab(
        task_name,
        new_task_name,
        new_vocab=new_vocab,
        new_output_features=new_output_features,
        add_to_seqio_registry=add_to_seqio_registry,
    )
    new_tasks_and_rates.append((new_task, rate))

  new_mix = dp.Mixture(
      new_mixture_or_task_name,
      new_tasks_and_rates,
      default_rate=None,
      sample_fn=mixture_or_task._sample_fn,
  )
  if add_to_seqio_registry:
    dp.MixtureRegistry.add_provider(new_mixture_or_task_name, new_mix)
  return new_mix


class TruncatedDatasetProvider(dp.DataSource):
  """Wraps a dataset provider, truncating its data using `ds.take(N)`."""

  def __init__(
      self,
      child: dp.DataSource,
      split_sizes: Mapping[str, int],
      shuffle_buffer_size: Optional[int] = None,
  ):
    self.child = child
    self.split_sizes = split_sizes
    self.shuffle_buffer_size = (
        shuffle_buffer_size if shuffle_buffer_size else dp.SHUFFLE_BUFFER_SIZE
    )

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

  @functools.lru_cache()
  def list_shards(self, split: str) -> Sequence[str]:
    """See base class for documentation."""
    return self.child.list_shards(split)

  def get_dataset(
      self,
      split: str = tfds.Split.TRAIN,
      shuffle: bool = True,
      seed: Optional[int] = None,
      shard_info: Optional[dp.ShardInfo] = None,
      *,  # remaining args are out of order from parent
      sequence_length: Optional[Mapping[str, int]] = None,  # Unused
      use_cached: bool = False,  # Unused
      num_epochs: Optional[int] = 1,  # Unused
  ) -> tf.data.Dataset:
    """See base class for documentation."""
    if split not in self.split_sizes:
      return self.child.get_dataset(
          split=split, shuffle=shuffle, seed=seed, shard_info=shard_info
      )

    max_items: int = self.split_sizes[split]
    ds = self.child.get_dataset(
        split=split,
        # Never shuffle since we can't guarantee reproducibility without it
        # (unless we further require deterministic seqio).
        shuffle=False,
        seed=seed,
        shard_info=shard_info,
    )
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
    mixture_or_task: Union[dp.Task, dp.Mixture, str],
    new_mixture_or_task_name: str,
    *,
    split_sizes: Mapping[str, int],
    add_to_seqio_registry: bool = True,
) -> Union[dp.Task, dp.Mixture]:
  """Creates a new Task/Mixture from a given Task/Mixture with less data.

  This can be used for creating smaller subsets of datasets for quick evaluation
  and few-shot fine-tuning datasets.

  Args:
    mixture_or_task: The original Task or Mixture, or the name of a registered
      Task or Mixture.
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
  if isinstance(mixture_or_task, str):
    mixture_or_task = dp.get_mixture_or_task(mixture_or_task)

  if isinstance(mixture_or_task, dp.Task):
    # This is a `Task`.
    new_task = dp.Task(
        new_mixture_or_task_name,
        source=TruncatedDatasetProvider(
            mixture_or_task.source,
            split_sizes=split_sizes,
            shuffle_buffer_size=mixture_or_task._shuffle_buffer_size,
        ),
        output_features=mixture_or_task.output_features,
        preprocessors=mixture_or_task.preprocessors,
        postprocess_fn=mixture_or_task.postprocessor,
        metric_fns=mixture_or_task.metric_fns,
        metric_objs=mixture_or_task.metric_objs,
        shuffle_buffer_size=mixture_or_task._shuffle_buffer_size,
    )
    if add_to_seqio_registry:
      dp.TaskRegistry.add_provider(new_mixture_or_task_name, new_task)
    return new_task
  else:
    # This is a Mixture. Create and register new sub-Tasks/Mixtures with the
    # provided vocab/output_features, then create a new Mixture.
    new_tasks_and_rates = []
    for task_name, rate in mixture_or_task._task_to_rate.items():
      new_task = mixture_or_task_with_truncated_data(
          task_name,
          f"{new_mixture_or_task_name}.{task_name}",
          split_sizes=split_sizes,
          add_to_seqio_registry=add_to_seqio_registry,
      )
      new_tasks_and_rates.append((new_task, rate))

    new_mix = dp.Mixture(
        new_mixture_or_task_name,
        new_tasks_and_rates,
        default_rate=None,
        sample_fn=mixture_or_task._sample_fn,
    )
    if add_to_seqio_registry:
      dp.MixtureRegistry.add_provider(new_mixture_or_task_name, new_mix)
    return new_mix


def mixture_with_missing_task_splits_removed(
    mixture_name: str,
    split: str,
    new_mixture_name: str,
    *,
    add_to_seqio_registry: bool = True,
) -> dp.Mixture:
  """Creates a new mixture removing all subtasks missing the given split.

  In Mixture.get_dataset(...), if a subtask is missing the desired split, it is
  ignored. This means that actual mixing rates could be different from what is
  desired, although it is helpful in defining super-Mixtures that contain
  multiple splits. This helper provides a way to split these super-Mixtures in
  per-split Mixtures by taking a Mixture and a split, and creating a new Mixture
  removing all subtasks missing that split. Mixing rates for other subtasks
  remain unchanged.

  Args:
    mixture_name: The name of the original Mixture.
    split: The split for which to check valid sub-tasks.
    new_mixture_name: The name of the new Mixture.
    add_to_seqio_registry: If True, adds the new Mixture to the SeqIO Registry.

  Returns:
    The new `Mixture` object.
  """
  og_mix: dp.Mixture = dp.get_mixture_or_task(mixture_name)  # pytype: disable=annotation-type-mismatch  # always-use-return-annotations
  new_tasks_and_rates = []
  for task_name, rate in og_mix._task_to_rate.items():
    subtask: dp.Task = dp.get_mixture_or_task(task_name)  # pytype: disable=annotation-type-mismatch  # always-use-return-annotations
    if split in subtask.splits:
      new_tasks_and_rates.append((subtask.name, rate))
  new_mix = dp.Mixture(
      new_mixture_name,
      new_tasks_and_rates,
      default_rate=None,
      sample_fn=og_mix._sample_fn,
  )
  if add_to_seqio_registry:
    dp.MixtureRegistry.add_provider(new_mixture_name, new_mix)
  return new_mix
