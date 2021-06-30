# Copyright 2021 The SeqIO Authors.
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

"""Utilities for Beam pipelines."""

from typing import Any, Iterable, Mapping, Optional, Tuple, Union
import uuid

import apache_beam as beam
import numpy as np


class _IndexBundlesDoFn(beam.DoFn):
  """Generates sequentual indices for items in each processing bundle.

  Also emits a final count for each bundle.
  """

  def start_bundle(self):
    self._count = 0
    self._bundle_id = uuid.uuid4()

  def process(self, item: Any):
    yield beam.TaggedOutput('item', (self._bundle_id, (self._count, item)))
    self._count += 1

  def finish_bundle(self):
    yield beam.TaggedOutput(
        'count',
        beam.utils.windowed_value.WindowedValue(
            (self._bundle_id, self._count),
            -1,
            [beam.transforms.window.GlobalWindow()]))


class MergeIndexes(beam.PTransform):
  """Merges multiple indexes into a global index.

  Assumes that the elements in the input PCollection is of the form
  `(index_key, (index, item))`, where the combined elements for each `index_key`
  are assumed to produce a set of `index` values that span the range `[0, N)`
  such that `N` is the number of elements with that key.

  Results in a PCollection with elements of the form `(index, item)`,
  where the combined elements represent all input items and produce a set of
  `index` values that span the range `[0, M)` such that `M` is the size of the
  input PCollection.


  Notes:
    * There is no test of the assumptions for the input, which must be true to
      produce the expected output.
    * The output will be stable (deterministic) for a given input.
  """

  def __init__(self,
               offset: int,
               index_sizes: Optional[Mapping[Any, int]] = None):
    """MergeIndexes constructor.

    Args:
      offset: a global offset to add to the final indices.
      index_sizes: an optional mapping from index keys to size to use. If not
        provided, it will be computed by using the maximum index for each key
        as its size.
    """

    self._offset = offset
    self._index_sizes = index_sizes

  def expand(
      self, indexed_items: beam.PCollection[Tuple[Any, Tuple[int, Any]]]
  ) -> beam.PCollection[Tuple[int, Any]]:
    if self._index_sizes is not None:
      index_sizes = self._index_sizes
    else:
      index_sizes = (
          indexed_items
          | beam.Map(lambda x: (x[0], x[1][0] + 1))
          | beam.combiners.Top.PerKey(1)
          | beam.Map(lambda x: (x[0], x[1][0]))
          | beam.combiners.ToDict())

    def _accumulate_offsets(size_map: Mapping[Any, int]):
      next_offset = 0
      offset_map = {}
      for k in sorted(size_map):
        offset_map[k] = next_offset
        next_offset += size_map[k]
      return offset_map

    index_offsets = index_sizes | beam.Map(_accumulate_offsets)

    def _assign_global_index(
        element: Tuple[Any, Tuple[int, Any]],
        offset_map: Mapping[Any, int],
        global_offset: int = 0
    ) -> Tuple[int, Any]:
      """Computes global index from  local index."""
      index_id, (local_index, item) = element
      return (global_offset + offset_map[index_id] + local_index, item)

    return (
        indexed_items |
        beam.Map(_assign_global_index,
                 offset_map=beam.pvalue.AsSingleton(index_offsets),
                 global_offset=self._offset))


class GenerateIds(beam.PTransform):
  """Generates indices for the given items in a consecutive range.

  No guarantees on ordering are provided, although they will not be random.

  """

  def __init__(self, offset: int = 0):
    self._offset = offset

  def expand(
      self, pcoll: beam.PCollection
  ) -> beam.PCollection[Tuple[int, Any]]:
    bundle_indexed_items, bundle_sizes = (
        pcoll | beam.ParDo(_IndexBundlesDoFn()).with_outputs('item', 'count'))

    return (
        bundle_indexed_items
        | MergeIndexes(
            offset=self._offset,
            index_sizes=bundle_sizes | beam.combiners.ToDict()))


class ShuffleIndices(beam.PTransform):
  """Shuffles the PCollection's indices based on a seed."""

  def __init__(self, seed):
    self._seed = seed

  def expand(
      self,
      pcoll: beam.PCollection[Tuple[int, Any]]
  ) -> beam.PCollection[Tuple[int, Any]]:

    def _shuffle_indices(size: int) -> Iterable[Tuple[int, int]]:
      """Emits pairs of (old index, new index)."""
      for i, j in enumerate(
          np.random.RandomState(seed=self._seed).permutation(size)):
        yield (i, j)

    shuffled_indices = (
        pcoll
        # Extract index
        | beam.Map(lambda x: x[0])
        # Get max index
        | beam.combiners.Top.Of(1)
        # Add one to get size
        | beam.Map(lambda x: x[0] + 1)
        # Shuffle indices
        | beam.ParDo(_shuffle_indices))

    def _update_index(
        element: Tuple[int, Mapping[str, Union[Iterable[int], Iterable[Any]]]]
    ) -> Tuple[int, Any]:
      old_index, joined_values = element
      new_indices = list(joined_values['new_index'])
      assert len(new_indices) == 1, f'{old_index}, {new_indices}'
      items = list(joined_values['item'])
      assert len(items) == 1, f'{old_index}, {items}'
      return new_indices[0], items[0]

    return ({'new_index': shuffled_indices, 'item': pcoll}
            | beam.CoGroupByKey()
            | beam.Map(_update_index))
