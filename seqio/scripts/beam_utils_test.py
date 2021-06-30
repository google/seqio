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

"""Tests for beam_utils."""

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
from apache_beam.testing.test_pipeline import TestPipeline
from seqio.scripts import beam_utils


class GenerateIds(absltest.TestCase):

  def test_merge_indexes(self):
    with TestPipeline() as p:
      mixed_index_items = p | 'items' >> beam.Create([
          ('idx0', (0, 'a')),  # 0
          ('idx1', (0, 'b')),  # 3
          ('idx0', (1, 'c')),  # 1
          ('idx0', (2, 'd')),  # 2
          ('idx2', (0, 'd')),  # 4
          ('idx2', (1, 'e')),   # 5
      ])
      globally_indexed_items = (
          mixed_index_items | beam_utils.MergeIndexes(offset=8))
      util.assert_that(
          globally_indexed_items,
          util.equal_to([
              (8, 'a'),
              (11, 'b'),
              (9, 'c'),
              (10, 'd'),
              (12, 'd'),
              (13, 'e'),
          ])
      )

  def test_merge_indexes_with_sizes(self):
    with TestPipeline() as p:
      mixed_index_items = p | 'items' >> beam.Create([
          ('idx0', (0, 'a')),  # 0
          ('idx1', (0, 'b')),  # 3
          ('idx0', (1, 'c')),  # 1
          ('idx0', (2, 'd')),  # 2
          ('idx2', (0, 'd')),  # 4
          ('idx2', (1, 'e')),   # 5
      ])
      globally_indexed_items = (
          mixed_index_items
          | beam_utils.MergeIndexes(
              offset=8,
              index_sizes=p | beam.Create([{'idx0': 10, 'idx1': 5, 'idx2': 3}]))
      )
      util.assert_that(
          globally_indexed_items,
          util.equal_to([
              (8, 'a'),
              (18, 'b'),
              (9, 'c'),
              (10, 'd'),
              (23, 'd'),
              (24, 'e'),
          ])
      )

  def test_generate_ids(self):
    with TestPipeline() as p:
      pcoll = p | beam.Create(range(999))
      indexed_pcoll = pcoll | beam_utils.GenerateIds(offset=42)
      indices = indexed_pcoll | beam.Map(lambda x: x[0])
      items = indexed_pcoll | beam.Map(lambda x: x[1])

      util.assert_that(
          indices, util.equal_to(range(42, 42 + 999)), label='assert_indices')
      util.assert_that(items, util.equal_to(range(999)), label='assert_items')

  def test_shuffle_indices(self):
    with TestPipeline() as p:
      indexed_items = p | beam.Create(enumerate('abca'))
      shuffled_items = indexed_items | beam_utils.ShuffleIndices(seed=42)
      util.assert_that(
          shuffled_items,
          util.equal_to([
              (1, 'a'),
              (3, 'b'),
              (0, 'c'),
              (2, 'a')
          ])
      )

  def test_shuffle_indices_with_seed(self):
    with TestPipeline() as p:
      indexed_items = p | beam.Create(enumerate('abca'))
      shuffled_items = indexed_items | beam_utils.ShuffleIndices(seed=48)
      util.assert_that(
          shuffled_items,
          util.equal_to([
              (2, 'a'),
              (3, 'b'),
              (1, 'c'),
              (0, 'a')
          ]),
      )


if __name__ == '__main__':
  absltest.main()
