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

"""Test for combined_sequence.py."""
import math

from absl.testing import absltest

from seqio.deterministic import lazylist


class LazylistTest(absltest.TestCase):

  def test_fingerprint(self):
    self.assertEqual(lazylist.fingerprint("foo"),
                     116648619004475542419127283777354274714)

  def test_pseudorandom_permutation(self):
    self.assertEqual(
        lazylist.pseudorandom_permutation(1000, 123, "foo"), 614)

    for n in range(100):
      permuted = [lazylist.pseudorandom_permutation(n, i, [n, "foo"])
                  for i in range(n)]
      self.assertCountEqual(list(range(n)), permuted)

  def test_counts_in_first_k(self):
    """Tests counts_in_first_k function."""
    proportions = [10, 5, 1]
    expected_counts = [63, 31, 6]
    k = 100
    counts = lazylist.interleave_counts_in_first_k(proportions, k)
    self.assertEqual(counts, expected_counts)

  def test_kth_element(self):
    """Tests interleave_counts_in_first_k function."""
    proportions = [10, 5, 1]
    num_values = 100
    counts = lazylist.interleave_counts_in_first_k(
        proportions, num_values)
    actual_values = [lazylist.interleave_kth_element(proportions, k)
                     for k in range(100)]
    expected_values = []
    for component_num, count in enumerate(counts):
      expected_values.extend([(component_num, i) for i in range(count)])
    self.assertCountEqual(actual_values, expected_values)

  def test_wrap(self):
    """Tests Import."""
    original = ["will", "code", 4, "coffee"]
    s = lazylist.Import(original)
    self.assertEqual(list(s), original)

  def test_reference(self):
    """Tests Reference."""
    two = lazylist.Reference("two", 2)
    self.assertEqual(two.length, 2)
    self.assertEqual(lazylist.all_sources(two), ["two"])
    self.assertEqual(two[0], ("two", 0))
    self.assertEqual(repr(two), "lazylist.Reference(source='two', length=2)")
    three = lazylist.Reference("three", lambda: 3)
    self.assertEqual(three.length, 3)
    self.assertEqual(lazylist.all_sources(three), ["three"])
    self.assertEqual(three[2], ("three", 2))

  def test_concat(self):
    """Tests Concat."""
    two = lazylist.Reference("two", 2)
    three = lazylist.Reference("three", 3)
    s = lazylist.Concat([two, three])
    self.assertEqual(s.length, 5)
    self.assertEqual(lazylist.all_sources(s), ["two", "three"])
    self.assertEqual(s[1], ("two", 1))
    self.assertEqual(s[4], ("three", 2))

  def test_repeat(self):
    """Tests Repeat."""
    s = lazylist.Reference("ten", 10).Repeat()
    self.assertEqual(s.length, math.inf)
    self.assertEqual(lazylist.all_sources(s), ["ten"])
    self.assertEqual(s[12345678901234567], ("ten", 7))

  def test_range(self):
    """Tests Range."""
    s = lazylist.Reference("ten", 10)[3:8]
    self.assertEqual(s.length, 5)
    self.assertEqual(lazylist.all_sources(s), ["ten"])
    self.assertEqual(s[4], ("ten", 7))

  def test_iid(self):
    """Tests IID."""
    s = lazylist.Import(range(16)).IID()
    self.assertEqual(s.length, math.inf)
    self.assertEqual(s[9999999], 10)
    samples = list(s[:1000])
    min_count = min([samples.count(i) for i in range(16)])
    self.assertEqual(min_count, 50)

  def test_shuffle(self):
    """Tests Shuffle."""
    s = lazylist.Import(range(10)).Shuffle(seed="foo")
    self.assertEqual(s.length, 10)
    self.assertEqual(s[1], 4)
    self.assertCountEqual(list(range(10)), list(s))

  def test_repeat_shuffle(self):
    """Tests RepeatShuffle."""
    s = lazylist.Import(range(10)).RepeatShuffle()[:1000]
    self.assertEqual(s.length, 1000)
    values = list(s)
    self.assertEqual(values[:10], list(range(10)))
    self.assertEqual(s[10], 4)
    for i in range(100):
      self.assertCountEqual(values[i * 10:(i + 1) * 10], list(range(10)))

  def test_interleave(self):
    """Tests Interleave."""
    ten = lazylist.Reference("ten", 10)
    five = lazylist.Reference("five", 5)
    s = lazylist.Interleave([ten.Repeat(), five.Repeat()], [10, 5])
    self.assertEqual(s[15 * 10 ** 100 + 8], ("five", 2))

  def test_interleave_once(self):
    """Tests InterleaveOnce."""
    ten = lazylist.Reference("ten", 10)
    five = lazylist.Reference("five", 5)
    s = lazylist.InterleaveOnce([ten, five])
    self.assertEqual(s.length, 15)
    self.assertEqual(lazylist.all_sources(s), ["ten", "five"])
    self.assertEqual(s[0], ("ten", 0))
    self.assertEqual(s[8], ("five", 2))
    self.assertCountEqual(list(s), list(ten) + list(five))


if __name__ == "__main__":
  absltest.main()
