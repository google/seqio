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

"""Lazylist library - virtual sequence manipulation for ML training datasets.

A Lazylist is an ultra-light immutable, possibly-infinite, virtual
sequence. Instead of storing its elements, a Lazylist knows:

 - its length (or a function for computing it)
 - a function from index to element

Elementary Lazylists are defined from scratch, e.g.

  ll = lazylist.Range(5)
  ll.length -> 5
  ll[2] -> 2
  list(ll) -> [0, 1, 2, 3, 4]

  ll = lazylist.Reference(source='foo', length=3)
  ll.length -> 3
  ll[0] -> ('foo', 0)
  list(ll) -> [('foo', 0), ('foo', 1), ('foo', 2)]

Some Lazylists are defined in terms of other lazylists, e.g.

  # repeat infinitely
  ll = lazylist.Repeat(lazylist.Range(10))
  ll.length -> math.inf
  ll[10000000000000000000000002] -> 2

  # concatenate two Lazylists
  ll = lazylist.Concat([lazylist.Reference('foo', 3),
                        lazylist.Reference('bar', 2)])
  ll.length -> 5
  list(ll) -> [('foo', 0), ('foo', 1), ('foo', 2), ('bar', 0), ('bar, 1)]

  # pseudorandom shuffle a Lazylist
  ll = lazylist.Shuffle(lazylist.Range(10))
  list(ll) -> [0, 7, 8, 4, 6, 1, 9, 3, 2, 5]

The intended use of this library is for defining deterministic
training sequences for machine learning.  Since data is generally
stored on disk, arbitrary visitation orders can be defined,
performance will depend on either efficient random access to the
training examples, or offline caching.

A Lazylist also as a `children` attribute containing the list of other
Lazylists used to define it. The is useful for inspecting Lazylist
objects. For example, a mixed dataset might be defined by
interleaving/shuffing several `lazylist.Refernce` lists pointing
to different source datasets. Rather than keeping a separate list of
source datasets, the Lazylist corresponding to the mixed dataset can
be inspected by the `all_sources()` function defined below.

  ll = lazylist.Shuffle(lazylist.Concat(
    [lazylist.Reference('foo', 3), lazylist.Reference('bar', 2)]))
  list(ll) -> [('foo', 0), ('bar', 0), ('foo', 2), ('bar', 1), ('foo', 1)]
  ll[3] -> ('bar', 1)
  lazylist.all_sources(ll) -> ['foo', 'bar']

The user may implement additional subclasses of Lazylist, in order to
experiment with different training regimens.

This file does not depend on anything else. Please keep it that
way. You can play with it from the command line:

python3 -i lazylist.py
list(Import(range(10)).Shuffle())
-> [0, 7, 8, 4, 6, 1, 9, 3, 2, 5]

"""

import hashlib
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union


def _fingerprint(*args) -> int:
  """A 128-bit fingerprint based on md5.

  For data shuffling - not for cryptography.

  Args:
    *args: any argument list that can be converted to a string
  Returns:
    an integer in [0, 2 ** 128)
  """
  return int.from_bytes(hashlib.md5(str(args).encode()).digest(), "little")


def _pseudorandom_permutation(n: int,
                              i: int,
                              seed: object) -> int:
  """Computes the position of `i` after a pseudorandom permutation on `[0, n)`.

  Based on Feistel ciphers.

  For data shuffling - not for cryptography.

  if i != j, then
  pseudorandom_permutation(n, i, seed) != pseudorandom_permutation(n, j, seed)

  Args:
    n: a positive integer
    i: an integer in [0, n)
    seed: a python object that can be converted to a string
  Returns:
    an integer in [0, n)
  """
  if not isinstance(n, int):
    raise ValueError("n must be an integer")

  if i < 0 or i >= n:
    raise ValueError("out of range")

  if n == 1:
    return 0

  # smallest k such that n-1 fits in 2k bits
  k = ((n-1).bit_length() + 1) // 2
  assert n <= 4 ** k
  # Permute repeatedly in [0, 4 ** k) until you land back in [0, n)
  # This constitutes a permutation of [0, n)
  while True:
    # Feistel ciper on 2k bits - i.e. a permutation of [0, 4 ** k)
    a, b = i // (2 ** k), i % (2 ** k)
    for r in range(3):
      a, b = b, a ^ (_fingerprint(b, r, seed) % (2 ** k))
    i = a * (2 ** k) + b
    if i < n:
      return int(i)


def _interleave_counts_in_first_k(proportions: Sequence[int],
                                  k: int) -> List[int]:
  """Formula for interleaving infinite sequences with given proportions.

  We are interleaving n infinite sequences (components) into one combined
  sequence.

  proportions (P) is a list of n integers, representing mixing proportions.

  mix(P, k, i) represents the number of examples from component i
  among the first k examples from the mixed sequence.  It is given by the
  following formula:

    mix(P, k, 0) = ceiling(k * P[0] / sum(P))
    mix(P, k, i>0) = mix(P[1:], k-mix(P, k, 0), i-1)

  Element k of the mixed sequence is equal to element m from component i iff:

    mix(P, k+1, i) == m+1  AND
    mix(P, k, i) == m

  _interleave_counts_in_first_k() computes the "mix" function described above.

  _interleave_kth_element() maps from the index in the combined sequence to
    identity of the component sequence and index in the component sequence.

  Args:
    proportions: a list/tuple of n integers (mixing proportions)
    k: number of elements of the mixed sequence

  Returns:
    counts of how many elements from each component sequence are used.
  """
  orig_k = k
  if not isinstance(k, int) or k < 0:
    raise ValueError("k must be a non-negative integer, got %s" % k)
  for p in proportions:
    if not isinstance(p, int) or p <= 0:
      raise ValueError("proportions must be positive integers, got %s" %
                       proportions)
  sum_remaining = sum(proportions)
  ret = []
  for p in proportions:
    num_not_from_first = (k * (sum_remaining - p)) // sum_remaining
    ret.append(k - num_not_from_first)
    k = num_not_from_first
    sum_remaining -= p
  assert k == 0
  assert sum(ret) == orig_k
  return ret


def _interleave_kth_element(proportions: Sequence[int],
                            k: int) -> Tuple[int, int]:
  """Formula for interleaving infinite sequences with given proportions.

  We are interleaving n infinite sequences (components) into one combined
  sequence.

  See the description in _interleave_counts_in_first_k() above.

  Args:
    proportions: a list/tuple of n integers (mixing proportions)
    k: index in the mixed sequence

  Returns:
    which_component: an integer in [0, n), representing the component index
    which_example: the index in the component sequence
  """
  new_counts = _interleave_counts_in_first_k(proportions, k + 1)
  old_counts = _interleave_counts_in_first_k(proportions, k)
  for which_component, (old_count,
                        new_count) in enumerate(zip(old_counts, new_counts)):
    if new_count > old_count:
      return which_component, old_count
  assert False


class Lazylist(object):
  """Superclass for Lazylist objects.

  See file docstring above.

  A Lazylist is a lightweight object representing an immutable,
  possibly-infinite seqeunce of elements, and providing efficient random access.

  Lazylist creation and random access should be though of as
  constant time and space operations.

  External interface:

    ll.length   -> number of elements (integer or math.inf)
    ll[i]       -> i-th Element

  Note: len(ll) produces an error for infinite sequences.

  Internal interface:

  Subclasses should implement _getitem(self, idx)

  In most cases, the length is computed lazily. This is to accommodate
  lightweight Lazylist construction in the case where the underlying
  source sequences require filesystem access to know their lengths.

  Subclasses whose lengths depend on the lengths of children should implement
  _compute_length(self).  Other subclasses can simply set _cached_length in the
  constructor.

  Convenience Constructors:

  Lazylists are built out of each other using the Lazylists
  subclasses in this file, or using user-defined Lazylists subclasses.
  For convenience, a few of these are implemented as methods of
  Lazylist.

  ll.Repeat()
  ll.Shuffle()
  ll.RepeatShuffle()
  ll.IID()
  ll[j:k], ll[j:], ll[:k] -> Lazylist containing a range of elements

  This allows easier-to-read syntax like
    `ll.Shuffle(seed=123).Repeat()[:1000000]`
  instead of
    `Range(Repeat(Shuffle(ll, seed=123)), stop=1000000)`

  """

  def __init__(self, children: List["Lazylist"]):
    """Create a Lazylist.

    Args:
      children: Lazylist object(s) used to build this Lazylist
    """
    self.children = list(children)
    self._cached_length = None

  @property
  def length(self) -> Union[int, float]:
    if self._cached_length is None:
      self._cached_length = self._compute_length()
    return self._cached_length

  def __len__(self):
    ret = self.length
    if ret == math.inf:
      raise ValueError(
          "len() cannot be used for infinite sequences. Use .length instead.")
    return ret

  def __getitem__(self, idx: int) -> object:
    """Returns the Element at the given index."""
    if isinstance(idx, int):
      self._verify_in_range(idx)
      return self._getitem(idx)
    elif isinstance(idx, slice):
      if idx.step is not None:
        raise NotImplementedError("stride not yet implemented")
      # convenience constructor for Slice() class
      return Slice(self,
                   0 if idx.start is None else idx.start,
                   idx.stop)
    else:
      raise ValueError("__getitem__ must get an int or a slice")

  def _compute_length(self) -> Union[int, float]:
    """Subclasses whose length depends on their children should implement."""
    raise NotImplementedError("not implemented")

  def _getitem(self, idx: int) -> object:
    """Return the element at the given index."""
    raise NotImplementedError("not implenented")

  def _verify_in_range(self, idx: int):
    """Verify that the index is in the range [0, self.length)."""
    if not isinstance(idx, int):
      raise ValueError("idx must be an integer, got %s" % idx)
    if idx < 0 or idx >= self.length:
      raise ValueError(
          "idx out of range - expected value in [0, %s), got %d"
          % (self.length, idx))

  def __iter__(self):
    """Iterate over items."""
    i = 0
    while i < self.length:
      yield self[i]
      i += 1

  @property
  def child(self) -> "Lazylist":
    """Convenience property for returning self.children[0]."""
    if isinstance(self.children, list) and len(self.children) == 1:
      return self.children[0]
    else:
      raise ValueError(
          "self.child requires that self.children be a one-element list")

  def Repeat(self):  # pylint: disable=invalid-name
    """Repeat forever in same order."""
    return Repeat(self)

  def Shuffle(self,  # pylint: disable=invalid-name
              seed: object = 0) -> "Lazylist":
    """Change order of a finite Lazylist."""
    return Shuffle(self, seed=seed)

  def RepeatShuffle(self,  # pylint: disable=invalid-name
                    seed: object = 0,
                    shuffle_first_epoch: bool = False) -> "Lazylist":
    """Repeat forever, but reshuffle each epoch except the first."""
    return RepeatShuffle(self,
                         seed=seed,
                         shuffle_first_epoch=shuffle_first_epoch)

  def IID(self,  # pylint: disable=invalid-name
          length: Union[int, float] = math.inf,
          seed: object = 0) -> "Lazylist":
    """Sample n examples independently and uniformly."""
    return IID(self, length=length, seed=seed)


class Import(Lazylist):
  """Lazylist version of any sequence object.

  Length and elements are the same as for the wrapped sequence.

  Example:
    Import(['foo', 'bar', 'baz'])
  """

  def __init__(self, sequence: Sequence[Any]):
    """Create a Import.

    Args:
      sequence: any sequence
    """
    super().__init__([])
    self.sequence = sequence
    self._cached_length = len(self.sequence)

  def _getitem(self, idx) -> object:
    if len(self.sequence) != self.length:
      raise ValueError("length of underlying sequence changed.")
    return self.sequence[idx]

  def __repr__(self) -> str:
    return "lazylist.Import(sequence=%s)" % repr(self.sequence)


class Reference(Lazylist):
  """A sequence of pairs [(source, 0), (source, 1) .. (source, length-1)].

  `source` is an opaque external object, e.g. a string.
  Element i of this sequence returns the pair (source, i).

  For example:

    ll = Reference(source='foo', length=3)
    ll.length -> 3
    list(ll) -> [('foo', 0), ('foo', 1), ('foo', 2)]

  Reference is useful in debugging/visualizing other Lazylist
  subclasses.  For example:

    ll = Concat([Reference('foo', 3), Reference('bar', 2)])
    ll.length -> 5
    list(ll) -> [('foo', 0), ('foo', 1), ('foo', 2), ('bar', 0), ('bar, 1)]
  """

  def __init__(self,
               source: object,
               length: Union[int, float, Callable[[], Union[int, float]]]):
    """Create a Reference.

    For lazy length computation, length can be a function .

    Args:
      source: any python object
      length: an integer or math.inf, or a callable returning such a value
    """
    super().__init__([])
    self.source = source
    if callable(length):
      self.length_fn = length
    else:
      self._cached_length = length

  def _compute_length(self) -> Union[int, float]:
    return self.length_fn()

  def _getitem(self, idx) -> object:
    return self.source, idx

  def __repr__(self) -> str:
    return "lazylist.Reference(source=%s, length=%s)" % (
        repr(self.source),
        repr(self._cached_length)
        if self._cached_length is not None else self.length_fn)


def all_sources(ll: Lazylist) -> List[Any]:
  """Given a Lazylist built from References, return all sources.

  Example:
    ll = Concat([Reference('foo', 3), Reference('bar', 2)])
    list(ll) -> [('foo', 0), ('foo', 1), ('foo', 2), ('bar', 0), ('bar, 1)]
    all_sources(ll) -> ['foo', 'bar']

  Args:
    ll: a Lazylist built out of Reference objects
  Returns:
    a list of all unique `source` objects from the Reference objects.
  """
  if isinstance(ll, Reference):
    return [ll.source]
  if not ll.children:
    raise ValueError("all leaves must be Reference")
  ret = []
  sources_set = set()
  for c in ll.children:
    for s in all_sources(c):
      if s not in sources_set:
        ret.append(s)
        sources_set.add(s)
  return ret


class Range(Lazylist):
  """The sequence of nonnegative integers [0, length).

  `length` can be a nonnegative integer or math.inf.
  """

  def __init__(self, length: Union[int, float]):
    super().__init__([])
    self._cached_length = length

  def _getitem(self, idx) -> object:
    return idx

  def __repr__(self) -> str:
    return "lazylist.Range(length=%s)" % self._cached_length


class Concat(Lazylist):
  """Concatenate multiple Lazylists into one.

  Example:
    ll = Concat([Reference('foo', 3), Reference('bar', 2)])
    ll.length -> 5
    list(ll) -> [('foo', 0), ('foo', 1), ('foo', 2), ('bar', 0), ('bar, 1)]
  """

  def _compute_length(self) -> Union[int, float]:
    if math.inf in [c.length for c in self.children[::-1]]:
      raise ValueError("only last child can be infinite")
    return sum([c.length for c in self.children])

  def _getitem(self, idx: int) -> object:
    for c in self.children:
      if idx < c.length:
        return c[idx]
      idx -= c.length
    assert False

  def __repr__(self) -> str:
    return "lazylist.Concat(children=%s)" % repr(self.children)


class Repeat(Lazylist):
  """Repeat the child sequence forever in same order.

  Example:
    ll = Repeat(Import(range(10)))
    ll.length -> math.inf
    ll[123456789123456789] -> 9
  """

  def __init__(self, child: Lazylist):
    super().__init__([child])
    self._cached_length = math.inf

  def _getitem(self, idx: int) -> object:
    if self.child.length == math.inf:
      raise ValueError("self.child.length must be finite")
    return self.child[idx % self.child.length]

  def __repr__(self) -> str:
    return "lazylist.Repeat(child=%s)" % repr(self.child)


class Slice(Lazylist):
  """A range of elements from the child Lazylist."""

  def __init__(self,
               child: Lazylist,
               start: int,
               stop: Optional[int]):
    """Create a Slice.

    Args:
      child: a Lazylist
      start: start index
      stop: optional end index
    """
    super().__init__([child])
    self.start = start
    self.stop = stop

  def _compute_length(self) -> Union[int, float]:
    if self.stop is None:
      self.stop = self.child.length
    if self.start < 0 or self.start > self.stop or self.stop > self.child.length:
      raise ValueError("[start, stop) must be contained in [0, child.length)")
    return self.stop - self.start

  def _getitem(self, idx: int) -> object:
    if self.stop is None:
      self.stop = self.child.length
    return self.child[idx + self.start]

  def __repr__(self) -> str:
    return "lazylist.Slice(child=%s, start=%s, stop=%s)" % (
        repr(self.child),
        repr(self.start),
        repr(self.stop))


class IID(Lazylist):
  """Uniform iid sampling.

  This is useful for creating an infinite sequence of IID examples.
  """

  def __init__(self,
               child: Lazylist,
               length: Union[int, float] = math.inf,
               seed: object = 0):
    """Create a IID.

    Args:
      child: a Lazylist
      length: an integer or math.inf
      seed: a python object that can be converted to a string
    """
    super().__init__([child])
    self._cached_length = length
    self.seed = seed

  def _getitem(self, idx: int) -> object:
    return self.child[_fingerprint(idx, self.seed) % self.child.length]

  def __repr__(self) -> str:
    return "lazylist.IID(child=%s, length=%s, seed=%s)" % (
        repr(self.child),
        repr(self._cached_length),
        repr(self.seed))


class Shuffle(Lazylist):
  """Shuffle a finite sequence deterministically."""

  def __init__(self,
               child: Lazylist,
               seed: object = 0):
    """Create a Shuffle.

    Args:
      child: a Lazylist of finite length
      seed: a python object that can be converted to a string
    """
    super().__init__([child])
    self.seed = seed

  def _compute_length(self) -> Union[int, float]:
    if self.child.length == math.inf:
      raise ValueError("Shuffle requires a finite dataset.")
    return self.child.length

  def _getitem(self, idx: int) -> object:
    return self.child[_pseudorandom_permutation(self.length, idx, self.seed)]

  def __repr__(self) -> str:
    return "lazylist.Shuffle(child=%s, seed=%s)" % (
        repr(self.child),
        repr(self.seed))


class RepeatShuffle(Lazylist):
  """Repeat forever - shuffle every epoch differently.

  To shuffle uniformly across epochs, use IID() instead.
  """

  def __init__(self,
               child: Lazylist,
               seed: object = 0,
               shuffle_first_epoch: bool = False):
    """Create a RepeatShuffle.

    Args:
      child: a Lazylist of finite length
      seed: a python object that can be converted to a string
      shuffle_first_epoch: whether to also shuffle the first epoch
    """
    super().__init__([child])
    self._cached_length = math.inf
    self.seed = seed
    self.shuffle_first_epoch = shuffle_first_epoch

  def _getitem(self, idx: int) -> object:
    if self.child.length == math.inf:
      raise ValueError("self.child.length must be finite")
    epoch_num = idx // self.child.length
    idx_in_epoch = idx % self.child.length
    if epoch_num == 0 and not self.shuffle_first_epoch:
      return self.child[idx_in_epoch]
    seed = (self.seed, epoch_num)
    return self.child[_pseudorandom_permutation(
        self.child.length, idx_in_epoch, seed)]

  def __repr__(self) -> str:
    return "lazylist.RepeatShuffle(child=%s, seed=%s, shuffle_first_epoch=%s)" % (
        repr(self.child),
        repr(self.seed),
        repr(self.shuffle_first_epoch),
    )


class Interleave(Lazylist):
  """Interleave (mix) sequences according to given proportions."""

  def __init__(self,
               children: List[Lazylist],
               proportions: List[int],
               length: Union[int, float] = math.inf):
    """Create a Interleave.

    Args:
      children: a list of Lazylist
      proportions: mixing rates (same length as children)
      length: an integer or math.inf
    """
    super().__init__(children)
    self.proportions = proportions
    self._cached_length = math.inf
    if len(children) != len(proportions):
      raise ValueError(
          "children and proportions must be lists of the same length.")

  def _getitem(self, idx: int) -> object:
    child_num, idx_in_child = _interleave_kth_element(self.proportions, idx)
    return self.children[child_num][idx_in_child]

  def __repr__(self) -> str:
    return "lazylist.Interleave(children=%s, proportions=%s, length=%s)" % (
        repr(self.children),
        repr(self.proportions),
        repr(self._cached_length),
    )


class InterleaveOnce(Lazylist):
  """Interleave finite data sequences, using every element once."""

  def _compute_length(self) -> Union[int, float]:
    ret = sum([c.length for c in self.children])
    if ret == math.inf:
      raise ValueError("InterleaveOnce requires finite sequence")
    return ret

  def _getitem(self, idx: int) -> object:
    proportions = [c.length for c in self.children]
    child_num, idx_in_child = _interleave_kth_element(proportions, idx)
    return self.children[child_num][idx_in_child]

  def __repr__(self) -> str:
    return "lazylist.InterleaveOnce(children=%s)" % repr(self.children)
