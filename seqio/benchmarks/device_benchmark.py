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

"""Microbenchmarks for SeqIO tpu functions."""

import functools

import google_benchmark
import jax
import jax.numpy as jnp
import numpy as np
import seqio
import t5.data.tasks  # pylint:disable=unused-import
import tensorflow_datasets as tfds

partial = functools.partial


_SOURCE_NUM_EXAMPLES = 1000


def requires_tpu(num_devices_required):
  """Helper to skip benchmarks that require TPUs."""

  def helper1(f):
    @functools.wraps(f)
    def helper2(state):
      if jax.device_count() < num_devices_required:
        state.skip_with_error(f'requires {num_devices_required} devices')
        return
      return f(state)

    return helper2

  return helper1


def _sum_of_squares(x):
  return jnp.sum(x**2)


_sum_of_squares_dx = jax.grad(_sum_of_squares)
_sum_of_squares_dx_jit = jax.jit(_sum_of_squares_dx)


@google_benchmark.register
@requires_tpu(2)
def wmt_generated_data_benchmark(state):
  """Loads a generated WMT dataset onto TPUs and performs a simple calculation."""
  with tfds.testing.mock_data(num_examples=_SOURCE_NUM_EXAMPLES):
    wmt_task = seqio.TaskRegistry.get('wmt19_ende_v003')
    ds = wmt_task.get_dataset(split='train')

  while state:
    for element in ds:
      for _, v in element.items():
        if isinstance(v, np.ndarray):
          if v.dtype == np.int64:
            v = v.astype(np.float32)
          # Transfer to device.
          x = jax.device_put(v)
          state.pause_timing()
          # Compile.
          _sum_of_squares_dx_jit(x).block_until_ready()
          state.resume_timing()
          # Run.
          _sum_of_squares_dx_jit(x).block_until_ready()


@google_benchmark.register
@requires_tpu(2)
def wmt_from_file_data_benchmark(state):
  """Loads a WMT dataset from file onto TPUs and performs a simple calculation."""
  wmt_task = seqio.TaskRegistry.get('wmt19_ende_v003')
  ds = wmt_task.get_dataset(split='train')

  element_count = 0
  while state:
    for element in ds:
      for _, v in element.items():
        if isinstance(v, np.ndarray):
          if v.dtype == np.int64:
            v = v.astype(np.float32)
          # Transfer to device.
          x = jax.device_put(v)
          state.pause_timing()
          # Compile.
          _sum_of_squares_dx_jit(x).block_until_ready()
          state.resume_timing()
          # Run.
          _sum_of_squares_dx_jit(x).block_until_ready()
      element_count += 1
      if element_count >= _SOURCE_NUM_EXAMPLES:
        break


if __name__ == '__main__':
  google_benchmark.main()
