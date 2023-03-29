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

"""Tests for asserts."""
from absl.testing import absltest

from seqio import dataset_providers
from seqio.test_utils import assert_dataset
from seqio.test_utils import DataInjector
from seqio.test_utils import FakeTaskTest
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()


# Note that the b'string' values are for PY3 to interpret as bytes literals,
# which match the tf.data.Dataset from tensor slices.
class TestUtilsTest(absltest.TestCase):

  def test_assert_dataset(self):
    first_dataset = tf.data.Dataset.from_tensor_slices(
        {'key1': ['val1'], 'key2': ['val2']}
    )
    ragged_tensor_dataset = tf.data.Dataset.from_tensor_slices(
        {'key': tf.ragged.constant([[[1.0, 2.0], [3.0]]])}
    )
    float_dataset = tf.data.Dataset.from_tensor_slices(
        {'float_key1': [0.2, 0.3], 'float_key2': [0.4, 0.5]}
    )

    # Equal
    assert_dataset(first_dataset, {'key1': [b'val1'], 'key2': [b'val2']})
    assert_dataset(
        first_dataset,
        {'key1': [b'val1'], 'key2': [b'val2']},
        expected_dtypes={'key1': tf.string},
    )

    # Equal ragged tensor
    assert_dataset(ragged_tensor_dataset, {'key': [[1.0, 2.0], [3.0]]})

    # Equal floating-point
    assert_dataset(
        float_dataset,
        [
            {'float_key1': 0.2, 'float_key2': 0.4},
            {'float_key1': 0.3, 'float_key2': 0.5},
        ],
    )

    # Close enough floating-point
    assert_dataset(
        float_dataset,
        [
            {'float_key1': 0.20000001, 'float_key2': 0.39999999},
            {'float_key1': 0.30000001, 'float_key2': 0.49999999},
        ],
    )

    # Unequal value
    with self.assertRaises(AssertionError):
      assert_dataset(first_dataset, {'key1': [b'val1'], 'key2': [b'val2x']})

    with self.assertRaises(AssertionError):
      assert_dataset(ragged_tensor_dataset, {'key': [[1.0], [3.0]]})

    # Unequal floating-point, not close enough.
    with self.assertRaises(AssertionError):
      assert_dataset(
          float_dataset,
          [
              {'float_key1': 0.201, 'float_key2': 0.399},
              {'float_key1': 0.301, 'float_key2': 0.499},
          ],
      )

    # Wrong dtype
    with self.assertRaises(AssertionError):
      assert_dataset(
          first_dataset,
          {'key1': [b'val1'], 'key2': [b'val2']},
          expected_dtypes={'key1': tf.int32},
      )

    # Additional key, value
    with self.assertRaises(AssertionError):
      assert_dataset(
          first_dataset,
          {'key1': [b'val1'], 'key2': [b'val2'], 'key3': [b'val3']},
      )

    # Additional key, value
    with self.assertRaises(AssertionError):
      assert_dataset(
          first_dataset,
          {'key1': [b'val1'], 'key2': [b'val2'], 'key3': [b'val3']},
      )


class TasksTest(FakeTaskTest):

  def test_data_injection(self):
    def ds_fn(split, shuffle_files):
      del shuffle_files
      data = {'train': {'data': b'not used'}}
      ds = tf.data.Dataset.from_tensors(data[split])
      return ds

    source = dataset_providers.FunctionDataSource(
        dataset_fn=ds_fn, splits=['train']
    )

    dataset_providers.TaskRegistry.add(
        'test_data_injection_task',
        source=source,
        preprocessors=[],
        output_features={},
        metric_fns=[],
    )

    data = {'train': {'data': b'This data is not used.'}}
    with DataInjector('test_data_injection_task', data):
      pass

    task = dataset_providers.TaskRegistry.get('test_data_injection_task')
    self.assertIs(source, task._source)


if __name__ == '__main__':
  absltest.main()
