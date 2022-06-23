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

"""Tests for helpers."""

import functools

from absl.testing import absltest
from seqio import dataset_providers as dp
from seqio import helpers
from seqio import preprocessors as pr
from seqio import test_utils
import tensorflow as tf

VOCAB1 = test_utils.sentencepiece_vocab(extra_ids=10)
VOCAB2 = test_utils.sentencepiece_vocab(extra_ids=20)

_SEQUENCE_LENGTH = {"inputs": 16, "targets": 8}


def _dataset_fn(split, shuffle_files=False, seed=None, data=None):
  del split, shuffle_files, seed
  exs = {"feature_a": data, "feature_b": data}
  return tf.data.Dataset.from_tensor_slices(exs)


class HelpersTest(test_utils.FakeTaskTest):

  def test_task_new_vocab(self):
    task_dataset_fn = functools.partial(_dataset_fn, data=["this is", "a test"])
    test_task = dp.TaskRegistry.add(
        "my_test_task",
        source=dp.FunctionDataSource(task_dataset_fn, splits=["train"]),
        preprocessors=[pr.tokenize],
        output_features={
            "feature_a": dp.Feature(VOCAB1),
            "feature_b": dp.Feature(VOCAB1, add_eos=False)
        })
    helpers.mixture_or_task_with_new_vocab(
        "my_test_task", "my_new_test_task", new_vocab=VOCAB2)
    new_task = dp.get_mixture_or_task("my_new_test_task")
    self.assertEqual(new_task.source, test_task.source)
    self.assertEqual(new_task.preprocessors, test_task.preprocessors)
    self.assertEqual(
        new_task.output_features, {
            "feature_a": dp.Feature(VOCAB2),
            "feature_b": dp.Feature(VOCAB2, add_eos=False)
        })

  def test_task_new_output_features(self):
    task_dataset_fn = functools.partial(_dataset_fn, data=["this is", "a test"])
    test_task = dp.TaskRegistry.add(
        "my_test_task",
        source=dp.FunctionDataSource(task_dataset_fn, splits=["train"]),
        preprocessors=[pr.tokenize],
        output_features={
            "feature_a": dp.Feature(VOCAB1),
            "feature_b": dp.Feature(VOCAB1, add_eos=False)
        })
    new_task = helpers.mixture_or_task_with_new_vocab(
        "my_test_task",
        "my_new_test_task",
        new_output_features={
            "feature_a": dp.Feature(VOCAB2, add_eos=False),
            "feature_b": dp.Feature(VOCAB1)
        },
        add_to_seqio_registry=False)
    self.assertNotIn("my_new_test_task", dp.TaskRegistry.names())
    self.assertEqual(new_task.source, test_task.source)
    self.assertEqual(new_task.preprocessors, test_task.preprocessors)
    self.assertEqual(
        new_task.output_features, {
            "feature_a": dp.Feature(VOCAB2, add_eos=False),
            "feature_b": dp.Feature(VOCAB1)
        })

  def test_mixture_new_vocab(self):
    # Step 1: Define test Tasks.
    test_dataset_fn1 = functools.partial(
        _dataset_fn, data=["this is", "a test"])
    test_dataset_fn2 = functools.partial(
        _dataset_fn, data=["this is", "another test"])
    og_output_features = {
        "feature_a": dp.Feature(VOCAB1),
        "feature_b": dp.Feature(VOCAB1, add_eos=False)
    }
    test_task1 = dp.TaskRegistry.add(
        "my_test_task1",
        source=dp.FunctionDataSource(test_dataset_fn1, splits=["train"]),
        preprocessors=[pr.tokenize],
        output_features=og_output_features)
    test_task2 = dp.TaskRegistry.add(
        "my_test_task2",
        source=dp.FunctionDataSource(test_dataset_fn2, splits=["train"]),
        preprocessors=[pr.tokenize],
        output_features=og_output_features)

    # Step 2: Define test Mixtures
    dp.MixtureRegistry.add(
        "my_test_mix1", [("my_test_task1", 0.5), "my_test_task2"],
        default_rate=1.0)
    dp.MixtureRegistry.add(
        "my_test_mix2", ["my_test_task1", "my_test_mix1"], default_rate=1.0)

    # Step 3: Call helper to convert the mixture
    new_mix = helpers.mixture_or_task_with_new_vocab(
        "my_test_mix2",
        "my_new_test_mix2",
        new_vocab=VOCAB2,
        add_to_seqio_registry=True)

    # Step 4: Get new Tasks and Mixtures from the Registry.
    new_mix = dp.get_mixture_or_task("my_new_test_mix2")
    new_submix = dp.get_mixture_or_task("my_new_test_mix2.my_test_mix1")
    new_submix_subtask1 = dp.get_mixture_or_task(
        "my_new_test_mix2.my_test_mix1.my_test_task1")
    new_submix_subtask2 = dp.get_mixture_or_task(
        "my_new_test_mix2.my_test_mix1.my_test_task2")
    new_subtask = dp.get_mixture_or_task("my_new_test_mix2.my_test_task1")

    # Step 5: Verify mixing rates for new mixtures.
    self.assertDictEqual(new_mix._task_to_rate, {
        "my_new_test_mix2.my_test_task1": 1.0,
        "my_new_test_mix2.my_test_mix1": 1.0
    })
    self.assertDictEqual(
        new_submix._task_to_rate, {
            "my_new_test_mix2.my_test_mix1.my_test_task1": 0.5,
            "my_new_test_mix2.my_test_mix1.my_test_task2": 1.0
        })

    # Step 6: Verify output features for new Tasks and Mixtures.
    expected_output_features = {
        "feature_a": dp.Feature(VOCAB2),
        "feature_b": dp.Feature(VOCAB2, add_eos=False)
    }
    self.assertDictEqual(new_mix.output_features, expected_output_features)
    self.assertDictEqual(new_submix.output_features, expected_output_features)
    self.assertDictEqual(new_submix_subtask1.output_features,
                         expected_output_features)
    self.assertDictEqual(new_submix_subtask2.output_features,
                         expected_output_features)
    self.assertDictEqual(new_subtask.output_features, expected_output_features)

    # Step 7: Verify source and preprocessors for new Tasks.
    self.assertEqual(new_submix_subtask1.source, test_task1.source)
    self.assertEqual(new_submix_subtask1.preprocessors,
                     test_task1.preprocessors)
    self.assertEqual(new_submix_subtask2.source, test_task2.source)
    self.assertEqual(new_submix_subtask2.preprocessors,
                     test_task2.preprocessors)
    self.assertEqual(new_subtask.source, test_task1.source)
    self.assertEqual(new_subtask.preprocessors, test_task1.preprocessors)

  def test_mixture_new_output_features(self):
    # Step 1: Define test Tasks.
    test_dataset_fn1 = functools.partial(
        _dataset_fn, data=["this is", "a test"])
    test_dataset_fn2 = functools.partial(
        _dataset_fn, data=["this is", "another test"])
    og_output_features = {
        "feature_a": dp.Feature(VOCAB1),
        "feature_b": dp.Feature(VOCAB1, add_eos=False)
    }
    test_task1 = dp.TaskRegistry.add(
        "my_test_task1",
        source=dp.FunctionDataSource(test_dataset_fn1, splits=["train"]),
        preprocessors=[pr.tokenize],
        output_features=og_output_features)
    test_task2 = dp.TaskRegistry.add(
        "my_test_task2",
        source=dp.FunctionDataSource(test_dataset_fn2, splits=["train"]),
        preprocessors=[pr.tokenize],
        output_features=og_output_features)

    # Step 2: Define test Mixtures
    dp.MixtureRegistry.add(
        "my_test_mix1", [("my_test_task1", 0.5), "my_test_task2"],
        default_rate=1.0)
    dp.MixtureRegistry.add(
        "my_test_mix2", ["my_test_task1", "my_test_mix1"], default_rate=1.0)

    # Step 3: Call helper to convert the mixture
    new_output_features = {
        "feature_a": dp.Feature(VOCAB2, add_eos=False),
        "feature_b": dp.Feature(VOCAB1)
    }
    new_mix = helpers.mixture_or_task_with_new_vocab(
        "my_test_mix2",
        "my_new_test_mix2",
        new_output_features=new_output_features,
        add_to_seqio_registry=False)

    # Step 4: Get new Tasks and Mixtures from the Registry.
    self.assertNotIn("my_new_test_mix2", dp.MixtureRegistry.names())
    new_submix = dp.get_mixture_or_task("my_new_test_mix2.my_test_mix1")
    new_submix_subtask1 = dp.get_mixture_or_task(
        "my_new_test_mix2.my_test_mix1.my_test_task1")
    new_submix_subtask2 = dp.get_mixture_or_task(
        "my_new_test_mix2.my_test_mix1.my_test_task2")
    new_subtask = dp.get_mixture_or_task("my_new_test_mix2.my_test_task1")

    # Step 5: Verify mixing rates for new mixtures.
    self.assertDictEqual(new_mix._task_to_rate, {
        "my_new_test_mix2.my_test_task1": 1.0,
        "my_new_test_mix2.my_test_mix1": 1.0
    })
    self.assertDictEqual(
        new_submix._task_to_rate, {
            "my_new_test_mix2.my_test_mix1.my_test_task1": 0.5,
            "my_new_test_mix2.my_test_mix1.my_test_task2": 1.0
        })

    # Step 6: Verify output features for new Tasks and Mixtures.
    self.assertDictEqual(new_mix.output_features, new_output_features)
    self.assertDictEqual(new_submix.output_features, new_output_features)
    self.assertDictEqual(new_submix_subtask1.output_features,
                         new_output_features)
    self.assertDictEqual(new_submix_subtask2.output_features,
                         new_output_features)
    self.assertDictEqual(new_subtask.output_features, new_output_features)

    # Step 7: Verify source and preprocessors for new Tasks.
    self.assertEqual(new_submix_subtask1.source, test_task1.source)
    self.assertEqual(new_submix_subtask1.preprocessors,
                     test_task1.preprocessors)
    self.assertEqual(new_submix_subtask2.source, test_task2.source)
    self.assertEqual(new_submix_subtask2.preprocessors,
                     test_task2.preprocessors)
    self.assertEqual(new_subtask.source, test_task1.source)
    self.assertEqual(new_subtask.preprocessors, test_task1.preprocessors)

  def test_mixture_or_task_with_new_vocab_invalid(self):
    task_dataset_fn = functools.partial(_dataset_fn, data=["this is", "a test"])
    dp.TaskRegistry.add(
        "my_test_task",
        source=dp.FunctionDataSource(task_dataset_fn, splits=["train"]),
        preprocessors=[pr.tokenize],
        output_features={
            "feature_a": dp.Feature(VOCAB1),
            "feature_b": dp.Feature(VOCAB1, add_eos=False)
        })
    with self.assertRaises(ValueError):  # None set
      helpers.mixture_or_task_with_new_vocab("my_test_task", "my_new_test_task")
    with self.assertRaises(ValueError):  # Both set
      helpers.mixture_or_task_with_new_vocab(
          "my_test_task",
          "my_new_test_task",
          new_vocab=VOCAB2,
          new_output_features={
              "feature_a": dp.Feature(VOCAB2, add_eos=False),
              "feature_b": dp.Feature(VOCAB1)
          })
    with self.assertRaises(ValueError):  # Incorrect feature set
      helpers.mixture_or_task_with_new_vocab(
          "my_test_task",
          "my_new_test_task",
          new_output_features={
              "feature_a": dp.Feature(VOCAB2),
          })
    with self.assertRaises(ValueError):  # Incompatible features
      helpers.mixture_or_task_with_new_vocab(
          "my_test_task",
          "my_new_test_task",
          new_output_features={
              "feature_a": dp.Feature(VOCAB2, rank=3),
              "feature_b": dp.Feature(VOCAB1)
          })

  def test_task_with_truncated_data(self):
    task_dataset_fn = functools.partial(_dataset_fn, data=["this is", "a test"])
    _ = dp.TaskRegistry.add(
        "my_test_task",
        source=dp.FunctionDataSource(task_dataset_fn, splits=["train"]),
        preprocessors=[pr.tokenize],
        output_features={
            "feature_a": dp.Feature(VOCAB1),
            "feature_b": dp.Feature(VOCAB1, add_eos=False)
        })
    helpers.mixture_or_task_with_truncated_data(
        "my_test_task", "my_new_test_task", split_sizes={"train": 1})
    new_task = dp.get_mixture_or_task("my_new_test_task")
    ds = new_task.get_dataset(_SEQUENCE_LENGTH, "train")
    examples = list(ds.as_numpy_iterator())
    self.assertEqual(len(examples), 1)
    self.assertEqual(examples[0]["feature_a_pretokenized"].decode("utf-8"),
                     "this is")


if __name__ == "__main__":
  absltest.main()
