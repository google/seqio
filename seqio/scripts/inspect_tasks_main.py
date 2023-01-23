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

r"""Python script to inspect Seqio tasks.

Usage:
====================
inspect_tasks_main \
--tasks=my_task_*,your_task \
--module_import=my.tasks \
--sequence_length="{'inputs':2048,'targets':2048}"
"""
import ast
import importlib
import pprint
import re

from absl import app
from absl import flags
import numpy as np
import seqio

FLAGS = flags.FLAGS
flags.DEFINE_list(
    "tasks",
    None,
    (
        "Regexes matching task(s) to inspect. Will inspect all registered if"
        " not specified."
    ),
)

flags.DEFINE_list("excluded_tasks", None, "Regexes matching task(s) to skip.")

flags.DEFINE_multi_string(
    "module_import",
    [],
    (
        "Modules to import. Use this, for example, to add new `Task`s to the "
        "global `TaskRegistry`."
    ),
)
flags.DEFINE_string(
    "sequence_length",
    "{'inputs': 32, 'targets': 32}",
    (
        "String representation of a dictionary for sequence length to be passed"
        " to Seqio `get_dataset`."
    ),
)

flags.DEFINE_bool(
    "decode_features",
    False,
    "If true, decode the output features using the vocabulary.",
)
flags.DEFINE_bool(
    "use_cached",
    False,
    "If true, use cached dataset. Required for DeterministicTask.",
)
flags.DEFINE_bool(
    "inspect_task_examples",
    True,
    "If true, inspects task examples one at a time.",
)
flags.DEFINE_bool(
    "print_length_statistics",
    False,
    "If true, print length statistics from each task.",
)
flags.DEFINE_multi_float(
    "length_percentiles",
    [0.5, 0.9, 0.95, 0.99],
    "Percentiles to use when `print_length_statistics` is `True`.",
)
flags.DEFINE_integer(
    "length_example_count",
    4096,
    (
        "Number of examples to compute length statistics when "
        + "`print_length_statistics is `True`."
    ),
)


def _print_length_statistics(task_or_mixture):
  """Utility function for printing length statistics of a feature."""
  print(f"* {task_or_mixture.name} Length Statistics *")
  percentiles = [int(p * 100) for p in FLAGS.length_percentiles]
  percentile_headers = [f"p{p}" for p in percentiles]
  print("split, feature, " + ", ".join(percentile_headers))
  sequence_length = ast.literal_eval(FLAGS.sequence_length)
  for split in task_or_mixture.splits:
    dataset = task_or_mixture.get_dataset(
        sequence_length=sequence_length,
        split=split,
        use_cached=FLAGS.use_cached,
        shuffle=False,
    )
    sizes = []
    features = list(sequence_length.keys())
    for e in dataset.take(FLAGS.length_example_count):
      sizes.append([e[feature].numpy().size for feature in features])
    all_lengths = np.transpose(np.percentile(sizes, percentiles, axis=0))
    for feature, lengths in zip(features, all_lengths):
      print(f"{split}, {feature}, " + ", ".join(f"{l:.1f}" for l in lengths))


def _import_modules(modules):
  """Function that imports an additional module."""
  for module in modules:
    if module:
      importlib.import_module(module)


def _inspect_task_or_mixture(task_or_mixture):
  """Utility function for testing."""
  print("* Found following splits. *")
  print("\n".join(f"  {split}" for split in task_or_mixture.splits))

  if (
      input(
          "Press any key to explore splits. Press 'c' to end inspection for"
          " this task/mixture."
      )
      != "c"
  ):
    for split in task_or_mixture.splits:
      sequence_length = ast.literal_eval(FLAGS.sequence_length)
      dataset = task_or_mixture.get_dataset(
          sequence_length=sequence_length,
          split=split,
          use_cached=FLAGS.use_cached,
          shuffle=False,
      )
      print(f"* Split: {split} *")
      for example in dataset:
        printed_example = {}
        for key, val in example.items():
          if FLAGS.decode_features and key in task_or_mixture.output_features:
            printed_example[key] = task_or_mixture.output_features[
                key
            ].vocabulary.decode_tf(val)
          else:
            printed_example[key] = val
        pprint.pprint(printed_example, indent=2)
        if (
            input(
                "Press any key for next example. Press 'c' to skip to next"
                " split."
            )
            == "c"
        ):
          break


def main(_) -> None:
  _import_modules(FLAGS.module_import)

  # Includes all names by default.
  included_regex = re.compile(r"(%s\Z)" % r"\Z|".join(FLAGS.tasks or [".*"]))
  # Excludes only empty names by default.
  excluded_regex = re.compile(
      r"(%s\Z)" % r"\Z|".join(FLAGS.excluded_tasks or [])
  )

  task_names = [
      ("Task", t)
      for t in seqio.TaskRegistry.names()
      if included_regex.match(t) and not excluded_regex.match(t)
  ]
  print("*** Found the following Seqio tasks. ***")
  print("\n".join(f"  {name}" for _, name in task_names))

  mixture_names = [
      ("Mixture", t)
      for t in seqio.MixtureRegistry.names()
      if included_regex.match(t) and not excluded_regex.match(t)
  ]
  print("*** Found the following Seqio mixtures. ***")
  print("\n".join(f"  {name}" for _, name in mixture_names))

  for typ, name in task_names + mixture_names:
    print(f"*** {typ}: {name} ***")
    task = seqio.get_mixture_or_task(name)
    if FLAGS.inspect_task_examples:
      _inspect_task_or_mixture(task)
    if FLAGS.print_length_statistics:
      _print_length_statistics(task)
    print(f"*** Done Inspecting {typ}: {name} ***")


if __name__ == "__main__":
  app.run(main)
