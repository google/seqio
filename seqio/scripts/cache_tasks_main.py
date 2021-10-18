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

# Lint as: python3
r"""Dumps preprocessed tasks as TFRecord of tf.Examples.

Usage:
====================
seqio_cache_tasks \
--tasks=my_task_*,your_task \
--excluded_tasks=my_task_5 \
--output_cache_dir=/path/to/cache_dir \
--module_import=my.tasks \
--alsologtostderr

"""

import importlib
import os
import re

from absl import app
from absl import flags
from absl import logging

import apache_beam as beam
import seqio
from seqio import beam_utils
import tensorflow.compat.v2 as tf



# Significantly speeds up preprocessing in tf1.
tf.compat.v1.enable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_list(
    "tasks", None,
    "Regexes matching task(s) to build a preprocessed dataset for. Will build "
    "all registered if not specified.")
flags.DEFINE_list(
    "excluded_tasks", None,
    "Regexes matching task(s) to skip.")
flags.DEFINE_string(
    "output_cache_dir", None,
    "The directory to output cached tasks to.")
flags.DEFINE_list(
    "tasks_additional_cache_dirs", [],
    "Additional directories to search for cached Tasks after checking the "
    "global caches and `output_cache_dir`.")
flags.DEFINE_multi_string(
    "module_import", [],
    "Modules to import. Use this, for example, to add new `Task`s to the "
    "global `TaskRegistry`.")
flags.DEFINE_list(
    "pipeline_options", ["--runner=DirectRunner"],
    "A comma-separated list of command line arguments to be used as options "
    "for the Beam Pipeline.")
flags.DEFINE_boolean(
    "overwrite", False,
    "If true, overwrite the cached task even if it exists in the cached "
    "directories.")


def _import_modules(modules):
  for module in modules:
    if module:
      importlib.import_module(module)


def run_pipeline(pipeline,
                 task_names,
                 cache_dir,
                 excluded_tasks=None,
                 modules_to_import=(),
                 overwrite=False,
                 completed_file_contents=""):
  """Run preprocess pipeline."""
  output_dirs = []
  # Includes all names by default.
  included_regex = re.compile(r"(%s\Z)" % r"\Z|".join(task_names or [".*"]))
  # Excludes only empty names by default.
  excluded_regex = re.compile(r"(%s\Z)" % r"\Z|".join(excluded_tasks or []))
  task_names = [
      t for t in seqio.TaskRegistry.names()
      if included_regex.match(t) and not excluded_regex.match(t)]
  if not task_names:
    logging.warning("No tasks have been selected from the task registry. "
                    "Please make sure that the tasks you want cached exist in "
                    "the task registry and haven't been excluded by the "
                    "--excluded_tasks flag.")
  for task_name in task_names:
    task = seqio.TaskRegistry.get(task_name)
    if not task.supports_caching:
      logging.info(
          "Skipping task that does not support caching: '%s'", task.name)
      continue

    task_cache_dir = task.cache_dir
    output_dir = os.path.join(
        cache_dir, seqio.get_task_dir_from_name(task.name))

    if task_cache_dir and not overwrite:
      logging.info("Skipping task '%s', which exists in cache dir: %s",
                   task.name, task_cache_dir)
      continue

    if task_cache_dir and overwrite:
      if task_cache_dir == output_dir:
        # We were asked to overwrite the data, and the given directory that we
        # should generate the data in already has the data, then delete it.
        logging.warning(
            "Overwriting already cached data for task '%s' in cache_dir %s",
            task.name, output_dir)
        tf.io.gfile.rmtree(output_dir)
      else:
        # Cannot overwrite, since cache_dir isn't same as task.cache_dir.
        logging.warning("Not overwriting data in task.cache_dir since it is "
                        "different from cache_dir - %s vs %s", task.cache_dir,
                        output_dir)
        continue

    if not task.splits:
      logging.warning("Skipping task '%s' with no splits.", task.name)
      continue

    # Log this task to the terminal.
    print("Caching task '%s' with splits: %s" % (task.name, task.splits))

    output_dirs.append(output_dir)
    completion_values = []

    if isinstance(task.source, seqio.FunctionDataSource):
      logging.warning(
          "Task '%s' using FunctionDataSource cannot be distributed. If your "
          "dataset is large, you may be able to speed up preprocessing by "
          "sharding it and using a TfdsSource, TFExampleSource, or "
          "TextLineSource instead.", task.name)

    for split in task.splits:
      label = "%s_%s" % (task.name, split)

      pat = beam_utils.PreprocessTask(
          task, split, modules_to_import=modules_to_import)
      num_shards = len(pat.shards)
      examples = pipeline | "%s_pat" % label >> pat
      completion_values.append(
          examples
          | "%s_write_tfrecord" % label >> beam_utils.WriteExampleTfRecord(
              seqio.get_cached_tfrecord_prefix(output_dir, split),
              num_shards=num_shards))
      completion_values.append(
          examples
          | "%s_info" % label >> beam_utils.GetInfo(num_shards)
          | "%s_write_info" % label >> beam_utils.WriteJson(
              seqio.get_cached_info_path(output_dir, split)))
      completion_values.append(
          examples
          | "%s_stats" % label >> beam_utils.GetStats(task.output_features)
          | "%s_write_stats" % label >> beam_utils.WriteJson(
              seqio.get_cached_stats_path(output_dir, split)))

    # After all splits for this task have completed, write COMPLETED files to
    # the task's output directory.
    _ = (completion_values
         | "%s_flatten_completion_values" % task.name >> beam.Flatten()
         | "%s_discard_completion_values" % task.name >> beam.Filter(
             lambda _: False)
         | "%s_write_completed_file" % task.name >> beam.io.textio.WriteToText(
             os.path.join(output_dir, "COMPLETED"),
             append_trailing_newlines=False, num_shards=1,
             shard_name_template="", header=completed_file_contents))

  return output_dirs


def main(_):
  flags.mark_flags_as_required(["output_cache_dir"])

  _import_modules(FLAGS.module_import)

  seqio.add_global_cache_dirs(
      [FLAGS.output_cache_dir] + FLAGS.tasks_additional_cache_dirs)

  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options)
  with beam.Pipeline(options=pipeline_options) as pipeline:
    tf.io.gfile.makedirs(FLAGS.output_cache_dir)
    unused_output_dirs = run_pipeline(
        pipeline,
        FLAGS.tasks,
        FLAGS.output_cache_dir,
        FLAGS.excluded_tasks,
        FLAGS.module_import,
        FLAGS.overwrite,
    )


def console_entry_point():
  app.run(main)

if __name__ == "__main__":
  console_entry_point()
