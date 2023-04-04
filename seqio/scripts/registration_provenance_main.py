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

r"""Python script to print where TaskRegistry.add calls are coming from.

Prints a TSV to stdout of each registered task (after importing modules from
--module_import), along with the callsites upstream. This is useful for
determining where any TaskRegistry.add calls exist at the module-level (and
are thus run when just importing that module).

Usage:
====================
registration_provenance \
  --registration_module_import=my.tasks
  --registration_provenance_out_filename=/some/file/path.tsv
"""
import importlib
from typing import List

from absl import app
from absl import flags
from seqio import task_registry_provenance_tracking
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_multi_string(
    "registration_module_import",
    [],
    (
        "Modules to import. Use this, for example, to add new `Task`s to the "
        "global `TaskRegistry`."
    ),
)

flags.DEFINE_string(
    "registration_provenance_out_filename",
    "/dev/stdout",  # Default is to print to command line.
    (
        "File name to which to write TSV of provenance. Use default to print to"
        " terminal."
    ),
)


def _import_modules(modules):
  """Function that imports an additional module."""
  for module in modules:
    if module:
      importlib.import_module(module)


def _get_lines_for_tsv() -> List[List[str]]:
  """Gets a TSV of provenance for all registered & tracked Tasks/Mixtures.

  Returns:
    List of TSV lines with header line.
    Columns of TSV lines are:
    provider_name    provider_type    filename_1:lineno    filename_2:lineno...

  Raises:
    ValueError if no provenances are found via
    task_registry_provenance_tracking.view_provenances()
  """
  if not task_registry_provenance_tracking.view_provenances():
    raise ValueError(
        "No task/mixture registrations found. Make sure you have turned on"
        " task_registry_provenance_tracking.turn_on_tracking() before importing"
        " any modules that may be registering tasks."
    )

  # The TSV manipulation could be more easily done with a tool like pandas,
  # but we choose instead to avoid bringing in pandas as a dependency for
  # just this small use.
  tsv_lines = []
  for (
      provider_name,
      (provider_type, provenance),
  ) in task_registry_provenance_tracking.view_provenances().items():
    num_columns_for_call_sites = len(provenance)
    current_line = [provider_name, provider_type] + provenance
    tsv_lines.append(current_line)

  header = ["provider_name", "provider_type"] + [
      f"call_site_{i}" for i in range(num_columns_for_call_sites)
  ]
  tsv_lines.insert(0, header)
  return tsv_lines


def _write_lines(tsv_lines: List[List[str]], out_path: str) -> None:
  with tf.io.gfile.GFile(out_path, "w") as f:
    for line in tsv_lines:
      f.write("\t".join(line) + "\n")


def main(_) -> None:
  task_registry_provenance_tracking.turn_on_tracking()
  _import_modules(FLAGS.registration_module_import)

  tsv_lines = _get_lines_for_tsv()
  _write_lines(tsv_lines, FLAGS.registration_provenance_out_filename)


if __name__ == "__main__":
  app.run(main)
