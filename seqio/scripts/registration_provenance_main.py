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
  --module_import=my.tasks
"""
import importlib

from absl import app
from absl import flags
from seqio import task_registry_provenance_tracking

FLAGS = flags.FLAGS
flags.DEFINE_multi_string(
    "module_import",
    [],
    (
        "Modules to import. Use this, for example, to add new `Task`s to the "
        "global `TaskRegistry`."
    ),
)


def _import_modules(modules):
  """Function that imports an additional module."""
  for module in modules:
    if module:
      importlib.import_module(module)


def main(_) -> None:
  task_registry_provenance_tracking.turn_on_tracking()
  _import_modules(FLAGS.module_import)

  tsv_lines = []
  for (
      provider_name,
      provenance,
  ) in task_registry_provenance_tracking.view_provenances().items():
    current_line = provider_name + "\t" + "\t".join(provenance)
    tsv_lines.append(current_line)

  print("\n".join(tsv_lines))


if __name__ == "__main__":
  app.run(main)
