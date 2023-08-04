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

"""Helper functions for lineage log utilization in the scope of seqio."""

from absl import logging

SEPARATOR = "__"
TASK_PARENT = "projects/experiment/datasets/seqio_task"


def create_task_entry(task):
  """Creates a task entry in Datahub via lineage log.

  Args:
    task: the task of type dataset_providers.Task to be piped through lineage
      log into Datahub as a new entry.
  """

  lineage_log.input(handle=_create_seqio_task_guri(task), intent="SEQIO_TASK")


def create_relationship_between_task_and_tfds_or_cns(task):
  """Creates an artifact to artifact relationship between a task and its respective datasource from TFDS or CNS."""

  print("dataset: ", task.source.tfds_dataset)
  if _create_tfds_guri_from_seqio_task(task):
    seqio_handle = lineage_log._SeqIOTaskHandle(  # pylint: disable=protected-access
        uri=_create_seqio_task_guri(task),
        data_source_path=_create_tfds_guri_from_seqio_task(task),
    )
  else:
    seqio_handle = lineage_log._SeqIOTaskHandle(  # pylint: disable=protected-access
        uri=_create_seqio_task_guri(task),
        data_source_path=_create_cns_guri_from_seqio_task(task),
    )

  lineage_log.input(handle=seqio_handle, intent="SEQIO_TASK_AND_TFDS_OR_CNS")


def _create_seqio_task_guri(task) -> str:
  """Creates a GUri for a seqio task."""
  task_local_id = _get_task_entry_local_id(task)

  return guri.ToString(
      guri_pb2.GUri(
          experimental=guri_pb2.Experimental(
              experimental_type="seqio_task",
              part1=task_local_id,
          )
      )
  )


def _create_tfds_guri_from_seqio_task(task) -> str | None:
  """Creates a GUri for a TFDS dataset from a seqio task."""
  if not task.source.tfds_dataset.name:
    return None

  tfds_path = task.source.tfds_dataset.name.replace(":", "/")
  tfds_namespace = datahub_lib.datahub_parent_for(tfds_path)

  return guri.ToString(
      guri_pb2.GUri(
          ml_data=guri_pb2.MlData(
              dataset_namespace=tfds_namespace.dataset, path=tfds_path
          )
      )
  )


def _create_cns_guri_from_seqio_task(task) -> str | None:
  """Creates a GUri for a CNS directory from a seqio task."""
  if not task.source.tfds_dataset.data_dir:
    return None

  return guri.ToString(
      guri_pb2.GUri(cns=guri_pb2.Cns(path=task.source.tfds_dataset.data_dir))
  )


def _get_task_entry_local_id(task) -> str | None:
  """Gets Datahub entry local id of the provided task.

  Args:
    task: Provided task.

  Returns:
    Entry id.
  """
  try:
    return task.name + SEPARATOR + _get_task_hash(task)
  except ValueError:
    logging.exception("Failed to generate task id.")
    return None


def _get_task_hash(task) -> str | None:
  """Gets the hash of a task.

  Args:
    task: task of type dataset_providers.Task

  Returns:
    Hash of the task

  Raises:
    ValueError: If the task contains unsupported data source.
  """

  source_description = ""
  if task.source.tfds_dataset.name:
    source_description = task.source.tfds_dataset.name + SEPARATOR
  if task.source.tfds_dataset.data_dir:
    source_description += task.source.tfds_dataset.data_dir

  if not source_description:
    raise ValueError(f"Unsupported data source! Task: {task}")

  # rate support can be added once mixture support is added

  return _compute_hash(source_description)


def _compute_hash(value: str) -> str:
  return str(fingerprint2011.Fingerprint2011(value.encode("utf-8")))


def _task_entry_id(entry_local_id: str) -> str:
  return f"{TASK_PARENT}/entries/{entry_local_id}"
