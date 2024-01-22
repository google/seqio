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

"""Utils for tracking where TaskRegistry.add was called.

TaskRegistry.add calls often exist at the module level, and as such,
this code is run at module import time. This usage pattern can make
it difficult to determine where a particular task or mixture is being
registered. This module helps to alleviate difficulty with this pattern.
"""

import inspect
import itertools
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple


# Type of inspect.currentframe()
Frame = Any


# TaskRegistry.add calls often exist at the module level, and as such,
# this code is run at module import time. This usage pattern can make
# it difficult to determine where a particular task or mixture is being
# registered. This constant determines the number of frames to look
# backward when TaskRegistry is tracking registration provenance.
# See turn_on_registration_provenance_tracking.
_MAX_NUM_FRAMES_TO_CAPTURE_FOR_PROVENANCE = 5

# pyformat: disable
_ProvenanceLookupType = Dict[
        str,            # Task or Mixture name in Registry.
        Tuple[
            str,        # Provider class type. "Task" or "Mixture" etc.
            List[str],  # List of filename:lineno where Task was registered.
        ]
    ]
# pyformat: enable
# If None, no tracking is done. Else, a dict from task/mixture name to
# (provider_type, provenance). See full type annotation for more description.
_PROVIDER_PROVENANCE_LOOKUP: Optional[_ProvenanceLookupType] = None


def turn_on_tracking() -> None:
  """Adds attribute to track where tasks/mixtures were registered."""
  global _PROVIDER_PROVENANCE_LOOKUP
  if _PROVIDER_PROVENANCE_LOOKUP is None:
    _PROVIDER_PROVENANCE_LOOKUP = {}
  _PROVIDER_PROVENANCE_LOOKUP = {}


def registration_provenance_tracking_is_on() -> bool:
  return _PROVIDER_PROVENANCE_LOOKUP is not None


def maybe_record_provenance(
    frame: Frame, name: str, provider_type: str
) -> None:
  if registration_provenance_tracking_is_on():
    # Disable pylint check because in this case, we know that
    # _PROVIDER_PROVENANCE_LOOKUP is a dict.
    _PROVIDER_PROVENANCE_LOOKUP[name] = (  # pylint: disable=unsupported-assignment-operation
        provider_type,
        informative_upstream_callsites_from_frame(frame),
    )


def _frame_to_neat_filename(frame: Frame) -> str:
  """Frame to filename.py, possibly excluding long, unuseful prefixes."""
  module = inspect.getmodule(frame)
  if module:
    to_return = module.__file__
    return to_return
  raise ValueError(
      "Expected to get the module of the frame"
      f" {frame=} {frame.f_code.co_filename=}, but was unable to do so."
  )


def _frame_to_neat_filename_and_line(frame: Frame) -> str:
  """Frame to filename.py:line_number."""
  filename = _frame_to_neat_filename(frame)
  return f"{filename}:{frame.f_lineno}"


def _f_back_iterator(frame: Frame) -> Iterable[Frame]:
  while frame:
    yield frame
    frame = frame.f_back


def _frame_likely_informative_for_provenance(frame: Frame) -> bool:
  """Guesses whether frame is in internal util code vs external callsite."""
  filename = frame.f_code.co_filename
  return not (
      "t5/data/dataset_providers.py" in filename
      or "seqio/dataset_providers.py" in filename
      or "seqio/task_registry_provenance_tracking.py" in filename
      or "importlib" in filename
      or "embedded module" in filename
  )


def informative_upstream_callsites_from_frame(
    cur_frame: Frame,
    max_num_frames: int = _MAX_NUM_FRAMES_TO_CAPTURE_FOR_PROVENANCE,
) -> List[str]:
  """Computes [filename.py:line_number] for registration provenance tracking.

  Args:
    cur_frame: e.g. output of inspect.currentframe()
    max_num_frames: the maximum number of filename.py:line_number to return.
      This parameter affects the length of time this function takes to run.

  Returns:
    List of filename.py:line_number, starting with the frame closest to
    TaskRegistry.add.
  """
  back_frame_itr = _f_back_iterator(cur_frame)
  informative_frame_itr = itertools.islice(
      (
          x
          for x in back_frame_itr
          if _frame_likely_informative_for_provenance(x)
      ),
      max_num_frames,
  )
  informative_frames = [*informative_frame_itr]
  informative_frame_locs = [
      _frame_to_neat_filename_and_line(x) for x in informative_frames
  ]

  return informative_frame_locs


def view_provenances() -> _ProvenanceLookupType:
  return (
      # Copy dict to prevent outside consumers from modifying private
      # attribute.
      # Disable pylint check because in this case we know that we have
      # a dictionary.
      dict(**_PROVIDER_PROVENANCE_LOOKUP)  # pylint: disable=not-a-mapping
      if isinstance(_PROVIDER_PROVENANCE_LOOKUP, dict)
      else {}
  )
