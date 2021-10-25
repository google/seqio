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

"""MetricValue objects to wrap results being returned by metric funcitons."""

import dataclasses
from typing import Optional, Union

import numpy as np
import tensorflow.compat.v2 as tf


@dataclasses.dataclass
class MetricValue:
  """A base method for the dataclasses that represent tensorboard values.

  Task `metric_fn`s should output `Mapping[str, MetricValue]` which will be
  written by a `Logger`.
  """


@dataclasses.dataclass
class Scalar(MetricValue):
  """The default tensorflow value, used for creating time series graphs."""
  value: Union[int, float]


@dataclasses.dataclass
class Text(MetricValue):
  """Text to output to tensorboard, markdown is rendered by tensorboard."""
  textdata: Union[str, bytes]


@dataclasses.dataclass
class Image(MetricValue):
  """An image to output to tensorboard.

  The format for the image array should match the format expected for the data
  parameter described
  [here](https://www.tensorflow.org/api_docs/python/tf/summary/image).
  """
  image: np.ndarray
  max_outputs: int = 3


@dataclasses.dataclass
class Audio(MetricValue):
  """An audio example to output to tensorboard.

  The format for the audio array should match the format expected for the data
  parameter described
  [here](https://www.tensorflow.org/api_docs/python/tf/summary/audio).
  """
  audiodata: np.ndarray
  sample_rate: int = 44100
  max_outputs: int = 3


@dataclasses.dataclass
class Histogram(MetricValue):
  """A histogram to output to tensorboard."""
  values: np.ndarray
  bins: Optional[int] = None


@dataclasses.dataclass
class Generic(MetricValue):
  """A raw tensor to output to tensorboard."""
  tensor: np.ndarray
  metadata: tf.compat.v1.SummaryMetadata
