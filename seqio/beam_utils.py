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

"""SeqIO Beam utilities."""

import functools
import importlib
import json
import operator
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from absl import logging
import apache_beam as beam
import apache_beam.metrics as metrics
import numpy as np
import seqio
import tensorflow.compat.v2 as tf

PROVENANCE_PREFIX = "provenance/"
TASK_PROVENANCE_KEY = PROVENANCE_PREFIX + "task"
SOURCE_SHARD_PROVENANCE_KEY = PROVENANCE_PREFIX + "source_shard"
SOURCE_SHARD_ID_PROVENANCE_KEY = PROVENANCE_PREFIX + "source_shard_index"
ID_WITHIN_SHARD_PROVENANCE_KEY = PROVENANCE_PREFIX + "index_within_shard"
PREPROCESSORS_SEED_PROVENANCE_KEY = PROVENANCE_PREFIX + "preprocessors_seed"
PROVENANCE_KEYS = [
    TASK_PROVENANCE_KEY, SOURCE_SHARD_PROVENANCE_KEY,
    SOURCE_SHARD_ID_PROVENANCE_KEY, ID_WITHIN_SHARD_PROVENANCE_KEY,
    PREPROCESSORS_SEED_PROVENANCE_KEY
]


def _import_modules(modules):
  for module in modules:
    if module:
      importlib.import_module(module)


class PreprocessTask(beam.PTransform):
  """Abstract base class to preprocess a Task.

  Returns a PCollection of example dicts containing Tensors.
  """

  def __init__(self,
               task: seqio.Task,
               split: str,
               *,
               preprocessors_seed: Optional[int] = None,
               modules_to_import: Sequence[str] = (),
               add_provenance: bool = False,
               tfds_data_dir: Optional[str] = None):
    """BasePreprocessTask constructor.

    Args:
      task: Task, the task to process.
      split: string, the split to process.
      preprocessors_seed: (Optional) int, a seed for stateless random ops in
        task preprocessing.
      modules_to_import: (Optional) list, modules to import.
      add_provenance: If True, provenance is added to each example.
      tfds_data_dir: (Optional) str, directory used to store datasets.
    """
    self._task = task
    self._split = split
    self._preprocessors_seed = preprocessors_seed
    self._modules_to_import = modules_to_import
    self._add_provenance = add_provenance
    self._tfds_data_dir = tfds_data_dir
    self.shards = list(enumerate(task.source.list_shards(split)))
    logging.info("%s %s shards: %s", task.name, split,
                 ", ".join(["%s" % f[1] for f in self.shards]))

  def _increment_counter(self, name):
    metrics.Metrics.counter(
        str("%s_%s" % (self._task.name, self._split)), name).inc()

  def _emit_examples(self, shard: Tuple[int, str]):
    """Emits examples keyed by shard number and index for a single shard."""
    _import_modules(self._modules_to_import)

    if self._tfds_data_dir:
      seqio.set_tfds_data_dir_override(self._tfds_data_dir)

    shard_index, shard_name = shard
    logging.info("Processing shard: %s", shard_name)
    self._increment_counter("input-shards")

    ds = self._task.source.get_dataset(
        split=self._split,
        shard_info=seqio.ShardInfo(
            index=shard_index, num_shards=len(self.shards)),
        shuffle=False)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Create a unique, deterministic preprocessors seed for each task and shard.
    shard_preprocessors_seed = shard_index + (self._preprocessors_seed or 0)

    ds = self._task.preprocess_precache(ds, seed=shard_preprocessors_seed)

    def _add_provenance(index_within_shard: int, ex: Dict[str, Any]):
      ex.update({
          TASK_PROVENANCE_KEY: self._task.name,
          SOURCE_SHARD_PROVENANCE_KEY: shard_name,
          SOURCE_SHARD_ID_PROVENANCE_KEY: shard_index,
          ID_WITHIN_SHARD_PROVENANCE_KEY: index_within_shard
      })
      if self._preprocessors_seed:
        ex.update({PREPROCESSORS_SEED_PROVENANCE_KEY: self._preprocessors_seed})
      return ex

    for i, ex in enumerate(ds.as_numpy_iterator()):
      if self._add_provenance:
        ex = _add_provenance(i, ex)
      self._increment_counter("examples")
      # Log every power of two.
      if i & (i - 1) == 0:
        logging.info("Example [%d] = %s", i, ex)
      yield ex

  def expand(self, pipeline):
    return (pipeline
            | "create_shards" >> beam.Create(self.shards)
            | "emit_examples" >> beam.FlatMap(self._emit_examples))


class WriteExampleTfRecord(beam.PTransform):
  """Writes examples (dicts) to a TFRecord of tf.Example protos."""

  def __init__(self, output_path: str, num_shards: Optional[int] = None):
    """WriteExampleTfRecord constructor.

    Args:
      output_path: string, path to the output TFRecord file (w/o shard suffix).
      num_shards: (optional) int, number of shards to output or None to use
        liquid sharding.
    """
    self._output_path = output_path
    self._num_shards = num_shards

  def expand(self, pcoll):
    return (pcoll
            | beam.Map(seqio.dict_to_tfexample)
            | beam.Reshuffle()
            | beam.io.tfrecordio.WriteToTFRecord(
                self._output_path,
                num_shards=self._num_shards,
                coder=beam.coders.ProtoCoder(tf.train.Example)))


class WriteJson(beam.PTransform):
  """Writes datastructures to file as JSON(L)."""

  def __init__(self, output_path: str, prettify: Optional[bool] = True):
    """WriteJson constructor.

    Args:
      output_path: string, path to the output JSON(L) file.
      prettify: bool, whether to write the outputs with sorted keys and
        indentation. Note this not be used if there are multiple records being
        written to the file (JSONL).
    """
    self._output_path = output_path
    self._prettify = prettify

  def _jsonify(self, el):
    if self._prettify:
      return json.dumps(el, sort_keys=True, indent=2)
    else:
      return json.dumps(el)

  def expand(self, pcoll):
    return (pcoll
            | beam.Map(self._jsonify)
            | "write_info" >> beam.io.WriteToText(
                self._output_path, num_shards=1, shard_name_template=""))


class GetInfo(beam.PTransform):
  """Computes info for dataset examples.

  Expects a single PCollections of examples.
  Returns a dictionary with information needed to read the data (number of
  shards, feature shapes and types)
  """

  def __init__(self, num_shards: int, exclude_provenance: bool = True):
    self._num_shards = num_shards
    self._exclude_provenance = exclude_provenance

  def _info_dict(self, ex: List[Dict[str, Any]]):
    if not ex:
      return {}
    assert len(ex) == 1
    ex = ex[0]
    info = {
        "num_shards": self._num_shards,
        "features": {},
        "seqio_version": seqio.__version__,
    }
    feature_dict = info["features"]
    for k, v in ex.items():
      if self._exclude_provenance and k.startswith(PROVENANCE_PREFIX):
        continue
      t = tf.constant(v)
      dtype = t.dtype.name
      shape = t.shape.as_list()
      # Keep all the dimensions but the first if t is not a scalar.
      if shape:
        shape = [None] + shape[1:]
      feature_dict[k] = {"shape": shape, "dtype": dtype}
    return info

  def expand(self, pcoll):
    return (pcoll
            | beam.combiners.Sample.FixedSizeGlobally(1)
            | beam.Map(self._info_dict))


class _CountTokens(beam.DoFn):
  """Returns token counts for each feature."""

  def __init__(self, output_features: Mapping[str, seqio.Feature]):
    self._output_features = output_features

  def setup(self):
    # Certain vocabularies are lazy loaded. Since we are running under beam we
    # try to do the loading only once in the setup phase.
    for feat in self._output_features.values():
      v = feat.vocabulary.eos_id
      v = feat.vocabulary.unk_id
      v = feat.vocabulary.pad_id
      del v

  def process(self, ex: Mapping[str, Any]) -> Iterable[Tuple[str, int]]:
    for name, feat in self._output_features.items():
      if (name in ex and isinstance(ex[name], np.ndarray) and
          ex[name].dtype in (np.int32, np.int64)):
        values = ex[name]
        conditions = []
        if feat.vocabulary.eos_id is not None:
          conditions.append((values != feat.vocabulary.eos_id))
        if feat.vocabulary.pad_id is not None:
          conditions.append((values != feat.vocabulary.pad_id))

        if conditions:
          valid_tokens = functools.reduce(operator.and_, conditions)
        else:
          # Assumes all values are valid tokens.
          valid_tokens = np.ones_like(values, dtype=bool)

        num_tokens = int(np.sum(valid_tokens))
        yield (f"{name}_tokens", num_tokens)


class _CountCharacters(beam.DoFn):
  """Returns character counts for each feature.

  This works with both tokenized (integer array) dataset and string dataset. For
  the former, each feature is detokenized and the string length is computed. For
  the latter, the bytes feature is decoded and the string length of that is
  returned.

  Example 1 (tokenized dataset):
  ```python

    Assume that these examples are generated with a vocab that decodes "ea" as
    [4, 5], etc.

    input_examples = [{
        # Decoded as "ea", i.e., length 2 string
        "inputs": np.array([4, 5]),
        # Decoded as "ea test", i.e., length 7 string
        "targets": np.array([4, 5, 10]),
    }, {
        # Decoded as "e", i.e., length 1 string
        "inputs": np.array([4]),
        # Decoded as "asoil", i.e., length 5 string. "1" is an EOS id.
        "targets": np.array([5, 6, 7, 8, 9, 1])
    }]

    This `DoFn` returns (yields each of the 4 elements in sequence):
      [("inputs_chars", 2), ("targets_chars", 7),
       ("inputs_chars", 1), ("targets_chars", 5)]
  ```

  Example 2 (string dataset):
  ```python

    input_examples = [{
        "text": b"this is a string of length 29"
    }, {
        "text": b"this is another string of length 35"
    }]

    This `DoFn` returns (yields each of the 2 elements in sequence):
      [("text_chars", 29), ("text_chars", 35)]
  ```
  """

  def __init__(self, output_features: Mapping[str, seqio.Feature]):
    self._output_features = output_features

  def setup(self):
    # Certain vocabularies are lazy loaded. Since we are running under beam we
    # try to do the loading only once in the setup phase.
    for feat in self._output_features.values():
      v = feat.vocabulary.eos_id
      v = feat.vocabulary.unk_id
      v = feat.vocabulary.pad_id
      del v

  def process(self, ex: Mapping[str, Any]) -> Iterable[Tuple[str, int]]:
    for name, feat in self._output_features.items():
      # We only compute the character length for the rank-1 integer array for
      # the feature using `seqio.SentencePieceVocabulary`.
      if (name in ex and isinstance(ex[name], np.ndarray) and
          ex[name].dtype in (np.int32, np.int64) and feat.rank == 1 and
          isinstance(feat.vocabulary, seqio.SentencePieceVocabulary)):
        value = ex[name]
        value = np.abs(value.astype(np.int32))
        decoded = feat.vocabulary.decode_tf(value).numpy().decode("utf-8")

      # If each example in the dataset has the type tf.string, its type
      # becomes `bytes` inside the `ds.as_numpy_iterator()`. This `DoFn` is
      # assumed to be applied to  the examples from such numpy iterator.
      elif name in ex and isinstance(ex[name], bytes):
        decoded = ex[name].decode("utf-8")
      else:
        continue

      yield (f"{name}_chars", len(decoded))


class GetStats(beam.PTransform):
  """Computes statistics for dataset examples.

  The `expand` method expects a PCollection of examples where each example is a
  dictionary of string identifiers (e.g. "inputs" and "targets") mapped to numpy
  array.

  Returns a dictionary with statistics (number of examples, number of tokens)
  prefixed by the identifiers.
  """

  def __init__(self,
               output_features: Mapping[str, seqio.Feature],
               task_ids: Optional[Mapping[str, Any]] = None):
    self._output_features = output_features
    self._task_ids = task_ids or {}

  def expand(self, pcoll):
    example_counts = (
        pcoll
        | "count_examples" >> beam.combiners.Count.Globally()
        | "key_example_counts" >> beam.Map(lambda x: ("examples", x))
        | "example_count_dict" >> beam.combiners.ToDict())

    token_counts = pcoll | "count_tokens" >> beam.ParDo(
        _CountTokens(self._output_features))
    total_tokens = (
        token_counts
        | "sum_tokens" >> beam.CombinePerKey(sum)
        | "token_count_dict" >> beam.combiners.ToDict())
    max_tokens = (
        token_counts
        | "max_tokens" >> beam.CombinePerKey(max)
        | "rename_max_stat" >>
        beam.Map(lambda x: (x[0].replace("tokens", "max_tokens"), x[1]))
        | "token_max_dict" >> beam.combiners.ToDict())

    # Compute the character length
    char_length = (
        pcoll
        | beam.ParDo(_CountCharacters(self._output_features))
        | "sum_characters" >> beam.CombinePerKey(sum)
        | "character_length_dict" >> beam.combiners.ToDict())

    def _merge_dicts(dicts):
      merged_dict = {}
      for d in dicts:
        assert not set(merged_dict).intersection(d)
        merged_dict.update(d)
      return merged_dict

    stats = [example_counts, total_tokens, max_tokens, char_length]
    if self._task_ids:
      # ids could be Tensors, cast to int.
      self._task_ids = {k: int(v) for k, v in self._task_ids.items()}
      task_ids_dict = {"task_ids": self._task_ids}
      task_ids = (
          pcoll
          | "sample_for_task_ids" >> beam.combiners.Sample.FixedSizeGlobally(1)
          | "create_task_ids" >> beam.Map(lambda _: task_ids_dict))
      stats.append(task_ids)

    return (stats
            | "flatten_counts" >> beam.Flatten()
            | "merge_stats" >> beam.CombineGlobally(_merge_dicts))
