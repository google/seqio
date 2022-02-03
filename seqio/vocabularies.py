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

# Lint as: python3
"""Vocabularies."""

import abc
import hashlib
from typing import Any, Dict, Iterable, Optional, Sequence, Union

from absl import logging
import cached_property
import tensorflow.compat.v2 as tf
import tensorflow_text as tf_text

from sentencepiece import sentencepiece_model_pb2
import sentencepiece as sentencepiece_processor

PAD_ID = 0


class Vocabulary(metaclass=abc.ABCMeta):
  """Abstract class for all vocabularies.

  Subclasses must implement methods for converting between strings and tokens
  both in pure python (`_encode`/`_decode`) and in TensorFlow
  (`_encode_tf`/`_decode_tf`).

  Subclasses are responsible for reserving PAD_ID=0 as well as optionally
  reserving EOS_ID and UNK_ID

  `_base_vocab_size` should account for PAD, EOS, and UNK but not `extra_ids`.
  """

  def __init__(self, extra_ids: int = 0):
    """Vocabulary constructor.

    Args:
      extra_ids: The number of extra IDs to reserve.
    """
    self._extra_ids = extra_ids or 0

  @abc.abstractproperty
  def eos_id(self) -> Optional[int]:
    raise NotImplementedError("need to implement eos_id")

  @property
  def pad_id(self) -> int:
    return PAD_ID

  @abc.abstractproperty
  def unk_id(self) -> Optional[int]:
    raise NotImplementedError("need to implement unk_id")

  @property
  def extra_ids(self) -> int:
    return self._extra_ids

  @property
  def vocab_size(self) -> int:
    """Vocabulary size, including extra ids."""
    return self._base_vocab_size + self.extra_ids

  @abc.abstractproperty
  def _base_vocab_size(self) -> int:
    """Vocabulary size, excluding extra ids but including PAD/EOS/UNK."""
    # TODO(fjord): add a check that pad_id and unk_id (if present)
    #   are less than _base_vocab_size.
    raise NotImplementedError

  @abc.abstractmethod
  def _encode(self, s: str) -> Sequence[int]:
    raise NotImplementedError

  def encode(self, s: Union[Sequence[int], str]) -> Sequence[int]:
    """Tokenizes string to an int sequence, without adding EOS."""
    return self._encode(s)

  @abc.abstractmethod
  def _decode(self, ids):
    raise NotImplementedError

  def decode(self, ids: Iterable[int]):
    """Detokenizes int32 iterable to a string, up through first EOS."""
    clean_ids = list(ids)

    if self.unk_id is not None:
      vocab_size = self._base_vocab_size
      clean_ids = [self.unk_id if i >= vocab_size else i for i in clean_ids]

    if self.eos_id is not None and self.eos_id in clean_ids:
      clean_ids = clean_ids[:clean_ids.index(self.eos_id) + 1]

    return self._decode(clean_ids)

  @abc.abstractmethod
  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    """Tokenizes string Scalar to an int32 Tensor, without adding EOS."""
    return self._encode_tf(s)

  @abc.abstractmethod
  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    """Detokenizes int32 batched Tensor through first EOS."""
    clean_ids = ids

    if self.unk_id is not None:
      clean_ids = tf.where(
          tf.less(clean_ids, self._base_vocab_size), clean_ids, self.unk_id)

    if self.eos_id is not None:
      # Replace everything after the first eos_id with pad_id.
      after_eos = tf.cumsum(
          tf.cast(tf.equal(clean_ids, self.eos_id), tf.int32),
          exclusive=True,
          axis=-1)
      clean_ids = tf.where(tf.cast(after_eos, tf.bool), self.pad_id, clean_ids)

    return self._decode_tf(clean_ids)


class PassThroughVocabulary(Vocabulary):
  """Vocabulary that passes through inputs unchanged."""

  def __init__(self, size: int, eos_id: Optional[int] = None):
    """PassThroughVocabulary constructor.

    Args:
      size: the full size of the vocabulary.
      eos_id: the end-of-sequence token.
    """
    self._size = size
    self._eos_id = eos_id
    super().__init__()

  @property
  def _base_vocab_size(self):
    return self._size

  def _encode(self, s: Sequence[int]) -> Sequence[int]:
    return s

  def _decode(self, ids: Sequence[int]) -> Sequence[int]:
    return ids

  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    return s

  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    return ids

  @property
  def eos_id(self) -> Optional[int]:
    return self._eos_id

  @property
  def unk_id(self) -> Optional[int]:
    return None

  def __eq__(self, other):
    if not isinstance(other, PassThroughVocabulary):
      return False
    return self._size == other._size and self.eos_id == other.eos_id


class SentencePieceVocabulary(Vocabulary):
  """Wrapper for nlp/sentencepiece encoder.

  Assumes the model was built using flags to reserve ID=0 for padding, ID=1 for
  EOS, and ID=2 for UNK.

  If using extra ids, you can represent them in string-form as `<extra_id_0>`,
  `<extra_id_1>`, etc. They will be indexed starting from the end of the
  vocabulary to match how the masking preprocessors are set up.

  IMPORTANT NOTE: these placeholders only work properly when they are used at
  word starts (e.g., "I like peanut butter and <extra_id_0> sandwiches." or
  "I like peanut butter and <extra_id_0>ly sandwiches" are both okay, but
  "I like peanut butter and jel<extra_id_0> sandwiches is not.").
  """

  def __init__(self,
               sentencepiece_model_file,
               extra_ids=None,
               force_preserve_repeated_whitespace=False):
    """Create a SentencePieceVocabulary.

    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels.

    Args:
      sentencepiece_model_file: a string
      extra_ids: an optional integer
      force_preserve_repeated_whitespace: if True, whitespaces are preserved
        regardless of how the SPM model was trained.
    """
    self._sentencepiece_model_file = sentencepiece_model_file
    self._tokenizer = None
    self._sp_model = None
    self.force_preserve_repeated_whitespace = force_preserve_repeated_whitespace
    super().__init__(extra_ids=extra_ids)

  def _load_model(self):
    """Load SPM and Python tokenizer."""
    # Handle cases where SP can't load the file, but gfile can.
    with tf.io.gfile.GFile(self._sentencepiece_model_file, "rb") as f:
      self._sp_model = f.read()
      # Add placeholder strings for extra IDs.
      model = sentencepiece_model_pb2.ModelProto.FromString(self._sp_model)
      if self.force_preserve_repeated_whitespace:
        model.normalizer_spec.remove_extra_whitespaces = False
        model.denormalizer_spec.remove_extra_whitespaces = False
      if self._extra_ids:
        # We name them in reverse order to match their use in span corruption.
        for i in reversed(range(self._extra_ids)):
          model.pieces.add(
              piece=f"▁<extra_id_{i}>",
              score=0.0,
              type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED
          )
      self._sp_model = model.SerializeToString()
    # Load Python tokenizer and ensure the EOS and PAD IDs are correct.
    self._tokenizer = sentencepiece_processor.SentencePieceProcessor()
    self._tokenizer.LoadFromSerializedProto(self._sp_model)
    if self._tokenizer.pad_id() != PAD_ID:
      logging.warning(
          "T5 library uses PAD_ID=%s, which is different from the "
          "sentencepiece vocabulary, which defines pad_id=%s", PAD_ID,
          self._tokenizer.pad_id())

  @property
  def eos_id(self) -> Optional[int]:
    return self.tokenizer.eos_id()

  @property
  def unk_id(self) -> Optional[int]:
    return self.tokenizer.unk_id()

  @property
  def sp_model(self):
    """Retrieve the SPM."""
    if self._sp_model is None:
      self._load_model()
    return self._sp_model

  @property
  def sentencepiece_model_file(self):
    return self._sentencepiece_model_file

  @property
  def tokenizer(self):
    """Returns the Python tokenizer."""
    if not self._tokenizer:
      self._load_model()
    return self._tokenizer

  @property
  def tf_tokenizer(self):
    """Instantiate and return a TF tokenizer."""
    return tf_text.SentencepieceTokenizer(model=self.sp_model)

  @property
  def vocab_size(self):
    return self._base_vocab_size

  @property
  def _base_vocab_size(self):
    """Number of ids (including 0=PAD, 1=EOS, and 2=UNK).

    Returns:
      an integer, the vocabulary size
    """
    return self.tokenizer.GetPieceSize()

  def _encode(self, s):
    """Encode a python string as a list of integers.

    Args:
      s: a string

    Returns:
      a list of integers (not terminated by EOS)
    """
    return self.tokenizer.EncodeAsIds(s)

  def _decode(self, ids):
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers (not terminated by EOS)

    Returns:
      a string
    """
    # convert all the extra ids (sentinels) to UNK=2
    ids = [
        self.tokenizer.unk_id() if i >= self.tokenizer.GetPieceSize() else i
        for i in ids
    ]
    return self.tokenizer.DecodeIds(ids)

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    This will be necessary for on-the-fly tokenization.

    Args:
      s: a tf.Scalar with dtype tf.string

    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    return self.tf_tokenizer.tokenize(s)

  def _decode_tf(self, ids):
    """Decode in TensorFlow.

    Args:
      ids: a 1d tf.Tensor with dtype tf.int32

    Returns:
      a tf Scalar with dtype tf.string
    """
    return self.tf_tokenizer.detokenize(ids)

  def __eq__(self, other):
    if not isinstance(other, SentencePieceVocabulary):
      return False
    try:
      their_md5 = hashlib.md5(other.sp_model).hexdigest()
    # If other has no sp_model attribute, we can't test for equality
    except AttributeError:
      return False
    our_md5 = hashlib.md5(self.sp_model).hexdigest()
    return our_md5 == their_md5

  def __str__(self) -> str:
    return (f"SentencePieceVocabulary(file={self._sentencepiece_model_file}, "
            f"extra_ids={self.extra_ids}, "
            f"spm_md5={hashlib.md5(self.sp_model).hexdigest()})")


class ClassificationSentencePieceVocabulary(SentencePieceVocabulary):
  """Wrapper over an existing SPM vocab.

  Use this vocabulary when you are trying to cast multi-class classification
  into a text-to-text format, but require each class to correspond to a single
  token. When class labels are multi-token, its not clear what the best way is
  to calculate the prediction probabilities. Single token classes make it
  easier to extract the probability scores  for each class. This class requries
  the input text to always a class label. During encoding, the class label is
  mapped to  a token in the SPM  vocab. During decoding, the class label
  corresponding to the token id is  returned. When defining a task, replace the
  output_vocabulary with  ClassificationSentencePieceVocabulary.
  """

  def __init__(self,
               sentencepiece_model_file,
               extra_ids=None,
               class_labels=None,
               use_default_ids=False):
    """Create a SentencePieceVocabulary.

    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels.

    Args:
      sentencepiece_model_file: path to sentencepiece model.
      extra_ids: an optional integer.
      class_labels: a list of class labels. Requires len(class_labels) <= size
        of the SPM vocab.
      use_default_ids: if True, the token ids corresponding to the class_labels
        will be used.  The user should only give labels that correspond to
        single token pieces. If set to False, each label is mapped to either
        extra_id tokens or tokens that are "rare" in the SPM corpus.
    """
    super().__init__(sentencepiece_model_file, extra_ids)
    if use_default_ids:
      self._class_label_to_vocab_token = {}
      for label in class_labels:
        token_ids = self.tokenizer.EncodeAsIds(label)
        if len(token_ids) != 1:
          raise ValueError(
              "If use_default_ids=True, class labels should correspond to"
              f"single token pieces only. However, {label} corresponds to"
              f"{token_ids}")
        self._class_label_to_vocab_token[label] = token_ids[0]
    else:
      num_pieces = self.tokenizer.GetPieceSize()
      # Tokens are already sorted by corpus frequency / score.
      # We start using them in decreasing order of their corpus frequency.
      # The most rare token is chosen for the first class label,
      # then second most rare token etc.
      # One exception is when the vocab has extra_ids. These are typically
      # tacked to the end of the vocab and will be used first,
      # followed by the non extra_ids.
      all_token_ids = [
          token_id for token_id in list(reversed(range(0, num_pieces)))
          if token_id not in [self.pad_id, self.unk_id, self.eos_id]
      ]
      # Chose as many tokens as there are classes.
      chosen_token_ids = all_token_ids[:len(class_labels)]
      self._class_label_to_vocab_token = {
          label: self.tokenizer.DecodeIds([token_id])
          for token_id, label in zip(chosen_token_ids, class_labels)
      }
      self._vocab_token_to_class_label = {
          v: k for k, v in self._class_label_to_vocab_token.items()
      }
    self.class_labels = list(self._class_label_to_vocab_token.keys())

  def _encode(self, s):
    return super()._encode(self._class_label_to_vocab_token[s])

  def _decode(self, ids):
    text = super()._decode(ids)
    return self._vocab_token_to_class_label.get(text, text)

  def _encode_tf(self, s):
    keys = tf.constant(list(self._class_label_to_vocab_token))
    values = tf.constant(list(self._class_label_to_vocab_token.values()))
    s = self.table_lookup(keys, values, s)
    return super()._encode_tf(s)

  def _decode_tf(self, ids):
    text = super()._decode_tf(ids)
    keys = tf.constant(list(self._vocab_token_to_class_label))
    values = tf.constant(list(self._vocab_token_to_class_label.values()))
    # If text is present in keys, its a vocab token that maps to a class label,
    # so we return the class label. If the text is not present in keys, it does
    # not correspond to any class label, so we return text as it is.
    return tf.cond(
        tf.equal(tf.size(tf.where(tf.equal(keys, text))), 0), lambda: text,
        lambda: tf.constant(self.table_lookup(keys, values, text)))

  def table_lookup(self, keys, values, query):
    """The TensorFlow equivalent of: {k: v for k,v in zip(keys, values)}[query].

    Note: this is an alternative to tf.lookup.StaticHashTable, for situations
    when the lookup table is very small, or when we are not able to run the
    initialization op that is necessary to use tf.lookup.StaticHashTable.

    Args:
      keys: a 1-D array of lookup table keys (TF Tensor).
      values: a 1-D array of lookup table values (TF Tensor).
      query: a scalar value to lookup in the table (TF Tensor).

    Returns:
      a scalar value, one of the elements in `values`.
    """
    occurrences = tf.where(tf.equal(keys, query))
    first_occurrence_idx = tf.reduce_min(occurrences)
    return values[first_occurrence_idx]

  @cached_property.cached_property
  def class_ids(self):
    return [
        self.tokenizer.EncodeAsIds(self._class_label_to_vocab_token[label])[0]
        for label in self.class_labels
    ]


class ByteVocabulary(Vocabulary):
  """Byte-level vocabulary.

  Encode/decode text directly to 256 "byte IDs" using UTF-8 encoding. Three
  special IDs are reserved (0=padding, 1=EOS, 2=UNK), so our encoded byte IDs
  are +3 greater than UTF-8 byte values.

  This is the vocabulary used by the ByT5 models:
  https://arxiv.org/abs/2105.13626
  """

  def __init__(self, extra_ids: int = 0):
    """Create a ByteVocabulary.

    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels.

    Args:
      extra_ids: an optional integer
    """
    self._byte_size = 256
    # The special tokens: 0=PAD, 1=EOS,and 2=UNK
    self._num_special_tokens = 3
    super().__init__(extra_ids=extra_ids)

  @property
  def eos_id(self) -> Optional[int]:
    return 1

  @property
  def unk_id(self) -> Optional[int]:
    return 2

  def _convert_strings_to_ids(self, s):
    """Convert a python string to integers based on UTF-8 encoding.

    Args:
      s: a string

    Returns:
      ids: a list of integers
    """
    return list(s.encode("utf-8"))

  def _convert_ids_to_strings(self, ids):
    """Convert ids to a python string based on UTF-8 encoding.

    Args:
      ids: a list of integers

    Returns:
      s: a string
    """
    return bytes(ids).decode("utf-8", errors="ignore")

  def _filter_non_string_ids(self, ids):
    """Filter special token ids and extra ids if there are any.

    Args:
      ids: a list of integers

    Returns:
      ids: a list of integers
    """
    lower_bound = self._num_special_tokens
    upper_bound = self._byte_size + self._num_special_tokens
    return [id for id in ids if lower_bound <= id < upper_bound]

  @property
  def _base_vocab_size(self):
    """Number of ids.

    Returns:
      an integer, the vocabulary size
    """
    return self._num_special_tokens + self._byte_size

  def _encode(self, s):
    """Encode a python string as a list of integers.

    To keep the first few ids for special tokens, increase ids by the number
    of special tokens.

    Args:
      s: a string

    Returns:
      a list of integers (not terminated by EOS)
    """
    ids = self._convert_strings_to_ids(s)
    return [i + self._num_special_tokens for i in ids]

  def _decode(self, ids):
    """Decode a list of integers to a python string.

    The special tokens of PAD, EOS, and UNK will not be represented in the
    output string. This is different from the SentencePieceVocabulary, where
    UNK will show up as a '?' character.

    Args:
      ids: a list of integers (not terminated by EOS)

    Returns:
      a string
    """

    ids = self._filter_non_string_ids(ids)
    ids = [i - self._num_special_tokens for i in ids]
    return self._convert_ids_to_strings(ids)

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    Args:
      s: a tf.Scalar with dtype tf.string

    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    tf_ids = tf.io.decode_raw(s, tf.uint8) + self._num_special_tokens
    return tf.dtypes.cast(tf_ids, tf.int32)

  def _decode_tf(self, ids):
    """Decode in TensorFlow.

    Args:
      ids: a 1d tf.Tensor with dtype tf.int32

    Returns:
      a tf Scalar with dtype tf.string
    """
    return tf.py_function(func=self.decode, inp=[ids], Tout=tf.string)

  def __eq__(self, other):
    if not isinstance(other, ByteVocabulary):
      return False
    return (self.extra_ids == other.extra_ids and
            self.eos_id == other.eos_id and self.unk_id == other.unk_id)


class FullCodepointVocabulary(Vocabulary):
  """Encodes and decodes text as codepoint sequences.

  This "vocabulary" is lexicon-free (i.e. it is static), and is an exhaustive
  representation of all codepoints. This is well-suited to encoders (especially
  with a hash-based embedding strategy) or a decoder that does not softmax over
  the whole vocabulary.

  A Unicode codepoint is effectively a single character. Unicode provides a
  well-defined mapping from the set of codepoint integers onto the set of all
  Unicode characters.
  """
  # While this should generally match `sys.maxunicode`, we want to provide this
  # as a constant to avoid architecture/system-dependent array overruns. If
  # downstream preprocessors choose to use `vocab_size-1` as a sentinel ID,
  # then this will still map such characters onto the Unicode private range on
  # planes 15-16. See:
  # https://en.wikipedia.org/wiki/Unicode#Code_planes_and_blocks.
  LARGEST_CODEPOINT = 0x10ffff  # Decimal: 1,114,111
  # Padding is always index zero. This means that the NULL character is
  # technically not embeddable. This seems fine according to all reasonable
  # interpretations of the NULL character as a past-end-of-string marker.
  PAD_CODEPOINT = 0
  # Special symbols are represented using codepoints values that are valid,
  # but designated as "Private Use", meaning that they will never by assigned
  # characters by the Unicode Consortium, and are thus safe for use here.
  EOS_CODEPOINT = 0xE005

  @property
  def eos_id(self) -> int:
    return self.EOS_CODEPOINT

  @property
  def pad_id(self) -> int:
    return self.PAD_CODEPOINT

  @property
  def unk_id(self) -> Optional[int]:
    # Because `FullCodepointVocabulary` exhaustively embeds all codepoints
    # possible in Unicode, unknown characters are not possible.
    return None

  @property
  def _base_vocab_size(self) -> int:
    return self.LARGEST_CODEPOINT

  def _encode(self, s: str) -> Sequence[int]:
    return [ord(i) for i in s]

  def _decode(self, ids: Sequence[int]) -> str:
    return "".join(chr(id_) for id_ in ids if id_ != self.EOS_CODEPOINT)

  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    return tf.strings.unicode_decode(s, input_encoding="UTF-8")

  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    return tf.strings.unicode_encode(ids, output_encoding="UTF-8")

  def __eq__(self, other):
    return isinstance(other, FullCodepointVocabulary)


class PartialCodepointVocabulary(Vocabulary):
  """Encodes and decodes text as a fixed set of codepoints.

  A Unicode codepoint is effectively a single character. Unicode provides a
  well-defined mapping from the set of codepoint integers onto the set of all
  Unicode characters.

  Unlike `FullCodepointVocabulary`, this uses only a subset of codepoints which
  are read in from a provided file. The format of the file is as decimal
  integers, where each integer is the codepoint integer as defined by Unicode.
  These can be obtained in Python 3 by converting a single character `str` to
  an `int` using `codepoint = ord(char)`.

  This sort of vocabulary is especially useful for decoder vocabularies where
  one might want to control the size of the output softmax and for encoders
  that do not use a hash embedding strategy.
  """

  # Padding is always index zero. This means that the NULL character is
  # technically not embeddable. This seems fine according to all reasonable
  # interpretations of the NULL character as a past-end-of-string marker.
  PAD_CODEPOINT = FullCodepointVocabulary.PAD_CODEPOINT
  # Special symbols are represented using codepoints values that are valid,
  # but designated as "Private Use", meaning that they will never by assigned
  # characters by the Unicode Consortium, and are thus safe for use here.
  EOS_CODEPOINT = FullCodepointVocabulary.EOS_CODEPOINT
  UNK_CODEPOINT = 0xE004

  PAD_ID = 0
  EOS_ID = 1
  UNK_ID = 2

  def __init__(self, codepoints: Sequence[int], extra_ids: int = 0):
    """Format of vocab file assumes one codepoint per line."""
    self._codepoint_to_id = {
        self.PAD_CODEPOINT: self.PAD_ID,
        self.EOS_CODEPOINT: self.EOS_ID,
        self.UNK_CODEPOINT: self.UNK_ID,
    }
    for codepoint in codepoints:
      if codepoint not in self._codepoint_to_id:
        self._codepoint_to_id[codepoint] = len(self._codepoint_to_id)
    self._id_to_codepoint = {v: k for k, v in self._codepoint_to_id.items()}

    self._codepoint_to_id_tf = PartialCodepointVocabulary.convert_dict_to_tf(
        self._codepoint_to_id, default_value=self.UNK_ID)
    self._id_to_codepoint_tf = PartialCodepointVocabulary.convert_dict_to_tf(
        self._id_to_codepoint, default_value=self.unk_id)
    super().__init__(extra_ids=extra_ids)

  @classmethod
  def create_from_file(cls, vocab_file: str, extra_ids: int = 0):
    codepoint_list = []
    with tf.io.gfile.GFile(vocab_file, "r") as f:
      for line in f:
        codepoint_list.append(int(line.strip()))
    return cls(codepoint_list, extra_ids)

  @property
  def eos_id(self) -> int:
    return self.EOS_ID

  @property
  def pad_id(self) -> int:
    return self.PAD_ID

  @property
  def unk_id(self) -> int:
    return self.UNK_ID

  @property
  def _base_vocab_size(self) -> int:
    return len(self._codepoint_to_id)

  @staticmethod
  def convert_dict_to_tf(
      d: Dict[Any, Any],
      default_value: Optional[Any] = None) -> tf.lookup.StaticHashTable:
    keys_tensor = tf.constant(list(d))
    vals_tensor = tf.constant(list(d.values()))
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=default_value)

  def _encode(self, s: str) -> Sequence[int]:
    output_ids = []
    for c in s:
      codepoint = ord(c)
      output_ids.append(self._codepoint_to_id.get(codepoint, self.unk_id))
    return output_ids

  def _decode(self, ids: Sequence[int]) -> str:
    output_str = ""
    for id_ in ids:
      codepoint = self._id_to_codepoint.get(id_, self.UNK_CODEPOINT)
      if codepoint == self.EOS_CODEPOINT:
        continue
      output_str += chr(codepoint)
    return output_str

  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    return self._codepoint_to_id_tf[tf.strings.unicode_decode(
        s, input_encoding="UTF-8")]

  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    return tf.strings.unicode_encode(
        self._id_to_codepoint_tf[ids], output_encoding="UTF-8")

  def __eq__(self, other):
    if not isinstance(other, PartialCodepointVocabulary):
      return False
    return (self._codepoint_to_id == other._codepoint_to_id and
            self.extra_ids == other.extra_ids)


class BertWordPieceVocabulary(Vocabulary):
  """Wrapper for Bert wordpiece encoder.

  This "vocabulary" wraps the tensorflow_text's BertTokenizer, which applies an
  end-to-end, text string to wordpiece tokenization.
  """

  def __init__(self,
               vocab_lookup_table: str,
               suffix_indicator: str = "##",
               max_bytes_per_word: int = 100,
               max_chars_per_token: Optional[int] = None,
               token_out_type: tf.dtypes.DType = tf.dtypes.int64,
               unknown_token: str = "[UNK]",
               split_unknown_characters: bool = False,
               lower_case: bool = False,
               keep_whitespace: bool = False,
               normalization_form: Optional[str] = None,
               preserve_unused_token: bool = False,
               pad_id: int = 0,
               start_of_sequence_id: int = 101,
               end_of_sequence_id: int = 102):
    r"""Create a Bert WordPieceVocabulary.

    Args:
      vocab_lookup_table: A lookup table implementing the LookupInterface
        containing the vocabulary of subwords or a string which is the file path
        to the vocab.txt file.
      suffix_indicator: (optional) The characters prepended to a wordpiece to
        indicate that it is a suffix to another subword. Default is '##'.
      max_bytes_per_word: (optional) Max size of input token. Default is 100.
      max_chars_per_token: (optional) Max size of subwords, excluding suffix
        indicator. If known, providing this improves the efficiency of decoding
        long words.
      token_out_type: (optional) The type of the token to return. This can be
        `tf.int64` IDs, or `tf.string` subwords. The default is `tf.int64`.
      unknown_token: (optional) The value to use when an unknown token is found.
        Default is "[UNK]". If this is set to a string, and `token_out_type` is
        `tf.int64`, the `vocab_lookup_table` is used to convert the
        `unknown_token` to an integer. If this is set to `None`,
        out-of-vocabulary tokens are left as is.
      split_unknown_characters: (optional) Whether to split out single unknown
        characters as subtokens. If False (default), words containing unknown
        characters will be treated as single unknown tokens.
      lower_case: bool - If true, a preprocessing step is added to lowercase the
        text, apply NFD normalization, and strip accents characters.
      keep_whitespace: bool - If true, preserves whitespace characters instead
        of stripping them away.
      normalization_form: If set to a valid value and lower_case=False, the
        input text will be normalized to `normalization_form`. See
        normalize_utf8() op for a list of valid values.
      preserve_unused_token: If true, text in the regex format
        `\\[unused\\d+\\]` will be treated as a token and thus remain preserved
        as is to be looked up in the vocabulary.
      pad_id: ID for the `[PAD]` token.
      start_of_sequence_id: ID for the `[CLS]` token.
      end_of_sequence_id: ID for the `[SEP]` token.
    """
    self._vocab_lookup_table = vocab_lookup_table
    self._suffix_indicator = suffix_indicator
    self._max_bytes_per_word = max_bytes_per_word
    self._max_chars_per_token = max_chars_per_token
    self._token_out_type = token_out_type
    self._unknown_token = unknown_token
    self._split_unknown_characters = split_unknown_characters
    self._lower_case = lower_case
    self._keep_whitespace = keep_whitespace
    self._normalization_form = normalization_form
    self._preserve_unused_token = preserve_unused_token
    self._tokenizer = tf_text.BertTokenizer(
        vocab_lookup_table=vocab_lookup_table,
        suffix_indicator=suffix_indicator,
        max_bytes_per_word=max_bytes_per_word,
        max_chars_per_token=max_chars_per_token,
        token_out_type=token_out_type,
        unknown_token=unknown_token,
        split_unknown_characters=split_unknown_characters,
        lower_case=lower_case,
        keep_whitespace=keep_whitespace,
        normalization_form=normalization_form,
        preserve_unused_token=preserve_unused_token,
    )
    self._vocab = self._tokenizer._wordpiece_tokenizer._vocab_lookup_table
    self._pad_id = pad_id
    self._unk_id = self._vocab.lookup(tf.constant(unknown_token)).numpy()
    self._sos_id = start_of_sequence_id
    self._eos_id = end_of_sequence_id
    with tf.io.gfile.GFile(vocab_lookup_table, "rb") as f:
      self._wp_model = f.read()
    # We won't pass in extra_ids for Bert vocabulary.
    super().__init__()

  @property
  def sos_id(self) -> Optional[int]:
    return self._sos_id

  @property
  def eos_id(self) -> Optional[int]:
    return self._eos_id

  @property
  def unk_id(self) -> Optional[int]:
    return self._unk_id

  @property
  def pad_id(self) -> Optional[int]:
    return self._pad_id

  @property
  def _base_vocab_size(self):
    """Returns the vocabulary size."""
    return self._vocab.size().numpy()

  @property
  def tokenizer(self):
    """Returns the Python tokenizer."""
    return self._tokenizer

  @property
  def tf_tokenizer(self):
    """Instantiate and return a TF tokenizer."""
    return self._tokenizer

  @property
  def vocab_size(self):
    return self._base_vocab_size

  def _encode(self, s):
    """Encode a python string as a list of integers.

    Args:
      s: a string

    Returns:
      a list of integers (not terminated by EOS)
    """
    return self._encode_tf(s).numpy()

  def _decode(self, ids):
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers (not terminated by EOS)

    Returns:
      a string
    """
    ids = tf.constant(ids)
    str_text = self._decode_tf(ids)
    return str_text.numpy().decode("UTF-8")

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    This will be necessary for on-the-fly tokenization.

    Args:
      s: a tf.Scalar with dtype tf.string

    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    tokens = self.tokenizer.tokenize(s)
    # Convert tf.RaggedTensor to tf.Tensor
    return tf.squeeze(tokens.to_tensor())

  def _decode_tf(self, ids):
    """Decode in TensorFlow.

    Args:
      ids: a 1d tf.Tensor with dtype tf.int32

    Returns:
      a tf Scalar with dtype tf.string
    """
    # Convert tf.Tensor to tf.RaggedTensor
    ids = tf.RaggedTensor.from_tensor(tf.expand_dims(ids, axis=1))
    tokens = self.tf_tokenizer.detokenize(ids)
    # Flatten tf.RaggedTensor and convert tokens into a string
    return tf.strings.join(tokens.flat_values, " ")

  def __eq__(self, other):
    try:
      their_md5 = hashlib.md5(other._wp_model).hexdigest()
      their_sos_id = other._sos_id
    except AttributeError:
      return False
    our_md5 = hashlib.md5(self._wp_model).hexdigest()
    return our_md5 == their_md5 and self._sos_id == their_sos_id
