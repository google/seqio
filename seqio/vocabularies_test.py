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

"""Tests for seqio.vocabularies."""

from absl.testing import absltest
import numpy as np
from seqio import test_utils
from seqio import vocabularies
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

mock = absltest.mock


def _decode_tf(vocab, tokens):
  return vocab.decode_tf(tf.constant(tokens, tf.int32)).numpy().decode("UTF-8")


class VocabularyTest(absltest.TestCase):

  TEST_STR = "Testing."
  TEST_IDS = [84, 101, 115, 116, 105, 110, 103, 46]

  class AsciiVocab(vocabularies.Vocabulary):

    def __init__(self, extra_ids=0, use_eos=True, use_unk=True):
      super().__init__(extra_ids=extra_ids)
      self._extra_ids = extra_ids
      self._use_eos = use_eos
      self._use_unk = use_unk

    @property
    def eos_id(self):
      return 1 if self._use_eos else None

    @property
    def unk_id(self):
      return 2 if self._use_unk else None

    @property
    def _base_vocab_size(self):
      return 128

    def _encode(self, s):
      return [ord(c) for c in s]

    def _decode(self, ids):
      return "".join("<eos>" if id == 1 else chr(id) for id in ids if id > 0)

    def _encode_tf(self, s):
      return tf.strings.unicode_decode(s, "UTF-8")

    def _decode_tf(self, ids):
      s = tf.strings.unicode_encode(ids, "UTF-8")
      s = tf.strings.regex_replace(s, chr(0), "")
      s = tf.strings.regex_replace(s, chr(1), "<eos>")
      return s

  def test_properties(self):
    test_vocab = self.AsciiVocab(use_eos=False, use_unk=True, extra_ids=10)
    self.assertEqual(test_vocab.extra_ids, 10)
    self.assertEqual(test_vocab.pad_id, 0)
    self.assertIsNone(test_vocab.eos_id)
    self.assertEqual(test_vocab.unk_id, 2)
    self.assertEqual(test_vocab.vocab_size, 128 + 10)

    test_vocab = self.AsciiVocab(use_eos=True, use_unk=False)
    self.assertEqual(test_vocab.extra_ids, 0)
    self.assertEqual(test_vocab.pad_id, 0)
    self.assertEqual(test_vocab.eos_id, 1)
    self.assertIsNone(test_vocab.unk_id)
    self.assertEqual(test_vocab.vocab_size, 128)

  def test_encode(self):
    test_vocab = self.AsciiVocab()
    self.assertSequenceEqual(test_vocab.encode(self.TEST_STR), self.TEST_IDS)
    self.assertSequenceEqual(
        tuple(test_vocab.encode_tf(self.TEST_STR).numpy()),
        self.TEST_IDS)

  def test_decode_unk_and_eos(self):
    test_vocab = self.AsciiVocab(use_eos=True, use_unk=True)
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 0, 10]
    test_str = "\x02" + self.TEST_STR + "\x7f\x02<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)
    self.assertEqual(_decode_tf(test_vocab, test_ids), test_str)

  def test_decode_unk_only(self):
    test_vocab = self.AsciiVocab(use_eos=False, use_unk=True, extra_ids=35)
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 33, 1]
    test_str = "\x02" + self.TEST_STR + "\x7f\x02<eos>!<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)
    self.assertEqual(_decode_tf(test_vocab, test_ids), test_str)

  def test_decode_eos_only(self):
    test_vocab = self.AsciiVocab(use_eos=True, use_unk=False)
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 33, 1]
    test_str = "¡" + self.TEST_STR + "\x7f¿<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)
    self.assertEqual(_decode_tf(test_vocab, test_ids), test_str)

    test_ids = [161] + self.TEST_IDS + [127, 191]
    test_str = "¡" + self.TEST_STR + "\x7f¿"
    self.assertEqual(test_vocab.decode(test_ids), test_str)
    self.assertEqual(_decode_tf(test_vocab, test_ids), test_str)

    test_ids = [1] + self.TEST_IDS
    test_str = "<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)
    self.assertEqual(_decode_tf(test_vocab, test_ids), test_str)

  def test_decode_no_unk_or_eos(self):
    test_vocab = self.AsciiVocab(use_eos=False, use_unk=False)
    test_ids = [161] + self.TEST_IDS +  [127, 191, 1, 33, 1]
    test_str = "¡" + self.TEST_STR + "\x7f¿<eos>!<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)
    self.assertEqual(_decode_tf(test_vocab, test_ids), test_str)

  def test_decode_tf_batch(self):
    test_vocab = self.AsciiVocab(use_eos=True, use_unk=True)
    test_ids = (
        [161] + self.TEST_IDS +  [127, 191, 1, 33, 1],
        [161] + self.TEST_IDS +  [1, 191, 1, 33, 1],
    )
    test_str = (
        "\x02" + self.TEST_STR + "\x7f\x02<eos>",
        "\x02" + self.TEST_STR + "<eos>",
    )
    decoded = [
        dec.decode("UTF-8") for dec in
        test_vocab.decode_tf(tf.constant(test_ids, tf.int32)).numpy()
    ]
    self.assertSequenceEqual(decoded, test_str)


class PassThroughVocabularyTest(absltest.TestCase):

  def test_no_eos(self):
    vocab = vocabularies.PassThroughVocabulary(size=128, eos_id=None)
    ids = list(range(2, 10))
    ids.insert(3, 1)
    self.assertIsNone(vocab.eos_id)
    self.assertEqual(128, vocab.vocab_size)
    self.assertSequenceEqual(ids, vocab.encode(ids))
    self.assertSequenceEqual(ids, vocab.decode(ids))
    ids_t = tf.constant([ids], tf.int32)
    np.testing.assert_equal(ids_t, vocab.encode_tf(ids_t).numpy())
    np.testing.assert_equal(ids_t, vocab.decode_tf(ids_t).numpy())

  def test_eos(self):
    vocab = vocabularies.PassThroughVocabulary(size=128, eos_id=1)
    ids = list(range(2, 10))
    ids.insert(3, 1)
    self.assertEqual(128, vocab.vocab_size)
    self.assertEqual(1, vocab.eos_id)
    self.assertSequenceEqual(ids, vocab.encode(ids))
    self.assertSequenceEqual(ids[0:4], vocab.decode(ids))
    ids_t = tf.constant([ids], tf.int32)
    np.testing.assert_equal(ids_t, vocab.encode_tf(ids_t).numpy())
    np.testing.assert_equal(
        [ids[0:4] + [0]*5], vocab.decode_tf(ids_t).numpy())

  def test_equal(self):
    vocab1 = vocabularies.PassThroughVocabulary(size=128)
    vocab2 = vocabularies.PassThroughVocabulary(size=128)
    self.assertEqual(vocab1, vocab2)

  def test_not_equal(self):
    vocab1 = vocabularies.PassThroughVocabulary(size=128, eos_id=None)
    vocab2 = vocabularies.PassThroughVocabulary(size=256, eos_id=None)
    vocab3 = vocabularies.PassThroughVocabulary(size=128, eos_id=1)
    self.assertNotEqual(vocab1, vocab2)
    self.assertNotEqual(vocab1, vocab3)


class SentencepieceVocabularyTest(absltest.TestCase):

  TEST_STRING = "this is a test"
  TEST_TOKENS = (11, 8, 6, 3, 8, 6, 3, 5, 10)
  UNK_STRING = " ⁇ "

  def test_vocab(self):
    vocab = test_utils.sentencepiece_vocab()
    self.assertEqual(26, vocab.vocab_size)
    self.assertSequenceEqual(self.TEST_TOKENS, vocab.encode(self.TEST_STRING))
    self.assertEqual(self.TEST_STRING, vocab.decode(self.TEST_TOKENS))
    self.assertSequenceEqual(
        self.TEST_TOKENS,
        tuple(vocab.encode_tf(self.TEST_STRING).numpy()))
    self.assertEqual(self.TEST_STRING, _decode_tf(vocab, self.TEST_TOKENS))

  def test_extra_ids(self):
    vocab = test_utils.sentencepiece_vocab(extra_ids=10)
    self.assertEqual(36, vocab.vocab_size)
    self.assertEqual("v", vocab.decode([25]))
    test_string = "<extra_id_0> <extra_id_1> v <extra_id_9>"
    test_tokens = (35, 34, 3, 25, 26)
    self.assertEqual(test_string, vocab.decode(test_tokens))
    self.assertEqual(test_string, _decode_tf(vocab, test_tokens))
    self.assertSequenceEqual(test_tokens, vocab.encode(test_string))
    self.assertSequenceEqual(
        test_tokens,
        tuple(vocab.encode_tf(test_string).numpy()))

  def test_force_repeated_whitespace_preservation(self):
    test_string = "a a  a   a"  # string with repeated whitespaces

    vocab = test_utils.sentencepiece_vocab(
        force_preserve_repeated_whitespace=True)
    self.assertEqual(test_string, vocab.decode(vocab.encode(test_string)))

    vocab = test_utils.sentencepiece_vocab()
    self.assertEqual("a a a a", vocab.decode(vocab.encode(test_string)))

  def test_equal(self):
    vocab1 = test_utils.sentencepiece_vocab()
    vocab2 = test_utils.sentencepiece_vocab()
    self.assertEqual(vocab1, vocab2)

  def test_not_equal(self):
    vocab1 = test_utils.sentencepiece_vocab()
    vocab2 = test_utils.sentencepiece_vocab(10)
    self.assertNotEqual(vocab1, vocab2)


class ByteVocabularyTest(absltest.TestCase):

  TEST_STRING = "this is a test"
  TEST_BYTE_IDS = (
      119, 107, 108, 118, 35, 108, 118, 35, 100, 35, 119, 104, 118, 119)

  def test_vocab(self):
    vocab = vocabularies.ByteVocabulary()
    self.assertEqual(259, vocab.vocab_size)
    self.assertSequenceEqual(self.TEST_BYTE_IDS, vocab.encode(self.TEST_STRING))
    self.assertEqual(self.TEST_STRING, vocab.decode(self.TEST_BYTE_IDS))
    self.assertEqual(
        self.TEST_BYTE_IDS,
        tuple(vocab.encode_tf(self.TEST_STRING).numpy()))
    self.assertEqual(self.TEST_STRING, _decode_tf(vocab, self.TEST_BYTE_IDS))

  def test_extra_ids(self):
    vocab = vocabularies.ByteVocabulary(extra_ids=10)
    self.assertEqual(269, vocab.vocab_size)
    self.assertEqual("a", vocab.decode([100]))
    self.assertEqual("", vocab.decode([268]))

  def test_out_of_vocab(self):
    vocab = vocabularies.ByteVocabulary()
    self.assertEqual(259, vocab.vocab_size)
    self.assertEqual("", vocab.decode([260]))

  def test_equal(self):
    vocab1 = vocabularies.ByteVocabulary()
    vocab2 = vocabularies.ByteVocabulary()
    self.assertEqual(vocab1, vocab2)

  def test_not_equal(self):
    vocab1 = vocabularies.ByteVocabulary()
    vocab2 = vocabularies.ByteVocabulary(10)
    self.assertNotEqual(vocab1, vocab2)


class FullCodepointVocabularyTest(absltest.TestCase):

  TEST_STRING = "this is a test"
  TEST_CODEPOINT_IDS = (116, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101,
                        115, 116)
  EOS_TEST_STRING = "this is a test" + chr(
      vocabularies.FullCodepointVocabulary.EOS_CODEPOINT)
  EOS_TEST_CODEPOINT_IDS = (116, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116,
                            101, 115, 116, 57349)

  def test_vocab(self):
    vocab = vocabularies.FullCodepointVocabulary()
    self.assertEqual(vocab.vocab_size, vocab.LARGEST_CODEPOINT)
    self.assertEqual(vocab.pad_id, vocab.PAD_CODEPOINT)
    self.assertEqual(vocab.eos_id, vocab.EOS_CODEPOINT)
    self.assertIsNone(vocab.unk_id)

  def test_encode_tf(self):
    vocab = vocabularies.FullCodepointVocabulary()
    self.assertEqual(self.TEST_CODEPOINT_IDS,
                     tuple(vocab.encode_tf(self.TEST_STRING).numpy()))
    self.assertEqual(self.EOS_TEST_CODEPOINT_IDS,
                     tuple(vocab.encode_tf(self.EOS_TEST_STRING).numpy()))

  def test_decode_tf(self):
    vocab = vocabularies.FullCodepointVocabulary()
    self.assertSequenceEqual(self.TEST_STRING,
                             _decode_tf(vocab, self.TEST_CODEPOINT_IDS))
    self.assertSequenceEqual(self.EOS_TEST_STRING,
                             _decode_tf(vocab, self.EOS_TEST_CODEPOINT_IDS))

  def test_encode(self):
    vocab = vocabularies.FullCodepointVocabulary()
    self.assertSequenceEqual(self.TEST_CODEPOINT_IDS,
                             vocab.encode(self.TEST_STRING))
    self.assertSequenceEqual(self.EOS_TEST_CODEPOINT_IDS,
                             vocab.encode(self.EOS_TEST_STRING))

  def test_decode(self):
    vocab = vocabularies.FullCodepointVocabulary()
    self.assertEqual(self.TEST_STRING, vocab.decode(self.TEST_CODEPOINT_IDS))
    self.assertEqual(self.TEST_STRING,
                     vocab.decode(self.EOS_TEST_CODEPOINT_IDS))

  def test_equal(self):
    vocab1 = vocabularies.FullCodepointVocabulary()
    vocab2 = vocabularies.FullCodepointVocabulary()
    self.assertEqual(vocab1, vocab2)


class PartialCodepointVocabularyTest(absltest.TestCase):

  TEST_STRING = "this is a test"
  TEST_CODEPOINT_IDS = (3, 4, 5, 6, 9, 5, 6, 9, 7, 9, 3, 8, 6, 3)

  UNK_TEST_STRING = "This is a test!"
  UNK_TEST_CODEPOINT_IDS = (2, 4, 5, 6, 9, 5, 6, 9, 7, 9, 3, 8, 6, 3, 2)
  UNK_ID = vocabularies.PartialCodepointVocabulary.UNK_CODEPOINT
  UNK_TEST_STRING_ENCODED = chr(UNK_ID) + "his is a test" + chr(UNK_ID)

  EOS_TEST_STRING = "this is a test" + chr(
      vocabularies.PartialCodepointVocabulary.EOS_CODEPOINT)
  EOS_TEST_CODEPOINT_IDS = (3, 4, 5, 6, 9, 5, 6, 9, 7, 9, 3, 8, 6, 3, 1)

  def setUp(self):
    super().setUp()
    self.char_points = [ord(i) for i in "thisae "]
    data = "\n".join(str(i) for i in self.char_points)
    self.char_points_file = self.create_tempfile(content=data)

  def test_vocab(self):
    vocab = vocabularies.PartialCodepointVocabulary.create_from_file(
        self.char_points_file.full_path)
    self.assertEqual(vocab.vocab_size, vocab.vocab_size)
    self.assertEqual(vocab.pad_id, vocab.PAD_ID)
    self.assertEqual(vocab.eos_id, vocab.EOS_ID)
    self.assertEqual(vocab.unk_id, vocab.UNK_ID)

  def test_vocab_constructor(self):
    vocab = vocabularies.PartialCodepointVocabulary(self.char_points)
    self.assertEqual(vocab.vocab_size, vocab.vocab_size)
    self.assertEqual(vocab.pad_id, vocab.PAD_ID)
    self.assertEqual(vocab.eos_id, vocab.EOS_ID)
    self.assertEqual(vocab.unk_id, vocab.UNK_ID)

  def test_encode_tf(self):
    vocab = vocabularies.PartialCodepointVocabulary.create_from_file(
        self.char_points_file.full_path)
    self.assertEqual(self.TEST_CODEPOINT_IDS,
                     tuple(vocab.encode_tf(self.TEST_STRING).numpy()))
    self.assertEqual(self.UNK_TEST_CODEPOINT_IDS,
                     tuple(vocab.encode_tf(self.UNK_TEST_STRING).numpy()))
    self.assertEqual(
        self.EOS_TEST_CODEPOINT_IDS,
        tuple(vocab.encode_tf(self.EOS_TEST_STRING).numpy()))

  def test_decode_tf(self):
    vocab = vocabularies.PartialCodepointVocabulary.create_from_file(
        self.char_points_file.full_path)
    self.assertSequenceEqual(self.TEST_STRING,
                             _decode_tf(vocab, self.TEST_CODEPOINT_IDS))
    self.assertSequenceEqual(self.EOS_TEST_STRING,
                             _decode_tf(vocab, self.EOS_TEST_CODEPOINT_IDS))
    self.assertSequenceEqual(self.UNK_TEST_STRING_ENCODED,
                             _decode_tf(vocab, self.UNK_TEST_CODEPOINT_IDS))

  def test_encode(self):
    vocab = vocabularies.PartialCodepointVocabulary.create_from_file(
        self.char_points_file.full_path)
    self.assertSequenceEqual(self.TEST_CODEPOINT_IDS,
                             vocab.encode(self.TEST_STRING))
    self.assertSequenceEqual(self.UNK_TEST_CODEPOINT_IDS,
                             vocab.encode(self.UNK_TEST_STRING))
    self.assertSequenceEqual(self.EOS_TEST_CODEPOINT_IDS,
                             vocab.encode(self.EOS_TEST_STRING))

  def test_decode(self):
    vocab = vocabularies.PartialCodepointVocabulary.create_from_file(
        self.char_points_file.full_path)
    self.assertEqual(self.TEST_STRING, vocab.decode(self.TEST_CODEPOINT_IDS))
    self.assertEqual(self.UNK_TEST_STRING_ENCODED,
                     vocab.decode(self.UNK_TEST_CODEPOINT_IDS))
    self.assertEqual(self.TEST_STRING,
                     vocab.decode(self.EOS_TEST_CODEPOINT_IDS))

  def test_not_equal(self):
    vocab1 = vocabularies.PartialCodepointVocabulary.create_from_file(
        self.char_points_file.full_path)
    vocab2 = vocabularies.PartialCodepointVocabulary.create_from_file(
        self.char_points_file.full_path, extra_ids=10)
    self.assertNotEqual(vocab1, vocab2)


class BertWordpieceVocabularyTest(absltest.TestCase):

  TEST_STRING = "this is a test"
  TEST_TOKENS = (106, 105, 104, 107)

  def test_vocab(self):

    vocab = test_utils.bertwordpiece_vocab()
    self.assertEqual(109, vocab.vocab_size)

    self.assertEqual(self.TEST_STRING, vocab.decode(self.TEST_TOKENS))
    self.assertEqual(self.TEST_STRING, _decode_tf(vocab, self.TEST_TOKENS))

    self.assertSequenceEqual(self.TEST_TOKENS,
                             tuple(vocab.encode(self.TEST_STRING)))
    self.assertSequenceEqual(self.TEST_TOKENS,
                             tuple(vocab.encode_tf(self.TEST_STRING).numpy()))

  def test_special_ids(self):
    # Set preserve_unused_token to True so that detokenization remains the
    # special ids.
    vocab = test_utils.bertwordpiece_vocab()
    test_string = "[CLS] [MASK] [UNK] [SEP]"
    test_tokens = (101, 103, 100, 102)
    self.assertEqual(test_string, vocab.decode(test_tokens))
    self.assertEqual(test_string, _decode_tf(vocab, test_tokens))

  def test_equal(self):
    vocab1 = test_utils.bertwordpiece_vocab()
    vocab2 = test_utils.bertwordpiece_vocab()
    self.assertEqual(vocab1, vocab2)

  def test_not_equal(self):
    vocab1 = test_utils.bertwordpiece_vocab()
    vocab2 = test_utils.bertwordpiece_vocab(start_of_sequence_id=100)
    self.assertNotEqual(vocab1, vocab2)


class ClassificationSentencepieceVocabularyTest(absltest.TestCase):

  TEST_STRING = "class_B"
  TEST_TOKENS = (34,)

  def test_vocab(self):
    cls_vocab = test_utils.classification_sentencepiece_vocab(
        extra_ids=10, class_labels=["class_A", "class_B", "class_C"])
    self.assertEqual(36, cls_vocab.vocab_size)

  def test_encode(self):
    cls_vocab = test_utils.classification_sentencepiece_vocab(
        extra_ids=10, class_labels=["class_A", "class_B", "class_C"])
    self.assertSequenceEqual(self.TEST_TOKENS,
                             cls_vocab.encode(self.TEST_STRING))

  def test_encode_tf(self):
    cls_vocab = test_utils.classification_sentencepiece_vocab(
        extra_ids=10, class_labels=["class_A", "class_B", "class_C"])
    self.assertSequenceEqual(
        self.TEST_TOKENS,
        tuple(cls_vocab.encode_tf(tf.constant(self.TEST_STRING)).numpy()))

  def test_decode(self):
    cls_vocab = test_utils.classification_sentencepiece_vocab(
        extra_ids=10, class_labels=["class_A", "class_B", "class_C"])
    self.assertEqual(self.TEST_STRING, cls_vocab.decode(self.TEST_TOKENS))

  def test_decode_tf(self):
    cls_vocab = test_utils.classification_sentencepiece_vocab(
        extra_ids=10, class_labels=["class_A", "class_B", "class_C"])
    self.assertEqual(self.TEST_STRING, _decode_tf(cls_vocab, self.TEST_TOKENS))


if __name__ == "__main__":
  absltest.main()
