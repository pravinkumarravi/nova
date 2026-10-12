"""Unit tests for Byte-Pair Encoding (BPE) subword tokenizer."""

from pathlib import Path
import pytest

from nova.tokenizer.bpe import BPETokenizer
from nova.tokenizer.tokenizer import SimpleTokenizer


def test_bpe_training_and_zero_oov():
    corpus = [
        "What is your name?",
        "My name is Nova.",
        "Who are you?",
        "How can you help me?",
    ]
    bpe = BPETokenizer(num_merges=40)
    bpe.train_from_corpus(corpus)

    assert bpe.vocab_size > len(bpe.special_tokens)
    assert len(bpe.merges) > 0

    # Test on unseen slang with short forms: "who r u?"
    # SimpleTokenizer will produce <UNK> for "r" and "u"
    simple_tok = SimpleTokenizer()
    simple_tok.train_from_corpus(corpus)
    simple_ids = simple_tok.encode("who r u?", add_special_tokens=False)
    assert simple_tok.unk_token_id in simple_ids  # Has <UNK>!

    # BPETokenizer will decompose "r" and "u" into base characters, so ZERO <UNK>!
    bpe_ids = bpe.encode("who r u?", add_special_tokens=False)
    assert bpe.unk_token_id not in bpe_ids  # No <UNK> whatsoever!


def test_bpe_encode_decode_roundtrip():
    corpus = [
        "hello world",
        "good morning nova",
        "tell me the time",
        "what is today's date",
    ]
    bpe = BPETokenizer(num_merges=30)
    bpe.train_from_corpus(corpus)

    test_sentence = "what is the time?"
    encoded_ids = bpe.encode(test_sentence, add_special_tokens=True)

    assert encoded_ids[0] == bpe.bos_token_id
    assert encoded_ids[-1] == bpe.eos_token_id

    decoded_text = bpe.decode(encoded_ids, skip_special_tokens=True)
    assert decoded_text.lower() == test_sentence.lower()


def test_bpe_save_and_load(tmp_path: Path):
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "neural language models are built from scratch",
    ]
    bpe = BPETokenizer(num_merges=25)
    bpe.train_from_corpus(corpus)

    bpe.save(tmp_path)

    loaded = BPETokenizer.load(tmp_path)
    assert loaded.vocab_size == bpe.vocab_size
    assert len(loaded.merges) == len(bpe.merges)

    sample = "quick brown fox"
    assert loaded.encode(sample) == bpe.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == bpe.decode(bpe.encode(sample))
