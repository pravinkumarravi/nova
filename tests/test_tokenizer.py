"""Tests for SimpleTokenizer and Vocabulary."""

import tempfile
from pathlib import Path
import pytest

from nova.tokenizer.vocabulary import Vocabulary
from nova.tokenizer.tokenizer import SimpleTokenizer


def test_vocabulary_special_tokens():
    vocab = Vocabulary()
    assert vocab.pad_id == 0
    assert vocab.unk_id == 1
    assert vocab.bos_id == 2
    assert vocab.eos_id == 3
    assert len(vocab) == 4


def test_vocabulary_build_and_lookup():
    vocab = Vocabulary()
    tokens = [["hello", "world"], ["hello", "nova"]]
    vocab.build_from_tokens(tokens)

    assert "hello" in vocab
    assert "world" in vocab
    assert "nova" in vocab
    assert vocab.token_to_index("unseen_word") == vocab.unk_id
    assert vocab.index_to_token(vocab.unk_id) == "<UNK>"


def test_vocabulary_save_load(tmp_path: Path):
    vocab = Vocabulary()
    vocab.add_token("test", count=5)
    save_file = tmp_path / "vocab.json"
    vocab.save(save_file)

    loaded = Vocabulary.load(save_file)
    assert len(loaded) == len(vocab)
    assert loaded.token_to_index("test") == vocab.token_to_index("test")


def test_tokenizer_encoding_decoding():
    tok = SimpleTokenizer()
    corpus = ["What is your name?", "My name is Nova."]
    tok.train_from_corpus(corpus)

    text = "What is your name?"
    token_ids = tok.encode(text, add_special_tokens=True)
    assert token_ids[0] == tok.bos_token_id
    assert token_ids[-1] == tok.eos_token_id

    decoded = tok.decode(token_ids, skip_special_tokens=True)
    assert "what is your name?" in decoded.lower()


def test_tokenizer_unknown_and_empty():
    tok = SimpleTokenizer()
    tok.train_from_corpus(["hello world"])

    # Empty string
    assert tok.tokenize("") == []
    assert tok.encode("") == []
    assert tok.decode([]) == ""

    # Unknown tokens
    ids = tok.encode("quantum physics", add_special_tokens=False)
    assert ids == [tok.unk_token_id, tok.unk_token_id]


def test_tokenizer_save_load(tmp_path: Path):
    tok = SimpleTokenizer()
    tok.train_from_corpus(["good morning", "good evening"])
    tok.save(tmp_path)

    loaded = SimpleTokenizer.load(tmp_path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("good morning") == tok.encode("good morning")
