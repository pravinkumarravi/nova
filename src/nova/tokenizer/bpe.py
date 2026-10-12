"""Educational Byte-Pair Encoding (BPE) Subword Tokenizer for Nova."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


class BPETokenizer:
    """Educational Byte-Pair Encoding (BPE) Subword Tokenizer.
    
    Learns iterative subword merges from character-level representations.
    Eliminates Out-Of-Vocabulary (<UNK>) issues by falling back to base characters
    for unseen words, slang, or abbreviations (e.g., 'who r u?').
    """

    END_OF_WORD = "</w>"
    WORD_SPLIT_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(
        self,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        num_merges: int = 150,
    ) -> None:
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.num_merges = num_merges

        # Special tokens
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]

        # Learned merge rules: tuple of (token1, token2) -> rank index (lower rank = higher priority)
        self.merges: Dict[Tuple[str, str], int] = {}
        # Vocabulary: token string -> token ID
        self.token_to_id: Dict[str, int] = {}
        # Inverse vocabulary: token ID -> token string
        self.id_to_token: Dict[int, str] = {}

        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        self.token_to_id.clear()
        self.id_to_token.clear()
        for i, tok in enumerate(self.special_tokens):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def normalize(self, text: str) -> str:
        """Normalizes input string by trimming and lowercasing."""
        return text.strip().lower() if text else ""

    def pre_tokenize(self, text: str) -> List[str]:
        """Extracts words and punctuation from normalized string."""
        norm = self.normalize(text)
        if not norm:
            return []
        return self.WORD_SPLIT_REGEX.findall(norm)

    def _get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Counts frequencies of adjacent symbol pairs in word representations."""
        pairs = Counter()
        for word_tuple, freq in vocab.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    def _merge_vocab(
        self,
        pair: Tuple[str, str],
        v_in: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, ...], int]:
        """Merges all occurrences of a specific pair across the vocabulary table."""
        v_out: Dict[Tuple[str, ...], int] = {}
        p1, p2 = pair
        merged_symbol = p1 + p2

        for word_tuple, freq in v_in.items():
            new_word: List[str] = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == p1 and word_tuple[i + 1] == p2:
                    new_word.append(merged_symbol)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            v_out[tuple(new_word)] = freq

        return v_out

    def train_from_corpus(
        self,
        corpus: Sequence[str],
        num_merges: Optional[int] = None,
    ) -> None:
        """Learns subword vocabulary and merge operations from a raw text corpus."""
        target_merges = num_merges or self.num_merges
        self._init_special_tokens()

        # Step 1: Pre-tokenize corpus and count word frequencies
        word_counts = Counter()
        for text in corpus:
            words = self.pre_tokenize(text)
            for w in words:
                # Append end-of-word symbol to preserve word boundaries
                symbols = tuple(list(w) + [self.END_OF_WORD])
                word_counts[symbols] += 1

        # Step 2: Initialize base alphabet with corpus symbols and standard ASCII punctuation
        alphabet: Set[str] = set()
        for symbols in word_counts.keys():
            alphabet.update(symbols)

        # Include standard printable characters to ensure full English/punctuation coverage
        import string
        for ch in string.ascii_lowercase + string.digits + string.punctuation:
            alphabet.add(ch)
            alphabet.add(ch + self.END_OF_WORD)

        # Register base characters in vocabulary (sorted for determinism)
        for char in sorted(list(alphabet)):
            if char not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char

        # Step 3: Iteratively find and merge the most frequent pair
        current_vocab = dict(word_counts)
        self.merges.clear()

        for merge_idx in range(target_merges):
            pairs = self._get_stats(current_vocab)
            if not pairs:
                break

            # Best pair = highest frequency, tie-break alphabetically
            best_pair = max(pairs.items(), key=lambda x: (x[1], -len(x[0][0]), x[0]))[0]
            if pairs[best_pair] < 1:
                break

            # Record merge rule
            self.merges[best_pair] = merge_idx
            merged_symbol = best_pair[0] + best_pair[1]

            if merged_symbol not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[merged_symbol] = idx
                self.id_to_token[idx] = merged_symbol

            current_vocab = self._merge_vocab(best_pair, current_vocab)

    def _tokenize_word(self, word: str) -> List[str]:
        """Encodes an individual word into its constituent subwords using learned merge rules."""
        symbols = list(word) + [self.END_OF_WORD]

        while len(symbols) > 1:
            # Find all adjacent pairs
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]

            # Find the pair with the lowest merge rank (highest priority)
            mergeable_pairs = [(p, self.merges[p]) for p in pairs if p in self.merges]
            if not mergeable_pairs:
                break

            best_pair = min(mergeable_pairs, key=lambda x: x[1])[0]
            p1, p2 = best_pair
            merged_str = p1 + p2

            # Apply merge to symbols list
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == p1 and symbols[i + 1] == p2:
                    new_symbols.append(merged_str)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def tokenize(self, text: str) -> List[str]:
        """Splits input string into a list of BPE subwords."""
        words = self.pre_tokenize(text)
        subwords: List[str] = []
        for w in words:
            subwords.extend(self._tokenize_word(w))
        return subwords

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """Encodes text to a list of integer token IDs."""
        subwords = self.tokenize(text)
        token_ids: List[int] = []

        for sub in subwords:
            tid = self.token_to_id.get(sub, self.unk_token_id)
            token_ids.append(tid)

        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decodes integer token IDs back into readable English text."""
        special_ids = {
            self.pad_token_id,
            self.unk_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }

        subwords: List[str] = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            sub = self.id_to_token.get(tid, "")
            if sub:
                subwords.append(sub)

        # Concatenate and handle word boundary marker </w>
        full_text = "".join(subwords).replace(self.END_OF_WORD, " ").strip()

        # Re-attach standard punctuation naturally
        reconstructed = []
        for token in full_text.split():
            if token in {".", ",", "!", "?", ":", ";", "'"} and reconstructed:
                reconstructed[-1] = f"{reconstructed[-1]}{token}"
            else:
                reconstructed.append(token)

        return " ".join(reconstructed)

    def save(self, directory: str | Path) -> None:
        """Saves BPE merge table and vocabulary to disk."""
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)

        config_data = {
            "type": "BPETokenizer",
            "vocab_size": len(self.token_to_id),
            "num_merges": len(self.merges),
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "end_of_word": self.END_OF_WORD,
        }

        with open(dirpath / "bpe_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        # Save merges in order of rank
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        with open(dirpath / "bpe_merges.txt", "w", encoding="utf-8") as f:
            for (p1, p2), _ in sorted_merges:
                f.write(f"{p1} {p2}\n")

        # Save vocabulary
        with open(dirpath / "bpe_vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str | Path) -> "BPETokenizer":
        """Loads a BPETokenizer from saved artifacts."""
        dirpath = Path(directory)
        cfg_file = dirpath / "bpe_config.json"
        merges_file = dirpath / "bpe_merges.txt"
        vocab_file = dirpath / "bpe_vocab.json"

        if not (cfg_file.exists() and merges_file.exists() and vocab_file.exists()):
            raise FileNotFoundError(f"Missing BPE artifacts in {dirpath}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        tokenizer = cls(
            pad_token=cfg.get("pad_token", "<PAD>"),
            unk_token=cfg.get("unk_token", "<UNK>"),
            bos_token=cfg.get("bos_token", "<BOS>"),
            eos_token=cfg.get("eos_token", "<EOS>"),
            num_merges=cfg.get("num_merges", 150),
        )

        with open(vocab_file, "r", encoding="utf-8") as f:
            tokenizer.token_to_id = json.load(f)
            tokenizer.id_to_token = {int(idx): tok for tok, idx in tokenizer.token_to_id.items()}

        tokenizer.merges.clear()
        with open(merges_file, "r", encoding="utf-8") as f:
            for rank, line in enumerate(f):
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        tokenizer.merges[(parts[0], parts[1])] = rank

        return tokenizer
