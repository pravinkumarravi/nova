# Educational Guide: Byte-Pair Encoding (BPE) Subword Tokenization

This guide explains the motivation, mathematics, and implementation of Byte-Pair Encoding (BPE) in Nova, and demonstrates how it solves the Out-Of-Vocabulary (OOV) problem for internet slang, abbreviations, and unseen words.

---

## 1. Why Word-Level Tokenizers Fail on Human Slang

In a standard word-level tokenizer:
- Every unique word is an atomic token in the vocabulary.
- If a user types an abbreviation or slang word not explicitly in the training set (e.g., `"who r u?"` or `"how r u doing"`), the words `"r"` and `"u"` **do not exist** in the vocabulary.
- The tokenizer maps them to `<UNK>` (Unknown, ID `1`):
  $$\text{"who r u?"} \longrightarrow [\text{"who"}, \mathbf{<UNK>}, \mathbf{<UNK>}, \text{"?"}]$$
- Because the words are replaced with `<UNK>`, the neural model loses their identities and produces noisy or erroneous classifications.

---

## 2. How Byte-Pair Encoding (BPE) Solves This

Byte-Pair Encoding (Sennrich et al., 2016) is a subword tokenization algorithm used in modern LLMs (GPT-2, GPT-3, GPT-4, LLaMA, RoBERTa).

Instead of treating words as indivisible atoms:
1. **Base Alphabet:** The initial vocabulary contains all individual characters (letters, digits, punctuation).
2. **Iterative Merging:** Frequent consecutive character pairs (e.g., `'i'` + `'s'` $\to$ `'is'`) are iteratively merged into new subword tokens.
3. **Graceful Fallback:** If a novel word appears (like `"u"`, `"r"`, or `"subword"`), it never produces `<UNK>`; it simply decomposes into its base character units or known subwords!

---

## 3. Mathematical & Algorithmic Formulation

### Training Phase:
Let the corpus be represented as word-frequency pairs with an end-of-word boundary marker `</w>`:
$$\mathcal{V}_0 = \{ (w, f_w) \mid w \in \text{corpus} \}$$

At each merge step $k \in \{1, \dots, K\}$:
1. Count adjacent bigram frequencies across all symbol tuples:
   $$\text{pair}^* = \arg\max_{(p_1, p_2)} \sum_{w \in \mathcal{V}} f_w \cdot \text{count}((p_1, p_2) \in w)$$
2. Record merge rule:
   $$\mathcal{M}[\text{pair}^*] = k, \quad \text{new\_token} = p_1 \circ p_2$$
3. Replace all occurrences of $(p_1, p_2)$ in $\mathcal{V}$ with the merged symbol.

### Inference / Encoding Phase:
Given an input word $w = [c_1, c_2, \dots, c_m, \text{</w>}]$:
1. Find all consecutive pairs $(c_i, c_{i+1})$.
2. If any pair exists in merge table $\mathcal{M}$, find the one with the lowest rank (highest priority).
3. Merge that pair and repeat until no mergeable pairs remain.
4. Return the resulting subword sequence.

---

## 4. Empirical Comparison in Nova

Run the comparison CLI command:
```powershell
python -m nova.cli compare-tokenizers --text "who r u? how r u doing?"
```

### Observed Output:
```text
=================================================================
TOKENIZER COMPARISON: WORD-LEVEL VS BYTE-PAIR ENCODING (BPE)
=================================================================
Input Query: "who r u? how r u doing?"

1. Word-Level Tokenizer (SimpleTokenizer):
   Tokens      : ['who', 'r', 'u', '?', 'how', 'r', 'u', 'doing', '?']
   Token IDs   : [43, 1, 1, 7, 42, 1, 1, 1, 7]
   OOV Count   : 5 (<UNK> DETECTED)
   Decoded     : "who <UNK> <UNK>? how <UNK> <UNK> <UNK>?"

2. Byte-Pair Encoding Subword Tokenizer (BPETokenizer):
   Subwords    : ['who</w>', 'r</w>', 'u', '</w>', '?</w>', 'how</w>', 'r</w>', 'u', '</w>', 'd', 'o', 'ing</w>', '?</w>']
   Token IDs   : [125, 116, 121, 59, 66, 249, 116, 121, 59, 87, 109, 208, 66]
   OOV Count   : 0 (ZERO OOV - 100% Character/Subword Fallback)
   Decoded     : "who r u? how r u doing?"
=================================================================
```

---

## 5. Artifacts and Persistence

BPE artifacts are saved in `artifacts/bpe_tokenizer/`:
- `bpe_config.json`: Configuration, vocabulary size, merge count.
- `bpe_merges.txt`: Ranked merge pairs in priority order.
- `bpe_vocab.json`: Learned subword-to-ID mapping.
