# Nova: The Mathematics of Neural Language Models & AI Agents

This document explains the mathematical foundations of Nova. Every concept includes the mathematical formulation, variable definitions, a concrete numerical or algorithmic example, the corresponding Python implementation in the codebase, a suggested experiment, and the expected observation.

---

## 1. Text Normalization

- **Equation / Operation:**  
  $$\text{Norm}(s) = \text{strip}(\text{case\_fold}(s))$$
- **Variables:**  
  $s$: Input raw text string from user.  
  $\text{Norm}(s)$: Canonical string representation.
- **Worked Example:**  
  `"  What is your Name?  "` $\to$ `"what is your name?"`
- **Source Code:** [`SimpleTokenizer.normalize()` in `src/nova/tokenizer/tokenizer.py`](file:///d:/AI/hello_ai/src/nova/tokenizer/tokenizer.py#L48-L55)
- **Experiment:** Run `SimpleTokenizer().normalize("  HELLO Nova!  ")`.
- **Expected Observation:** Returns `"hello nova!"` without surrounding whitespace.

---

## 2. Tokenization

- **Equation / Operation:**  
  $$\mathcal{T}(s) = [t_1, t_2, \dots, t_N], \quad t_i \in \Sigma^*$$
- **Variables:**  
  $s$: Normalized string.  
  $t_i$: Word or punctuation token.
- **Worked Example:**  
  `"hello, nova!"` $\to$ `['hello', ',', 'nova', '!']`
- **Source Code:** [`SimpleTokenizer.tokenize()` in `src/nova/tokenizer/tokenizer.py`](file:///d:/AI/hello_ai/src/nova/tokenizer/tokenizer.py#L57-L62)
- **Experiment:** Pass `"What is 10:30?"` to `tokenize()`.
- **Expected Observation:** Yields `['what', 'is', '10', ':', '30', '?']`.

---

## 3. Vocabulary Construction

- **Equation / Operation:**  
  $$V = \{ \text{special\_tokens} \} \cup \{ w \in \text{corpus} \mid \text{freq}(w) \ge \tau \}$$
- **Variables:**  
  $V$: Vocabulary set.  
  $\tau$: Minimum occurrence frequency threshold.
- **Worked Example:**  
  If special tokens are `[<PAD>, <UNK>, <BOS>, <EOS>]` and corpus contains `"hello nova"` (frequency 1), then $|V| = 4 + 2 = 6$.
- **Source Code:** [`Vocabulary.build_from_tokens()` in `src/nova/tokenizer/vocabulary.py`](file:///d:/AI/hello_ai/src/nova/tokenizer/vocabulary.py#L65-L77)
- **Experiment:** Inspect vocabulary using `python -m nova.cli inspect-vocab`.
- **Expected Observation:** Special tokens occupy indices 0 to 3, followed by most frequent corpus words.

---

## 4. Token-to-ID Conversion

- **Equation / Operation:**  
  $$f_{\text{enc}}(t) = \begin{cases} \text{index}(t) & \text{if } t \in V \\ \text{index}(\text{UNK}) & \text{otherwise} \end{cases}$$
- **Variables:**  
  $t$: Token string.  
  $\text{index}(\text{UNK})$: ID reserved for out-of-vocabulary tokens (1 in Nova).
- **Worked Example:**  
  `"hello"` $\to 4$, `"unseen_word"` $\to 1$ (`<UNK>`).
- **Source Code:** [`Vocabulary.token_to_index()` in `src/nova/tokenizer/vocabulary.py`](file:///d:/AI/hello_ai/src/nova/tokenizer/vocabulary.py#L79-L81)
- **Experiment:** Encode `"quantum mechanics"`.
- **Expected Observation:** Both tokens map to ID 1 without crashing.

---

## 5. Embedding Vectors

- **Equation / Operation:**  
  $$\mathbf{e}_t = \mathbf{E}[f_{\text{enc}}(t)], \quad \mathbf{E} \in \mathbb{R}^{|V| \times d_{\text{model}}}$$
- **Variables:**  
  $\mathbf{E}$: Learnable token embedding lookup matrix.  
  $d_{\text{model}}$: Embedding dimension (e.g., 64).  
  $\mathbf{e}_t \in \mathbb{R}^{d_{\text{model}}}$: Dense continuous vector representation of token $t$.
- **Worked Example:**  
  With $d_{\text{model}} = 4$, ID 2 retrieves row 2: `[0.12, -0.45, 0.88, 0.01]`.
- **Source Code:** [`nn.Embedding` in `src/nova/models/language_model.py`](file:///d:/AI/hello_ai/src/nova/models/language_model.py#L35-L36)
- **Experiment:** Inspect `model.token_embeddings.weight.shape`.
- **Expected Observation:** Returns `torch.Size([vocab_size, 64])`.

---

## 6. Positional Embeddings

- **Equation / Operation:**  
  $$\mathbf{x}_i = \mathbf{e}_{t_i} + \mathbf{P}_i, \quad \mathbf{P} \in \mathbb{R}^{L_{\text{max}} \times d_{\text{model}}}$$
- **Variables:**  
  $i \in \{0, \dots, T-1\}$: Discrete sequence position index.  
  $\mathbf{P}_i$: Position embedding vector.
- **Worked Example:**  
  Token `"nova"` at position 0 receives $\mathbf{e}_{\text{nova}} + \mathbf{P}_0$; at position 3 it receives $\mathbf{e}_{\text{nova}} + \mathbf{P}_3$, allowing attention to distinguish word order.
- **Source Code:** [`nn.Embedding` for positions in `src/nova/models/language_model.py`](file:///d:/AI/hello_ai/src/nova/models/language_model.py#L36)
- **Experiment:** Pass two sequences with identical tokens in reverse order without positional embeddings.
- **Expected Observation:** Pure self-attention without positional encoding treats them identically.

---

## 7. Query, Key, and Value Matrices

- **Equation / Operation:**  
  $$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$
- **Variables:**  
  $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$: Sequence representation.  
  $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$: Learned projection weight matrices.
- **Worked Example:**  
  Input $\mathbf{X} \in \mathbb{R}^{8 \times 64}$ projected by $\mathbf{W}_Q \in \mathbb{R}^{64 \times 64}$ produces $\mathbf{Q} \in \mathbb{R}^{8 \times 64}$.
- **Source Code:** [`MultiHeadAttention.__init__()` projections in `src/nova/models/attention.py`](file:///d:/AI/hello_ai/src/nova/models/attention.py#L31-L36)
- **Experiment:** Print shapes of `q, k, v` inside `MultiHeadAttention.forward()`.
- **Expected Observation:** Dimensions match `(batch_size, n_heads, seq_len, d_k)`.

---

## 8. Scaled Dot-Product Attention

- **Equation / Operation:**  
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
- **Variables:**  
  $d_k = d_{\text{model}} / h$: Head dimension (e.g., $64 / 4 = 16$).  
  $\sqrt{d_k}$: Scaling factor preventing dot products from growing excessively large, which pushes softmax into vanishing-gradient regimes.
- **Worked Example:**  
  If $d_k = 16$, $\sqrt{d_k} = 4$. Dot product score $12.0$ scales to $12.0 / 4 = 3.0$ before softmax.
- **Source Code:** [`MultiHeadAttention.forward()` in `src/nova/models/attention.py`](file:///d:/AI/hello_ai/src/nova/models/attention.py#L64-L88)
- **Experiment:** Compare softmax outputs with and without $1/\sqrt{d_k}$ on high-dimensional vectors.
- **Expected Observation:** Unscaled dot products produce sharp one-hot distributions with zero gradients.

---

## 9. Causal Attention Masks

- **Equation / Operation:**  
  $$\mathbf{M}_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}, \quad \mathbf{S}_{\text{masked}} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}$$
- **Variables:**  
  $\mathbf{M} \in \mathbb{R}^{T \times T}$: Lower-triangular causal attention mask.
- **Worked Example:**  
  For sequence length $T=3$:  
  $$\mathbf{M} = \begin{pmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{pmatrix}$$  
  Since $e^{-\infty} = 0$, position 0 pays 0% attention to positions 1 and 2.
- **Source Code:** [`CausalSelfAttention` in `src/nova/models/attention.py`](file:///d:/AI/hello_ai/src/nova/models/attention.py#L70-L74)
- **Experiment:** Run `tests/test_attention.py::test_causal_masking_prevents_future_leakage`.
- **Expected Observation:** Changing future tokens at position 3 has exactly zero effect on representations at positions 0, 1, and 2.

---

## 10. Multi-Head Attention

- **Equation / Operation:**  
  $$\text{MHA}(\mathbf{X}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}_O, \quad \text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i)$$
- **Variables:**  
  $h$: Number of parallel attention heads (e.g., 4).  
  $\mathbf{W}_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$: Final linear projection.
- **Worked Example:**  
  With $d_{\text{model}} = 64$ and $h = 4$, each head operates on subspace $d_k = 16$. Four outputs of size 16 are concatenated into 64.
- **Source Code:** [`MultiHeadAttention` in `src/nova/models/attention.py`](file:///d:/AI/hello_ai/src/nova/models/attention.py#L12-L101)
- **Experiment:** Train with 1 head vs 4 heads on the intent classifier.
- **Expected Observation:** 4 heads converge faster by attending to different syntactic and semantic positions simultaneously.

---

## 11. Residual Connections

- **Equation / Operation:**  
  $$\mathbf{x}_{\text{out}} = \mathbf{x} + \mathcal{F}(\mathbf{x})$$
- **Variables:**  
  $\mathbf{x}$: Input to sub-layer.  
  $\mathcal{F}(\mathbf{x})$: Sub-layer transformation (Attention or FFN).
- **Worked Example:**  
  During backpropagation, $\frac{\partial \mathbf{x}_{\text{out}}}{\partial \mathbf{x}} = \mathbf{I} + \frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}}$, guaranteeing an unbroken highway for gradients.
- **Source Code:** [`TransformerBlock.forward()` in `src/nova/models/transformer.py`](file:///d:/AI/hello_ai/src/nova/models/transformer.py#L48-L56)
- **Experiment:** Disable residual connections in a 4-layer model.
- **Expected Observation:** Gradients vanish and training loss stagnates.

---

## 12. Layer Normalization

- **Equation / Operation:**  
  $$\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}, \quad \mu = \frac{1}{d} \sum_{k=1}^d x_k, \quad \sigma^2 = \frac{1}{d} \sum_{k=1}^d (x_k - \mu)^2$$
- **Variables:**  
  $\mu, \sigma$: Mean and variance computed across embedding features for a single token.  
  $\boldsymbol{\gamma}, \boldsymbol{\beta}$: Learnable scale and shift parameters.  
  $\epsilon = 10^{-5}$: Numerical stability constant.
- **Worked Example:**  
  Vector `[10.0, 20.0, 30.0]` has $\mu=20, \sigma^2=66.67$. Normalized values have $\mu=0$ and $\sigma=1$.
- **Source Code:** [`nn.LayerNorm` in `src/nova/models/transformer.py`](file:///d:/AI/hello_ai/src/nova/models/transformer.py#L32-L38)
- **Experiment:** Print token vector variance before and after `ln1`.
- **Expected Observation:** Output variance is normalized close to 1.0.

---

## 13. Feed-Forward Neural Networks

- **Equation / Operation:**  
  $$\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$
- **Variables:**  
  $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$: Expands dimension (e.g., $64 \to 256$).  
  $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$: Contracts dimension back ($256 \to 64$).  
  $\text{GELU}(z) \approx 0.5z(1 + \tanh(\sqrt{2/\pi}(z + 0.044715z^3)))$: Gaussian Error Linear Unit.
- **Source Code:** [`FeedForward` in `src/nova/models/transformer.py`](file:///d:/AI/hello_ai/src/nova/models/transformer.py#L11-L24)
- **Experiment:** Change $d_{\text{ff}}$ from 256 to 64 and observe parameter count.
- **Expected Observation:** Model parameters decrease significantly while training capacity is reduced.

---

## 14. Logits and Softmax

- **Equation / Operation:**  
  $$p_i = \text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
- **Variables:**  
  $\mathbf{z} \in \mathbb{R}^K$: Unnormalized output logits from linear projection head.  
  $p_i$: Probability score where $\sum p_i = 1$ and $p_i \in (0, 1)$.
- **Worked Example:**  
  Logits `[2.0, 1.0, 0.1]` $\to e^{\mathbf{z}} \approx [7.39, 2.72, 1.11]$, sum $= 11.22 \to \mathbf{p} = [0.659, 0.242, 0.099]$.
- **Source Code:** [`F.softmax` in `src/nova/models/language_model.py`](file:///d:/AI/hello_ai/src/nova/models/language_model.py#L143) and [`intent_classifier.py`](file:///d:/AI/hello_ai/src/nova/models/intent_classifier.py#L108)
- **Experiment:** Scale logits by temperature $T$: $\text{softmax}(\mathbf{z} / T)$.
- **Expected Observation:** As $T \to 0$, distribution approaches $\text{argmax}$ (one-hot).

---

## 15. Cross-Entropy Loss

- **Equation / Operation:**  
  $$\mathcal{L}_{\text{CE}} = -\sum_{k=1}^K y_k \log(p_k) = -\log(p_{\text{target}})$$
- **Variables:**  
  $y_k \in \{0, 1\}$: Ground-truth target indicator.  
  $p_{\text{target}}$: Softmax probability assigned to the true label.
- **Worked Example:**  
  If true token probability $p_{\text{target}} = 0.9$, $\mathcal{L} = -\ln(0.9) \approx 0.105$. If $p_{\text{target}} = 0.1$, $\mathcal{L} = -\ln(0.1) \approx 2.302$.
- **Source Code:** [`F.cross_entropy` in `src/nova/models/language_model.py`](file:///d:/AI/hello_ai/src/nova/models/language_model.py#L104-L108)
- **Experiment:** Run `tests/test_language_model.py::test_language_model_forward`.
- **Expected Observation:** Finite positive scalar loss returned for valid input batches.

---

## 16. Next-Token Prediction

- **Equation / Operation:**  
  $$\mathcal{L}_{\text{LM}}(\theta) = -\frac{1}{T} \sum_{t=1}^T \log P_\theta(w_t \mid w_1, w_2, \dots, w_{t-1})$$
- **Variables:**  
  $w_1, \dots, w_{t-1}$: Input prompt history.  
  $w_t$: Target next token to predict.
- **Worked Example:**  
  Input: `[<BOS>, "what", "is", "your"]` $\to$ Targets: `["what", "is", "your", "name"]`.
- **Source Code:** [`LanguageModelDataset.__getitem__()` in `src/nova/data/dataset.py`](file:///d:/AI/hello_ai/src/nova/data/dataset.py#L47-L59)
- **Experiment:** Inspect inputs `x` and targets `y` from `LanguageModelDataset`.
- **Expected Observation:** $y[i] = x[i+1]$ for all tokens in non-padded range.

---

## 17. Forward Propagation

- **Equation / Operation:**  
  $$\mathbf{h}_0 = \mathbf{E}_{\text{tok}} + \mathbf{E}_{\text{pos}}, \quad \mathbf{h}_\ell = \text{Block}_\ell(\mathbf{h}_{\ell-1}), \quad \hat{\mathbf{y}} = \mathbf{W}_{\text{head}} \text{LN}(\mathbf{h}_L)$$
- **Variables:**  
  $\mathbf{h}_\ell$: Hidden representation at layer $\ell \in \{1, \dots, L\}$.
- **Worked Example:**  
  Tokens transform sequentially across 2 transformer blocks to produce final vocabulary logits.
- **Source Code:** [`DecoderLanguageModel.forward()` in `src/nova/models/language_model.py`](file:///d:/AI/hello_ai/src/nova/models/language_model.py#L82-L109)
- **Experiment:** Pass dummy tensor through `model(x)` and check output dimensions.
- **Expected Observation:** Logits tensor has shape `(batch_size, seq_len, vocab_size)`.

---

## 18. Computational Graphs

- **Equation / Operation:**  
  A directed acyclic graph (DAG) where nodes represent tensors and operations (edges) track input dependencies.
- **Variables:**  
  Node $v_i = \phi_i(\text{Parents}(v_i))$.
- **Source Code:** Implicitly constructed by PyTorch tensors when `requires_grad=True`.
- **Experiment:** Print `loss.grad_fn`.
- **Expected Observation:** Displays `<NllLossBackward0>` or `<CrossEntropyBackward0>`.

---

## 19. Automatic Differentiation

- **Equation / Operation:**  
  Reverse-mode accumulation applying the chain rule:  
  $$\frac{\partial \mathcal{L}}{\partial x} = \sum_{y \in \text{Children}(x)} \frac{\partial \mathcal{L}}{\partial y} \frac{\partial y}{\partial x}$$
- **Source Code:** [`loss.backward()` in `src/nova/training/train_language_model.py`](file:///d:/AI/hello_ai/src/nova/training/train_language_model.py#L115)
- **Experiment:** Execute `loss.backward()` and check `model.blocks[0].attn.q_proj.weight.grad`.
- **Expected Observation:** Gradient tensor contains non-zero float values matching weight shape.

---

## 20. Backpropagation and Gradients

- **Equation / Operation:**  
  $$\nabla_\theta \mathcal{L} = \left[ \frac{\partial \mathcal{L}}{\partial \theta_1}, \frac{\partial \mathcal{L}}{\partial \theta_2}, \dots, \frac{\partial \mathcal{L}}{\partial \theta_P} \right]^T$$
- **Variables:**  
  $\nabla_\theta \mathcal{L}$: Gradient vector pointing in direction of steepest increase in loss.
- **Source Code:** [`loss.backward()` in `src/nova/training/train_language_model.py`](file:///d:/AI/hello_ai/src/nova/training/train_language_model.py#L115)
- **Experiment:** Run `tests/test_language_model.py::test_language_model_backward_and_optimization`.
- **Expected Observation:** All parameters with `requires_grad=True` receive gradient updates.

---

## 21. Gradient Descent and AdamW

- **Equation / Operation:**  
  $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \mathbf{g}_t, \quad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) \mathbf{g}_t^2$$  
  $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}, \quad \theta_t = \theta_{t-1} - \eta \left( \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$
- **Variables:**  
  $\mathbf{m}_t$: First moment (momentum).  
  $\mathbf{v}_t$: Second moment (adaptive learning rate).  
  $\lambda$: Weight decay parameter (0.01).  
  $\eta$: Base learning rate (0.001).
- **Source Code:** [`torch.optim.AdamW` in `src/nova/training/train_language_model.py`](file:///d:/AI/hello_ai/src/nova/training/train_language_model.py#L86-L90)
- **Experiment:** Train with AdamW vs standard SGD without momentum.
- **Expected Observation:** AdamW converges smoothly in fewer epochs due to per-parameter adaptive step sizes.

---

## 22. Gradient Clipping

- **Equation / Operation:**  
  $$\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{C}{\max(C, \|\mathbf{g}\|_2)}, \quad \|\mathbf{g}\|_2 = \sqrt{\sum_i g_i^2}$$
- **Variables:**  
  $C$: Maximum allowed gradient norm (default: 1.0).
- **Worked Example:**  
  If total gradient norm $\|\mathbf{g}\|_2 = 5.0$ and $C = 1.0$, every gradient is scaled by $1.0 / 5.0 = 0.2$.
- **Source Code:** [`torch.nn.utils.clip_grad_norm_` in `src/nova/training/train_language_model.py`](file:///d:/AI/hello_ai/src/nova/training/train_language_model.py#L118)
- **Experiment:** Remove gradient clipping during early high-learning-rate training.
- **Expected Observation:** Gradient spikes can cause loss to explode into `NaN`.

---

## 23. Training versus Evaluation Mode

- **Equation / Operation:**  
  - In `train()`: Dropout randomly zeros neurons with probability $p$; LayerNorm computes batch statistics.  
  - In `eval()`: Dropout is disabled (identity scaling $\times 1$); LayerNorm uses fixed statistics.
- **Source Code:** `model.train()` and `model.eval()` toggles across all training and inference loops.
- **Experiment:** Compare model forward pass output variance in `eval()` vs `train()` mode.
- **Expected Observation:** `eval()` outputs are deterministic across multiple calls with the same input.

---

## 24. Overfitting and Generalization

- **Equation / Operation:**  
  $$\text{Generalization Gap} = \mathcal{L}_{\text{validation}} - \mathcal{L}_{\text{training}}$$
- **Worked Example:**  
  If train loss drops to 0.05 while validation loss rises from 0.40 to 1.50, the model is memorizing training samples rather than learning generalizable linguistic patterns.
- **Source Code:** [`TrainingLogger.log_epoch()` tracking in `src/nova/training/metrics.py`](file:///d:/AI/hello_ai/src/nova/training/metrics.py#L74-L104)
- **Experiment:** Train with 10 examples for 100 epochs vs full dataset for 30 epochs.
- **Expected Observation:** Tiny dataset overfits to near-zero loss but fails on novel test prompts.

---

## 25. Autoregressive Inference

- **Equation / Operation:**  
  $$\mathbf{w}_{1:T+1} = [\mathbf{w}_{1:T}, \hat{w}_{T+1}], \quad \hat{w}_{T+1} \sim P_\theta(\cdot \mid \mathbf{w}_{1:T})$$
- **Variables:**  
  Each newly generated token is appended to the context window and fed back into the model in a loop until `<EOS>` is predicted or `max_new_tokens` is reached.
- **Source Code:** [`DecoderLanguageModel.generate()` in `src/nova/models/language_model.py`](file:///d:/AI/hello_ai/src/nova/models/language_model.py#L111-L157)
- **Experiment:** Run `python -m nova.cli generate-lm --prompt "User: What is your name?\nNova:"`.
- **Expected Observation:** Model iteratively outputs `"My"`, `"name"`, `"is"`, `"Nova"`, `"."`.

---

## 26. Temperature and Sampling

- **Equation / Operation:**  
  $$p_i(T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
- **Variables:**  
  $T > 0$: Sampling temperature.  
  $T \to 0$: Greedy argmax selection (deterministic).  
  $T > 1.0$: Uniform distribution (diverse, higher entropy).
- **Source Code:** [`DecoderLanguageModel.generate()` in `src/nova/models/language_model.py`](file:///d:/AI/hello_ai/src/nova/models/language_model.py#L137-L148)
- **Experiment:** Compare generations at `--temp 0.1` vs `--temp 1.8`.
- **Expected Observation:** High temperature produces novel or nonsensical word sequences; low temperature produces repetitive, high-probability tokens.

---

## 27. Intent Classification

- **Equation / Operation:**  
  $$P(\text{intent} = c \mid \mathbf{x}) = \text{softmax}(\mathbf{W}_c \bar{\mathbf{h}} + \mathbf{b}_c)$$
- **Variables:**  
  $\bar{\mathbf{h}} = \frac{1}{\sum m_i} \sum_{i=1}^T m_i \mathbf{h}_i$: Mean pooled representation over non-padding tokens.  
  $c \in \{1, \dots, 12\}$: Target intent class.
- **Source Code:** [`NeuralIntentClassifier.forward()` in `src/nova/models/intent_classifier.py`](file:///d:/AI/hello_ai/src/nova/models/intent_classifier.py#L82-L99)
- **Experiment:** Pass `"What time is it now?"` to classifier.
- **Expected Observation:** Highest probability assigned to `ASK_CURRENT_TIME`.

---

## 28. Confidence Thresholds

- **Equation / Operation:**  
  $$\text{Action} = \begin{cases} \text{Route}(c^*) & \text{if } \max_k p_k \ge \theta_{\text{conf}} \\ \text{UNKNOWN} & \text{if } \max_k p_k < \theta_{\text{conf}} \end{cases}$$
- **Variables:**  
  $\theta_{\text{conf}}$: Minimum confidence threshold (default: 0.65).  
  $c^* = \text{argmax}_k p_k$: Highest scoring intent.
- **Source Code:** [`evaluate_confidence()` in `src/nova/agent/confidence.py`](file:///d:/AI/hello_ai/src/nova/agent/confidence.py#L7-L38)
- **Experiment:** Submit an out-of-domain query like `"What is the capital of France?"` with `--threshold 0.70`.
- **Expected Observation:** Confidence score falls below threshold or routes directly to `UNKNOWN`.

---

## 29. Tool Routing

- **Equation / Operation:**  
  $$\mathcal{R}: \mathcal{I} \to \mathcal{T} \cup \{ \emptyset \}$$  
  $$\mathcal{R}(\text{ASK\_CURRENT\_TIME}) = \text{get\_current\_time}, \quad \mathcal{R}(\text{ASK\_CURRENT\_DATE}) = \text{get\_current\_date}$$
- **Variables:**  
  $\mathcal{I}$: Intent set.  
  $\mathcal{T}$: Allowlisted tool set.
- **Source Code:** [`AgentRouter.INTENT_TO_TOOL` in `src/nova/agent/router.py`](file:///d:/AI/hello_ai/src/nova/agent/router.py#L18-L22)
- **Experiment:** Run chat in debug mode (`--debug`) asking for the time.
- **Expected Observation:** Debug trace displays `Tool Invoked: get_current_time` and `Tool Result: {'formatted_value': '10:30 AM IST'}`.

---

## 30. Evaluation Metrics

- **Equations:**  
  $$\text{Accuracy} = \frac{\sum_{c} \text{TP}_c}{N}$$  
  $$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$  
  $$\text{F1}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}, \quad \text{Macro-F1} = \frac{1}{C} \sum_{c=1}^C \text{F1}_c$$
- **Variables:**  
  $\text{TP}_c$: True Positives for class $c$.  
  $\text{FP}_c$: False Positives.  
  $\text{FN}_c$: False Negatives.  
  $C$: Number of classes.
- **Source Code:** [`compute_classification_metrics()` in `src/nova/training/metrics.py`](file:///d:/AI/hello_ai/src/nova/training/metrics.py#L10-L65)
- **Experiment:** Run `python -m nova.cli evaluate`.
- **Expected Observation:** Full tabular breakdown of accuracy, precision, recall, macro F1, and confusion matrix on the held-out test split.
