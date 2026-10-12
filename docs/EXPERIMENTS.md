# Nova Experimental Learning Exercises

This guide presents 18 hands-on experiments designed to teach the inner workings of neural language models, attention mechanisms, tokenizers, and AI agent architectures.

---

### Experiment 1: Overfitting on a Tiny Dataset
- **Hypothesis:** When trained on a tiny dataset of 5 examples, a neural network will rapidly memorize the sequences and achieve near-zero training loss, but will fail to generalize to novel inputs.
- **Command:**
  ```powershell
  python -c "from nova.training.train_language_model import train_language_model; from nova.config import load_config; cfg=load_config(); train_language_model(cfg, epochs_override=50)"
  ```
- **Expected Observation:** Training loss plunges towards $0.05$, but test perplexity on novel sequences remains high.
- **Explanation:** Without sufficient data variety, high-capacity neural networks memorize exact sample positions rather than learning structural grammatical rules.

---

### Experiment 2: Increasing Dataset Diversity
- **Hypothesis:** Expanding linguistic variety (polite prefixes, alternative word orders) improves held-out test evaluation accuracy.
- **Command:**
  ```powershell
  python -m nova.cli audit-data
  python -m nova.cli evaluate
  ```
- **Expected Observation:** Test set accuracy exceeds 95% on diverse expressions of the same intent.
- **Explanation:** Linguistic augmentation forces self-attention heads to focus on key semantic tokens (e.g., `"time"`, `"name"`) rather than memorizing leading words.

---

### Experiment 3: Changing Vocabulary Size and Inspecting Out-of-Vocabulary (OOV)
- **Hypothesis:** Increasing `min_freq` shrinks vocabulary size and causes rare words to map to `<UNK>`, impacting accuracy.
- **Command:**
  ```powershell
  python -m nova.cli inspect-vocab --top-n 20
  ```
- **Expected Observation:** Less frequent tokens are replaced by index 1 (`<UNK>`). If critical intent words become `<UNK>`, classification accuracy degrades.
- **Explanation:** The vocabulary defines the discrete symbol space. Any token outside this finite set loses its individual semantic identity.

---

### Experiment 4: Modifying Embedding Dimension ($d_{\text{model}}$)
- **Hypothesis:** Increasing $d_{\text{model}}$ from 64 to 128 increases total model parameters quadratically in the attention projection matrices ($\mathcal{O}(d_{\text{model}}^2)$).
- **Command:**
  Modify `d_model: 128` in `config/default.yaml` and run:
  ```powershell
  python -m nova.cli info
  ```
- **Expected Observation:** Total parameters in the Decoder LM jump from ~126k to ~350k+.
- **Explanation:** In Transformer blocks, the $Q, K, V, O$ linear layers each have weight matrices of size $d_{\text{model}} \times d_{\text{model}}$.

---

### Experiment 5: Changing the Number of Transformer Layers
- **Hypothesis:** Stacking more Transformer layers increases model depth and expressive capacity, but increases training latency linearly per epoch.
- **Command:**
  Edit `n_layers: 4` in `config/default.yaml` and run:
  ```powershell
  python -m nova.cli train-lm --epochs 5
  ```
- **Expected Observation:** Per-epoch duration approximately doubles compared to a 2-layer model.
- **Explanation:** Each token must sequentially pass through attention and feed-forward operations at each layer during forward and backward passes.

---

### Experiment 6: Varying Attention Heads ($n_{\text{heads}}$)
- **Hypothesis:** Splitting $d_{\text{model}} = 64$ into 8 heads ($d_k = 8$) versus 2 heads ($d_k = 32$) changes the subspace granularity without altering the total parameter count.
- **Command:**
  Edit `n_heads: 8` in `config/default.yaml` and verify:
  ```powershell
  python -m nova.cli info
  ```
- **Expected Observation:** Parameter count remains identical because $\sum_h (d_{\text{model}} \times d_k) = d_{\text{model}}^2$.
- **Explanation:** Multi-head attention projects into $h$ subspaces of dimension $d_{\text{model}} / h$, keeping the concatenated matrix size constant.

---

### Experiment 7: Disabling Positional Embeddings
- **Hypothesis:** Without positional embeddings, self-attention is permutation-invariant; the model cannot distinguish between `"who are you"` and `"you are who"`.
- **Command:**
  Run unit test:
  ```powershell
  pytest tests/test_attention.py
  ```
- **Expected Observation:** The order of tokens does not change dot-product attention scores unless position vectors $\mathbf{P}_i$ are added to token embeddings.
- **Explanation:** Scaled dot-product attention computes set-based operations over vectors. Positional encodings break this symmetry.

---

### Experiment 8: Comparing Causal Attention with Bidirectional Attention
- **Hypothesis:** A causal mask prevents token $i$ from attending to tokens $j > i$. Removing it allows future tokens to leak into past predictions.
- **Command:**
  Run:
  ```powershell
  pytest tests/test_attention.py::test_causal_masking_prevents_future_leakage
  ```
- **Expected Observation:** In `CausalSelfAttention`, changing token 3 leaves token 0, 1, and 2 outputs strictly identical.
- **Explanation:** The lower-triangular mask sets upper-triangular logits to $-\infty$, ensuring $e^{-\infty} = 0$ after softmax.

---

### Experiment 9: Changing the Learning Rate
- **Hypothesis:** Setting learning rate too high ($\eta = 0.1$) causes divergence/loss explosion; setting it too low ($\eta = 10^{-5}$) stalls convergence.
- **Command:**
  Modify `learning_rate: 0.1` in `config/default.yaml` and run:
  ```powershell
  python -m nova.cli train-lm --epochs 5
  ```
- **Expected Observation:** Loss fluctuates wildly or produces `NaN` due to gradient overshooting.
- **Explanation:** AdamW updates parameters proportional to $\eta$. If the step size exceeds the curvature radius of the loss surface, optimization diverges.

---

### Experiment 10: Comparing Training and Validation Loss Curves
- **Hypothesis:** A healthy training run exhibits decreasing validation loss alongside training loss, stabilizing without diverging upward.
- **Command:**
  Inspect history logs:
  ```powershell
  Get-Content logs/language_model_history.csv
  ```
- **Expected Observation:** Training loss decreases from ~4.1 to ~0.17; validation loss follows closely and plateaus at ~0.16.
- **Explanation:** The absence of a rising validation loss curve confirms that regularizations (dropout = 0.1, weight decay = 0.01) prevent overfitting.

---

### Experiment 11: Inspecting Model Predictions Before and After Training
- **Hypothesis:** Before training, randomly initialized weights produce gibberish token predictions; after training, coherent dialogue structures emerge.
- **Command:**
  Generate text with randomly initialized model vs saved checkpoint:
  ```powershell
  python -m nova.cli generate-lm --prompt "User: What is your name?`nNova:"
  ```
- **Expected Observation:** Trained model outputs: `"My name is Nova. I am a small AI agent trained to understand English questions."`
- **Explanation:** Cross-entropy backpropagation aligns output projection weights with English bigram and n-gram conditional probabilities.

---

### Experiment 12: Comparing Greedy Decoding vs Temperature Sampling
- **Hypothesis:** Greedy decoding ($T = 0$) is completely deterministic, while high temperature ($T = 1.5$) samples lower-probability tokens.
- **Command:**
  ```powershell
  python -m nova.cli generate-lm --temp 0.0 --prompt "User: Hello!`nNova:"
  python -m nova.cli generate-lm --temp 1.8 --prompt "User: Hello!`nNova:"
  ```
- **Expected Observation:** $T=0.0$ returns the exact same string every run. $T=1.8$ produces varied or unconventional continuations.
- **Explanation:** Scaling logits by $1/T$ flattens the probability distribution as $T \to \infty$, approaching uniform random selection.

---

### Experiment 13: Comparing Neural Classifier with Rule-Based Baseline
- **Hypothesis:** The Neural Classifier handles rephrased queries that do not match exact keywords, outperforming the rule matcher on unseen test data.
- **Command:**
  ```powershell
  python -m nova.cli evaluate
  ```
- **Expected Observation:** Neural Classifier achieves ~98-100% accuracy; rule baseline achieves ~70%.
- **Explanation:** Self-attention learns distributed continuous semantic representations across all tokens rather than relying on brittle surface substring matches.

---

### Experiment 14: Changing Confidence Threshold
- **Hypothesis:** Raising confidence threshold to 0.95 increases false rejections of valid questions; lowering it to 0.20 risks falsely accepting out-of-domain inputs.
- **Command:**
  ```powershell
  python -m nova.cli chat --threshold 0.98
  ```
- **Expected Observation:** Marginal inputs are routed to `UNKNOWN` fallback.
- **Explanation:** Softmax confidence represents peak distribution sharpness. Calibration curves dictate precision-recall trade-offs.

---

### Experiment 15: Disabling Dynamic Tools
- **Hypothesis:** A pure language model cannot know the real current time or date without an external tool call; it can only hallucinate training examples (e.g., `"10:30 AM"`).
- **Command:**
  Compare the tool-augmented agent output with the raw LM output:
  ```powershell
  python -m nova.cli chat --debug
  python -m nova.cli generate-lm --prompt "User: What is the time?`nNova:"
  ```
- **Expected Observation:** The agent returns the live system clock time in Asia/Kolkata (`10:21 AM IST`). The raw LM reproduces static training dialogue text.
- **Explanation:** Neural weights are static after training. Real-time dynamic ground truth requires grounded tool execution.

---

### Experiment 16: Ambiguous Questions and Routing Diagnostics
- **Hypothesis:** Asking overlapping queries like `"tell me about the clock"` tests boundary distinctions between `ASK_CURRENT_TIME` and `UNKNOWN`.
- **Command:**
  ```powershell
  python -m nova.cli chat --debug
  ```
  Input: `"Tell me about the clock."`
- **Expected Observation:** Debug trace displays softmax probabilities across competing classes and shows which intent was selected.
- **Explanation:** Ambiguous inputs distribute probability mass across multiple classes, reducing peak confidence.

---

### Experiment 17: Checkpoint Persistence and Reloading
- **Hypothesis:** Saving model weights, config, and vocabulary to disk allows a completely fresh process to load and run inference identically.
- **Command:**
  ```powershell
  python tests/test_live_smoke.py
  ```
- **Expected Observation:** Clean startup from disk without retraining, producing verified outputs across all supported intents.
- **Explanation:** Serialized state dicts store all learned tensor weights $(\mathbf{W}, \mathbf{b})$.

---

### Experiment 18: Modifying the Random Training Seed
- **Hypothesis:** Changing random seed from 42 to 123 generates a slightly different dataset partition and random weight initialization, but converges to similar final metrics.
- **Command:**
  Edit `seed: 123` in `config/default.yaml` and re-run:
  ```powershell
  python -m nova.cli generate-data
  python -m nova.cli train-intent --epochs 10
  ```
- **Expected Observation:** Slight variation in per-epoch loss trajectories, but convergence to near-identical accuracy.
- **Explanation:** Stochastic gradient descent with AdamW is robust across different initial random weight configurations on convex or well-behaved manifolds.
