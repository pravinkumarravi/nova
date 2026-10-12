# Nova Model Training Pipelines

This guide details the explicit training procedures for both neural models in Nova: the **Decoder-Only Language Model** and the **Neural Intent Classifier**.

---

## 1. Overview & Training Philosophy

Nova uses pure, transparent PyTorch training loops without high-level abstraction frameworks (like Hugging Face Trainer or PyTorch Lightning) that obscure backpropagation, gradient calculation, or tensor manipulation.

Both models are initialized with **random weights** (no pretrained models or external weights) and trained entirely locally on CPU.

---

## 2. Decoder Language Model Training Pipeline

### Mathematical Objective
The model minimizes cross-entropy loss over next-token predictions across teacher-forced conversation sequences:
$$\mathcal{L}_{\text{LM}}(\theta) = -\frac{1}{T} \sum_{t=1}^T \log P_\theta(w_t \mid w_{<t})$$

### Data Pipeline
- Padded to `max_seq_len = 64`
- Input sequence $\mathbf{x} = [w_0, w_1, \dots, w_{T-1}]$
- Target sequence $\mathbf{y} = [w_1, w_2, \dots, w_T]$
- `ignore_index = 0` (`<PAD>` ID) ensures padding tokens do not contribute to gradients or loss values.

### Step-by-Step Execution
1. Forward pass: `logits, loss = model(x, targets=y)`
2. Zero gradients: `optimizer.zero_grad()`
3. Backward pass: `loss.backward()` (automatic differentiation through causal self-attention layers)
4. Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
5. Optimizer update: `optimizer.step()` (AdamW with $\beta_1=0.9, \beta_2=0.999$, weight decay 0.01)

### Execution Command
```powershell
python -m nova.cli train-lm
```

Default hyperparameters:
- Embedding dimension: 64
- Transformer layers: 2
- Attention heads: 4
- Feed-forward dimension: 256
- Sequence length: 64
- Learning rate: 0.001
- Batch size: 16
- Epochs: 35
- Duration on CPU: ~45 seconds

---

## 3. Neural Intent Classifier Training Pipeline

### Mathematical Objective
Minimizes multi-class cross-entropy loss over the 12 supported intent classes:
$$\mathcal{L}_{\text{intent}}(\theta) = -\sum_{c=1}^{12} y_c \log(\text{softmax}(\mathbf{z})_c)$$

### Architecture & Pooling
- Bidirectional Transformer (non-causal attention allows tokens to attend to the entire question context).
- Mean pooling over active non-padded tokens extracts a sequence-level semantic vector.
- Output projection maps pooled vector to class logits.

### Evaluation Metrics
At each epoch and after training on the held-out test split, Nova computes:
- Accuracy
- Macro Precision, Macro Recall, Macro F1
- Per-intent precision, recall, and support
- 12x12 Confusion Matrix

### Execution Command
```powershell
python -m nova.cli train-intent
```

Default hyperparameters:
- Embedding dimension: 64
- Transformer layers: 2
- Attention heads: 4
- Sequence length: 32
- Learning rate: 0.001
- Batch size: 16
- Epochs: 30
- Duration on CPU: ~40 seconds

---

## 4. Artifact & Checkpoint Directory Structure

Saved artifacts in `artifacts/`:

```
artifacts/
├── language_model/
│   ├── best_model.pt           # Checkpoint with lowest validation loss
│   ├── last_model.pt           # Final epoch model weights
│   ├── tokenizer_config.json   # Tokenizer configuration
│   └── vocab.json              # Learned vocabulary & token counts
└── intent_classifier/
    ├── best_model.pt           # Checkpoint with highest validation Macro-F1
    ├── labels.json             # Intent string to integer index mapping
    ├── evaluation_report.json  # Comprehensive test split metrics
    ├── tokenizer_config.json   # Tokenizer configuration
    └── vocab.json              # Learned vocabulary
```

Training logs (JSON and CSV) are saved in `logs/`:
- `logs/language_model_history.json` and `.csv`
- `logs/intent_classifier_history.json` and `.csv`
