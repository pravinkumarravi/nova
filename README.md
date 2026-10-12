# Nova: Build a Small English Language Model and AI Agent from Scratch

Nova is a fully transparent, educational AI project implemented completely from scratch using Python and PyTorch. Its purpose is to teach how a language model learns English, processes user input, predicts outputs, and calls external tools without relying on pretrained weights, black-box frameworks, or commercial LLM APIs.

```
       +-------------------------------------------------------------+
       |                         USER INPUT                          |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |               Input Normalization & Tokenizer               |
       |  (Whitespace, case folding, regex tokens, vocabulary IDs)   |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |               Neural Intent Classifier (from scratch)       |
       |        Bidirectional Transformer + Mean Pooling (64-dim)    |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |             Confidence Evaluation (Threshold: 0.65)         |
       |   Passed? ──No──> Safe Fallback: UNKNOWN (Out-of-Scope)     |
       +-------------------------------------------------------------+
                                      | Yes
                                      v
       +-------------------------------------------------------------+
       |                       Agent Router                          |
       |  • Conversational Intent ──> Controlled Response Generator  |
       |  • Time/Date Query       ──> Allowlisted Local Tool Call    |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |            Tool Execution (Asia/Kolkata timezone)           |
       |       get_current_time  /  get_current_date (ZoneInfo)      |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                      TERMINAL OUTPUT                        |
       +-------------------------------------------------------------+
```

---

## 1. What Nova Can and Cannot Do

### What Nova Can Do:
- **Understand supported domain questions:** Accurately classifies and responds to questions about its identity, name, capabilities, model architecture, offline status, and greetings.
- **Provide live date and time:** Queries local system time and formats real-time answers in the `Asia/Kolkata` timezone (e.g., `10:21 AM IST`, `Monday, 12 October 2026`).
- **Reject out-of-domain questions safely:** Refuses general knowledge, mathematics, and arbitrary queries without pretending to know or hallucinating answers.
- **Generate raw autoregressive text:** Includes an independently trainable Decoder-Only Transformer that predicts next tokens causally from scratch.
- **Provide complete internal transparency:** Offers an interactive debug mode that displays token IDs, softmax confidence scores, and tool invocation metadata.

### What Nova Cannot Do:
- Nova **cannot** answer general knowledge questions (e.g., "What is the capital of France?").
- Nova is **not** a multi-billion-parameter general LLM; it is a compact, inspectable educational architecture designed to run on standard CPUs.
- Nova does **not** execute arbitrary code or use `eval()`; all tools are strictly allowlisted.

---

## 2. Project Structure

```
nova/
├── README.md                          # Project documentation and guide
├── pyproject.toml                     # Project packaging configuration
├── requirements.txt                   # Dependency specification
├── .gitignore                         # Git ignore configuration
├── config/
│   ├── default.yaml                   # Model and pipeline hyperparameters
│   └── identity.yaml                  # Agent identity and canned responses
├── data/
│   ├── generated/                     # Synthetic JSONL datasets and manifest
│   │   ├── dataset_manifest.json
│   │   ├── intent_train.jsonl
│   │   ├── intent_validation.jsonl
│   │   ├── intent_test.jsonl
│   │   └── language_model_train.jsonl
│   └── processed/
├── artifacts/
│   ├── language_model/                # LM weights, config, and vocabulary
│   └── intent_classifier/             # Classifier weights, labels, and report
├── logs/                              # Training history (JSON and CSV)
├── src/
│   └── nova/
│       ├── __init__.py
│       ├── config.py                  # YAML and dataclass settings loader
│       ├── cli.py                     # Unified CLI with subcommands
│       ├── data/
│       │   ├── generate_data.py       # Programmatic synthetic data generator
│       │   ├── dataset.py             # PyTorch Dataset wrappers
│       │   └── audit.py               # Dataset auditor and leakage detector
│       ├── tokenizer/
│       │   ├── vocabulary.py          # Vocabulary and special tokens
│       │   └── tokenizer.py           # Word and punctuation tokenizer
│       ├── models/
│       │   ├── attention.py           # Multi-Head and Causal Self-Attention
│       │   ├── transformer.py         # Transformer Block (Pre-LN + FFN)
│       │   ├── language_model.py      # Decoder-Only Causal Language Model
│       │   └── intent_classifier.py   # Bidirectional Intent Classifier
│       ├── training/
│       │   ├── train_language_model.py
│       │   ├── train_intent_classifier.py
│       │   ├── checkpoint.py          # State dict save/load verification
│       │   └── metrics.py             # Accuracy, F1, confusion matrix, logger
│       ├── inference/
│       │   ├── generate.py            # Autoregressive text generator
│       │   └── load_model.py          # Checkpoint and artifact loaders
│       ├── agent/
│       │   ├── router.py              # Pipeline coordinator
│       │   ├── response_generator.py  # Factual response synthesizer
│       │   └── confidence.py          # Softmax confidence evaluator
│       └── tools/
│           ├── registry.py            # Sandboxed tool registry
│           └── date_time.py           # Deterministic Date/Time in Asia/Kolkata
├── tests/                             # Full automated test suite (29 tests)
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   ├── test_attention.py
│   ├── test_language_model.py
│   ├── test_intent_classifier.py
│   ├── test_tools.py
│   ├── test_router.py
│   ├── test_integration.py
│   └── test_live_smoke.py
└── docs/                              # Comprehensive technical documentation
    ├── ARCHITECTURE.md                # System design and component data flow
    ├── MATHEMATICS.md                 # 30 mathematical concepts with equations
    ├── DATASET.md                     # Synthetic data design and schemas
    ├── TRAINING.md                    # Training loops and loss objectives
    ├── INFERENCE.md                   # Agent and LM execution modes
    ├── WINDOWS_SETUP.md               # PowerShell setup instructions
    ├── TROUBLESHOOTING.md             # Common Windows & PyTorch solutions
    └── EXPERIMENTS.md                 # 18 guided experimental learning exercises
```

---

## 3. Quick Start (Windows & PowerShell)

### Step 1: Create and Activate Virtual Environment
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 3: Set PYTHONPATH
```powershell
$env:PYTHONPATH = "src"
```

### Step 4: Run the Complete Test Suite
```powershell
pytest -v tests
```

---

## 4. Command-Line Interface (CLI)

Nova provides a single, consistent CLI with dedicated subcommands:

```powershell
# 1. Generate synthetic datasets programmatically
python -m nova.cli generate-data

# 2. Audit datasets for zero data leakage
python -m nova.cli audit-data

# 3. Inspect vocabulary and top token frequencies
python -m nova.cli inspect-vocab --top-n 15

# 4. Train the Decoder Language Model from scratch
python -m nova.cli train-lm

# 5. Train the Neural Intent Classifier from scratch
python -m nova.cli train-intent

# 6. Evaluate on held-out test data (Neural vs Baseline)
python -m nova.cli evaluate

# 7. Start the interactive agent chat
python -m nova.cli chat

# 8. Start chat in debug mode (shows tokens, logits, tool output)
python -m nova.cli chat --debug

# 9. Run direct text generation with the Decoder Language Model
python -m nova.cli generate-lm --prompt "User: What is your name?`nNova:"

# 10. Display system and model parameter counts
python -m nova.cli info
```

---

## 5. Live Demonstration Results

### Real Dialogue Excerpt
```text
User: What is your name?
Nova: My name is Nova. I am a small AI agent trained to understand a limited set of English questions.

User: Who are you?
Nova: I am Nova, an educational AI agent.

User: Are you a human?
Nova: No. I am an AI program running on your computer.

User: What can you do?
Nova: I can answer questions about my identity and capabilities, respond to greetings, and retrieve the current date and time.

User: Can you tell me the time?
Nova: It is currently 10:21 AM IST.

User: What is today's date?
Nova: Today is Monday, 12 October 2026.

User: Hello!
Nova: Hello! I am Nova. How can I help you?

User: What is the capital of France?
Nova: I have not been trained to answer general knowledge questions yet. I can answer questions about myself and my supported capabilities.
```

---

## 6. Evaluation Performance

Evaluated on the held-out test split (234 unseen examples, zero leakage):

| Model | Accuracy | Macro F1 | Notes |
| :--- | :--- | :--- | :--- |
| **Neural Intent Classifier** | **100.00%** | **1.0000** | Transformer encoder with bidirectional self-attention |
| **Rule-Based Baseline** | **70.94%** | **0.7410** | Pattern/keyword matching heuristic |

---

## 7. Educational Documentation

Detailed reference guides are located in `docs/`:

- [`docs/ARCHITECTURE.md`](file:///d:/AI/hello_ai/docs/ARCHITECTURE.md): Component interactions, security boundaries, and data flow.
- [`docs/MATHEMATICS.md`](file:///d:/AI/hello_ai/docs/MATHEMATICS.md): All 30 core mathematical concepts explained with formulas, code references, and worked examples.
- [`docs/DATASET.md`](file:///d:/AI/hello_ai/docs/DATASET.md): Synthetic data generation, distribution, and zero-leakage guarantees.
- [`docs/TRAINING.md`](file:///d:/AI/hello_ai/docs/TRAINING.md): Training pipelines, loss objectives, and optimizer configurations.
- [`docs/INFERENCE.md`](file:///d:/AI/hello_ai/docs/INFERENCE.md): Agent and language model execution modes.
- [`docs/WINDOWS_SETUP.md`](file:///d:/AI/hello_ai/docs/WINDOWS_SETUP.md): Windows setup, execution policies, and package management.
- [`docs/TROUBLESHOOTING.md`](file:///d:/AI/hello_ai/docs/TROUBLESHOOTING.md): Solutions for common setup and runtime issues.
- [`docs/EXPERIMENTS.md`](file:///d:/AI/hello_ai/docs/EXPERIMENTS.md): 18 guided experimental learning exercises with hypotheses and commands.

---

## 8. Limitations & Future Milestones

- **Vocabulary:** Nova currently uses a word-level tokenizer. Future iterations can introduce Byte-Pair Encoding (BPE) subword tokenization.
- **Domain Scope:** Nova intentionally bounds its scope to identity, capabilities, and date/time queries to demonstrate reliable agent behavior without hallucinations.
- **Hardware Requirement:** Nova runs entirely on CPU in seconds without requiring an external GPU or internet connection.
