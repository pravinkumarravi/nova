# Nova Dataset Design, Generation, and Audit

This document explains the synthetic dataset generation pipeline, data schemas, linguistic variation mechanics, and anti-leakage audit procedures implemented for Nova.

---

## 1. Design Philosophy

Nova does not rely on external scraped corpora or proprietary datasets. Instead, all training and evaluation datasets are generated **programmatically and deterministically** from scratch using Python scripts ([`src/nova/data/generate_data.py`](file:///d:/AI/hello_ai/src/nova/data/generate_data.py)).

Key objectives:
1. **Linguistic Diversity:** Rather than simple word substitution, Nova generates diverse question formulations, politeness prefixes, casing variations, punctuation varieties, and natural rephrasings.
2. **Strict Stratified Splitting:** 70% Training, 15% Validation, 15% Test.
3. **Guaranteed Zero Data Leakage:** Case-insensitive, punctuation-stripped canonical keys ensure that no evaluation example (or near-identical duplicate) ever appears in the training split.
4. **Balanced Distribution:** Every supported intent receives an equal number of unique samples.
5. **Robust UNKNOWN & OOD Examples:** Explicitly includes diverse unsupported queries (general knowledge, mathematics, creative writing, science) to train the model to recognize its own operational boundaries.

---

## 2. Dataset Files & Formats

All datasets are serialized in UTF-8 encoded JSONL (JSON Lines) format within `data/generated/`:

| File | Purpose | Sample Count |
| :--- | :--- | :--- |
| `language_model_train.jsonl` | Autoregressive dialogue training for Decoder LM | 600 records |
| `intent_train.jsonl` | Training set for Neural Intent Classifier | 1,008 records (84 per intent) |
| `intent_validation.jsonl` | Hyperparameter tuning and model checkpointing | 216 records (18 per intent) |
| `intent_test.jsonl` | Final held-out generalization benchmark | 216 records (18 per intent) |
| `dataset_manifest.json` | Provenance metadata, random seeds, and split counts | Single JSON document |

### Record Schemas

#### A. Intent Record
```json
{
  "text": "Could you please tell me what time it is?",
  "intent": "ASK_CURRENT_TIME"
}
```

#### B. Language Model Dialogue Record
```json
{
  "text": "User: What is your name?\nNova: My name is Nova. I am a small AI agent trained to understand English questions."
}
```

---

## 3. Supported Intent Taxonomy

Nova supports 12 distinct intents:

1. `GREETING`: Salutations and introductory greetings ("Hello", "Hi there", "Good morning").
2. `ASK_NAME`: Inquiries regarding the agent's name ("What is your name?", "What should I call you?").
3. `ASK_IDENTITY`: Inquiries regarding agent nature and human status ("Who are you?", "Are you a human?", "What are you?").
4. `ASK_CAPABILITIES`: Questions about features and tasks ("What can you do?", "What are your capabilities?").
5. `ASK_CURRENT_TIME`: Queries requesting live system time in Asia/Kolkata ("What time is it now?", "Check the clock").
6. `ASK_CURRENT_DATE`: Queries requesting live system date and weekday in Asia/Kolkata ("What is today's date?", "What day is it?").
7. `ASK_MODEL_TYPE`: Technical architecture queries ("What is your model architecture?", "Are you a transformer?").
8. `ASK_SUPPORTED_TOPICS`: Scope inquiries ("What topics do you support?", "What can I ask you?").
9. `ASK_OFFLINE_STATUS`: Offline and privacy status queries ("Do you run offline?", "Do you need internet?").
10. `THANKS`: Gratitude expressions ("Thank you", "Thanks a lot", "Much appreciated").
11. `GOODBYE`: Exit and farewell phrases ("Goodbye", "Bye Nova", "See you later").
12. `UNKNOWN`: Out-of-domain, unsupported, or general knowledge questions ("What is the capital of France?", "Solve 2+2").

---

## 4. Linguistic Variation Generation Engine

For each intent, base question templates are systematically expanded using:

1. **Politeness & Conversational Prefixes:**
   `["", "please ", "could you please ", "can you ", "would you ", "kindly ", "hey ", "nova ", "excuse me ", "i want to know ", "tell me "]`
2. **Punctuation Suffixes:**
   `["", "?", "!", "."]`
3. **Case Variation:**
   Natural lowercase, sentence casing, and capitalization variations.
4. **Canonical Deduplication:**
   Each candidate string is mapped to a normalized key $\text{key} = \text{alnum\_only}(\text{lower}(s))$. If the key has already been assigned to any split or intent, it is rejected to prevent memorization.

---

## 5. Audit Results & Leakage Verification

Auditing is performed via `python -m nova.cli audit-data`. The auditor verifies:
- Duplicates within each split = 0
- Overlap between Train and Validation = 0
- Overlap between Train and Test = 0
- Overlap between Validation and Test = 0
- Balance across all 12 intents = Exactly 84 / 18 / 18 samples per intent

### Audit Summary Output:
```json
{
  "status": "PASSED",
  "counts": {
    "language_model_train": 600,
    "intent_train": 1008,
    "intent_validation": 216,
    "intent_test": 216
  },
  "duplicates_within_splits": {
    "intent_train": 0,
    "intent_validation": 0,
    "intent_test": 0
  },
  "data_leakage": {
    "train_val_overlap": 0,
    "train_test_overlap": 0,
    "val_test_overlap": 0
  }
}
```
Status: **PASSED (Zero data leakage)**
