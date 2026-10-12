"""Synthetic training data generator for Nova language model and intent classifier.

Generates diverse question-and-answer pairs across all supported intents,
splits without leakage into train/val/test, and saves JSONL records with a manifest.
"""

from datetime import datetime
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple


INTENTS = [
    "GREETING",
    "ASK_NAME",
    "ASK_IDENTITY",
    "ASK_HUMAN",
    "ASK_CAPABILITIES",
    "ASK_CURRENT_TIME",
    "ASK_CURRENT_DATE",
    "ASK_MODEL_TYPE",
    "ASK_SUPPORTED_TOPICS",
    "ASK_OFFLINE_STATUS",
    "THANKS",
    "GOODBYE",
    "UNKNOWN",
]

# Base question patterns per intent
INTENT_PATTERNS: Dict[str, List[str]] = {
    "GREETING": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "greetings",
        "hey there",
        "hello there",
        "hi nova",
        "hello nova",
        "hey nova",
        "howdy",
        "hiya",
        "welcome",
        "nice to meet you",
        "good day",
        "salutations",
        "pleased to meet you",
    ],
    "ASK_NAME": [
        "what is your name",
        "what's your name",
        "tell me your name",
        "may i know your name",
        "can you tell me your name",
        "what should i call you",
        "what do you call yourself",
        "do you have a name",
        "who do i have the pleasure of speaking with",
        "give me your name",
        "your name please",
        "what are you called",
        "by what name are you known",
        "could you share your name",
        "tell me who you are called",
        "what is the name given to you",
        "state your name",
    ],
    "ASK_IDENTITY": [
        "who are you",
        "who're you",
        "what are you",
        "are you real",
        "are you a bot",
        "are you an ai",
        "are you an artificial intelligence",
        "can you introduce yourself",
        "please introduce yourself",
        "tell me about yourself",
        "describe yourself",
        "explain who you are",
        "what kind of assistant are you",
        "who created you",
        "give me an introduction of yourself",
        "are you a computer program",
        "what is your identity",
    ],
    "ASK_HUMAN": [
        "are you a human",
        "are you human",
        "are you a person",
        "are you a real person",
        "are you a real human",
        "are you a human being",
        "are you a living person",
        "are you a living human",
        "are you human or robot",
        "are you an actual person",
    ],
    "ASK_CAPABILITIES": [
        "what can you do",
        "what are your capabilities",
        "what are your features",
        "what functions do you support",
        "how can you help me",
        "what can you help me with",
        "tell me what you can do",
        "list your capabilities",
        "what are you capable of",
        "what tasks can you perform",
        "show me what you can do",
        "what skills do you have",
        "what are your abilities",
        "explain your capabilities",
        "can you do anything useful",
        "give me a summary of your features",
        "what services do you offer",
    ],
    "ASK_CURRENT_TIME": [
        "what time is it",
        "what is the time",
        "what's the time",
        "tell me the time",
        "can you tell me the time",
        "could you tell me the current time",
        "what time is it now",
        "current time please",
        "do you know what time it is",
        "what is the current time in kolkata",
        "give me the current time",
        "tell me what time it is right now",
        "can i get the current time",
        "check the clock for me",
        "what does the clock say",
        "time check please",
        "what hour is it currently",
        "could you check the time",
    ],
    "ASK_CURRENT_DATE": [
        "what is today's date",
        "what's today's date",
        "what is the date today",
        "tell me today's date",
        "what date is it",
        "what's the date",
        "can you tell me the date",
        "what day is it today",
        "what day of the week is it",
        "what is the current date",
        "tell me the current date",
        "give me today's date",
        "which date is it today",
        "today's date please",
        "check today's date",
        "what is today's day and date",
        "tell me the date right now",
    ],
    "ASK_MODEL_TYPE": [
        "what model are you",
        "what kind of model are you",
        "what is your model architecture",
        "how were you built",
        "are you a transformer",
        "what neural network do you use",
        "how do you work",
        "tell me about your architecture",
        "are you an llm",
        "what language model powers you",
        "are you a pretrained model",
        "describe your neural architecture",
        "are you built with pytorch",
        "how are you trained",
        "explain your model structure",
    ],
    "ASK_SUPPORTED_TOPICS": [
        "what topics do you support",
        "what questions can i ask you",
        "what can i ask you about",
        "which topics can you handle",
        "tell me your supported topics",
        "what are your supported questions",
        "what kinds of things can you answer",
        "what are the questions you know",
        "list the topics you understand",
        "what are your supported intents",
        "what domain do you cover",
        "which queries can you process",
    ],
    "ASK_OFFLINE_STATUS": [
        "are you offline",
        "do you run offline",
        "do you need internet",
        "are you running locally",
        "do you use an external api",
        "are you connected to the cloud",
        "can you work without internet",
        "is this running on my computer",
        "do you send my data anywhere",
        "are you a local model",
        "does this model need web access",
        "are you completely offline",
    ],
    "THANKS": [
        "thank you",
        "thanks",
        "thank you very much",
        "thanks a lot",
        "thanks so much",
        "many thanks",
        "i appreciate it",
        "i appreciate your help",
        "thank you nova",
        "thanks nova",
        "much appreciated",
        "that is helpful thank you",
        "great thanks",
        "awesome thank you",
        "thanks a million",
    ],
    "GOODBYE": [
        "goodbye",
        "bye",
        "bye bye",
        "see you later",
        "see ya",
        "farewell",
        "talk to you later",
        "have a good day",
        "good night",
        "catch you later",
        "exit",
        "quit",
        "bye nova",
        "goodbye nova",
        "see you soon",
        "have a great rest of your day",
    ],
    "UNKNOWN": [
        "what is the capital of france",
        "who is the president of the united states",
        "how far is the moon",
        "what is the meaning of life",
        "can you write a poem about love",
        "how do airplanes fly",
        "solve this math problem two plus two",
        "tell me a recipe for chocolate cake",
        "who painted the mona lisa",
        "what is quantum physics",
        "play my favorite song",
        "how is the weather in new york today",
        "book a flight to london",
        "can you write python code for web scraping",
        "translate this sentence into spanish",
        "what is photosynthesis",
        "how old is the universe",
        "who won the world cup in 2022",
        "tell me a funny joke",
        "can you order pizza for me",
        "what is the speed of light",
        "explain general relativity",
        "what is machine learning in healthcare",
        "how do i fix a flat tire",
        "summarize the latest news",
        "what is the stock price of apple",
        "write an essay on climate change",
        "can you play chess with me",
        "what is the tallest mountain in the world",
        "how does a rocket engine work",
        "how many continents are there",
        "who invented the telephone",
        "what is the distance to mars",
    ],
}

POLITE_PREFIXES = [
    "",
    "please ",
    "could you please ",
    "can you ",
    "would you ",
    "kindly ",
    "hey ",
    "nova ",
    "excuse me ",
    "i want to know ",
    "can you please ",
    "tell me ",
    "do you know ",
    "i would like to ask ",
    "may i ask ",
]

PUNCTUATION_SUFFIXES = [
    "",
    "?",
    "!",
    ".",
]


def generate_intent_dataset(
    seed: int = 42,
    target_per_intent: int = 120,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Generates strictly non-overlapping, deduplicated train, validation, and test intent datasets."""
    rng = random.Random(seed)

    train_data: List[Dict[str, str]] = []
    val_data: List[Dict[str, str]] = []
    test_data: List[Dict[str, str]] = []

    global_seen_normalized: Set[str] = set()

    for intent, bases in INTENT_PATTERNS.items():
        intent_unique_samples: List[str] = []
        intent_normalized_seen: Set[str] = set()

        # Generate unique variations
        attempts = 0
        while len(intent_unique_samples) < target_per_intent and attempts < target_per_intent * 50:
            attempts += 1
            base = rng.choice(bases)
            prefix = rng.choice(POLITE_PREFIXES)
            suffix = rng.choice(PUNCTUATION_SUFFIXES)

            # Natural concatenation
            candidate = f"{prefix}{base}".strip()
            if suffix and not candidate.endswith(suffix):
                candidate = f"{candidate}{suffix}"

            # Normalize for deduplication check (lowercase, collapsed spaces, stripped punctuation)
            norm_key = "".join(c for c in candidate.lower() if c.isalnum() or c.isspace())
            norm_key = " ".join(norm_key.split())

            if norm_key and norm_key not in global_seen_normalized and norm_key not in intent_normalized_seen:
                intent_normalized_seen.add(norm_key)
                global_seen_normalized.add(norm_key)
                intent_unique_samples.append(candidate)

        # Shuffle deterministic
        rng.shuffle(intent_unique_samples)

        # Strict 70% / 15% / 15% partition with disjoint slices
        n = len(intent_unique_samples)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        train_slice = intent_unique_samples[:n_train]
        val_slice = intent_unique_samples[n_train : n_train + n_val]
        test_slice = intent_unique_samples[n_train + n_val :]

        for text in train_slice:
            train_data.append({"text": text, "intent": intent})
        for text in val_slice:
            val_data.append({"text": text, "intent": intent})
        for text in test_slice:
            test_data.append({"text": text, "intent": intent})

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)

    return train_data, val_data, test_data


def generate_lm_conversations(
    seed: int = 42,
    total_examples: int = 600,
) -> List[Dict[str, str]]:
    """Generates educational conversation dialogues for language model training."""
    rng = random.Random(seed)

    dialogue_templates = [
        # GREETING
        ("Hello!", "Hello! I am Nova. How can I help you?"),
        ("Hi there!", "Hello! I am Nova, an educational AI agent."),
        ("Good morning Nova.", "Good morning! How can I assist you today?"),
        ("Hey Nova!", "Hello! What can I help you with today?"),
        ("Greetings!", "Greetings! I am Nova. Feel free to ask me questions."),
        # ASK_NAME
        ("What is your name?", "My name is Nova. I am a small AI agent trained to understand English questions."),
        ("Tell me your name.", "My name is Nova."),
        ("Who are you called?", "I am called Nova."),
        ("What should I call you?", "You can call me Nova."),
        # ASK_IDENTITY
        ("Who are you?", "I am Nova, an educational AI agent."),
        ("Are you a human?", "No. I am an AI program running on your computer."),
        ("What are you?", "I am Nova, a small neural AI agent built to demonstrate language processing."),
        ("Can you introduce yourself?", "I am Nova. I am a small AI agent running locally on your computer."),
        # ASK_CAPABILITIES
        ("What can you do?", "I can answer questions about my identity and capabilities, respond to greetings, and retrieve the current date and time."),
        ("What are your capabilities?", "I can answer questions about myself, handle greetings, and query the system clock for date and time."),
        ("How can you help me?", "You can ask me about my name, identity, architecture, or ask for the live time and date."),
        # ASK_MODEL_TYPE
        ("What model are you?", "I am a small Transformer-based language model trained from scratch."),
        ("What is your architecture?", "I am built with a multi-head causal self-attention Transformer."),
        ("How were you trained?", "I was initialized with random weights and trained using next-token cross-entropy loss."),
        # ASK_OFFLINE_STATUS
        ("Are you running offline?", "Yes. I run entirely locally on your computer without external internet access."),
        ("Do you need internet?", "No. I run completely offline on your local machine."),
        # ASK_SUPPORTED_TOPICS
        ("What topics do you support?", "I support questions about my name, identity, capabilities, architecture, and live date and time."),
        # THANKS
        ("Thank you!", "You are very welcome!"),
        ("Thanks for your help.", "Happy to help anytime!"),
        ("I appreciate it.", "You are welcome! Let me know if you need anything else."),
        # GOODBYE
        ("Goodbye!", "Goodbye! Have a great day!"),
        ("Bye Nova.", "Farewell! Feel free to talk again anytime."),
        ("See you later.", "Goodbye! Take care!"),
        # UNKNOWN
        ("What is the capital of France?", "I have not been trained to answer general knowledge questions yet. I can answer questions about myself and my supported capabilities."),
        ("Can you write a poem about space?", "I have not been trained to answer general knowledge questions yet. I can answer questions about myself and my supported capabilities."),
        ("What is quantum physics?", "I have not been trained to answer general knowledge questions yet. I can answer questions about myself and my supported capabilities."),
        # DATE & TIME
        ("What is the time?", "It is currently 10:30 AM IST."),
        ("Can you tell me the time?", "The current time is 10:30 AM IST."),
        ("What is today's date?", "Today is Monday, 12 October 2026."),
        ("Tell me today's date.", "Today is Monday, 12 October 2026."),
    ]

    examples: List[Dict[str, str]] = []
    while len(examples) < total_examples:
        u, r = rng.choice(dialogue_templates)
        if rng.random() > 0.5:
            u_clean = u.lower()
        else:
            u_clean = u
        text = f"User: {u_clean}\nNova: {r}"
        examples.append({"text": text})

    return examples


def save_jsonl(records: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_all_datasets(
    data_dir: str = "data/generated",
    seed: int = 42,
    target_per_intent: int = 120,
    total_lm_examples: int = 600,
) -> Dict[str, any]:
    base_dir = Path(data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    train_intent, val_intent, test_intent = generate_intent_dataset(
        seed=seed,
        target_per_intent=target_per_intent,
    )

    lm_train = generate_lm_conversations(
        seed=seed,
        total_examples=total_lm_examples,
    )

    lm_train_path = base_dir / "language_model_train.jsonl"
    intent_train_path = base_dir / "intent_train.jsonl"
    intent_val_path = base_dir / "intent_validation.jsonl"
    intent_test_path = base_dir / "intent_test.jsonl"
    manifest_path = base_dir / "dataset_manifest.json"

    save_jsonl(lm_train, lm_train_path)
    save_jsonl(train_intent, intent_train_path)
    save_jsonl(val_intent, intent_val_path)
    save_jsonl(test_intent, intent_test_path)

    intent_counts: Dict[str, Dict[str, int]] = {}
    for split_name, dataset in [("train", train_intent), ("val", val_intent), ("test", test_intent)]:
        counts: Dict[str, int] = {}
        for item in dataset:
            counts[item["intent"]] = counts.get(item["intent"], 0) + 1
        intent_counts[split_name] = counts

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "seed": seed,
        "intents": INTENTS,
        "target_per_intent": target_per_intent,
        "files": {
            "language_model_train": {
                "path": str(lm_train_path),
                "count": len(lm_train),
            },
            "intent_train": {
                "path": str(intent_train_path),
                "count": len(train_intent),
            },
            "intent_validation": {
                "path": str(intent_val_path),
                "count": len(val_intent),
            },
            "intent_test": {
                "path": str(intent_test_path),
                "count": len(test_intent),
            },
        },
        "intent_distribution": intent_counts,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest
