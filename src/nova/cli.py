"""Command-Line Interface (CLI) for Nova."""

import argparse
import json
from pathlib import Path
import sys
import torch

from nova.config import load_config
from nova.data.generate_data import generate_all_datasets
from nova.data.audit import audit_datasets
from nova.tokenizer.tokenizer import SimpleTokenizer
from nova.training.train_language_model import train_language_model
from nova.training.train_intent_classifier import train_intent_classifier
from nova.inference.load_model import load_language_model, load_agent_router
from nova.inference.generate import generate_text
from nova.training.metrics import compute_classification_metrics
from nova.data.generate_data import INTENTS
from nova.models.intent_classifier import RuleBasedBaselineClassifier


def cmd_generate_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    print(f"Generating synthetic datasets using seed={cfg.seed}...")
    manifest = generate_all_datasets(
        data_dir=cfg.paths["data_dir"],
        seed=cfg.seed,
    )
    print("Synthetic data generated successfully.")
    print(f"Manifest written to: {cfg.paths['manifest_path']}")
    for name, info in manifest["files"].items():
        print(f"  - {name}: {info['count']} records -> {info['path']}")


def cmd_audit_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    print(f"Auditing datasets in {cfg.paths['data_dir']}...")
    report = audit_datasets(cfg.paths["data_dir"])
    print(json.dumps(report, indent=2))
    if report.get("status") == "PASSED":
        print("\nAudit Status: PASSED (Zero data leakage between splits)")
    else:
        print("\nAudit Status: WARNING/FAILED")


def cmd_inspect_vocab(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if not path.exists():
        print(f"Directory not found: {path}")
        return

    try:
        tok = SimpleTokenizer.load(path)
        info = tok.vocabulary.inspect(top_n=args.top_n)
        print(f"Tokenizer at {path}:")
        print(f"Vocabulary Size: {info['size']}")
        print(f"Special Tokens: {info['special_tokens']}")
        print(f"\nTop {args.top_n} Most Frequent Tokens:")
        for tok_name, count in info["top_tokens"]:
            print(f"  {tok_name:15s} : {count}")
    except Exception as e:
        print(f"Error inspecting vocabulary: {e}")


def cmd_train_lm(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    epochs = args.epochs or cfg.language_model.epochs
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Training Language Model on {device}...")
    summary = train_language_model(config=cfg, epochs_override=epochs, device=device)
    print(f"Model checkpoint: {summary['checkpoint_path']}")


def cmd_train_intent(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    epochs = args.epochs or cfg.intent_classifier.epochs
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Training Intent Classifier on {device}...")
    summary = train_intent_classifier(config=cfg, epochs_override=epochs, device=device)
    print(f"Model checkpoint: {summary['checkpoint_path']}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Evaluating models on held-out test data ({device})...")

    test_path = Path(cfg.paths["intent_test_data"])
    if not test_path.exists():
        print(f"Test data not found at {test_path}. Generate data first.")
        return

    # Load test records
    test_records = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))

    # Evaluate neural classifier
    router = load_agent_router(
        artifact_dir=cfg.paths["intent_artifact_dir"],
        device=device,
    )
    baseline = RuleBasedBaselineClassifier()

    neural_correct = 0
    baseline_correct = 0
    total = len(test_records)

    neural_preds = []
    baseline_preds = []
    targets = []

    label_to_idx = {name: idx for idx, name in enumerate(INTENTS)}

    for rec in test_records:
        text = rec["text"]
        target = rec["intent"]
        targets.append(label_to_idx.get(target, 0))

        # Neural evaluation
        trace = router.process(text)
        pred_intent = trace["effective_intent"]
        neural_preds.append(label_to_idx.get(pred_intent, 0))
        if pred_intent == target:
            neural_correct += 1

        # Baseline evaluation
        base_intent, _ = baseline.predict(text)
        baseline_preds.append(label_to_idx.get(base_intent, 0))
        if base_intent == target:
            baseline_correct += 1

    neural_metrics = compute_classification_metrics(targets, neural_preds, INTENTS)
    base_metrics = compute_classification_metrics(targets, baseline_preds, INTENTS)

    print("\n" + "=" * 50)
    print("EVALUATION COMPARISON: NEURAL VS BASELINE")
    print("=" * 50)
    print(f"Test Samples: {total}")
    print(f"Neural Classifier Accuracy : {neural_metrics['accuracy'] * 100:.2f}% | Macro F1: {neural_metrics['macro_f1']:.4f}")
    print(f"Baseline Matcher Accuracy  : {base_metrics['accuracy'] * 100:.2f}% | Macro F1: {base_metrics['macro_f1']:.4f}")
    print("=" * 50)


def cmd_chat(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    try:
        router = load_agent_router(
            artifact_dir=cfg.paths["intent_artifact_dir"],
            confidence_threshold=args.threshold or cfg.agent.confidence_threshold,
            device=device,
        )
    except Exception as e:
        print(f"Error loading agent model: {e}")
        print("Please train the models first: python -m nova.cli train-intent")
        return

    print("=" * 60)
    print("Nova AI Agent (Educational English Language Model)")
    print(f"Mode: {'DEBUG' if args.debug else 'STANDARD'} | Timezone: Asia/Kolkata")
    print("Type your question, or 'exit' / 'quit' to end.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Nova: Goodbye! Have a great day!")
            break

        trace = router.process(user_input)

        if args.debug:
            print("\n--- DEBUG TRACE ---")
            print(f"Original Input    : {trace['original_input']}")
            print(f"Normalized Input  : {trace['normalized_input']}")
            print(f"Tokens            : {trace['tokens']}")
            print(f"Token IDs         : {trace['token_ids']}")
            print(f"Predicted Intent  : {trace['predicted_intent']}")
            print(f"Confidence        : {trace['confidence']} (Passed: {trace['passed_threshold']})")
            print(f"Selected Action   : {trace['selected_action']}")
            if trace['tool_invocation']:
                print(f"Tool Invoked      : {trace['tool_invocation']}")
                print(f"Tool Result       : {trace['tool_result']}")
            print("-------------------")

        print(f"Nova: {trace['final_response']}")


def cmd_generate_lm(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    try:
        model, tokenizer = load_language_model(
            artifact_dir=cfg.paths["lm_artifact_dir"],
            device=device,
        )
    except Exception as e:
        print(f"Error loading language model: {e}")
        print("Please train the model first: python -m nova.cli train-lm")
        return

    prompt = args.prompt
    print(f"Prompt: {prompt}")
    print(f"Generating up to {args.max_tokens} tokens (temp={args.temp}, top_k={args.top_k})...\n")

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        device=device,
    )
    print("Generated Text:")
    print("-" * 40)
    print(output)
    print("-" * 40)


def cmd_info(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    print("Nova System Information:")
    print("------------------------")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch Version: {torch.__version__}")

    # Inspect LM if present
    lm_dir = Path(cfg.paths["lm_artifact_dir"])
    if (lm_dir / "best_model.pt").exists():
        try:
            model, tok = load_language_model(lm_dir)
            print(f"\nLanguage Model:")
            print(f"  Parameters: {model.get_num_params():,}")
            print(f"  Vocab Size: {tok.vocab_size}")
            print(f"  Layers: {model.n_layers}, Heads: {model.n_heads}, d_model: {model.d_model}")
        except Exception as e:
            print(f"  Error loading LM: {e}")
    else:
        print("\nLanguage Model: Not trained yet.")

    # Inspect Intent Classifier if present
    intent_dir = Path(cfg.paths["intent_artifact_dir"])
    if (intent_dir / "best_model.pt").exists():
        try:
            router = load_agent_router(intent_dir)
            cls_model = router.classifier
            n_params = sum(p.numel() for p in cls_model.parameters())
            print(f"\nIntent Classifier:")
            print(f"  Parameters: {n_params:,}")
            print(f"  Classes: {cls_model.num_classes} ({len(INTENTS)} intents)")
            print(f"  Vocab Size: {cls_model.vocab_size}")
        except Exception as e:
            print(f"  Error loading Intent Classifier: {e}")
    else:
        print("\nIntent Classifier: Not trained yet.")


def cmd_web(args: argparse.Namespace) -> None:
    from nova.web.server import run_server
    run_server(host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nova",
        description="Nova - Educational English Language Model & AI Agent built from scratch",
    )
    parser.add_argument("--config", default="config/default.yaml", help="Path to config file")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # generate-data
    subparsers.add_parser("generate-data", help="Generate synthetic training datasets")

    # audit-data
    subparsers.add_parser("audit-data", help="Audit generated datasets for leaks & statistics")

    # inspect-vocab
    p_vocab = subparsers.add_parser("inspect-vocab", help="Inspect tokenizer vocabulary")
    p_vocab.add_argument("--path", default="artifacts/intent_classifier", help="Path to tokenizer dir")
    p_vocab.add_argument("--top-n", type=int, default=15, help="Number of top tokens to display")

    # train-lm
    p_tlm = subparsers.add_parser("train-lm", help="Train DecoderLanguageModel from scratch")
    p_tlm.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    p_tlm.add_argument("--cpu", action="store_true", help="Force CPU training")

    # train-intent
    p_tint = subparsers.add_parser("train-intent", help="Train NeuralIntentClassifier from scratch")
    p_tint.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    p_tint.add_argument("--cpu", action="store_true", help="Force CPU training")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate models on test dataset")
    p_eval.add_argument("--cpu", action="store_true", help="Force CPU evaluation")

    # chat
    p_chat = subparsers.add_parser("chat", help="Start interactive agent chat")
    p_chat.add_argument("--debug", action="store_true", help="Show full intermediate pipeline trace")
    p_chat.add_argument("--threshold", type=float, default=None, help="Override confidence threshold")
    p_chat.add_argument("--cpu", action="store_true", help="Force CPU")

    # generate-lm
    p_gen = subparsers.add_parser("generate-lm", help="Run direct text generation with LM")
    p_gen.add_argument("--prompt", default="User: What is your name?\nNova:", help="Text prompt")
    p_gen.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    p_gen.add_argument("--max-tokens", type=int, default=25, help="Maximum new tokens to generate")
    p_gen.add_argument("--top-k", type=int, default=5, help="Top-k filtering")
    p_gen.add_argument("--cpu", action="store_true", help="Force CPU")

    # info
    subparsers.add_parser("info", help="Display model architecture and metadata")

    # web
    p_web = subparsers.add_parser("web", help="Start Nova Web UI with streaming response output")
    p_web.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    p_web.add_argument("--port", type=int, default=8000, help="Port to listen on")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "generate-data": cmd_generate_data,
        "audit-data": cmd_audit_data,
        "inspect-vocab": cmd_inspect_vocab,
        "train-lm": cmd_train_lm,
        "train-intent": cmd_train_intent,
        "evaluate": cmd_evaluate,
        "chat": cmd_chat,
        "generate-lm": cmd_generate_lm,
        "info": cmd_info,
        "web": cmd_web,
    }

    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
