"""End-to-end smoke test script demonstrating Nova agent queries against saved checkpoints."""

from nova.inference.load_model import load_agent_router


def main():
    router = load_agent_router("artifacts/intent_classifier")
    questions = [
        "What is your name?",
        "Who are you?",
        "Are you a human?",
        "What can you do?",
        "Can you tell me the time?",
        "What is today's date?",
        "Hello!",
        "What is the capital of France?",
    ]

    print("=" * 60)
    print("NOVA LIVE AGENT INFERENCE SMOKE TEST")
    print("=" * 60)

    for q in questions:
        trace = router.process(q)
        print(f"User: {q}")
        print(f"Nova: {trace['final_response']}")
        print(f"  [Intent: {trace['effective_intent']} | Confidence: {trace['confidence']:.4f} | Tool: {trace['tool_invocation']}]")
        print("-" * 60)


if __name__ == "__main__":
    main()
