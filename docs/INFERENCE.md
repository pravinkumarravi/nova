# Nova Inference Guide

This guide describes how to run inference with Nova across its two operating modes: the **Agent Chat Interface** (with neural intent routing and tool execution) and the **Decoder Language Model** (direct autoregressive text generation).

---

## 1. Interactive Agent Mode

In this mode, user queries pass through the complete processing pipeline: input normalization, tokenization, neural intent classification, confidence threshold evaluation, sandboxed tool calling (if authorized), and response synthesis.

### Launching the Agent
```powershell
$env:PYTHONPATH = "src"
python -m nova.cli chat
```

### Example Interaction:
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
Nova: It is currently 10:20 AM IST.

User: What is today's date?
Nova: Today is Monday, 12 October 2026.

User: What is the capital of France?
Nova: I have not been trained to answer general knowledge questions yet. I can answer questions about myself and my supported capabilities.
```

---

## 2. Agent Debug Mode

To inspect the complete internal telemetry and mathematical transformations performed at every step of the pipeline, pass the `--debug` flag:

```powershell
python -m nova.cli chat --debug
```

### Example Debug Trace:
```text
User: What time is it now?

--- DEBUG TRACE ---
Original Input    : What time is it now?
Normalized Input  : what time is it now?
Tokens            : ['what', 'time', 'is', 'it', 'now', '?']
Token IDs         : [2, 45, 12, 9, 31, 88, 14, 3]
Predicted Intent  : ASK_CURRENT_TIME
Confidence        : 0.9532 (Passed: True)
Selected Action   : invoke_tool:get_current_time
Tool Invoked      : get_current_time
Tool Result       : {'tool_name': 'get_current_time', 'timezone': 'Asia/Kolkata', 'time': '10:20:45', 'formatted_value': '10:20 AM IST', 'iso': '2026-10-12T10:20:45+05:30'}
-------------------
Nova: It is currently 10:20 AM IST.
```

---

## 3. Direct Decoder Language Model Inference

You can run the Decoder-Only Transformer directly without passing through the intent classifier or tool registry. This allows you to observe raw autoregressive token predictions, beam search / sampling dynamics, and model limitations.

### Launch Command:
```powershell
python -m nova.cli generate-lm --prompt "User: What is your name?`nNova:" --temp 0.8 --max-tokens 30 --top-k 5
```

### Options:
- `--prompt`: Starting prefix string for generation.
- `--temp`: Softmax temperature ($T < 1.0$ is greedy/deterministic, $T > 1.0$ increases entropy).
- `--max-tokens`: Maximum count of new tokens to predict.
- `--top-k`: Filters probability distribution to top $K$ highest-likelihood tokens.

---

## 4. Programmatic Python API

You can also embed Nova directly into your Python scripts:

```python
from nova.inference.load_model import load_agent_router, load_language_model
from nova.inference.generate import generate_text

# 1. Agent Router with Tool Calling
router = load_agent_router("artifacts/intent_classifier")
trace = router.process("What time is it?")
print(trace["final_response"])

# 2. Raw Language Model Generation
lm_model, lm_tok = load_language_model("artifacts/language_model")
output = generate_text(lm_model, lm_tok, prompt="User: Who are you?\nNova:")
print(output)
```
