import urllib.request
import json

queries = [
    "how r u?",
    "who r u?",
    "what time is it?",
    "today is?",
    "what can you do?",
    "goodbye",
]

for q in queries:
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/chat/stream",
        data=json.dumps({"prompt": q}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data:") and "final_response" in line_str:
                d = json.loads(line_str[5:].strip())
                trace = d.get("trace", {})
                intent = trace.get("predicted_intent")
                conf = trace.get("confidence", 0.0)
                norm = trace.get("normalized_input")
                resp_text = d.get("final_response")
                print(f"Query: '{q}' -> norm: '{norm}' -> intent: {intent} ({conf*100:.1f}%) -> '{resp_text}'")
