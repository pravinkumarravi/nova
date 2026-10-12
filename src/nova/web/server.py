"""Lightweight multi-threaded HTTP server supporting Server-Sent Events (SSE) streaming for Nova."""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import mimetypes
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, Generator, Optional
import urllib.parse
import torch

from nova.config import load_config
from nova.inference.load_model import load_agent_router, load_language_model
from nova.tokenizer.tokenizer import SimpleTokenizer
from nova.models.language_model import DecoderLanguageModel
from nova.agent.router import AgentRouter


STATIC_DIR = Path(__file__).parent / "static"


class NovaServerState:
    """Singleton holding cached models and config."""

    def __init__(self) -> None:
        self.cfg = load_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.router: Optional[AgentRouter] = None
        self.lm_model: Optional[DecoderLanguageModel] = None
        self.lm_tokenizer: Optional[SimpleTokenizer] = None
        self._lock = threading.Lock()

    def get_router(self) -> AgentRouter:
        with self._lock:
            if self.router is None:
                self.router = load_agent_router(
                    artifact_dir=self.cfg.paths["intent_artifact_dir"],
                    device=self.device,
                )
            return self.router

    def get_lm(self) -> tuple[DecoderLanguageModel, SimpleTokenizer]:
        with self._lock:
            if self.lm_model is None or self.lm_tokenizer is None:
                self.lm_model, self.lm_tokenizer = load_language_model(
                    artifact_dir=self.cfg.paths["lm_artifact_dir"],
                    device=self.device,
                )
            return self.lm_model, self.lm_tokenizer


state = NovaServerState()


def stream_lm_tokens(
    model: DecoderLanguageModel,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    top_k: int = 5,
    device: str = "cpu",
) -> Generator[str, None, None]:
    """Yields each generated token autoregressively as it is predicted."""
    model.eval()
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + tokens
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= model.max_seq_len else idx[:, -model.max_seq_len :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0.05:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled_logits = logits / max(temperature, 0.05)
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")
                probs = torch.softmax(scaled_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)
            token_int = next_id.item()

            if token_int == tokenizer.eos_token_id:
                break

            token_str = tokenizer.vocabulary.index_to_token(token_int)
            yield token_str


class NovaHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Handles static files, API metadata, and SSE streaming responses."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/":
            self.send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        elif parsed.path == "/api/info":
            self.send_json_info()
            return
        elif parsed.path.startswith("/static/"):
            rel_path = parsed.path[len("/static/") :]
            target_file = STATIC_DIR / rel_path
            if target_file.exists() and target_file.is_file():
                mime, _ = mimetypes.guess_type(str(target_file))
                self.send_file(target_file, mime or "application/octet-stream")
                return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/chat/stream":
            self.handle_stream_chat()
            return

        self.send_error(404, "Endpoint not found")

    def send_file(self, file_path: Path, content_type: str) -> None:
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def send_json_info(self) -> None:
        try:
            router = state.get_router()
            lm_model, lm_tok = state.get_lm()

            info = {
                "name": "Nova",
                "version": "0.1.0",
                "device": state.device,
                "timezone": "Asia/Kolkata",
                "language_model": {
                    "parameters": lm_model.get_num_params(),
                    "vocab_size": lm_tok.vocab_size,
                    "layers": lm_model.n_layers,
                    "heads": lm_model.n_heads,
                    "d_model": lm_model.d_model,
                },
                "intent_classifier": {
                    "parameters": sum(p.numel() for p in router.classifier.parameters()),
                    "vocab_size": router.tokenizer.vocab_size,
                    "intents": list(router.idx_to_label.values()),
                },
                "tools": list(router.tool_registry.list_tools().keys()),
            }
            body = json.dumps(info).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            err = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err)

    def handle_stream_chat(self) -> None:
        """Streams Server-Sent Events (SSE) back to client."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8"))
        except Exception:
            payload = {}

        prompt = payload.get("prompt", "").strip()
        mode = payload.get("mode", "agent")  # 'agent' or 'lm'
        temperature = float(payload.get("temperature", 0.8))
        top_k = int(payload.get("top_k", 5))
        threshold = float(payload.get("threshold", 0.65))

        # Setup SSE headers
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def send_event(event_type: str, data: Dict[str, Any]) -> bool:
            try:
                msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                self.wfile.write(msg.encode("utf-8"))
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False

        if not prompt:
            send_event("error", {"message": "Empty prompt received."})
            send_event("done", {})
            return

        if mode == "lm":
            # Direct autoregressive language model stream
            try:
                lm_model, lm_tok = state.get_lm()
                send_event("start", {"mode": "lm", "prompt": prompt})

                full_generated = []
                for token in stream_lm_tokens(
                    model=lm_model,
                    tokenizer=lm_tok,
                    prompt=prompt,
                    max_new_tokens=45,
                    temperature=temperature,
                    top_k=top_k,
                    device=state.device,
                ):
                    full_generated.append(token)
                    # Natural spacing
                    space = " " if token not in {".", ",", "!", "?", ":", ";", "'"} else ""
                    if not send_event("token", {"text": space + token}):
                        return
                    time.sleep(0.04)  # Smooth typewriter cadence

                send_event("done", {"text": " ".join(full_generated)})
            except Exception as e:
                send_event("error", {"message": str(e)})
                send_event("done", {})

        else:
            # Full Agent routing with telemetry steps and stream output
            try:
                router = state.get_router()
                router.confidence_threshold = threshold

                send_event("start", {"mode": "agent", "prompt": prompt})

                # Telemetry Step 1: Normalization
                norm_text = router.tokenizer.normalize(prompt)
                time.sleep(0.05)
                send_event("step", {
                    "step": "normalization",
                    "title": "Text Normalization",
                    "detail": f"Processed: '{norm_text}'",
                })

                # Telemetry Step 2: Tokenization
                tokens = router.tokenizer.tokenize(norm_text)
                token_ids = router.tokenizer.encode(norm_text, add_special_tokens=True)
                time.sleep(0.05)
                send_event("step", {
                    "step": "tokenization",
                    "title": "Vocabulary Tokenization",
                    "detail": f"{len(tokens)} tokens identified",
                    "tokens": tokens,
                    "token_ids": token_ids,
                })

                # Run full router process
                trace = router.process(prompt)

                # Telemetry Step 3: Intent Classification
                time.sleep(0.06)
                send_event("step", {
                    "step": "classification",
                    "title": "Neural Intent Classification",
                    "intent": trace["predicted_intent"],
                    "confidence": trace["confidence"],
                    "passed": trace["passed_threshold"],
                    "effective_intent": trace["effective_intent"],
                })

                # Telemetry Step 4: Tool Invocation (if any)
                if trace["tool_invocation"]:
                    time.sleep(0.08)
                    send_event("step", {
                        "step": "tool_execution",
                        "title": f"Invoking Tool: {trace['tool_invocation']}",
                        "tool": trace["tool_invocation"],
                        "result": trace["tool_result"],
                    })

                # Stream the final response word by word
                response_words = trace["final_response"].split()
                for i, word in enumerate(response_words):
                    prefix = " " if i > 0 else ""
                    if not send_event("token", {"text": prefix + word}):
                        return
                    time.sleep(0.035)

                send_event("done", {
                    "final_response": trace["final_response"],
                    "trace": trace,
                })
            except Exception as e:
                send_event("error", {"message": str(e)})
                send_event("done", {})
            finally:
                self.close_connection = True


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Starts the Nova Web UI server."""
    # Preload models
    print(f"Preloading Nova neural models on {state.device}...")
    state.get_router()
    state.get_lm()
    print("Models preloaded successfully.")

    server_address = (host, port)
    httpd = HTTPServer(server_address, NovaHTTPRequestHandler)
    print(f"\n==================================================")
    print(f"  Nova Web UI running at: http://{host}:{port}")
    print(f"  Streaming SSE endpoint: http://{host}:{port}/api/chat/stream")
    print(f"  Press Ctrl+C to stop the server.")
    print(f"==================================================\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Nova Web UI server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port=port)
