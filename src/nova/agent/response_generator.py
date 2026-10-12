"""Controlled response generator mapping validated intents and tool results to English text."""

from pathlib import Path
import random
from typing import Any, Dict, List, Optional
import yaml


class ResponseGenerator:
    """Generates natural, verified responses for supported intents and tool executions.
    
    Reads responses from configuration files (identity.yaml) to ensure deterministic,
    safe agent behavior without unconstrained hallucination.
    """

    def __init__(self, config_path: str | Path = "config/identity.yaml") -> None:
        self.config_path = Path(config_path)
        self.identity_data = self._load_config()
        self.responses: Dict[str, List[str]] = self.identity_data.get("responses", {})
        self.agent_name = self.identity_data.get("name", "Nova")

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {
                "name": "Nova",
                "responses": {
                    "UNKNOWN": [
                        "I have not been trained to answer general knowledge questions yet. I can answer questions about myself and my supported capabilities."
                    ]
                },
            }
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def generate(
        self,
        intent: str,
        tool_result: Optional[Dict[str, Any]] = None,
        tool_error: Optional[str] = None,
    ) -> str:
        """Constructs response based on intent and optional tool execution output."""
        if tool_error:
            return f"I encountered an error while trying to process your request: {tool_error}"

        # Dynamic tool output formatting
        if intent == "ASK_CURRENT_TIME":
            if tool_result and "formatted_value" in tool_result:
                return f"It is currently {tool_result['formatted_value']}."
            return "Unable to determine current time."

        if intent == "ASK_CURRENT_DATE":
            if tool_result and "formatted_value" in tool_result:
                return f"Today is {tool_result['formatted_value']}."
            return "Unable to determine today's date."

        candidates = self.responses.get(intent)
        if not candidates:
            # Fallback to UNKNOWN
            candidates = self.responses.get(
                "UNKNOWN",
                ["I have not been trained to answer general knowledge questions yet. I can answer questions about myself and my supported capabilities."]
            )

        return candidates[0]
