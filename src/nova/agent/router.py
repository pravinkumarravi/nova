"""Pipeline coordinator and agent router for Nova."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch

from nova.tokenizer.tokenizer import SimpleTokenizer
from nova.models.intent_classifier import NeuralIntentClassifier, RuleBasedBaselineClassifier
from nova.agent.confidence import evaluate_confidence
from nova.agent.response_generator import ResponseGenerator
from nova.tools.registry import ToolRegistry, default_registry


class AgentRouter:
    """Coordinates input processing, neural classification, tool invocation, and responses."""

    INTENT_TO_TOOL = {
        "ASK_CURRENT_TIME": "get_current_time",
        "ASK_CURRENT_DATE": "get_current_date",
    }

    def __init__(
        self,
        classifier: NeuralIntentClassifier,
        tokenizer: SimpleTokenizer,
        idx_to_label: Dict[int, str],
        confidence_threshold: float = 0.65,
        response_generator: Optional[ResponseGenerator] = None,
        tool_registry: Optional[ToolRegistry] = None,
        device: str = "cpu",
    ) -> None:
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.idx_to_label = idx_to_label
        self.confidence_threshold = confidence_threshold
        self.response_generator = response_generator or ResponseGenerator()
        self.tool_registry = tool_registry or default_registry
        self.baseline_classifier = RuleBasedBaselineClassifier()
        self.device = device

        self.classifier.eval()

    def process(
        self,
        user_input: str,
        now_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Executes full pipeline and returns response with detailed execution trace.
        
        Args:
            user_input: Raw text string from user.
            now_fn: Optional mock datetime supplier for deterministic tool testing.
            
        Returns:
            Dictionary containing final_response and intermediate debug steps.
        """
        # Step 1: Input validation & normalization
        normalized_input = self.tokenizer.normalize(user_input)

        if not normalized_input:
            empty_resp = "I did not receive any input. How can I help you?"
            return {
                "original_input": user_input,
                "normalized_input": "",
                "tokens": [],
                "token_ids": [],
                "predicted_intent": "UNKNOWN",
                "confidence": 1.0,
                "passed_threshold": True,
                "effective_intent": "UNKNOWN",
                "selected_action": "direct_response",
                "tool_invocation": None,
                "tool_result": None,
                "final_response": empty_resp,
            }

        # Step 2: Tokenization
        tokens = self.tokenizer.tokenize(normalized_input)
        token_ids = self.tokenizer.encode(normalized_input, add_special_tokens=True)

        # Step 3: Neural intent classification
        max_len = self.classifier.max_seq_len
        pad_id = self.tokenizer.pad_token_id
        if len(token_ids) > max_len:
            padded_ids = token_ids[:max_len]
        else:
            padded_ids = token_ids + [pad_id] * (max_len - len(token_ids))

        input_tensor = torch.tensor([padded_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, _ = self.classifier(input_tensor)
            probs = torch.softmax(logits, dim=-1)
            confidence_val, pred_idx_tensor = torch.max(probs, dim=-1)

        pred_idx = pred_idx_tensor.item()
        confidence = float(confidence_val.item())
        predicted_intent = self.idx_to_label.get(pred_idx, "UNKNOWN")

        # Baseline comparison (for diagnostics)
        baseline_intent, baseline_conf = self.baseline_classifier.predict(normalized_input)

        # Step 4: Confidence threshold & routing
        effective_intent, passed_threshold, conf_meta = evaluate_confidence(
            predicted_intent=predicted_intent,
            confidence=confidence,
            threshold=self.confidence_threshold,
        )

        # Step 5 & 6: Intent-to-action mapping and tool execution
        tool_name = self.INTENT_TO_TOOL.get(effective_intent)
        tool_result = None
        tool_error = None

        if tool_name is not None and self.tool_registry.is_registered(tool_name):
            selected_action = f"invoke_tool:{tool_name}"
            kwargs = {}
            if now_fn is not None:
                kwargs["now_fn"] = now_fn
            exec_res = self.tool_registry.execute(tool_name, **kwargs)
            if exec_res.get("success"):
                tool_result = exec_res.get("result")
            else:
                tool_error = exec_res.get("error", "Tool execution failed")
        else:
            selected_action = "direct_response"

        # Step 7: Response generation
        final_response = self.response_generator.generate(
            intent=effective_intent,
            tool_result=tool_result,
            tool_error=tool_error,
        )

        return {
            "original_input": user_input,
            "normalized_input": normalized_input,
            "tokens": tokens,
            "token_ids": token_ids,
            "predicted_intent": predicted_intent,
            "confidence": round(confidence, 4),
            "baseline_intent": baseline_intent,
            "baseline_confidence": round(baseline_conf, 4),
            "passed_threshold": passed_threshold,
            "effective_intent": effective_intent,
            "selected_action": selected_action,
            "tool_invocation": tool_name,
            "tool_result": tool_result,
            "tool_error": tool_error,
            "final_response": final_response,
        }
