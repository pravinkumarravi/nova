"""Confidence evaluation and calibration handling for Nova."""

from typing import Any, Dict, Tuple


def evaluate_confidence(
    predicted_intent: str,
    confidence: float,
    threshold: float = 0.65,
) -> Tuple[str, bool, Dict[str, Any]]:
    """Evaluates neural intent confidence against configured threshold.
    
    Important Note on Calibration:
    Standard softmax output is monotonic with logits but is typically uncalibrated
    and overconfident on out-of-domain inputs. If confidence is below threshold,
    we route safely to UNKNOWN.
    
    Args:
        predicted_intent: The intent label predicted by the classifier.
        confidence: Top softmax probability (0.0 to 1.0).
        threshold: Minimum confidence required to accept predicted intent.
        
    Returns:
        Tuple of (effective_intent, passed_threshold, metadata)
    """
    passed = confidence >= threshold

    effective_intent = predicted_intent if passed else "UNKNOWN"

    metadata = {
        "raw_intent": predicted_intent,
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "passed_threshold": passed,
        "effective_intent": effective_intent,
        "calibration_warning": (
            "Softmax score is an uncalibrated heuristic; values may be overconfident."
            if confidence > 0.9 and predicted_intent == "UNKNOWN"
            else None
        ),
    }

    return effective_intent, passed, metadata
