"""Direct autoregressive generation with DecoderLanguageModel."""

from typing import Optional
import torch

from nova.tokenizer.tokenizer import SimpleTokenizer
from nova.models.language_model import DecoderLanguageModel


def generate_text(
    model: DecoderLanguageModel,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_k: Optional[int] = 5,
    device: str = "cpu",
) -> str:
    """Generates autoregressive text continuation from prompt string."""
    model.eval()
    tokens = tokenizer.encode(prompt, add_special_tokens=False)

    # Prefix with BOS
    input_ids = [tokenizer.bos_token_id] + tokens
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
        )

    out_list = out_ids[0].cpu().tolist()
    # Decode only the generated part or the entire continuation
    decoded_text = tokenizer.decode(out_list, skip_special_tokens=True)
    return decoded_text
