"""Structured generation result type for the ClaimGuard generator (Step 17).

Deliberately generic: `GenerationResult` carries only prompt text,
generated text, and generation metadata - NEVER a FEVER/HaluEval/
RAGTruth/TruthfulQA label, gold evidence, or a hallucination span. The
generator module must remain independently callable and dataset-agnostic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GenerationResult:
    """The full, serializable output of one `qwen3.generate()` call.

    Carries everything needed to reproduce the call (user_input,
    system_instruction, generation_config, model_identifier) plus the
    generated answer and basic token/latency accounting. `generated_text`
    is the FINAL answer only - no hidden reasoning/thinking content is
    ever included here (see `qwen3.py`'s `enable_thinking` handling)."""

    user_input: str
    system_instruction: str | None
    generated_text: str
    model_identifier: str
    prompt_token_count: int
    generated_token_count: int
    generation_config: dict[str, Any]
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
