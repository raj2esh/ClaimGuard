"""ClaimGuard verification: load the Step 12 binary verifier checkpoint and
score (premise, hypothesis) pairs (Step 15).

Uses ONLY the already-trained Step 12 checkpoint
(experiments/verifier_binary_final/final/) - does NOT retrain, fine-tune,
or otherwise modify the verifier. `claimguard.verifier.binary_model.
assert_binary_head` is reused to hard-verify the loaded checkpoint is
still the genuine 2-class (entailment/contradiction) architecture before
any inference happens - never assumed.

## Binary label mapping (unchanged from Step 12, no neutral)

    entailment    -> "supported"
    contradiction -> "contradicted"

There is no neutral output - Step 11 established no legitimate neutral
training source exists, so none is invented at inference time either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import config as cg_config
from ..verifier.binary_model import BINARY_ID2LABEL, assert_binary_head

VERIFIER_LABEL_TO_PIPELINE_LABEL = {
    "entailment": "supported",
    "contradiction": "contradicted",
}


def load_verifier(
    checkpoint_dir: Path | str | None = None, device: str = "cuda", dtype: str = "bfloat16",
) -> tuple[Any, Any, dict[str, Any]]:
    """Load the Step 12 binary verifier checkpoint + tokenizer. Verifies
    (does not assume) the loaded model is still a genuine 2-class
    entailment/contradiction head via `assert_binary_head`. Moves the
    model to `device`/`dtype` explicitly (unchanged from Step 12's own
    training precision - bf16 - not altered here) and sets eval mode -
    the checkpoint alone loads onto CPU/fp32 by default via
    `from_pretrained`, which would be needlessly slow for repeated
    inference and would silently diverge from the precision the model was
    actually trained/evaluated in.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint_dir = str(checkpoint_dir or cg_config.resolve_path("experiments/verifier_binary_final/final"))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device=cuda but CUDA is not available.")

    torch_dtype = getattr(torch, dtype)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    assert_binary_head(model)  # hard guard - never assume the checkpoint is still binary
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    if id2label != BINARY_ID2LABEL:
        raise RuntimeError(
            f"Loaded verifier checkpoint's id2label {id2label} does not match the expected "
            f"binary scheme {BINARY_ID2LABEL} - refusing to score with an unverified mapping."
        )

    verified_info = {
        "checkpoint_dir": checkpoint_dir,
        "id2label": id2label,
        "num_labels": model.config.num_labels,
        "n_params": sum(p.numel() for p in model.parameters()),
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
    }
    return model, tokenizer, verified_info


def verify_pair(
    model, tokenizer, premise: str, hypothesis: str, max_length: int = 256
) -> dict[str, Any]:
    """Score one (premise, hypothesis) pair. Returns entailment/contradiction
    probabilities (softmax over the 2 real logits - never a fabricated
    neutral score) and the argmax predicted label (both the raw verifier
    label and the pipeline-facing supported/contradicted label).
    """
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(premise, hypothesis, truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits.float(), dim=-1)
    entailment_prob = float(probs[BINARY_ID2LABEL_INDEX["entailment"]])
    contradiction_prob = float(probs[BINARY_ID2LABEL_INDEX["contradiction"]])
    predicted_verifier_label = "entailment" if entailment_prob >= contradiction_prob else "contradiction"
    return {
        "entailment_probability": entailment_prob,
        "contradiction_probability": contradiction_prob,
        "verifier_label": predicted_verifier_label,
        "pipeline_label": VERIFIER_LABEL_TO_PIPELINE_LABEL[predicted_verifier_label],
    }


BINARY_ID2LABEL_INDEX = {v: k for k, v in BINARY_ID2LABEL.items()}


def verify_candidates(
    model, tokenizer, query: str, candidates: list[dict[str, Any]], max_length: int = 256
) -> list[dict[str, Any]]:
    """Score every candidate (premise=candidate text, hypothesis=query) and
    return enriched candidate dicts (original fields preserved, plus the
    verifier fields from verify_pair). Empty candidates returns []."""
    enriched = []
    for c in candidates:
        result = verify_pair(model, tokenizer, premise=c["text"], hypothesis=query, max_length=max_length)
        enriched.append({**c, **result})
    return enriched
