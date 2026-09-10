"""Qwen3-8B generator wrapper (Step 17): strictly the answer-generation
component of the future `query -> generator -> candidate answer ->
[Step 18] retrieval -> reranking -> verifier -> decision -> correction`
architecture. This module implements ONLY the generator box - no
correction/regeneration loop, no retrieval, no verification.

Model identity (`Qwen/Qwen3-8B`, bfloat16, cuda) is read from
`configs/models.yaml`'s `generator.primary` section - the project's
single source of truth for model selection (see
`CLAIMGUARD_MODEL_SELECTION.md`) - never hard-coded here. Generation
behavior (max_new_tokens, do_sample, seed, chat-template/thinking-mode
handling) is read from `configs/generator_baseline.yaml`.

Loads the published Qwen3-8B weights AS-IS: no quantization (the
approved research protocol specifies BF16, not a quantized mode), no
fine-tuning, no LoRA/PEFT, no adapter merging, no fallback to a
different generator model.

The chat-template handling here (`return_dict=True`, `dtype=` not
`torch_dtype=`, the `enable_thinking` kwarg with a TypeError fallback)
reuses the exact pattern already validated in Step 5A's
`scripts/smoke_test_qwen.py`, including the bug fixes found there - not
re-derived from scratch.
"""

from __future__ import annotations

import random
import time
from typing import Any

from .. import config as cg_config
from .types import GenerationResult


def load_generator(
    model_name: str | None = None, dtype: str | None = None, device: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load the approved Qwen3-8B generator + tokenizer. Model identity
    defaults to `configs/models.yaml`'s `generator.primary` section
    unless explicitly overridden by the caller (tests only). Never
    quantizes, fine-tunes, or adds LoRA/PEFT - loads the published
    weights unmodified."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    models_cfg = cg_config.load_models_config()
    gen_cfg = models_cfg["generator"]["primary"]
    model_name = model_name or gen_cfg["name"]
    dtype = dtype or gen_cfg.get("dtype", "bfloat16")
    device = device or gen_cfg.get("device", "cuda")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device=cuda but CUDA is not available.")

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch_dtype,
        device_map="auto" if device == "cuda" else device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    info = {
        "model_identifier": model_name,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "n_params": sum(p.numel() for p in model.parameters()),
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
    }
    return model, tokenizer, info


def unload_generator(model) -> None:
    """Explicitly release GPU resources held by a loaded generator.
    Callers (smoke tests) should call this when finished rather than
    leaving multiple generator instances resident on the GPU."""
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_chat_inputs(
    tokenizer, user_input: str, system_instruction: str | None,
    enable_thinking: bool, fallback_on_unsupported_kwarg: bool,
) -> tuple[Any, Any, bool]:
    """Format (system?, user) messages via the tokenizer's OWN chat
    template - never a manually-assumed prompt format. Returns
    (input_ids, attention_mask, fallback_triggered). `fallback_triggered`
    is True only if the installed tokenizer's chat template does not
    accept the `enable_thinking` kwarg and the plain-kwarg retry path was
    used instead (documented, not silently swallowed)."""
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": user_input})

    common_kwargs = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    fallback_triggered = False
    try:
        encoded = tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **common_kwargs)
    except TypeError:
        if not fallback_on_unsupported_kwarg:
            raise
        fallback_triggered = True
        encoded = tokenizer.apply_chat_template(messages, **common_kwargs)
    return encoded["input_ids"], encoded["attention_mask"], fallback_triggered


def _build_generation_kwargs(
    max_new_tokens: int, do_sample: bool, temperature: float | None, top_p: float | None,
) -> dict[str, Any]:
    """Isolated for direct unit testing: `temperature`/`top_p` are ONLY
    passed to `model.generate` when `do_sample=True` - passing them
    alongside `do_sample=False` is meaningless and some `transformers`
    versions warn about it."""
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
    return kwargs


def score_choice_log_likelihood(
    model, tokenizer, question: str, choice: str, system_instruction: str | None = None,
    enable_thinking: bool = False, fallback_on_unsupported_kwarg: bool = True,
) -> dict[str, Any]:
    """Score a candidate answer choice's log-likelihood conditioned on
    `question`, via a SINGLE teacher-forced forward pass - no generation,
    no sampling, no weight update. This is a standard, weight-preserving
    usage of the frozen causal LM (the same forward-pass mechanism
    `generate()` uses internally, just without autoregressive decoding) -
    added for Step 20's TruthfulQA MC1/MC2/MC0 evaluation, which requires
    ranking pre-written answer choices by likelihood rather than
    generating free text.

    Uses the SAME chat-template formatting as `generate()` for the
    question, so scoring and generation share an identical prompt format.
    Returns the raw (un-normalized) SUMMED log-probability of the
    choice's tokens - the standard convention for MC-style ranking - plus
    the token count for an optional length-normalized view.

    Knows nothing about TruthfulQA/FEVER/RAGTruth/HaluEval labels - only
    generic question/choice text; the caller supplies whichever choice
    text it wants scored."""
    import torch

    prompt_ids, _prompt_mask, fallback_triggered = build_chat_inputs(
        tokenizer, question, system_instruction, enable_thinking, fallback_on_unsupported_kwarg,
    )
    choice_ids = tokenizer(choice, add_special_tokens=False, return_tensors="pt")["input_ids"]

    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    choice_ids = choice_ids.to(device)

    input_ids = torch.cat([prompt_ids, choice_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = prompt_ids.shape[1]
    choice_len = choice_ids.shape[1]

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    if choice_len == 0:
        return {"choice": choice, "log_likelihood": 0.0, "num_tokens": 0, "avg_log_likelihood": None,
                "chat_template_fallback_triggered": fallback_triggered}

    # logits[0, t, :] predicts the token AT position t+1, so the choice's
    # own tokens are predicted by logits[prompt_len-1 : prompt_len+choice_len-1].
    relevant_logits = logits[0, prompt_len - 1: prompt_len + choice_len - 1, :]
    log_probs = torch.log_softmax(relevant_logits.float(), dim=-1)
    target_ids = choice_ids[0]
    token_log_probs = log_probs[torch.arange(choice_len, device=device), target_ids]
    total_log_likelihood = float(token_log_probs.sum())

    return {
        "choice": choice, "log_likelihood": total_log_likelihood, "num_tokens": int(choice_len),
        "avg_log_likelihood": total_log_likelihood / choice_len,
        "chat_template_fallback_triggered": fallback_triggered,
    }


def generate(
    model, tokenizer, user_input: str, model_identifier: str,
    system_instruction: str | None = None, max_new_tokens: int = 256, do_sample: bool = False,
    temperature: float | None = None, top_p: float | None = None, seed: int | None = 42,
    enable_thinking: bool = False, fallback_on_unsupported_kwarg: bool = True,
) -> GenerationResult:
    """Generate ONE response for `user_input`. Deterministic by default
    (`do_sample=False` plus a fixed seed). Returns ONLY the final
    generated answer text via `skip_special_tokens=True` decoding of
    JUST the newly generated tokens (`output_ids[0][prompt_len:]`) - the
    prompt itself is never echoed back, and no hidden reasoning/thinking
    content is extracted or stored (`enable_thinking=False` by default).

    Knows nothing about FEVER/HaluEval/RAGTruth/TruthfulQA labels, gold
    evidence, or hallucination spans - the only inputs are generic prompt
    text and generation parameters."""
    import torch

    if seed is not None:
        _set_seed(seed)

    input_ids, attention_mask, fallback_triggered = build_chat_inputs(
        tokenizer, user_input, system_instruction, enable_thinking, fallback_on_unsupported_kwarg,
    )
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_token_count = input_ids.shape[1]

    gen_kwargs = _build_generation_kwargs(max_new_tokens, do_sample, temperature, top_p)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    latency = time.time() - start

    generated_ids = output_ids[0][prompt_token_count:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return GenerationResult(
        user_input=user_input,
        system_instruction=system_instruction,
        generated_text=generated_text,
        model_identifier=model_identifier,
        prompt_token_count=int(prompt_token_count),
        generated_token_count=int(generated_ids.shape[0]),
        generation_config={
            "max_new_tokens": max_new_tokens, "do_sample": do_sample, "temperature": temperature,
            "top_p": top_p, "seed": seed, "enable_thinking": enable_thinking,
            "chat_template_fallback_triggered": fallback_triggered,
        },
        latency_seconds=latency,
    )
