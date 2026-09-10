"""ClaimGuard verifier: binary (entailment/contradiction) classification head.

Step 12 constructs a GENUINE 2-class classification head for the project's
Step 3/5B-selected NLI model
(MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli), rather than
training its original pretrained 3-class (entailment/neutral/contradiction)
head while only ever supplying labels 0 and 2. Training the original head
that way would leave the 'neutral' output dimension entirely without
supervision while still reporting (and shipping) a misleading 3-class
architecture - Step 11 already established the verifier is scientifically
binary for the current data; the model's actual classification head must
say the same thing, not just the training loop.

Empirically verified (not assumed) architecture, via direct model
introspection (see PROJECT_REPORT.md Step 12): the model is a
DebertaV2ForSequenceClassification with three top-level trainable
submodules - `deberta` (the shared transformer encoder, unaffected by
label count), `pooler` (a ContextPooler: a single Linear(1024, 1024)
shared across all labels, also unaffected by label count), and
`classifier` (a single Linear(1024, num_labels) - the ONLY submodule whose
shape depends on the number of labels). Confirmed original id2label =
{0: 'entailment', 1: 'neutral', 2: 'contradiction'} (matches the Step 5B/9
finding). 435,064,835 total parameters at 3-class; the classifier itself
accounts for exactly 3,075 of those (1024*3 weights + 3 bias) - everything
else is the shared encoder+pooler.

Binary head construction strategy (documented here, not silently chosen):
    1. Load the ORIGINAL pretrained 3-class model once, to read its actual
       classifier weight rows for 'entailment' and 'contradiction' (at
       whatever index they empirically occupy in ITS config - verified via
       `verified_source_label_indices`, never assumed to be [0, 2]).
    2. Separately construct a new model via
       `AutoModelForSequenceClassification.from_pretrained(...,
       num_labels=2, ignore_mismatched_sizes=True)` - this loads the
       pretrained encoder+pooler weights unchanged (their shape doesn't
       depend on num_labels, so `ignore_mismatched_sizes` never touches
       them) and randomly reinitializes ONLY the now-mismatched-shape
       classifier layer.
    3. Immediately overwrite that random classifier with the two rows
       captured in step 1 - so the binary head's actual starting point is
       "the pretrained model's own entailment/contradiction judgment,"
       not a fresh random head, and never a hidden interim random state
       that anything downstream could observe.
    4. Verify (not assume) that the encoder+pooler weights in the new
       model are bit-identical to the original pretrained model's, so
       "encoder/pooler reused" is a checked fact in the returned report,
       not a claim.

If this architecture-specific verification ever fails against a future
model swap, `build_binary_model` raises loudly rather than guessing at an
unfamiliar architecture's classifier location.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

BINARY_LABEL2ID = {"entailment": 0, "contradiction": 1}
BINARY_ID2LABEL = {0: "entailment", 1: "contradiction"}


def verified_source_label_indices(model_name: str, required_labels: set[str]) -> dict[str, int]:
    """Read the ORIGINAL pretrained model's own id2label/label2id from its
    config - used ONLY to know which row of the pretrained classifier
    corresponds to which label, so those specific rows can be transplanted
    into the new binary head. Does NOT define the new binary scheme itself
    (that is BINARY_LABEL2ID above, fixed by the project's Step 12
    protocol) - refuses to guess if a required label is absent from the
    model's own config.
    """
    config = AutoConfig.from_pretrained(model_name)
    id2label = {int(k): v.lower() for k, v in config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    missing = required_labels - set(label2id)
    if missing:
        raise ValueError(
            f"Model {model_name!r}'s id2label {id2label} is missing required label(s) "
            f"{sorted(missing)} - cannot determine which classifier row(s) to reuse."
        )
    return {label: label2id[label] for label in required_labels}


def assert_binary_head(model: torch.nn.Module) -> None:
    """Hard guard against accidentally training/using a non-binary head.
    Raises AssertionError (loud, not a silent log line) if num_labels != 2
    or the id2label mapping doesn't exactly match the project's binary
    protocol. Intended to be called right after model construction AND
    again at evaluation/checkpoint-load time, so a future accidental
    3-class configuration can never silently pass as "the binary
    verifier".
    """
    num_labels = getattr(model.config, "num_labels", None)
    if num_labels != 2:
        raise AssertionError(
            f"Expected a 2-class (binary) classification head, found num_labels={num_labels}. "
            "Refusing to proceed - this would silently train/evaluate the wrong architecture "
            "(Step 11 established the verifier is scientifically binary for this data)."
        )
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if id2label != BINARY_ID2LABEL:
        raise AssertionError(
            f"Model id2label {id2label} does not match the project's binary protocol "
            f"{BINARY_ID2LABEL}."
        )


def build_binary_model(
    model_name: str, dtype: torch.dtype | None = None
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build a genuine 2-class (entailment/contradiction) classification
    head for `model_name`, reusing the pretrained encoder+pooler in full
    and initializing the new classifier's two rows from the ORIGINAL
    pretrained 3-class classifier's entailment and contradiction rows
    (verified indices, never assumed to be [0, 2]) - not a randomly
    initialized head, and not a guess at which pretrained weights
    correspond to which label.

    Returns (model, report) where report documents exactly what was
    reused vs. reinitialized, with the encoder/pooler-reuse claim
    independently VERIFIED (bit-identical weight comparison), for the
    experiment's reproducibility record.
    """
    source_indices = verified_source_label_indices(model_name, {"entailment", "contradiction"})
    print(f"Verified source classifier row indices (from the model's own config): {source_indices}")

    print(f"Loading original pretrained 3-class model to extract classifier rows: {model_name}")
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype)
    classifier = pretrained_model.classifier
    if not isinstance(classifier, torch.nn.Linear):
        raise RuntimeError(
            f"Expected a single `classifier` nn.Linear submodule on {model_name}'s architecture "
            f"({type(pretrained_model).__name__}) - found {type(classifier).__name__} instead. "
            "Refusing to guess how to transplant classifier weights for an architecture this "
            "code has not verified."
        )
    if classifier.out_features != 3:
        raise RuntimeError(
            f"Expected the original classifier to have 3 output labels, found "
            f"{classifier.out_features} - source_indices {source_indices} would not be "
            "meaningful against this shape."
        )

    entailment_row = classifier.weight.data[source_indices["entailment"]].clone()
    entailment_bias = classifier.bias.data[source_indices["entailment"]].clone()
    contradiction_row = classifier.weight.data[source_indices["contradiction"]].clone()
    contradiction_bias = classifier.bias.data[source_indices["contradiction"]].clone()

    print("Building binary (2-class) model - fresh classifier shape, reused pretrained "
          f"encoder+pooler: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=BINARY_ID2LABEL,
        label2id=BINARY_LABEL2ID,
        ignore_mismatched_sizes=True,  # only the classifier's shape actually differs (3 -> 2)
        dtype=dtype,
    )
    assert_binary_head(model)

    # Verify the encoder+pooler claim rather than assume it: bit-identical
    # weights between the freshly-constructed binary model and the
    # original pretrained model, for every parameter EXCEPT the
    # (expectedly different-shaped) classifier.
    mismatched = []
    pretrained_params = dict(pretrained_model.named_parameters())
    for name, param in model.named_parameters():
        if name.startswith("classifier."):
            continue
        ref = pretrained_params.get(name)
        if ref is None or ref.shape != param.shape or not torch.equal(ref.data, param.data):
            mismatched.append(name)
    if mismatched:
        raise RuntimeError(
            f"Encoder/pooler reuse verification FAILED for {len(mismatched)} parameter(s), "
            f"e.g. {mismatched[:5]} - the binary model's non-classifier weights are not "
            "bit-identical to the original pretrained model. Refusing to report "
            "'encoder/pooler reused' as fact when it isn't verified true."
        )
    print(f"Verified: all non-classifier parameters ({len(pretrained_params) - 2} tensors) are "
          "bit-identical between the original pretrained model and the new binary model.")

    del pretrained_model  # the 3-class model is no longer needed past this point

    with torch.no_grad():
        model.classifier.weight.data[0].copy_(entailment_row)
        model.classifier.bias.data[0].copy_(entailment_bias)
        model.classifier.weight.data[1].copy_(contradiction_row)
        model.classifier.bias.data[1].copy_(contradiction_bias)

    report = {
        "strategy": "reuse_pretrained_encoder_and_pooler_plus_transplanted_classifier_rows",
        "description": (
            "Encoder and pooler weights reused unchanged from the pretrained checkpoint "
            "(verified bit-identical, not assumed). The new 2-row classifier is initialized "
            "from the ORIGINAL pretrained 3-class classifier's entailment and contradiction "
            "rows (verified indices from the model's own config) - not randomly initialized, "
            "and not trained from scratch."
        ),
        "source_model": model_name,
        "source_label_indices": source_indices,
        "encoder_reused": True,
        "encoder_reuse_verified_bit_identical": True,
        "pooler_reused": True,
        "pooler_reuse_verified_bit_identical": True,
        "classifier_shape_before": [3, 1024],
        "classifier_shape_after": [2, 1024],
        "classifier_rows_randomly_initialized": False,
        "new_label2id": BINARY_LABEL2ID,
        "new_id2label": BINARY_ID2LABEL,
    }
    return model, report
