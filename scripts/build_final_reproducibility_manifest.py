"""Step 22: build the final reproducibility manifest by reading the
live environment plus already-saved artifacts from Steps 9-21. No model
calls, no new experiments. Only SHA-256 hashes of small config/metadata
files are computed (not multi-gigabyte model weight files - infeasible
and not standard practice within this step's scope); this is stated
explicitly rather than implied.

Usage:
    python scripts/build_final_reproducibility_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

ROOT = cg_config.PROJECT_ROOT
OUT_DIR = cg_config.resolve_path("data/processed/final")
MANIFEST_PATH = OUT_DIR / "final_reproducibility_manifest.json"

HASHED_FILES = [
    "configs/default.yaml", "configs/models.yaml", "configs/verifier_binary_final.yaml",
    "configs/reranker_baseline.yaml", "configs/integration_baseline.yaml", "configs/decision_policy.yaml",
    "configs/generator_baseline.yaml", "configs/correction_baseline.yaml",
    "experiments/verifier_binary_final/final/config.json",
]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load(path: str) -> dict:
    with (ROOT / path).open() as f:
        return json.load(f)


def _git_state() -> dict:
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True)
    log = subprocess.run(["git", "log"], cwd=ROOT, capture_output=True, text=True)
    return {
        "git_status_short": status.stdout.strip().splitlines(),
        "git_log": log.stdout.strip() or log.stderr.strip(),
        "commits_exist": log.returncode == 0 and bool(log.stdout.strip()),
    }


def _test_count() -> int | None:
    result = subprocess.run(
        ["python", "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT,
        capture_output=True, text=True, timeout=600,
    )
    for line in result.stderr.splitlines():
        if line.startswith("Ran "):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def main() -> int:
    import torch

    verifier_repro = _load("experiments/verifier_binary_final/final/reproducibility.json")
    models_cfg = cg_config.load_models_config()

    ragtruth_manifest = _load("data/processed/evaluation/ragtruth/ragtruth_eval_manifest.json")
    truthfulqa_manifest = _load("data/processed/evaluation/truthfulqa/truthfulqa_eval_manifest.json")
    ablation_manifest = _load("data/processed/evaluation/ablations/ablation_run_manifest.json")

    file_hashes = {rel: _sha256(ROOT / rel) for rel in HASHED_FILES}

    manifest = {
        "project": "ClaimGuard", "steps_completed": "1-21", "report_step": "22 (final documentation)",
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "frozen_model_identifiers": {
            "generator": models_cfg["generator"]["primary"]["name"],
            "generator_dtype": models_cfg["generator"]["primary"]["dtype"],
            "verifier_base_model": verifier_repro["model_name"],
            "verifier_checkpoint": "experiments/verifier_binary_final/final",
            "verifier_n_params": verifier_repro["n_total_params"],
            "embedding_model": models_cfg["embedding"]["primary"]["name"],
            "reranker_model": models_cfg["reranker"]["primary"]["name"],
        },
        "seeds": {
            "dataset_manifest_seed": 42, "retrieval_eval_seed": 42, "generation_seed": 42,
            "ragtruth_eval_seed": "N/A (full test split evaluated, no sampling)",
            "note": "Seed 42 used consistently across dataset splitting, retrieval-eval sampling, "
                    "and deterministic generation throughout Steps 9-21.",
        },
        "dataset_sources": {
            "FEVER": "fever/fever (Hugging Face), CC BY-SA 3.0 + GNU FDL",
            "HaluEval": "HaluEval (original repository), MIT",
            "RAGTruth": "RAGTruth (original repository), MIT",
            "TruthfulQA": "sylinrl/TruthfulQA (original repository), Apache 2.0",
        },
        "evaluation_counts": {
            "ragtruth_test_responses": ragtruth_manifest.get("n_responses_evaluated"),
            "truthfulqa_questions": truthfulqa_manifest.get("n_questions_evaluated"),
            "ablation_subset_responses": ablation_manifest.get("subset_size"),
            "ablation_full_ragtruth_test_size": ablation_manifest.get("full_ragtruth_test_size"),
        },
        "config_and_checkpoint_file_hashes_sha256": file_hashes,
        "hash_scope_note": (
            "Only small config/metadata files were hashed (see list above). Full model weight files "
            "(multi-GB safetensors shards) were NOT hashed - infeasible within this step's scope and "
            "not standard practice for a documentation-only step; model identity/provenance is instead "
            "established via the exact HF model identifiers and the verified architecture checks "
            "already performed in Steps 9-17 (e.g. assert_binary_head, bit-identical encoder checks)."
        ),
        "key_artifact_paths": [
            "data/processed/evaluation/ragtruth/ragtruth_eval_results.json",
            "data/processed/evaluation/truthfulqa/truthfulqa_eval_results.json",
            "data/processed/evaluation/ablations/ablation_results.json",
            "data/processed/final/final_results_tables.json",
            "experiments/verifier_binary_final/final/",
        ],
        "git_state": _git_state(),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Saved {MANIFEST_PATH}")
    print(json.dumps(manifest["git_state"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
