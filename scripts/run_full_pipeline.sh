#!/usr/bin/env bash
# ClaimGuard — full pipeline driver.
#
# Runs the documented ClaimGuard pipeline stages (see REPRODUCIBILITY.md) in
# order, using exactly the scripts already in this repository — nothing
# here invents a new CLI or new logic, it only sequences existing scripts.
#
# PREREQUISITES (see REPRODUCIBILITY.md for full detail):
#   - `claimguard` conda env active, `pip install -e . --no-deps` done.
#   - Raw datasets (FEVER, HaluEval, RAGTruth, TruthfulQA) already acquired
#     and placed under data/raw/ (§5-10) — this script does NOT download
#     them for you; their licenses require you to obtain them yourself.
#   - A CUDA GPU for stages 4 onward (verifier training and every
#     evaluation stage); stages 1-3 are CPU-only.
#
# This is a convenience driver, not a guarantee of one-command reproduction:
# the full run (stage 8 in particular) took 15+ hours of GPU time when this
# project originally ran it — see PROJECT_REPORT.md Steps 19-21.
#
# Usage:
#   scripts/run_full_pipeline.sh --list              # show all stages, exit
#   scripts/run_full_pipeline.sh                      # run every stage, full scale
#   scripts/run_full_pipeline.sh --from 5             # resume from stage 5
#   scripts/run_full_pipeline.sh --only 8             # run just stage 8
#   scripts/run_full_pipeline.sh --dry-run            # print commands, run nothing
#   scripts/run_full_pipeline.sh --fast               # small/pilot-scale run
#                                                      # (uses each stage's
#                                                      # existing --smoke-test /
#                                                      # --pilot / --subset-size
#                                                      # flag, where the
#                                                      # underlying script
#                                                      # supports one - NOT a
#                                                      # substitute for the
#                                                      # real evaluation, just
#                                                      # a fast plumbing check)
#   scripts/run_full_pipeline.sh -y                   # skip the confirmation prompt
#
# Flags can be combined, e.g.: --fast --from 4 -y

set -euo pipefail
cd "$(dirname "$0")/.."

FAST=0
DRY_RUN=0
FROM=1
ONLY=""
ASSUME_YES=0
LIST_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) FAST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    --list) LIST_ONLY=1; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

# Stage table: "number|name|full command|fast command (or empty = same as full)"
STAGES=(
  "1|Dataset acquisition/inspection (FEVER, HaluEval, RAGTruth, TruthfulQA + FEVER wiki_pages)|python scripts/inspect_fever.py && python scripts/inspect_halueval.py && python scripts/inspect_ragtruth.py && python scripts/inspect_truthfulqa.py && python scripts/acquire_inspect_wiki_pages.py && python scripts/build_wiki_pages_index.py && python scripts/resolve_fever_evidence.py|"
  "2|Dataset manifest + contamination detection|python scripts/build_dataset_manifest.py && python scripts/detect_contamination.py|"
  "3|Verifier train/dev pool|python scripts/build_verifier_pool.py && python scripts/validate_verifier_dataset.py|"
  "4|Verifier training (binary entailment/contradiction head)|python -m claimguard.verifier.train_binary --config configs/verifier_binary_final.yaml|python -m claimguard.verifier.train_binary --config configs/verifier_binary_final.yaml --smoke-test"
  "5|Retrieval corpus + embeddings + FAISS index|python scripts/build_retrieval_corpus.py && python scripts/embed_retrieval_corpus.py && python scripts/build_faiss_index.py|python scripts/build_retrieval_corpus.py && python scripts/embed_retrieval_corpus.py --smoke-test && python scripts/build_faiss_index.py"
  "6|Retrieval + reranker recall@K evaluation|python scripts/evaluate_retrieval.py && python scripts/evaluate_reranker.py|"
  "7|Integration (retrieval+rerank+verify) + calibration/decision-policy evaluation|python scripts/evaluate_integration.py|"
  "8|RAGTruth + TruthfulQA end-to-end evaluation (SLOW: 15+ hours at full scale)|python scripts/evaluate_ragtruth.py && python scripts/evaluate_truthfulqa.py && python scripts/analyze_ragtruth_results.py && python scripts/analyze_truthfulqa_results.py|python scripts/evaluate_ragtruth.py --pilot 20 && python scripts/evaluate_truthfulqa.py --pilot 20 && python scripts/analyze_ragtruth_results.py && python scripts/analyze_truthfulqa_results.py"
  "9|Controlled ablations|python scripts/run_ablations.py && python scripts/analyze_ablations.py|python scripts/run_ablations.py --pilot 15 && python scripts/analyze_ablations.py"
  "10|Final results tables, reproducibility manifest, figures (Step 22)|python scripts/generate_final_results_tables.py && python scripts/build_final_reproducibility_manifest.py && python scripts/generate_final_figures.py|"
  "11|Test suite|python -m unittest discover -s tests|"
)

if [[ $LIST_ONLY -eq 1 ]]; then
  echo "ClaimGuard pipeline stages:"
  for s in "${STAGES[@]}"; do
    IFS='|' read -r num name _ _ <<< "$s"
    echo "  $num. $name"
  done
  exit 0
fi

echo "============================================================"
echo "ClaimGuard full pipeline driver"
echo "============================================================"
echo "Mode:      $([[ $FAST -eq 1 ]] && echo 'FAST (pilot/smoke-test scale)' || echo 'FULL (matches originally reported results)')"
echo "Dry run:   $([[ $DRY_RUN -eq 1 ]] && echo 'yes - no commands will actually execute' || echo 'no')"
if [[ -n "$ONLY" ]]; then
  echo "Running:   stage $ONLY only"
else
  echo "Running:   stages $FROM through 11"
fi
echo
echo "This does NOT download raw datasets for you (see REPRODUCIBILITY.md"
echo "Section 5) and, in FULL mode, stage 8 alone can take 15+ hours on a"
echo "single GPU. This will NOT modify any already-reported research"
echo "result files unless you explicitly overwrite them by running this."
echo "============================================================"

if [[ $ASSUME_YES -eq 0 && $DRY_RUN -eq 0 ]]; then
  read -r -p "Continue? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

for s in "${STAGES[@]}"; do
  IFS='|' read -r num name full_cmd fast_cmd <<< "$s"

  if [[ -n "$ONLY" && "$num" != "$ONLY" ]]; then continue; fi
  if [[ -z "$ONLY" && "$num" -lt "$FROM" ]]; then continue; fi

  cmd="$full_cmd"
  if [[ $FAST -eq 1 && -n "$fast_cmd" ]]; then cmd="$fast_cmd"; fi

  echo
  echo "------------------------------------------------------------"
  echo "Stage $num: $name"
  echo "  \$ $cmd"
  echo "------------------------------------------------------------"

  if [[ $DRY_RUN -eq 1 ]]; then
    continue
  fi

  start=$(date +%s)
  eval "$cmd"
  end=$(date +%s)
  echo "Stage $num finished in $((end - start))s."
done

echo
echo "Pipeline driver finished."
