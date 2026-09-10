"""Step 22: generate the final presentation figures (1-8) using ONLY
already-measured data from data/processed/final/final_results_tables.json
(itself a pure re-aggregation of Steps 9-21's validated artifacts - see
that file's "provenance" field). No model calls, no new experiments, no
recomputation of any metric with different logic. Two of the eight
figures (1: architecture, 2: protocol/isolation, 8: failure model) are
schematic diagrams, not data plots - they are static representations of
already-documented system structure, not new findings.

All bar-chart axes start at 0 (no truncation) and every panel labels its
sample size (n) explicitly.

Usage:
    python scripts/generate_final_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

ROOT = cg_config.PROJECT_ROOT
TABLES_PATH = cg_config.resolve_path("data/processed/final/final_results_tables.json")
FIG_DIR = cg_config.resolve_path("data/processed/final/figures")


def _load_tables() -> dict:
    with TABLES_PATH.open() as f:
        return json.load(f)


def _box(ax, xy, w, h, text, fc="#dbe7f5"):
    ax.add_patch(mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02", facecolor=fc, edgecolor="#333333", linewidth=1.2,
    ))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.4))


def fig1_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    stages = [
        "Input\n(claim/question\n+ context)", "FAISS\nretrieval\n(bge-large-en)",
        "Reranker\n(bge-reranker\n-large)", "Verifier\n(DeBERTa-v3\nbinary head)",
        "Decision\npolicy\n(accept/correct/\nabstain)", "Generator +\ncorrection loop\n(Qwen3-8B)",
        "Final\noutput",
    ]
    x = 0.2
    w, h, gap = 1.35, 1.4, 0.18
    for i, s in enumerate(stages):
        _box(ax, (x, 1.05), w, h, s)
        if i < len(stages) - 1:
            _arrow(ax, x + w, 1.05 + h / 2, x + w + gap, 1.05 + h / 2)
        x += w + gap
    ax.set_title("Figure 1. ClaimGuard pipeline architecture (schematic - see PROJECT_REPORT_FINAL.md §5)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig2_protocol_isolation(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4)
    ax.axis("off")
    _box(ax, (0.3, 2.2), 3.6, 1.2, "INFERENCE PATH\nretrieval → verifier → decision\npolicy → generator/correction\n(gold label NEVER passed in)",
         fc="#dcefe1")
    _box(ax, (5.1, 2.2), 3.6, 1.2, "EVALUATION PATH\nreads gold label ONLY to\nlabel the OUTPUT record\nafter inference completes",
         fc="#f5e6d3")
    _arrow(ax, 3.9, 2.8, 5.1, 2.8)
    ax.text(4.5, 3.15, "output +\nprediction", ha="center", fontsize=8)
    _box(ax, (2.6, 0.3), 3.8, 1.2,
         "Enforcement: AST-based static import-exclusion tests\n+ mock-based behavioral tests "
         "(sentinel gold value\nasserted absent from every inference call argument)",
         fc="#f0f0f0")
    _arrow(ax, 6.9, 2.2, 5.5, 1.5)
    ax.set_title("Figure 2. Gold-label isolation protocol (schematic - enforced by tests/, see §6-7)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig3_retrieval_recall(t: dict, out: Path) -> None:
    r3 = t["table3_retrieval_reranking"]
    ks = ["1", "5", "10", "20"]
    faiss = [r3["faiss_only"][f"recall_at_{k}"] for k in ks]
    rerank = [r3["faiss_plus_reranker"][f"recall_at_{k}"] for k in ks]
    n = r3["faiss_only"]["n_eval_claims"]
    x = range(len(ks))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    width = 0.35
    ax.bar([i - width / 2 for i in x], faiss, width, label="FAISS only", color="#4c72b0")
    ax.bar([i + width / 2 for i in x], rerank, width, label="FAISS + reranker", color="#dd8452")
    ax.set_xticks(list(x), [f"recall@{k}" for k in ks])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall")
    ax.set_title(f"Figure 3. Retrieval recall: FAISS vs. + reranker (n={n} claims)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig4_ragtruth_baseline_vs_final(t: dict, out: Path) -> None:
    r4 = t["table4_end_to_end_ragtruth"]
    labels = ["attempt0\n(pre-correction)", "final\n(post-correction)"]
    acc = [r4["attempt0_accuracy"], r4["final_accuracy"]]
    f1 = [r4["attempt0_macro_f1"], r4["final_macro_f1"]]
    n = r4["n_responses"]
    baseline = r4["majority_class_baseline"]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    width = 0.35
    ax.bar([i - width / 2 for i in x], acc, width, label="Accuracy", color="#4c72b0")
    ax.bar([i + width / 2 for i in x], f1, width, label="Macro F1", color="#dd8452")
    ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.2,
               label=f"majority-class baseline ({baseline:.3f})")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Figure 4. RAGTruth: attempt0 vs. final outcome (n={n} responses)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig5_ablation_comparison(t: dict, out: Path) -> None:
    r6 = t["table6_controlled_ablations"]
    conds = ["A_reranked", "B_no_reranker", "C_faiss_top1", "D_no_correction_REUSED", "E_verify_only_REUSED"]
    accs = [r6[c]["accuracy"] for c in conds]
    f1s = [r6[c]["macro_f1"] for c in conds]
    ns = [r6[c]["n"] for c in conds]
    x = range(len(conds))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.35
    ax.bar([i - width / 2 for i in x], accs, width, label="Accuracy", color="#4c72b0")
    ax.bar([i + width / 2 for i in x], f1s, width, label="Macro F1", color="#dd8452")
    ax.set_xticks(list(x), [f"{c}\n(n={n})" for c, n in zip(conds, ns)], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Figure 5. Controlled ablations: detection accuracy/macro F1 by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig6_verifier_confidence_by_outcome(t: dict, out: Path) -> None:
    r7 = t["table7_failure_source_evidence"]["verifier_overconfidence_full_2700"]
    order = ["accept_correct_negative", "correct_correct_positive",
             "accept_wrong_missed_hallucination", "correct_wrong_false_positive"]
    labels = ["ACCEPT,\ncorrect", "CORRECT,\ncorrect", "ACCEPT,\nWRONG\n(missed halluc.)", "CORRECT,\nWRONG\n(false positive)"]
    means = [r7[k]["mean_confidence"] for k in order]
    ns = [r7[k]["n"] for k in order]
    colors = ["#55a868", "#55a868", "#c44e52", "#c44e52"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(order)), means, color=colors)
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order)), labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean verifier confidence")
    ax.set_title("Figure 6. Verifier confidence is similarly high whether decision\nwas correct or wrong (full RAGTruth test, n=2700)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig7_correction_benefit_vs_harm(t: dict, out: Path) -> None:
    cats = t["table4_end_to_end_ragtruth"]["correction_outcome_categories"]
    order = ["hallucinated_corrected_successfully", "hallucinated_still_flagged", "hallucinated_missed",
              "not_hallucinated_preserved", "not_hallucinated_degraded"]
    labels = ["hallucinated,\ncorrected\nsuccessfully", "hallucinated,\nstill flagged", "hallucinated,\nmissed",
              "not hallucinated,\npreserved", "not hallucinated,\nDEGRADED"]
    vals = [cats[k] for k in order]
    colors = ["#55a868", "#dd8452", "#c44e52", "#55a868", "#c44e52"]
    total = sum(cats.values())
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(range(len(order)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 8, str(v), ha="center", fontsize=9)
    ax.set_xticks(range(len(order)), labels, fontsize=8)
    ax.set_title(f"Figure 7. Correction-loop outcome breakdown (n={total} RAGTruth responses)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig8_failure_model(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    _box(ax, (0.2, 3.6), 2.6, 1.0, "FEVER/Wikipedia\nretrieval corpus vs.\nRAGTruth source domains\n(news/business/QA)", fc="#f5e6d3")
    _box(ax, (3.4, 3.6), 2.6, 1.0, "Retrieved evidence often\ntopically irrelevant to\nRAGTruth responses", fc="#f5e6d3")
    _box(ax, (6.6, 3.6), 3.0, 1.0, "Verifier scores irrelevant\nevidence with same high\nconfidence as relevant evidence\n(overconfidence, not miscalibration)", fc="#f9d5d3")
    _box(ax, (3.4, 1.9), 2.6, 1.0, "Decision policy over-triggers\ncorrection (65.3% attempt0\ncorrect-rate)", fc="#f9d5d3")
    _box(ax, (6.6, 1.9), 3.0, 1.0, "Correction sometimes degrades\nalready-correct responses\n(923/2700 not_hallucinated_degraded)", fc="#f9d5d3")
    _arrow(ax, 2.8, 4.1, 3.4, 4.1)
    _arrow(ax, 6.0, 4.1, 6.6, 4.1)
    _arrow(ax, 8.1, 3.6, 4.7, 2.9)
    _arrow(ax, 4.7, 1.9, 6.6, 2.4)
    ax.text(5.0, 0.9,
            "Evidence-supported failure model - built from Steps 14/16/19/21 measurements.\n"
            "This is NOT a proven causal DAG: arrows denote evidence-supported association/plausible\n"
            "mechanism, not demonstrated causal effect sizes. See PROJECT_REPORT_FINAL.md §17.",
            ha="center", fontsize=8, style="italic")
    ax.set_title("Figure 8. Final evidence-supported failure model", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    t = _load_tables()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_architecture(FIG_DIR / "figure1_architecture.png")
    fig2_protocol_isolation(FIG_DIR / "figure2_protocol_isolation.png")
    fig3_retrieval_recall(t, FIG_DIR / "figure3_retrieval_recall.png")
    fig4_ragtruth_baseline_vs_final(t, FIG_DIR / "figure4_ragtruth_baseline_vs_final.png")
    fig5_ablation_comparison(t, FIG_DIR / "figure5_ablation_comparison.png")
    fig6_verifier_confidence_by_outcome(t, FIG_DIR / "figure6_verifier_confidence_by_outcome.png")
    fig7_correction_benefit_vs_harm(t, FIG_DIR / "figure7_correction_benefit_vs_harm.png")
    fig8_failure_model(FIG_DIR / "figure8_failure_model.png")
    print(f"Saved 8 figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
