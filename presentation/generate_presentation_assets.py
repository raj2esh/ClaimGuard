"""Generate diagram PNGs for the ClaimGuard guide presentation.
Pure documentation asset generation - no model calls, no experiments,
no numerical values invented (only used where drawn as static schematic
labels, matching already-established facts from PROJECT_REPORT_FINAL.md).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).resolve().parent / "pres_assets"
OUT.mkdir(exist_ok=True)

NAVY = "#1f3a5f"
TEAL = "#2f7a78"
AMBER = "#c97a2b"
RED = "#b3453c"
GRAY = "#5b6470"
LIGHT_BLUE = "#dbe7f5"
LIGHT_TEAL = "#dcefe1"
LIGHT_AMBER = "#f5e6d3"
LIGHT_RED = "#f9d5d3"
LIGHT_GRAY = "#eceff2"


def box(ax, xy, w, h, text, fc=LIGHT_BLUE, ec=NAVY, fs=10.5, fw="normal", tc="#1a1a1a"):
    ax.add_patch(mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=fc, edgecolor=ec, linewidth=1.4,
    ))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, linespacing=1.35)


def varrow(ax, x, y0, y1, color=GRAY):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))


def harrow(ax, x0, x1, y, color=GRAY):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------
# Slide 2 — motivation flow (vertical, 4 boxes)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.2, 6.2))
ax.set_xlim(0, 4.2); ax.set_ylim(0, 6.2); ax.axis("off")
labels = ["LLM", "Fluent Answer", '"Is it actually supported?"', "Evidence Verification"]
colors = [LIGHT_BLUE, LIGHT_BLUE, LIGHT_AMBER, LIGHT_TEAL]
y = 5.0
for lab, c in zip(labels, colors):
    box(ax, (0.3, y), 3.6, 0.85, lab, fc=c, fs=12)
    if y > 0.3:
        varrow(ax, 2.1, y - 0.15, y - 0.45)
    y -= 1.55
save(fig, "slide2_motivation.png")

# ---------------------------------------------------------------------
# Slide 3 — research problem pipeline (horizontal, failure points marked)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 2.6))
ax.set_xlim(0, 11.5); ax.set_ylim(0, 2.6); ax.axis("off")
stages = ["Answer", "Evidence\nRetrieval", "Evidence\nRanking", "Verification", "Decision", "Correction"]
x = 0.2; w, h, gap = 1.65, 1.0, 0.22
for i, s in enumerate(stages):
    box(ax, (x, 1.0), w, h, s, fc=LIGHT_BLUE if i == 0 else LIGHT_TEAL, fs=10.5)
    ax.plot(x + w / 2, 2.15, marker="*", markersize=14, color=RED, zorder=5)
    if i < len(stages) - 1:
        harrow(ax, x + w, x + w + gap, 1.5)
    x += w + gap
ax.text(5.75, 0.35, "* every stage can introduce failure — reliability depends on the WHOLE chain, not any single stage",
        ha="center", fontsize=9.5, style="italic", color=GRAY)
save(fig, "slide3_pipeline.png")

# ---------------------------------------------------------------------
# Slide 5 — full architecture (vertical, stage-labeled)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 10.0))
ax.set_xlim(0, 6.6); ax.set_ylim(0, 10.0); ax.axis("off")
stages = [
    ("User Query", LIGHT_GRAY, None),
    ("Qwen3-8B", LIGHT_BLUE, "GENERATE"),
    ("Candidate Answer", LIGHT_GRAY, None),
    ("BGE-large Embedding", LIGHT_TEAL, "RETRIEVE"),
    ("FAISS Retrieval  →  Top-20 Evidence", LIGHT_TEAL, "RETRIEVE"),
    ("BGE Reranker  →  Top-5 Evidence", LIGHT_AMBER, "RERANK"),
    ("DeBERTa-v3-large Verifier", LIGHT_RED, "VERIFY"),
    ("Step 16 Decision Policy", "#e6ddf2", "DECIDE"),
    ("ACCEPT / CORRECT / ABSTAIN", "#e6ddf2", "DECIDE"),
    ("Correction Loop (Qwen3-8B, bounded)", "#f2ddef", "CORRECT"),
    ("Final Answer", LIGHT_GRAY, None),
]
y = 9.35
box_h = 0.62
step = 0.855
for i, (lab, c, tag) in enumerate(stages):
    box(ax, (1.5, y), 3.6, box_h, lab, fc=c, fs=9.8)
    if tag:
        ax.text(5.35, y + box_h / 2, tag, ha="left", va="center", fontsize=9, fontweight="bold", color=NAVY)
    if i < len(stages) - 1:
        varrow(ax, 3.3, y - 0.06, y - (step - box_h) + 0.06)
    y -= step
save(fig, "slide5_architecture.png")

# ---------------------------------------------------------------------
# Slide 7 — contamination protocol flow
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 4.6))
ax.set_xlim(0, 11.5); ax.set_ylim(0, 4.6); ax.axis("off")
chain = ["HaluEval\nsummarization", "100 exact\ndocument overlaps", "RAGTruth\ntest", "200 expanded\nrecords removed", "Final contamination-\ncontrolled verifier pool"]
x = 0.15; w, h, gap = 2.05, 1.1, 0.15
for i, s in enumerate(chain):
    c = LIGHT_RED if i in (1, 3) else LIGHT_BLUE
    box(ax, (x, 2.7), w, h, s, fc=c, fs=9.8)
    if i < len(chain) - 1:
        harrow(ax, x + w, x + w + gap, 3.25)
    x += w + gap
box(ax, (1.2, 0.35), 9.1, 1.1,
    "TRAIN / DEV   ≠   RAGTruth TEST   ≠   TruthfulQA\nenforced by RoleViolationError guards + AST import-exclusion tests + behavioral sentinel tests",
    fc=LIGHT_TEAL, fs=10.5)
save(fig, "slide7_contamination.png")

# ---------------------------------------------------------------------
# Slide 9 — retrieval pipeline (simple horizontal)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 1.5))
ax.set_xlim(0, 11); ax.set_ylim(0, 1.5); ax.axis("off")
stages = ["BGE-large\nEmbedding", "FAISS\nIndexFlatIP", "Top-20\nCandidates", "BGE\nReranker", "Top-5\nEvidence"]
x = 0.1; w, h, gap = 1.95, 0.95, 0.2
for i, s in enumerate(stages):
    box(ax, (x, 0.28), w, h, s, fc=LIGHT_TEAL if i < 2 else LIGHT_AMBER if i in (3,) else LIGHT_GRAY, fs=9.8)
    if i < len(stages) - 1:
        harrow(ax, x + w, x + w + gap, 0.75)
    x += w + gap
save(fig, "slide9_retrieval_pipeline.png")

# ---------------------------------------------------------------------
# Slide 11 — decision / correction flow
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 6.7))
ax.set_xlim(0, 9.5); ax.set_ylim(0, 6.7); ax.axis("off")
chain = ["Verifier Score", "Decision Policy", "ACCEPT / CORRECT / ABSTAIN", "Correction (if required)", "Re-verification"]
y = 5.85
for i, s in enumerate(chain):
    box(ax, (2.2, y), 5.1, 0.75, s, fc=LIGHT_BLUE if i < 2 else "#e6ddf2", fs=10.5)
    if i < len(chain) - 1:
        varrow(ax, 4.75, y - 0.08, y - 0.24)
    y -= 1.0
box(ax, (0.6, 0.15), 8.3, 0.95,
    'Observation: Attempt-0 ABSTAIN = 0% across RAGTruth and TruthfulQA\n"The system is rarely uncertain — even when it is wrong."',
    fc=LIGHT_RED, fs=10.5, fw="bold")
save(fig, "slide11_decision_correction.png")

# ---------------------------------------------------------------------
# Slide 13 — RAGTruth failure: evidence-domain mismatch
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 6.2))
ax.set_xlim(0, 11.5); ax.set_ylim(0, 6.2); ax.axis("off")
box(ax, (0.3, 4.9), 3.4, 1.0, "ClaimGuard corpus:\nFEVER / Wikipedia", fc=LIGHT_BLUE, fs=10.5)
box(ax, (7.8, 4.9), 3.4, 1.0, "RAGTruth:\nnews, business listings,\nQA passages", fc=LIGHT_AMBER, fs=10.5)
box(ax, (3.75, 3.55), 4.0, 0.9, "Little genuine evidence overlap", fc=LIGHT_GRAY, fs=10.5)
ax.annotate("", xy=(4.9, 4.45), xytext=(2.0, 4.9), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.8))
ax.annotate("", xy=(6.6, 4.45), xytext=(9.5, 4.9), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.8))
box(ax, (3.75, 2.3), 4.0, 0.9, "Topical but irrelevant evidence retrieved", fc=LIGHT_RED, fs=10.5)
varrow(ax, 5.75, 3.55, 3.2)
box(ax, (3.75, 1.05), 4.0, 0.9, "Verifier overconfidence\non irrelevant evidence", fc=LIGHT_RED, fs=10.5)
varrow(ax, 5.75, 2.3, 1.95)
box(ax, (3.75, -0.2), 4.0, 0.9, "Incorrect decisions", fc=LIGHT_RED, fs=10.5, fw="bold")
varrow(ax, 5.75, 1.05, 0.7)
ax.set_ylim(-0.3, 6.2)
ax.text(9.7, 0.55, "0 mechanical\nretrieval failures\n(n=300)", ha="center", fontsize=9, color=GRAY,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY))
ax.text(9.7, 2.75, "relevance failure:\nNOT directly measurable\n(no gold-relevance\nannotation in RAGTruth)", ha="center", fontsize=9, color=GRAY,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY))
save(fig, "slide13_domain_mismatch.png")

# ---------------------------------------------------------------------
# Slide 16 — evidence-supported failure model
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 8.9))
ax.set_xlim(0, 7.6); ax.set_ylim(0, 8.9); ax.axis("off")
chain = [
    "DOMAIN-MISMATCHED\nCORPUS",
    "WEAK / IRRELEVANT\nEVIDENCE",
    "VERIFIER\nOVERCONFIDENCE",
    "NEAR-ZERO\nABSTENTION",
    "WRONG DECISION",
    "HARMFUL CORRECTION",
]
y = 8.0
for i, s in enumerate(chain):
    box(ax, (0.4, y), 4.0, 0.85, s, fc=LIGHT_RED if i >= 2 else LIGHT_AMBER, fs=11, fw="bold")
    if i < len(chain) - 1:
        varrow(ax, 2.4, y - 0.08, y - 0.28)
    y -= 1.15
side = [
    ("Verifier confidently wrong", "38.3% of diagnostic subset"),
    ("Harmful correction", "33.7%"),
    ("Retrieval mechanical failure", "0%"),
    ("Reranking", "not a meaningful bottleneck"),
]
ys = 6.6
for label, val in side:
    ax.text(6.9, ys, label, ha="center", fontsize=9.3, color=GRAY, wrap=True)
    ax.text(6.9, ys - 0.35, val, ha="center", fontsize=10.5, fontweight="bold", color=NAVY)
    ys -= 1.55
ax.text(3.8, 0.15, 'Evidence-supported failure model — NOT a proven causal DAG',
        ha="center", fontsize=10, style="italic", color=GRAY)
save(fig, "slide16_failure_model.png")

print("All presentation diagrams saved to", OUT)
