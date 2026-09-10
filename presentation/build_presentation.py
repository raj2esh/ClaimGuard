"""Build the ClaimGuard research-guide presentation (PPTX).

Every number in this file is copied verbatim from validated Step 9-22
source artifacts (see PROJECT_REPORT_FINAL.md / final_results_tables.json /
final_reproducibility_manifest.json) - nothing is invented, tuned, or
rerun. This is a documentation-generation script only.
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "pres_assets"
OUT_PATH = HERE / "ClaimGuard_Guide_Presentation.pptx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
TEAL = RGBColor(0x2F, 0x7A, 0x78)
AMBER = RGBColor(0xC9, 0x7A, 0x2B)
RED = RGBColor(0xB3, 0x45, 0x3C)
GRAY = RGBColor(0x5B, 0x64, 0x70)
DARK = RGBColor(0x20, 0x20, 0x20)
LIGHTROW = RGBColor(0xF2, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
NAVY_FADE = RGBColor(0xDB, 0xE7, 0xF5)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def new_slide():
    return prs.slides.add_slide(BLANK)


def add_title(slide, title, subtitle=None):
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.32), Inches(12.2), Inches(1.05))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = NAVY
    if subtitle:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.size = Pt(14)
        r2.font.italic = True
        r2.font.color.rgb = GRAY
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.18), Inches(12.2), Pt(2.2))
    line.fill.solid()
    line.fill.fore_color.rgb = NAVY
    line.line.fill.background()
    return box


def add_bullets(slide, bullets, left, top, width, height, font_size=16, color=DARK, space_after=10):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        run = p.add_run()
        run.text = "•  " + b
        run.font.size = Pt(font_size)
        run.font.color.rgb = color
    return box


def add_textbox(slide, text, left, top, width, height, size=14, color=DARK, bold=False,
                 italic=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    return box


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def add_picture_fit(slide, filename, left, top, max_w, max_h, center_h=True):
    path = ASSETS / filename
    with Image.open(path) as im:
        iw, ih = im.size
    ratio = iw / ih
    w = max_w
    h = int(w / ratio)
    if h > max_h:
        h = max_h
        w = int(h * ratio)
    l = left
    if center_h:
        l = left + int((max_w - w) / 2)
    slide.shapes.add_picture(str(path), l, top, width=w, height=h)
    return w, h


def add_table(slide, left, top, width, height, header, rows, col_widths=None, font_size=13,
              header_size=13):
    n_rows = len(rows) + 1
    n_cols = len(header)
    gt = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = gt.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            table.columns[i].width = cw
    for j, h in enumerate(header):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(header_size)
                r.font.color.rgb = WHITE
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHTROW if i % 2 == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.size = Pt(font_size)
                    r.font.color.rgb = DARK
    return gt


def add_bar_chart(slide, left, top, width, height, categories, series, title=None,
                   value_axis_title=None, chart_type=XL_CHART_TYPE.COLUMN_CLUSTERED,
                   number_format='0.0"%"', legend=True):
    chart_data = CategoryChartData()
    chart_data.categories = categories
    for name, values in series:
        chart_data.add_series(name, values)
    gframe = slide.shapes.add_chart(chart_type, left, top, width, height, chart_data)
    chart = gframe.chart
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(11)
    if title:
        chart.has_title = True
        chart.chart_title.text_frame.text = title
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = NAVY
    else:
        chart.has_title = False
    plot = chart.plots[0]
    plot.has_data_labels = True
    plot.data_labels.number_format = number_format
    plot.data_labels.number_format_is_linked = False
    plot.data_labels.font.size = Pt(10)
    cat_axis = chart.category_axis
    cat_axis.tick_labels.font.size = Pt(11)
    val_axis = chart.value_axis
    val_axis.tick_labels.font.size = Pt(10)
    if value_axis_title:
        val_axis.has_title = True
        val_axis.axis_title.text_frame.text = value_axis_title
        val_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
    colors = [NAVY, AMBER, TEAL, RED]
    for i, s in enumerate(plot.series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = colors[i % len(colors)]
    return gframe


def footer(slide, text):
    add_textbox(slide, text, Inches(0.55), Inches(7.18), Inches(12.2), Inches(0.28),
                size=9, color=GRAY, italic=True)


# =====================================================================
# SLIDE 1 — TITLE
# =====================================================================
s = new_slide()
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
bg.fill.solid()
bg.fill.fore_color.rgb = WHITE
bg.line.fill.background()
bg.shadow.inherit = False
top_bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(0.18))
top_bar.fill.solid()
top_bar.fill.fore_color.rgb = NAVY
top_bar.line.fill.background()
add_textbox(s, "ClaimGuard", Inches(1.0), Inches(2.55), Inches(11.3), Inches(1.2),
            size=54, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
add_textbox(s, "Evidence-Grounded Hallucination Detection and Correction\nfor LLM-Generated Answers",
            Inches(1.0), Inches(3.75), Inches(11.3), Inches(1.0),
            size=20, color=DARK, align=PP_ALIGN.CENTER)
add_textbox(s, "Design, Evaluation, Ablation, and Failure Analysis",
            Inches(1.0), Inches(4.55), Inches(11.3), Inches(0.6),
            size=15, italic=True, color=GRAY, align=PP_ALIGN.CENTER)
add_textbox(s, "Research Progress Presentation — Steps 1-22 Complete",
            Inches(1.0), Inches(6.5), Inches(11.3), Inches(0.5),
            size=13, color=GRAY, align=PP_ALIGN.CENTER)
set_notes(s, (
    "Good [morning/afternoon]. I'm going to walk you through ClaimGuard — the hallucination-"
    "detection and self-correction pipeline I've been building. This is a full progress "
    "presentation: what the system is, how I evaluated it end-to-end, and importantly, what "
    "I found when I tested it honestly. I want to say up front that this is not a 'we solved "
    "hallucination detection' talk — it's a 'here's what we built, here's what happened when "
    "we tested it rigorously, and here's what that tells us' talk. That's deliberate, and I'll "
    "explain why it's actually the more useful outcome scientifically."
))

# =====================================================================
# SLIDE 2 — MOTIVATION
# =====================================================================
s = new_slide()
add_title(s, "Why Hallucination Detection Matters")
add_picture_fit(s, "slide2_motivation.png", Inches(0.7), Inches(1.5), Inches(4.6), Inches(5.5))
add_bullets(s, [
    "LLMs generate fluent, confident-sounding text.",
    "Fluency does NOT guarantee factual correctness.",
    "Retrieval-augmented verification can potentially detect unsupported claims.",
    "But a reliable system must not just retrieve evidence — it must correctly judge "
    "whether that evidence actually supports the answer.",
], Inches(5.7), Inches(2.0), Inches(6.9), Inches(4.5), font_size=18, space_after=18)
footer(s, "ClaimGuard — Motivation")
set_notes(s, (
    "The starting problem is simple: LLMs are fluent, but fluency and factual correctness are "
    "two different things. A model can produce a confident, well-formed sentence that's "
    "simply wrong. The natural fix people reach for is retrieval-augmented verification — go "
    "get evidence, check the claim against it. But that's not automatically a solved problem "
    "either, because the hard part isn't just retrieving SOME evidence — it's correctly "
    "judging whether the evidence you got actually supports the claim. That judgment step is "
    "where a lot of the interesting failure modes in this project ended up living, and it's "
    "the thread that runs through the whole talk."
))

# =====================================================================
# SLIDE 3 — RESEARCH PROBLEM
# =====================================================================
s = new_slide()
add_title(s, "Research Problem")
add_textbox(s, '"Can an evidence-grounded pipeline reliably detect and correct hallucinated '
               'LLM responses?"', Inches(0.7), Inches(1.35), Inches(11.9), Inches(0.7),
            size=19, italic=True, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
add_picture_fit(s, "slide3_pipeline.png", Inches(0.7), Inches(2.3), Inches(11.9), Inches(3.2))
add_bullets(s, [
    "Answer -> Evidence Retrieval -> Evidence Ranking -> Verification -> Decision -> Correction.",
    "Every single stage can introduce failure — a wrong retrieval, a bad rerank, an "
    "overconfident verifier, a bad decision, or a harmful correction.",
    "End-to-end reliability is a property of the WHOLE chain, not any one stage in isolation.",
], Inches(0.7), Inches(5.7), Inches(11.9), Inches(1.6), font_size=16, space_after=8)
footer(s, "ClaimGuard — Research Problem")
set_notes(s, (
    "So the research question is: can an evidence-grounded pipeline reliably detect and "
    "correct hallucinated responses? The pipeline has six stages, and I want to flag early "
    "that every one of these stages is a place where things can go wrong — a bad retrieval, "
    "a reranker that reorders things unhelpfully, an overconfident verifier, a decision policy "
    "that flags the wrong thing, or a correction step that actually makes the answer worse. "
    "The point of building this carefully was to be able to isolate WHICH of these stages "
    "actually drives the end-to-end result, rather than just reporting one aggregate number."
))

# =====================================================================
# SLIDE 4 — RESEARCH GAP / HYPOTHESIS
# =====================================================================
s = new_slide()
add_title(s, "Research Gap")
add_bullets(s, [
    "Many pipelines are evaluated component-by-component — retrieval recall here, verifier "
    "accuracy there — but end-to-end reliability depends on how these components INTERACT.",
], Inches(0.7), Inches(1.4), Inches(11.9), Inches(1.0), font_size=17)
eq_box = s.shapes.add_textbox(Inches(0.7), Inches(2.5), Inches(11.9), Inches(1.3))
tf = eq_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
run = p.add_run()
run.text = "Retrieval quality  +  Verifier calibration  +  Decision uncertainty  +  Correction behavior"
run.font.size = Pt(19); run.font.bold = True; run.font.color.rgb = NAVY
p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
r2 = p2.add_run(); r2.text = "=  End-to-end reliability"
r2.font.size = Pt(22); r2.font.bold = True; r2.font.color.rgb = TEAL
add_textbox(s, '"Reliable hallucination detection requires both relevant evidence and a '
               'verifier whose confidence reflects actual evidence support."',
            Inches(1.3), Inches(4.3), Inches(10.7), Inches(1.1),
            size=18, italic=True, color=DARK, align=PP_ALIGN.CENTER)
add_textbox(s, "This is the project's working hypothesis — tested, not assumed proven "
               "universally.", Inches(1.3), Inches(5.5), Inches(10.7), Inches(0.6),
            size=13, italic=True, color=GRAY, align=PP_ALIGN.CENTER)
footer(s, "ClaimGuard — Research Gap")
set_notes(s, (
    "The gap I wanted to address is that a lot of prior work reports retrieval recall or "
    "verifier accuracy as separate numbers, but doesn't test how those components interact "
    "once you chain them together with a decision policy and a correction loop. My working "
    "hypothesis going in was that reliable detection needs BOTH relevant evidence AND a "
    "verifier whose confidence actually tracks whether the evidence supports the claim — "
    "not just one or the other. I want to be careful here: this is the hypothesis I tested, "
    "not something I'm claiming is universally proven. The rest of the talk is basically the "
    "test of this hypothesis."
))

# =====================================================================
# SLIDE 5 — ARCHITECTURE
# =====================================================================
s = new_slide()
add_title(s, "ClaimGuard Architecture")
add_picture_fit(s, "slide5_architecture.png", Inches(3.6), Inches(1.3), Inches(6.1), Inches(6.0))
footer(s, "ClaimGuard — Architecture")
set_notes(s, (
    "This is the full architecture. A user query goes to Qwen3-8B to generate a candidate "
    "answer. That answer is embedded with BGE-large and searched against a FAISS index to "
    "get the top-20 evidence candidates. A BGE reranker narrows that to the top 5. The "
    "DeBERTa-v3-large verifier scores each candidate for entailment or contradiction. The "
    "Step-16 decision policy turns those scores into ACCEPT, CORRECT, or ABSTAIN. If CORRECT, "
    "a bounded correction loop regenerates the answer using the retrieved evidence and "
    "re-verifies, up to a fixed attempt budget. I'll come back to almost every one of these "
    "boxes later in the talk, because the failure analysis traces back through this exact "
    "chain."
))

# =====================================================================
# SLIDE 6 — DATASETS
# =====================================================================
s = new_slide()
add_title(s, "Datasets and Their Roles")
add_table(s, Inches(0.7), Inches(1.5), Inches(11.9), Inches(2.0),
          ["Dataset", "Purpose", "Usage"],
          [
              ["FEVER", "Verifier / retrieval development", "Train / Dev"],
              ["HaluEval", "Verifier development supplement", "Train / Dev"],
              ["RAGTruth", "End-to-end hallucination evaluation", "Held-out TEST (never trained on)"],
              ["TruthfulQA", "Truthfulness evaluation", "Evaluation-only"],
          ], font_size=15, header_size=15)
add_bullets(s, [
    "RAGTruth and TruthfulQA were NEVER used for training — enforced structurally, not just by convention.",
    "FEVER evidence sentences were resolved against acquired Wikipedia wiki_pages (no invented text).",
    "HaluEval records were normalized to a common (premise, claim, label) schema before pooling.",
    "TruthfulQA has no official train/test split — used entirely as an evaluation-only benchmark.",
    "RAGTruth's test split is source-document separated from any other split (no source document leaks across splits).",
], Inches(0.7), Inches(3.85), Inches(11.9), Inches(3.2), font_size=15, space_after=8)
footer(s, "ClaimGuard — Datasets")
set_notes(s, (
    "Four datasets, each with a specific role. FEVER and HaluEval are for developing the "
    "verifier — training and dev. RAGTruth is the primary held-out test for the FULL pipeline "
    "— 2,700 responses that were never touched during training. TruthfulQA is a secondary, "
    "evaluation-only adversarial stress test. The important point on this slide is that "
    "RAGTruth and TruthfulQA are held out — not just informally, but with actual guards in "
    "the code that raise an error if anyone tries to build a training pool from them. I'll "
    "show that mechanism on the next slide."
))

# =====================================================================
# SLIDE 7 — CONTAMINATION
# =====================================================================
s = new_slide()
add_title(s, "Data Integrity and Contamination Control")
add_picture_fit(s, "slide7_contamination.png", Inches(0.7), Inches(1.4), Inches(11.9), Inches(3.6))
add_bullets(s, [
    "AST-based import-exclusion guards + behavioral sentinel tests confirm gold labels never reach inference.",
    "Source-level train/test separation verified directly, not assumed.",
    "Deterministic evaluation: identical inputs reproduce identical outputs, re-checked after every major run.",
], Inches(0.7), Inches(5.3), Inches(11.9), Inches(1.7), font_size=15, space_after=8)
footer(s, "ClaimGuard — Data Integrity")
set_notes(s, (
    "This slide is here because I think it matters for how much you should trust everything "
    "that follows. We found that 100 HaluEval summarization records exactly overlapped 100 "
    "RAGTruth test source documents. Since each RAGTruth source has 6 model responses, that "
    "expanded to 600 affected response records. We excluded 200 pool records from training — "
    "never touched RAGTruth test itself. Beyond that, there are structural guards: if any "
    "script tries to build a training pool that includes RAGTruth test or TruthfulQA, it "
    "raises an error. And there are tests that literally run the real inference code with a "
    "sentinel gold-label value and confirm that value never appears in any model call "
    "argument. This is what lets me trust the negative results I'm about to show you — they're "
    "not an artifact of leakage."
))

# =====================================================================
# SLIDE 8 — VERIFIER DEVELOPMENT
# =====================================================================
s = new_slide()
add_title(s, "Binary NLI Verifier")
add_textbox(s, "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
            Inches(0.7), Inches(1.35), Inches(11.9), Inches(0.5), size=17, bold=True, color=TEAL)
add_bullets(s, [
    "435M parameters, binary entailment / contradiction classification head.",
    "Trained on FEVER (resolved evidence + claim) and HaluEval (grounding context + response).",
    "NO fabricated neutral-class examples — binary formulation chosen because no scientifically "
    "valid neutral-premise source was available for this project's constraints.",
], Inches(0.7), Inches(2.0), Inches(6.6), Inches(3.0), font_size=15.5, space_after=12)
add_table(s, Inches(7.6), Inches(2.0), Inches(5.0), Inches(1.6),
          ["Metric", "Value"],
          [["Accuracy", "96.20%"], ["Macro F1", "95.85%"], ["Weighted F1", "96.21%"]],
          font_size=15, header_size=15)
add_textbox(s, "Gold-premise development evaluation — this is NOT the same as end-to-end "
               "pipeline performance (see Slide 12).", Inches(7.6), Inches(3.75), Inches(5.0),
            Inches(1.1), size=13, italic=True, bold=True, color=RED)
footer(s, "ClaimGuard — Verifier Development")
set_notes(s, (
    "The verifier is a fine-tuned DeBERTa-v3-large, reduced to a binary entailment/"
    "contradiction head — 435 million parameters. Important detail: we didn't fabricate a "
    "neutral class. We investigated whether a legitimate 'not enough info' class could be "
    "built and found no scientifically valid source for it within this project's data, so "
    "the verifier stays strictly binary and the 'insufficient evidence' case is instead "
    "handled downstream by the decision policy's ABSTAIN outcome. On its own dev set — gold "
    "evidence paired with the right claim — it looks very strong: 96.2% accuracy, 95.85% "
    "macro F1. But I want to flag immediately: this is a GOLD-PREMISE evaluation. It tells "
    "you the verifier is good AT THE NLI TASK ITSELF. It does not tell you how the full "
    "pipeline performs once retrieval is uncertain — that's a completely different number, "
    "coming up later."
))

# =====================================================================
# SLIDE 9 — RETRIEVAL & RERANKING
# =====================================================================
s = new_slide()
add_title(s, "Evidence Retrieval Pipeline")
add_picture_fit(s, "slide9_retrieval_pipeline.png", Inches(0.9), Inches(1.35), Inches(11.5), Inches(1.15))
add_bar_chart(s, Inches(0.7), Inches(2.75), Inches(11.9), Inches(4.1),
              ["Recall@1", "Recall@5", "Recall@10", "Recall@20"],
              [("FAISS only", [28.00, 56.85, 66.20, 73.25]),
               ("FAISS + Reranker", [24.45, 59.05, 70.10, 73.25])],
              title="Recall@K — FAISS vs. FAISS + Reranker (n=2,000 held-out claims)",
              value_axis_title="Recall (%)")
footer(s, "ClaimGuard — Retrieval & Reranking  |  Reranking hurts Top-1 but improves Top-5/Top-10 — not universally better.")
set_notes(s, (
    "Retrieval: BGE-large embeddings into a FAISS flat index, top-20 candidates, then a BGE "
    "reranker narrows to the top 5. On a 2,000-claim held-out evaluation set, FAISS alone "
    "gets 28% recall at 1, climbing to 73.25% at 20. Adding the reranker actually LOWERS "
    "recall at 1 — 24.45% versus 28% — while improving recall at 5 and 10. This is a real, "
    "repeated finding across multiple steps of this project, not noise. So I want to be "
    "careful about the framing: the reranker is not 'better' or 'worse' in some universal "
    "sense — it trades off top-1 precision for slightly broader coverage in the top 5-10. "
    "Whether that trade-off matters downstream is exactly what the ablation study later "
    "tests directly."
))

# =====================================================================
# SLIDE 10 — INTEGRATED PIPELINE
# =====================================================================
s = new_slide()
add_title(s, "Retrieval + Reranking + Verification")
add_table(s, Inches(0.7), Inches(1.4), Inches(5.8), Inches(2.6),
          ["Integrated Metric", "Value"],
          [
              ["Verifier-selected evidence hit rate", "31.80%"],
              ["Verifier classification accuracy", "88.60%"],
              ["Entailment F1", "92.85%"],
              ["Contradiction F1", "71.92%"],
          ], font_size=14.5, header_size=14.5)
add_textbox(s, "NOT comparable to Slide 8's gold-premise 95.85% macro F1 — this includes "
               "retrieval/reranking error propagation by construction.",
            Inches(0.7), Inches(4.15), Inches(5.8), Inches(1.1), size=12.5, italic=True,
            bold=True, color=RED)
add_bar_chart(s, Inches(6.8), Inches(1.4), Inches(5.9), Inches(4.2),
              ["Gold SUPPORT\nevidence", "Gold REFUTE\nevidence", "Non-gold\nevidence"],
              [("Mean entailment confidence", [98.26, 8.07, 69.68])],
              title="Verifier confidence: gold vs. non-gold evidence (n=10,000 candidates)",
              value_axis_title="Mean entailment prob. (%)", legend=False)
add_textbox(s, "Verifier shows high confidence even on NON-GOLD evidence — this is the "
               "first sign of the discrimination problem explored in Slides 13 and 16.",
            Inches(0.7), Inches(5.7), Inches(12.0), Inches(0.9), size=14, bold=True, color=AMBER)
footer(s, "ClaimGuard — Integrated Pipeline")
set_notes(s, (
    "Once retrieval, reranking, and the verifier are chained together, the picture already "
    "starts to shift. The verifier's own SELECTED evidence — the one it scores highest — "
    "only actually matches gold evidence 31.8% of the time. Overall classification accuracy "
    "in this pipeline setting is 88.6%, but that's dominated by the entailment class — "
    "entailment F1 is 92.85%, contradiction F1 drops to 71.92%. And here's the key number: "
    "look at the confidence chart. Gold SUPPORT evidence gets 98.26% mean confidence — great. "
    "Gold REFUTE evidence correctly gets pushed down to 8.07%. But NON-gold evidence — "
    "evidence that's just topically similar but not actually the right supporting sentence — "
    "still gets 69.68% mean confidence. That's uncomfortably close to genuine support. This "
    "is the first concrete sign of the verifier discrimination problem that becomes central "
    "later in the talk."
))

# =====================================================================
# SLIDE 11 — DECISION POLICY & CORRECTION
# =====================================================================
s = new_slide()
add_title(s, "From Verification to Correction")
add_picture_fit(s, "slide11_decision_correction.png", Inches(2.0), Inches(1.3), Inches(9.3), Inches(6.0))
footer(s, "ClaimGuard — Decision Policy & Correction")
set_notes(s, (
    "The verifier's score feeds a deterministic decision policy that outputs ACCEPT, CORRECT, "
    "or ABSTAIN. If CORRECT, the correction loop regenerates the answer using the retrieved "
    "evidence and re-verifies — bounded to a small number of attempts so it can never run "
    "forever. Here's the important observation that sets up the rest of the talk: across the "
    "real RAGTruth and TruthfulQA evaluations, the attempt-0 ABSTAIN rate is exactly 0%. The "
    "policy never once says 'I'm not sure.' It always picks ACCEPT or CORRECT, even when — as "
    "we'll see — it's frequently wrong. That's the headline observation for this slide: the "
    "system is rarely uncertain, even when it should be."
))

# =====================================================================
# SLIDE 12 — RAGTRUTH RESULTS
# =====================================================================
s = new_slide()
add_title(s, "RAGTruth: End-to-End Evaluation")
add_bar_chart(s, Inches(0.55), Inches(1.35), Inches(6.7), Inches(4.0),
              ["Attempt 0", "Final", "Majority\nBaseline"],
              [("Accuracy", [55.9, 58.0, 65.1])],
              title="Detection accuracy vs. majority-class baseline (n=2,700)",
              value_axis_title="Accuracy (%)", legend=False)
add_bar_chart(s, Inches(7.35), Inches(1.35), Inches(5.6), Inches(4.0),
              ["Corrected\nsuccessfully", "Degraded by\ncorrection"],
              [("Responses (n=2,700)", [25, 923])],
              title="Correction outcomes (full RAGTruth test set)",
              value_axis_title="Count", legend=False, number_format="0")
add_textbox(s, 'ClaimGuard does NOT demonstrate reliable hallucination detection on RAGTruth. '
               'Majority baseline (65.1%) > Attempt 0 (55.9%) > Final (58.0%).',
            Inches(0.55), Inches(5.55), Inches(12.4), Inches(1.2), size=17, bold=True,
            color=RED, align=PP_ALIGN.CENTER)
footer(s, "ClaimGuard — RAGTruth Results  |  Macro F1 tracks accuracy closely at both attempt0 (0.559) and final (0.580).")
set_notes(s, (
    "This is the central result of the whole project, and I'm not going to soften it: on "
    "RAGTruth's 2,700-response held-out test set, ClaimGuard's attempt-0 accuracy is 55.9%, "
    "final accuracy after correction is 58.0% — and the trivial majority-class baseline, "
    "just always guessing 'not hallucinated', gets 65.1%. The full pipeline underperforms a "
    "baseline that does no work at all. And on the correction side: only 25 hallucinated "
    "responses were successfully fixed, while 923 originally-correct responses were degraded "
    "by the correction step. I want to say this plainly and not bury it: ClaimGuard does not "
    "demonstrate reliable hallucination detection on RAGTruth. The rest of the talk is about "
    "why, because that diagnosis is where the actual scientific value of this project is."
))

# =====================================================================
# SLIDE 13 — WHY RAGTRUTH FAILED
# =====================================================================
s = new_slide()
add_title(s, "RAGTruth Failure: Evidence-Domain Mismatch")
add_picture_fit(s, "slide13_domain_mismatch.png", Inches(0.5), Inches(1.3), Inches(12.3), Inches(6.0))
footer(s, "ClaimGuard — Failure Diagnosis")
set_notes(s, (
    "Here's the diagnosis. ClaimGuard's retrieval corpus is built from FEVER and Wikipedia. "
    "RAGTruth's actual source material is news articles, business listings, and QA passages "
    "— a genuinely different domain. So there's little real evidence overlap, which means "
    "retrieval returns evidence that's topically adjacent but not actually the right "
    "supporting text. The verifier then scores that irrelevant-but-topical evidence "
    "overconfidently, leading to incorrect decisions. Two diagnostic details matter here: "
    "zero of the 300 diagnostic responses had a MECHANICAL retrieval failure — retrieval "
    "never returned empty-handed. But we could NOT directly measure retrieval RELEVANCE "
    "failure, because RAGTruth doesn't have gold-relevance annotations the way FEVER does. "
    "That's an honest, disclosed gap — I did not invent a relevance metric to paper over it."
))

# =====================================================================
# SLIDE 14 — TRUTHFULQA RESULTS
# =====================================================================
s = new_slide()
add_title(s, "TruthfulQA Evaluation")
add_textbox(s, "790 questions across 37 categories", Inches(0.7), Inches(1.35), Inches(11.9),
            Inches(0.5), size=16, bold=True, color=TEAL)
add_table(s, Inches(0.7), Inches(2.0), Inches(5.6), Inches(1.9),
          ["Multiple-choice metric", "Score"],
          [["MC1", "34.05%"], ["MC2", "55.05%"], ["MC0", "44.18%"]], font_size=15, header_size=15)
add_table(s, Inches(6.7), Inches(2.0), Inches(5.9), Inches(2.2),
          ["Free-form decision", "Attempt 0", "Final"],
          [["ACCEPT", "80.0%", "95.3%"], ["CORRECT", "20.0%", "-"], ["ABSTAIN", "0.0%", "4.7%"]],
          font_size=14, header_size=14)
add_textbox(s, 'IMPORTANT: ACCEPT/CORRECT rates are NOT presented as truthfulness accuracy.',
            Inches(0.7), Inches(4.5), Inches(11.9), Inches(0.5), size=15, bold=True, color=RED)
add_textbox(s, '"No approved automatic free-form truthfulness judge was available; therefore '
               'free-form truthfulness was not artificially scored." This is a strength of '
               'the methodology, not something to hide.',
            Inches(0.7), Inches(5.1), Inches(11.9), Inches(1.3), size=15, italic=True, color=DARK)
footer(s, "ClaimGuard — TruthfulQA")
set_notes(s, (
    "TruthfulQA: 790 questions, 37 categories, evaluation-only. On the multiple-choice "
    "metrics — which measure Qwen3's own answer-likelihood calibration, independent of the "
    "pipeline — MC1 is 34.05%, MC2 is 55.05%, MC0 is 44.18%. On the free-form side, where the "
    "full pipeline runs, attempt-0 accepts 80% of answers outright and the final accept rate "
    "climbs to 95.3%. Now, the important thing I want to be very explicit about: I am NOT "
    "presenting those ACCEPT rates as a truthfulness score. There is no approved automatic "
    "judge for scoring free-form generated text in this project — no human raters, no "
    "GPT-judge access — and I was explicit that I would not invent a semantic-similarity "
    "heuristic to fill that gap. So this is a disclosed limitation, not a hidden one, and I'd "
    "actually call that methodological honesty a strength rather than a weakness of the "
    "project."
))

# =====================================================================
# SLIDE 15 — CONTROLLED ABLATIONS
# =====================================================================
s = new_slide()
add_title(s, "What Actually Causes the Failure?")
add_bar_chart(s, Inches(0.55), Inches(1.35), Inches(12.3), Inches(4.15),
              ["A: Reranked\n(n=300)", "B: No Reranker\n(n=300)", "C: FAISS Top-1\n(n=300)",
               "D: No Correction\n(n=2,700)", "E: Verify-only\n(n=2,700)"],
              [("Accuracy", [58.67, 58.67, 51.00, 55.89, 48.74]),
               ("Macro F1", [58.13, 58.13, 48.74, 55.89, 48.11])],
              title="Controlled ablation comparison", value_axis_title="%")
add_bullets(s, [
    "Reranker changes WHICH evidence is selected (100% of responses) but does not change "
    "downstream metrics (A vs. B identical to 4 decimals).",
    "FAISS Top-1 only is measurably worse — candidate POOL SIZE matters more than ordering.",
    "Correction contributes some net signal (D vs. E) but is not the root problem.",
], Inches(0.55), Inches(5.65), Inches(12.3), Inches(1.6), font_size=14, space_after=6)
footer(s, "ClaimGuard — Controlled Ablations")
set_notes(s, (
    "So which component actually matters? I ran controlled ablations to find out. Condition "
    "A is the full pipeline with reranking, on a 300-response subset — 58.67% accuracy. "
    "Condition B removes the reranker entirely — 58.67% accuracy, identical to four decimal "
    "places on macro F1 too. So the reranker changes WHICH evidence gets selected for every "
    "single response, but it changes NOTHING about the final classification outcome. "
    "Condition C uses only the single top FAISS candidate, no top-5 — and that's measurably "
    "worse, 51% accuracy. So candidate POOL SIZE matters, ordering doesn't. Conditions D and "
    "E reuse the full 2,700-response data: D is no-correction-at-all, E is verify-only. "
    "Verify-only is worse than the full pipeline, meaning correction does contribute some net "
    "signal — but as we'll see next, it's not fixing the root cause."
))

# =====================================================================
# SLIDE 16 — FAILURE ANALYSIS
# =====================================================================
s = new_slide()
add_title(s, "Evidence-Supported Failure Model")
add_picture_fit(s, "slide16_failure_model.png", Inches(0.6), Inches(1.3), Inches(7.6), Inches(6.0),
                center_h=False)
add_bullets(s, [
    "Verifier confidently wrong: 38.3% of diagnostic subset.",
    "Harmful correction: 33.7%.",
    "Retrieval mechanical failure: 0%.",
    "Reranking: not a meaningful bottleneck.",
], Inches(8.5), Inches(2.2), Inches(4.3), Inches(3.0), font_size=15, space_after=14)
footer(s, "ClaimGuard — Failure Analysis")
set_notes(s, (
    "This is the synthesis diagram. Domain-mismatched corpus leads to weak or irrelevant "
    "retrieved evidence, which the verifier then scores overconfidently. Combined with "
    "near-zero abstention, that produces wrong decisions, and wrong decisions feed a "
    "correction loop that sometimes does more harm than good. The numbers on the right back "
    "this up: 38.3% of the diagnostic subset was a case where the verifier was confidently — "
    "meaning above 90% confidence — WRONG. 33.7% of corrections were harmful. Retrieval's "
    "MECHANICAL failure rate is 0% — it always returns something. And reranking, as we just "
    "saw, is not a meaningful bottleneck. I want to be careful about how I label this diagram: "
    "it's an evidence-supported failure model, built from measurements across several steps of "
    "this project. It is explicitly NOT a proven causal DAG — I have correlational, "
    "controlled-ablation evidence for each link, not a formal causal identification."
))

# =====================================================================
# SLIDE 17 — KEY FINDINGS & CONCLUSION
# =====================================================================
s = new_slide()
add_title(s, "Key Findings")
add_bullets(s, [
    "ClaimGuard did not outperform the RAGTruth majority baseline.",
    "FEVER/Wikipedia evidence is poorly matched to RAGTruth's actual source domains.",
    "The verifier is substantially overconfident on irrelevant evidence.",
    "The decision policy produces essentially no abstention.",
    "Correction harm is largely downstream of incorrect verification decisions, not an "
    "independently broken correction mechanism.",
], Inches(0.7), Inches(1.5), Inches(11.9), Inches(3.3), font_size=17, space_after=14)
box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(5.0), Inches(11.9), Inches(1.8))
box.fill.solid(); box.fill.fore_color.rgb = NAVY_FADE; box.line.color.rgb = NAVY; box.line.width = Pt(1.2)
tf = box.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.25); tf.margin_top = Inches(0.15)
p = tf.paragraphs[0]; r = p.add_run(); r.text = "FINAL CONCLUSION"
r.font.bold = True; r.font.size = Pt(14); r.font.color.rgb = NAVY
p2 = tf.add_paragraph()
r2 = p2.add_run()
r2.text = ('"ClaimGuard demonstrates the importance of evidence relevance and calibrated '
           'uncertainty in end-to-end hallucination detection."')
r2.font.italic = True; r2.font.size = Pt(16); r2.font.color.rgb = DARK
footer(s, "ClaimGuard — Key Findings  |  The current implementation does not solve hallucination detection.")
set_notes(s, (
    "Five findings to take away. One, ClaimGuard did not beat the majority baseline on its "
    "primary benchmark — that's the headline negative result. Two, the FEVER/Wikipedia "
    "corpus is a poor match for RAGTruth's actual domains. Three, the verifier is "
    "substantially overconfident specifically on irrelevant evidence. Four, the decision "
    "policy essentially never abstains. Five, correction-caused harm is a downstream "
    "consequence of the earlier failures, not an independent bug in the correction logic "
    "itself. Putting that together, the conclusion I'd draw is that this project demonstrates "
    "the importance of evidence relevance and calibrated uncertainty for this kind of "
    "pipeline — not that the current implementation has solved hallucination detection. I'm "
    "not claiming that, and I don't want to leave that impression."
))

# =====================================================================
# SLIDE 18 — LIMITATIONS & FUTURE WORK
# =====================================================================
s = new_slide()
add_title(s, "Limitations and Future Research")
add_textbox(s, "LIMITATIONS", Inches(0.7), Inches(1.35), Inches(5.8), Inches(0.4), size=15,
            bold=True, color=RED)
add_bullets(s, [
    "Domain-mismatched retrieval corpus.",
    "Verifier overconfidence on out-of-domain evidence.",
    "Near-zero-abstain decision policy.",
    "No approved free-form TruthfulQA judge.",
    "RAGTruth retrieval relevance not directly measurable.",
    "TruthfulQA MC may reflect benchmark memorization.",
    "Limited ablation subset for A/B/C (300/2,700 responses).",
    "Single primary verifier training run.",
], Inches(0.7), Inches(1.8), Inches(5.8), Inches(4.9), font_size=13.5, space_after=6)
add_textbox(s, "FUTURE WORK (not implemented / not solved)", Inches(6.8), Inches(1.35),
            Inches(5.8), Inches(0.4), size=15, bold=True, color=TEAL)
add_bullets(s, [
    "Domain-matched evidence retrieval.",
    "Broader evidence corpus.",
    "Verifier calibration for out-of-domain evidence.",
    "Explicit uncertainty / abstention mechanism.",
    "Evidence-quality scoring.",
    "Improved evidence aggregation.",
    "Correction gating (only correct on reliable signals).",
    "Independent free-form truthfulness evaluation.",
], Inches(6.8), Inches(1.8), Inches(5.8), Inches(4.9), font_size=13.5, space_after=6)
footer(s, "ClaimGuard — Limitations and Future Work")
set_notes(s, (
    "None of these limitations are hidden — they're listed explicitly. The biggest one, "
    "which I keep coming back to, is the domain-mismatched retrieval corpus. On the future "
    "work side, I want to be careful: none of these eight directions have been attempted or "
    "solved in this project. They're the concrete, evidence-based next steps that fall out of "
    "the failure diagnosis — things like domain-matched retrieval, verifier calibration "
    "specifically for out-of-domain evidence, an actual abstention mechanism that fires when "
    "it should, and gating correction so it only triggers on reliable signals. This is a "
    "roadmap, not a claim of partial completion."
))

# =====================================================================
# SLIDE 19 — REPRODUCIBILITY / FINAL STATUS
# =====================================================================
s = new_slide()
add_title(s, "Reproducibility & Project Status")
add_bullets(s, [
    "Deterministic inference — re-checked after every major evaluation run.",
    "Dataset isolation, contamination checks, and leakage guards enforced structurally.",
    "411+ tests passing through Step 21 (427 after Step 22's final documentation checks).",
    "Frozen evaluation configurations — no evaluation-set tuning.",
    "No model-weight changes during evaluation or documentation stages.",
    "Complete experiment artifacts preserved (manifests, results, reproducibility metadata).",
    "GitHub-ready source repository.",
], Inches(0.7), Inches(1.5), Inches(11.9), Inches(4.3), font_size=16.5, space_after=12)
box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(5.9), Inches(11.9), Inches(1.0))
box.fill.solid(); box.fill.fore_color.rgb = NAVY_FADE; box.line.color.rgb = NAVY; box.line.width = Pt(1.2)
tf = box.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.25)
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = '"Experimental cycle complete; next work is architectural improvement, not retrospective tuning."'
r.font.italic = True; r.font.bold = True; r.font.size = Pt(16); r.font.color.rgb = NAVY
footer(s, "ClaimGuard — Reproducibility & Status")
set_notes(s, (
    "On reproducibility: inference is deterministic and that was re-checked after every "
    "major run, not assumed once and forgotten. Dataset isolation and contamination checks "
    "are enforced structurally, as I showed earlier. There are over 400 tests passing "
    "throughout the project. Evaluation configurations are frozen — nothing was tuned against "
    "the RAGTruth or TruthfulQA results themselves. No model weights were changed once "
    "evaluation started. And the full set of experiment artifacts is preserved, so every "
    "number in this talk traces back to a saved file, not something recomputed on the fly. "
    "The project's status right now is that the experimental cycle is complete — the next "
    "phase is architectural redesign based on the failure diagnosis, not more tuning of the "
    "current system."
))

# =====================================================================
# SLIDE 20 — CLOSING
# =====================================================================
s = new_slide()
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
bg.fill.solid(); bg.fill.fore_color.rgb = WHITE; bg.line.fill.background(); bg.shadow.inherit = False
add_textbox(s, "ClaimGuard: From Building a Detector to\nUnderstanding Why It Fails",
            Inches(1.0), Inches(1.2), Inches(11.3), Inches(1.4), size=30, bold=True, color=NAVY,
            align=PP_ALIGN.CENTER)
add_bullets(s, [
    "Evidence relevance is fundamental.",
    "Confidence must reflect evidence quality.",
    "Correction should be gated by reliable uncertainty.",
], Inches(2.5), Inches(3.1), Inches(8.3), Inches(2.0), font_size=19, space_after=16)
add_textbox(s, "Next step: redesign the evidence + uncertainty layer based on the "
               "experimentally identified failure modes.", Inches(1.5), Inches(5.4),
            Inches(10.3), Inches(0.9), size=16, italic=True, color=TEAL, align=PP_ALIGN.CENTER)
add_textbox(s, "(FUTURE WORK — not yet started)", Inches(1.5), Inches(6.05), Inches(10.3),
            Inches(0.5), size=12, bold=True, color=RED, align=PP_ALIGN.CENTER)
set_notes(s, (
    "To close: this project went from building a hallucination detector to understanding, in "
    "a rigorous and evidence-backed way, why it fails. Three things I'd want you to remember: "
    "evidence relevance is fundamental — you can't verify against evidence that isn't "
    "actually about the claim. Confidence has to reflect evidence quality, not just pattern-"
    "match to something that looks similar. And correction should be gated by reliable "
    "uncertainty, not triggered every time the policy has any doubt at all. The next step is "
    "to redesign the evidence and uncertainty layer based on exactly what this failure "
    "analysis identified — and I want to be clear that's future work, not something already "
    "started. Happy to take questions."
))

# =====================================================================
# APPENDIX — ANTICIPATED GUIDE QUESTIONS (5 slides, 4 Q&A each)
# =====================================================================
QA = [
    ("Why did you choose DeBERTa?",
     "It's already trained on FEVER-NLI + MNLI + ANLI + LingNLI + WANLI (885K pairs), MIT "
     "licensed, and its 3-way labels map directly onto SUPPORTS/REFUTES/NEI. Small footprint "
     "(<2GB VRAM) and a strong pretrained entailment representation we reuse via encoder/"
     "pooler transplant rather than training from scratch."),
    ("Why Qwen3-8B?",
     "Apache 2.0 license, strong instruction-following, and ~16GB BF16 weights — the "
     "smallest footprint among comparable 8B-class candidates, leaving headroom to run the "
     "whole pipeline (embedder + reranker + verifier + generator) concurrently on the shared GPU."),
    ("Why BGE retrieval?",
     "The task is English-only, so we didn't need bge-m3's multilingual/sparse machinery. "
     "bge-large-en-v1.5 is small (<1.5GB), integrates natively with sentence-transformers, "
     "and has established retrieval quality."),
    ("Why use FEVER for verifier development?",
     "It's a large, well-established claim-verification dataset (145K+ raw claims) with a "
     "SUPPORTS/REFUTES/NEI structure that maps directly onto the entailment/contradiction "
     "verification task, with a clear license (CC BY-SA 3.0 + GNU FDL)."),

    ("Why is the verifier binary rather than 3-class?",
     "Step 11 investigated a neutral-class strategy and found no legitimate, non-fabricated "
     "source of neutral examples within this project's constraints. Rather than invent "
     "synthetic neutral premises, we kept the verifier strictly binary and let the decision "
     "policy's ABSTAIN outcome handle 'insufficiently supported' downstream instead."),
    ("Why was RAGTruth kept completely held out?",
     "It's the project's cleanest end-to-end effectiveness benchmark — FEVER and TruthfulQA "
     "are old, ubiquitous benchmarks likely present in generator pretraining data, so RAGTruth "
     "gives the least contaminated signal. This is enforced by RoleViolationError guards plus "
     "AST and behavioral tests, not just convention."),
    ("Why is RAGTruth performance below the majority baseline?",
     "Controlled ablations trace it to a retrieval-corpus domain mismatch (FEVER/Wikipedia vs. "
     "RAGTruth's news/business/QA text) compounded by a verifier discrimination failure on "
     "out-of-domain evidence and a decision policy that essentially never abstains — a "
     "compounding, not single-component, failure."),
    ("Why not simply add RAGTruth documents to the retrieval corpus?",
     "That's identified future work, not attempted here — it would directly test the domain-"
     "mismatch hypothesis, but needs careful protocol design so it tests generalization rather "
     "than becoming a leakage shortcut. Explicitly out of scope for the steps completed so far."),

    ("Why did the reranker not help?",
     "The controlled ablation shows the reranker changes WHICH evidence is selected for 100% "
     "of responses, but produces no measurable change in downstream classification metrics — "
     "reordering already-irrelevant candidates (due to corpus mismatch) can't make them relevant."),
    ("Why does the verifier become overconfident?",
     "It's well-calibrated ONLY on FEVER gold evidence. On out-of-domain evidence that's "
     "topically/lexically similar to a claim, it scores that evidence with the same near-1.0 "
     "confidence as genuine gold evidence — a discrimination failure, not a raw calibration "
     "failure, since there's no non-gold ground truth to penalize this during training."),
    ("Why is there zero abstention?",
     "The decision policy's fixed thresholds essentially never produce a 'neither confident' "
     "outcome, because the verifier's own confidence distribution is extreme and bimodal — it "
     "rarely outputs a genuinely ambiguous, mid-range score in practice."),
    ("Why does correction sometimes make answers worse?",
     "Correction triggers whenever the decision policy says CORRECT. Many of those triggers "
     "are false positives — confidently-wrong flags on already-correct answers — so the "
     "correction loop faithfully 'fixes' answers that didn't need fixing. This is a downstream "
     "consequence of upstream miscalibration, not an independent defect in the correction logic."),

    ("Why was TruthfulQA not given a free-form truthfulness score?",
     "No approved automatic judge exists in this project (no human raters, no GPT-judge "
     "access), and inventing a semantic-similarity heuristic was explicitly out of scope. This "
     "is a disclosed protocol gap, not silently filled with a fabricated metric."),
    ("What is the main contribution if the final system does not outperform the baseline?",
     "A rigorous, contamination-checked, gold-isolated end-to-end integration and evaluation "
     "that isolates WHY this architecture fails out-of-domain, via controlled ablation — a "
     "scientifically useful negative/diagnostic result and a concrete roadmap, not a working "
     "detector."),
    ("What would you change in the next version?",
     "Rebuild/augment the retrieval corpus with in-domain sources; test whether verifier "
     "overconfidence persists when retrieval is held at known-good quality; move to a learned/"
     "calibrated decision policy for out-of-domain evidence; add an approved free-form "
     "truthfulness judge; add per-evidence relevance annotation."),
    ("Is the problem the verifier or the retrieval system?",
     "Both, interacting. Ablations show identical overconfidence regardless of evidence-"
     "selection method, so the failure is upstream of any single retrieval choice — but the "
     "corpus's domain mismatch is what supplies the irrelevant-but-similar evidence the "
     "verifier then mis-scores. It's a corpus + verifier interaction, not one broken component."),

    ("How do you know there was no data leakage?",
     "An exact-match, field-aware contamination sweep found and removed 200 HaluEval pool "
     "records overlapping RAGTruth test sources. RoleViolationError guards confirm RAGTruth "
     "test and TruthfulQA can never enter the training pool, and gold-label isolation is "
     "enforced via AST import-exclusion plus mock-based behavioral tests."),
    ("What is the strongest result of the project?",
     "The verifier's in-domain quality: 96.20% accuracy / 0.9585 macro F1 on a 16,890-example "
     "dev set, with clean calibration (ECE=0.0092) on gold evidence — plus the methodological "
     "rigor (contamination control, gold isolation, cross-validated determinism) that makes "
     "the failure diagnosis itself trustworthy."),
    ("What is the biggest limitation?",
     "The retrieval corpus (FEVER/Wikipedia) was never rebuilt to match RAGTruth's or "
     "TruthfulQA's actual source domains — the single largest, identified, and addressable "
     "limitation."),
    ("What would be the next experiment?",
     "Re-analyze Step 15/16's existing FEVER-domain artifacts to test whether verifier "
     "overconfidence persists even when retrieval quality is held at a known-good level — "
     "answerable from already-existing data, no new collection needed."),
]

assert len(QA) == 20

for page in range(5):
    s = new_slide()
    add_title(s, f"Appendix — Anticipated Guide Questions ({page + 1}/5)",
              "Answers use only established project evidence — no invented claims.")
    top = Inches(1.55)
    for i in range(4):
        q, a = QA[page * 4 + i]
        qbox = s.shapes.add_textbox(Inches(0.6), top, Inches(12.1), Inches(0.4))
        qtf = qbox.text_frame; qtf.word_wrap = True
        qp = qtf.paragraphs[0]
        qr = qp.add_run(); qr.text = f"Q{page * 4 + i + 1}.  {q}"
        qr.font.bold = True; qr.font.size = Pt(15); qr.font.color.rgb = NAVY
        abox = s.shapes.add_textbox(Inches(0.9), top + Inches(0.38), Inches(11.8), Inches(0.75))
        atf = abox.text_frame; atf.word_wrap = True
        ap = atf.paragraphs[0]
        ar = ap.add_run(); ar.text = a
        ar.font.size = Pt(12.5); ar.font.color.rgb = DARK
        top += Inches(1.2)
    footer(s, "ClaimGuard — Appendix (Q&A)  |  Not part of the 15-20 minute core presentation.")
    set_notes(s, "Reference slide — use only if the guide asks one of these questions. Not part "
                 "of the timed core presentation.")

prs.save(OUT_PATH)
print(f"Saved presentation with {len(prs.slides._sldIdLst)} slides to {OUT_PATH}")
