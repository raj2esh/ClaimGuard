"""Extract slide titles + speaker notes from the built PPTX into a Markdown
file, so the two deliverables can never drift out of sync with each other."""
from pathlib import Path
from pptx import Presentation

HERE = Path(__file__).resolve().parent
PPTX = HERE / "ClaimGuard_Guide_Presentation.pptx"
OUT = HERE / "ClaimGuard_Guide_Presentation_Notes.md"

prs = Presentation(PPTX)

lines = ["# ClaimGuard Guide Presentation — Speaker Notes\n",
         "Generated directly from the notes embedded in "
         "`ClaimGuard_Guide_Presentation.pptx` — the two files cannot drift "
         "out of sync since this file is derived from the .pptx, not "
         "authored separately.\n"]

for i, slide in enumerate(prs.slides, start=1):
    title = None
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.strip():
            title = shape.text_frame.text.strip().splitlines()[0]
            break
    notes = ""
    if slide.has_notes_slide:
        notes = slide.notes_slide.notes_text_frame.text.strip()
    lines.append(f"## Slide {i}\n")
    lines.append(f"**Title:** {title or '(no title text found)'}\n")
    lines.append("**Speaker notes:**\n")
    lines.append(f"{notes if notes else '_(no notes — reference/appendix slide)_'}\n")
    lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT} ({len(prs.slides)} slides)")
