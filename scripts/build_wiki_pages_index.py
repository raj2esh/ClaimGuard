"""Step 10: build a deterministic FEVER wiki_pages resolution index.

Reads the raw, unmodified extracted wiki-pages JSONL files (109 real files,
after filtering the macOS AppleDouble artifacts confirmed in
scripts/acquire_inspect_wiki_pages.py) and builds a SQLite index mapping
(page_id, sentence_id) -> sentence text, so FEVER evidence references
(wiki_url + sentence_id) can be resolved deterministically - no fuzzy
matching. Also stores each page's full `text` field (the page's
introductory text, concatenated across all its sentences) - this is
REQUIRED, not optional: FEVER evidence references can carry
sentence_id == -1, a real annotation sentinel confirmed empirically (Step
10) to mean "the evidence is the page as a whole / not tied to one
specific indexed sentence" (100% of ~89k such references checked point at
pages that DO exist with real content - this is not a missing-data case).
Resolving sentence_id == -1 uses this stored page `text`, not a sentence
lookup (sentence indices never go negative, so a naive lookup would always
fail and misclassify a real annotation convention as "missing data").

Preserves exact page identity and sentence numbering as found in the real
data (0-based, per Step 10 inspection). Malformed line entries (~507 out of
42M, confirmed non-numeric-first-field artifacts) are skipped with a count,
not silently dropped without record.

CPU / disk only - no GPU, no model loading.

Usage:
    python scripts/build_wiki_pages_index.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

EXTRACT_DIR = cg_config.resolve_path("data/raw/fever/wiki_pages/extracted")
INDEX_PATH = cg_config.resolve_path("data/processed/fever/wiki_pages_index.sqlite")

BATCH_SIZE = 200_000


def real_files() -> list[Path]:
    all_jsonl = sorted(EXTRACT_DIR.rglob("*.jsonl"))
    return sorted(f for f in all_jsonl if not f.name.startswith("._"))


def main() -> int:
    files = real_files()
    print(f"Indexing {len(files)} real wiki-pages files -> {INDEX_PATH}")
    if not files:
        print("No real wiki-pages files found - run scripts/acquire_inspect_wiki_pages.py first.")
        return 1

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    conn = sqlite3.connect(str(INDEX_PATH))
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("""
        CREATE TABLE sentences (
            page_id TEXT NOT NULL,
            sentence_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            PRIMARY KEY (page_id, sentence_id)
        )
    """)
    conn.execute("""
        CREATE TABLE pages (
            page_id TEXT PRIMARY KEY,
            has_content INTEGER NOT NULL,
            text TEXT NOT NULL
        )
    """)

    start = time.time()
    total_pages = 0
    total_sentences = 0
    malformed_skipped = 0
    empty_id_skipped = 0
    duplicate_sentence_keys = 0
    buffer: list[tuple[str, int, str]] = []
    page_buffer: list[tuple[str, int, str]] = []

    def flush():
        nonlocal buffer, page_buffer, duplicate_sentence_keys
        if buffer:
            try:
                conn.executemany(
                    "INSERT INTO sentences (page_id, sentence_id, text) VALUES (?, ?, ?)", buffer
                )
            except sqlite3.IntegrityError:
                # Extremely rare (0 duplicate page_ids found in Step 10
                # inspection) - fall back to one-by-one insert so a single
                # collision doesn't drop the whole batch, and count them.
                for row in buffer:
                    try:
                        conn.execute(
                            "INSERT INTO sentences (page_id, sentence_id, text) VALUES (?, ?, ?)", row
                        )
                    except sqlite3.IntegrityError:
                        duplicate_sentence_keys += 1
            buffer = []
        if page_buffer:
            conn.executemany(
                "INSERT OR IGNORE INTO pages (page_id, has_content, text) VALUES (?, ?, ?)", page_buffer
            )
            page_buffer = []

    for fi, path in enumerate(files):
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                rec = json.loads(raw_line)
                page_id = rec.get("id", "")
                if not page_id:
                    empty_id_skipped += 1
                    continue
                total_pages += 1
                lines_field = rec.get("lines", "")
                page_text = rec.get("text", "") or ""
                has_content = 1 if lines_field else 0
                page_buffer.append((page_id, has_content, page_text))

                if not lines_field:
                    continue
                for line_entry in lines_field.split("\n"):
                    if not line_entry:
                        continue
                    parts = line_entry.split("\t")
                    if len(parts) < 2:
                        malformed_skipped += 1
                        continue
                    try:
                        sent_id = int(parts[0])
                    except ValueError:
                        malformed_skipped += 1
                        continue
                    sent_text = parts[1]
                    buffer.append((page_id, sent_id, sent_text))
                    total_sentences += 1

                if len(buffer) >= BATCH_SIZE:
                    flush()

        print(f"  ...{fi + 1}/{len(files)} files | {total_pages:,} pages | "
              f"{total_sentences:,} sentences | {time.time() - start:.0f}s elapsed")

    flush()
    conn.commit()

    print("Creating index on sentences(page_id, sentence_id) [already the PRIMARY KEY, "
          "adding an explicit covering index is unnecessary]...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_page ON pages(page_id)")
    conn.commit()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s.")
    print(f"Total pages indexed: {total_pages:,}")
    print(f"Total sentences indexed: {total_sentences:,}")
    print(f"Empty page id (skipped): {empty_id_skipped}")
    print(f"Malformed line entries (skipped, non-numeric sentence id): {malformed_skipped:,}")
    print(f"Duplicate (page_id, sentence_id) collisions: {duplicate_sentence_keys}")
    print(f"Index file size: {INDEX_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
