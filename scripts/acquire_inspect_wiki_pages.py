"""Step 10: FEVER wiki_pages acquisition and format inspection.

Downloads the official FEVER Wikipedia evidence corpus (wiki-pages.zip) from
the FEVER project's own site (fever.ai), preserves the original archive
untouched, extracts it (extraction is not a modification - the extracted
JSONL files are byte-identical to what's inside the zip), and inspects the
REAL format empirically - field names, sentence-line format, sentence-ID
indexing convention, page/sentence counts, duplicates, and encoding issues.
Nothing about the internal structure is assumed from folklore/documentation.

CPU / disk only - no GPU, no model loading.

Usage:
    python scripts/acquire_inspect_wiki_pages.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

RAW_DIR = cg_config.resolve_path("data/raw/fever/wiki_pages")
ZIP_PATH = RAW_DIR / "wiki-pages.zip"
EXTRACT_DIR = RAW_DIR / "extracted"

PRIMARY_URL = "https://fever.ai/download/fever/wiki-pages.zip"
FALLBACK_URL = "https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip"


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "claimguard-dataset-acquisition"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = resp.headers.get("Content-Length")
        print(f"  Content-Length: {total}")
        with dest.open("wb") as f:
            chunk_size = 1024 * 1024
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (50 * 1024 * 1024) < chunk_size:
                    print(f"  ...{downloaded / 1024 / 1024:.0f} MB downloaded")
    print(f"  Done: {dest.stat().st_size / 1024 / 1024:.1f} MB")


def acquire() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        print(f"[acquire] {ZIP_PATH} already present ({ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MB), "
              f"not re-downloading")
        return ZIP_PATH

    print(f"[acquire] downloading from primary source: {PRIMARY_URL}")
    try:
        _download(PRIMARY_URL, ZIP_PATH)
        print(f"[acquire] source used: {PRIMARY_URL} (official fever.ai site)")
        return ZIP_PATH
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"[acquire] primary source failed: {exc!r}")
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()

    print(f"[acquire] falling back to: {FALLBACK_URL} (the FEVER project's own S3 bucket, "
          f"fever.public - not a third-party mirror)")
    _download(FALLBACK_URL, ZIP_PATH)
    print(f"[acquire] source used: {FALLBACK_URL}")
    return ZIP_PATH


def extract(zip_path: Path) -> list[Path]:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        all_jsonl = sorted(EXTRACT_DIR.rglob("*.jsonl"))  # recursive - files may be nested in a subdir
        files = [f for f in all_jsonl if not f.name.startswith("._")]
        print(f"[extract] already extracted: {len(all_jsonl)} .jsonl-named entries present "
              f"({len(all_jsonl) - len(files)} filtered as macOS AppleDouble artifacts, "
              f"{len(files)} real data files), skipping re-extraction")
        return sorted(files)

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[extract] extracting {zip_path} -> {EXTRACT_DIR} (byte-identical to zip contents, not a modification)")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        print(f"[extract] {len(names)} entries in the archive")
        zf.extractall(EXTRACT_DIR)
    all_jsonl = sorted(EXTRACT_DIR.rglob("*.jsonl"))
    # The archive was packaged on macOS, which leaves AppleDouble
    # resource-fork sidecar files (binary, not JSON) named "._<original>"
    # alongside every real file - confirmed by inspecting the actual
    # filenames and the UnicodeDecodeError one produced, not assumed. These
    # are an artifact of how the zip was packaged, not part of FEVER's
    # actual data - filtered out here, not silently ignored.
    apple_double = [f for f in all_jsonl if f.name.startswith("._")]
    files = [f for f in all_jsonl if not f.name.startswith("._")]
    print(f"[extract] {len(all_jsonl)} .jsonl-named entries extracted; "
          f"{len(apple_double)} are macOS AppleDouble resource-fork artifacts (filtered out); "
          f"{len(files)} are real wiki-pages data files")
    return sorted(files)


def inspect_format(files: list[Path]) -> dict:
    _section("FORMAT INSPECTION (real data, nothing assumed)")
    print(f"Total files: {len(files)}")
    print(f"Sample filenames: {[f.name for f in files[:5]]}")

    # Inspect the first file's first few records raw.
    first_file = files[0]
    sample_records = []
    with first_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            sample_records.append(json.loads(line))

    print(f"\nField names in sample records: {[sorted(r.keys()) for r in sample_records]}")
    for r in sample_records[:2]:
        compact = {k: (v[:300] + "..." if isinstance(v, str) and len(v) > 300 else v) for k, v in r.items()}
        print(f"\nSample record: {compact}")

    # Inspect the raw 'lines' field format precisely - do not assume TSV/newline convention.
    if sample_records and "lines" in sample_records[0]:
        raw_lines = sample_records[0]["lines"]
        print(f"\nRaw 'lines' field (first 500 chars, repr to show exact whitespace/delimiters): {raw_lines[:500]!r}")

    return {"n_files": len(files), "sample_keys": sorted(sample_records[0].keys()) if sample_records else []}


def full_scan(files: list[Path]) -> dict:
    _section("FULL SCAN (all files, all pages) - counts, duplicates, empty fields")
    total_pages = 0
    total_sentences = 0
    empty_id_pages = 0
    empty_text_pages = 0
    empty_lines_pages = 0
    id_counter: Counter = Counter()
    sentence_id_min = None
    sentence_id_max = None
    non_sequential_sentence_ids = 0
    empty_sentence_texts = 0
    malformed_line_entries = 0
    sample_malformed = []

    for fi, path in enumerate(files):
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    rec = json.loads(raw_line)
                except json.JSONDecodeError:
                    malformed_line_entries += 1
                    continue
                total_pages += 1
                page_id = rec.get("id", "")
                id_counter[page_id] += 1
                if not page_id:
                    empty_id_pages += 1
                if not rec.get("text"):
                    empty_text_pages += 1
                lines_field = rec.get("lines", "")
                if not lines_field:
                    empty_lines_pages += 1
                    continue

                expected_next = 0
                for line_entry in lines_field.split("\n"):
                    if not line_entry:
                        continue
                    parts = line_entry.split("\t")
                    if len(parts) < 2:
                        malformed_line_entries += 1
                        if len(sample_malformed) < 5:
                            sample_malformed.append({"page_id": page_id, "raw_entry": line_entry[:200]})
                        continue
                    try:
                        sent_id = int(parts[0])
                    except ValueError:
                        malformed_line_entries += 1
                        if len(sample_malformed) < 5:
                            sample_malformed.append({"page_id": page_id, "raw_entry": line_entry[:200]})
                        continue
                    sent_text = parts[1]
                    total_sentences += 1
                    if sentence_id_min is None or sent_id < sentence_id_min:
                        sentence_id_min = sent_id
                    if sentence_id_max is None or sent_id > sentence_id_max:
                        sentence_id_max = sent_id
                    if sent_id != expected_next:
                        non_sequential_sentence_ids += 1
                    expected_next = sent_id + 1
                    if not sent_text.strip():
                        empty_sentence_texts += 1
        if (fi + 1) % 20 == 0:
            print(f"  ...scanned {fi + 1}/{len(files)} files, {total_pages:,} pages so far")

    duplicate_ids = {k: v for k, v in id_counter.items() if v > 1}

    print(f"\nTotal pages: {total_pages:,}")
    print(f"Total sentence lines: {total_sentences:,}")
    print(f"Empty page id: {empty_id_pages}")
    print(f"Empty text field: {empty_text_pages:,}")
    print(f"Empty lines field (no sentences at all): {empty_lines_pages:,}")
    print(f"Duplicate page ids: {len(duplicate_ids)} (showing up to 5): {dict(list(duplicate_ids.items())[:5])}")
    print(f"Sentence ID range observed: min={sentence_id_min}, max={sentence_id_max}")
    print(f"Sentence entries where sentence_id != expected sequential position: {non_sequential_sentence_ids:,}")
    print(f"Empty sentence text (sentence_id present, text blank): {empty_sentence_texts:,}")
    print(f"Malformed line entries (couldn't parse id\\ttext): {malformed_line_entries:,}")
    if sample_malformed:
        print(f"Sample malformed entries: {sample_malformed}")

    return {
        "total_pages": total_pages,
        "total_sentences": total_sentences,
        "empty_id_pages": empty_id_pages,
        "empty_text_pages": empty_text_pages,
        "empty_lines_pages": empty_lines_pages,
        "duplicate_page_ids": len(duplicate_ids),
        "sample_duplicate_ids": dict(list(duplicate_ids.items())[:20]),
        "sentence_id_min": sentence_id_min,
        "sentence_id_max": sentence_id_max,
        "non_sequential_sentence_ids": non_sequential_sentence_ids,
        "empty_sentence_texts": empty_sentence_texts,
        "malformed_line_entries": malformed_line_entries,
        "sample_malformed": sample_malformed,
    }


def main() -> int:
    _section("WIKI_PAGES ACQUISITION")
    print(f"Source: fever.ai's own site (primary) / fever.public S3 bucket (fallback) - "
          f"the FEVER project's own hosting, not a third-party mirror.")
    print(f"License: Wikipedia Copyright Policy / CC BY-SA 3.0 fallback (same as FEVER claims data, Step 6A).")
    print(f"Raw storage: {RAW_DIR}")

    zip_path = acquire()
    files = extract(zip_path)
    format_info = inspect_format(files)
    scan_info = full_scan(files)

    _section("SUMMARY")
    print(f"Zip: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Extracted files: {format_info['n_files']}")
    print(f"Total pages: {scan_info['total_pages']:,}")
    print(f"Total sentences: {scan_info['total_sentences']:,}")
    print("Inspection complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
