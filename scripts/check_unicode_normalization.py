import json
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config

REPORT_PATH = cg_config.resolve_path("data/processed/fever/wiki_resolution_report.json")
INDEX_PATH = cg_config.resolve_path("data/processed/fever/wiki_pages_index.sqlite")

with REPORT_PATH.open() as f:
    report = json.load(f)

missing_page_urls = []
for r in report["results"]:
    if r["status"] != "unresolved":
        continue
    for attempt in r["attempts"]:
        for s in attempt.get("sentences", []):
            if s["failure_reason"] == "missing_page":
                missing_page_urls.append(s["wiki_url"])

unique_missing = sorted(set(missing_page_urls))
print(f"Total missing_page failure instances: {len(missing_page_urls)}")
print(f"Unique missing page ids: {len(unique_missing)}")
print(f"Sample: {unique_missing[:20]}")

conn = sqlite3.connect(str(INDEX_PATH))

nfc_matches = 0
nfd_matches = 0
truly_absent = 0
truly_absent_samples = []

for page_id in unique_missing:
    nfc = unicodedata.normalize("NFC", page_id)
    nfd = unicodedata.normalize("NFD", page_id)
    found_nfc = conn.execute("SELECT 1 FROM pages WHERE page_id = ?", (nfc,)).fetchone() is not None
    found_nfd = conn.execute("SELECT 1 FROM pages WHERE page_id = ?", (nfd,)).fetchone() is not None
    if page_id != nfc and found_nfc:
        nfc_matches += 1
    elif page_id != nfd and found_nfd:
        nfd_matches += 1
    elif not found_nfc and not found_nfd:
        truly_absent += 1
        if len(truly_absent_samples) < 10:
            truly_absent_samples.append(page_id)

print(f"\nResolvable via NFC normalization (differs from original, found as NFC): {nfc_matches}")
print(f"Resolvable via NFD normalization (differs from original, found as NFD): {nfd_matches}")
print(f"Truly absent from the index under any normalization tried: {truly_absent}")
print(f"Sample truly-absent page ids: {truly_absent_samples}")

conn.close()
