import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config

REPORT_PATH = cg_config.resolve_path("data/processed/fever/wiki_resolution_report.json")

with REPORT_PATH.open() as f:
    report = json.load(f)

sentence_id_values = Counter()
page_missing_examples = []
neg_one_but_page_missing = 0

for r in report["results"]:
    if r["status"] != "unresolved":
        continue
    for attempt in r["attempts"]:
        for s in attempt.get("sentences", []):
            if s["failure_reason"] == "missing_sentence_id":
                sentence_id_values[s["sentence_id"]] += 1
            elif s["failure_reason"] == "missing_page":
                if len(page_missing_examples) < 10:
                    page_missing_examples.append({"wiki_url": s["wiki_url"], "sentence_id": s["sentence_id"]})

print("Distribution of sentence_id values among 'missing_sentence_id' failures (top 15):")
for val, count in sentence_id_values.most_common(15):
    print(f"  sentence_id={val}: {count:,}")

total_missing_sentence_failures = sum(sentence_id_values.values())
neg_one_count = sentence_id_values.get(-1, 0)
print(f"\nTotal missing_sentence_id failure instances: {total_missing_sentence_failures:,}")
print(f"Of which sentence_id == -1: {neg_one_count:,} ({neg_one_count / total_missing_sentence_failures * 100:.2f}%)")
print(f"Of which sentence_id >= 0 (genuinely missing from an existing page): "
      f"{total_missing_sentence_failures - neg_one_count:,}")

print(f"\nSample 'missing_page' failures (page not in index at all): {page_missing_examples}")

# For sentence_id == -1 cases specifically, check: does the referenced PAGE
# itself exist and have content in the index? (I.e. is -1 really "whole
# page, no sentence index" pointing at a real page, vs. a page that's
# ALSO missing entirely?)
import sqlite3
INDEX_PATH = cg_config.resolve_path("data/processed/fever/wiki_pages_index.sqlite")
conn = sqlite3.connect(str(INDEX_PATH))

neg_one_pages_checked = 0
neg_one_pages_exist = 0
neg_one_pages_have_text = 0
sample_neg_one_pages = []
for r in report["results"]:
    if r["status"] != "unresolved":
        continue
    for attempt in r["attempts"]:
        for s in attempt.get("sentences", []):
            if s["failure_reason"] == "missing_sentence_id" and s["sentence_id"] == -1:
                neg_one_pages_checked += 1
                if neg_one_pages_checked > 2000:
                    continue
                page_row = conn.execute("SELECT has_content FROM pages WHERE page_id = ?", (s["wiki_url"],)).fetchone()
                if page_row is not None:
                    neg_one_pages_exist += 1
                    if page_row[0]:
                        neg_one_pages_have_text += 1
                if len(sample_neg_one_pages) < 5:
                    sample_neg_one_pages.append(s["wiki_url"])

print(f"\nOf the first 2000 sentence_id=-1 references checked:")
print(f"  Referenced page EXISTS in wiki_pages index: {neg_one_pages_exist} / {min(neg_one_pages_checked,2000)}")
print(f"  Referenced page has non-empty content: {neg_one_pages_have_text} / {min(neg_one_pages_checked,2000)}")
print(f"  Sample page ids referenced with sentence_id=-1: {sample_neg_one_pages}")

conn.close()
