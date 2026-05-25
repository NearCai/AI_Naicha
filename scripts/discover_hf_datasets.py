"""Fast targeted probe of HF Hub for Chinese review datasets.

Probes a curated short list of dataset names (faster than full keyword search,
which can hang on slow per-dataset loads). Run with `python -u` for live
progress.

Outputs data/reviews/hf_dataset_survey.json with status per candidate:
  USABLE | DEPRECATED_SCRIPT | NOT_FOUND | GATED | FAIL
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


# Curated candidates — names I'm reasonably confident exist on HF, or that
# are explicitly mentioned in the project docstrings. Add more as you find them.
CANDIDATES = [
    # Already confirmed usable
    "XiangPan/waimai_10k",
    # Other plausible names
    "Setiaku/Chinese-Reviews",
    "luozhouyang/seamew-ChnSentiCorp",          # repurposed mirror
    "liyucheng/chinese_metaphor",
    "BAAI/COIG",
    "BAAI/IndustryCorpus_food_drink",
    "MMInstruction/M3IT",
    "shibing624/sharegpt_gpt4",                  # general but might have food
    "wangrui6/Zhihu-KOL",                        # Zhihu Q&A
    "kz919/multi_task_chinese_data",
    "shibing624/snli-zh",
    "fnlp/sharegpt-deepseek-chinese",
    "DUOMO/Aya",
    "OpenGVLab/MMPR-v1.2",
    "GoodBaiBai88/medical_chinese",
    "BAAI/Aquila-tts-train-zh",
    "uitnlp/SemEval-2020-Task-7",
    "BAAI/COIG-PC",
    "BAAI/COIG-PC-Lite",
    "wenge-research/wbpc-en-zh-300k",
    "Salesforce/wikitext",                       # control, English
]


def probe(name: str, timeout_total: float = 25.0) -> dict:
    """Try to load + peek a dataset. Returns status entry."""
    entry = {
        "id": name, "status": None, "columns": None,
        "first_sample": None, "error": None, "elapsed": None,
    }
    t0 = time.time()
    try:
        from datasets import load_dataset
        ds = load_dataset(name, split="train", streaming=True)
        first = next(iter(ds))
        entry["elapsed"] = round(time.time() - t0, 2)
        entry["columns"] = list(first.keys())[:8]
        for col in ("text", "review", "content", "comment", "Q",
                    "input", "instruction", "prompt", "question"):
            if col in first and isinstance(first[col], str):
                entry["first_sample"] = first[col][:200]
                break
        entry["status"] = "USABLE"
        return entry
    except Exception as e:
        entry["elapsed"] = round(time.time() - t0, 2)
        msg = str(e)[:250]
        entry["error"] = msg
        if "Dataset scripts are no longer supported" in msg:
            entry["status"] = "DEPRECATED_SCRIPT"
        elif ("doesn't exist" in msg or "not found" in msg
              or "404" in msg or "RepositoryNotFoundError" in msg):
            entry["status"] = "NOT_FOUND"
        elif "gated" in msg.lower() or "401" in msg or "403" in msg:
            entry["status"] = "GATED"
        else:
            entry["status"] = "FAIL"
        return entry


def main():
    sys.stdout.reconfigure(line_buffering=True)  # force live output
    print(f"Probing {len(CANDIDATES)} candidates...\n", flush=True)
    survey = []
    usable = []
    for i, name in enumerate(CANDIDATES, 1):
        print(f"[{i:>2}/{len(CANDIDATES)}] {name:<55s}", end=" ", flush=True)
        entry = probe(name)
        survey.append(entry)
        elapsed = entry["elapsed"] or 0
        if entry["status"] == "USABLE":
            print(f"USABLE  ({elapsed}s)  cols={entry['columns']}")
            print(f"     sample: {(entry['first_sample'] or '')[:100]}")
            usable.append(name)
        else:
            print(f"{entry['status']}  ({elapsed}s)")

    out = Path("data/reviews/hf_dataset_survey.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(survey, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    print(f"\n=== Summary ===")
    print(f"USABLE: {len(usable)}")
    for name in usable:
        print(f"  ✓ {name}")
    from collections import Counter
    other = Counter(e["status"] for e in survey if e["status"] != "USABLE")
    print(f"\nNot usable:")
    for s, n in other.most_common():
        print(f"  {s}: {n}")


if __name__ == "__main__":
    sys.exit(main() or 0)
