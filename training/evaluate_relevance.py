"""Check backend/relevance.py: what it keeps, and what it filters out.

    backend/venv/bin/python training/evaluate_relevance.py [--db]

Every comment in the labeled datasets is a product comment, so each one the
filter calls irrelevant is a false alarm - they're listed so the word lists
can be fixed. IRRELEVANT_EXAMPLES are comments that must be filtered out.
--db also lists which comments already analyzed in backend/dataset.db would
be filtered now. Doesn't load the sentiment model.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT.parent / "backend"
sys.path.insert(0, str(BACKEND))  # model_pipeline imports relevance/preprocessing by name

# Off-topic comments seen under real foundation videos.
IRRELEVANT_EXAMPLES = [
    "di ako pinayagan ni papa bumili",
    "hindi ako pinayagan ni papa bumili",
    "ano color ng hair mo mi?",
    "I  like your hair color",
    "link ng kwentas",
    "Ganda ng pants mi san mo nabili?",
    "hi pa help naman mag 600 followers huhy",
    "engagement is the key",
    "mag labas na kayo ng whitening soap HAHAHA",
    "anong shade ng lipstick mo",  # a shade, but of a lipstick
    "ang bango ng lotion na to, ang kinis ng balat ko",
    "hiyang ako sa serum nila, nawala pimples ko",
    "legit yung sunscreen walang white cast",
    "mascarra nyo po anong brand",
    "ganda ng eyeliner hindi nag smudge",
    "waterproof ba yung mascara?",
    "lip tint mo po?",
    "ganda ng kilay mo ate",
    "what song is this",
    "ako rin finger din bet ko HAhaha",
    "tangina ang sexy mo",
    "pakantot naman",
    "first ma",
    "first mii",
    "1st!!",
    "is that Anne Curtis?",
    "si Ivana ba yan",
    "hi ate notice me",
    "hello po",
    "HAHAHAHA",
    "🤣",
    "Us",
    ".dump",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_column(path: Path, column: str) -> list:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [row[column] for row in csv.DictReader(f) if row[column].strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", action="store_true", help="also list filtered comments in backend/dataset.db")
    args = p.parse_args()
    mp = load_module("model_pipeline", BACKEND / "model_pipeline.py")

    failures = 0
    for name, column in (("combined_dataset.csv", "text"), ("tiktok_comments_labeled.csv", "comment")):
        texts = read_column(ROOT / "raw" / name, column)
        flagged = [(t, why) for t in texts for ok, why in [mp.relevance_check(t)] if not ok]
        print(f"{name}: {len(texts) - len(flagged)}/{len(texts)} kept as relevant ({len(flagged)} false alarms)")
        for t, why in flagged:
            print(f"    - {t!r}: {why}")
        failures += len(flagged)

    missed = [t for t in IRRELEVANT_EXAMPLES if mp.relevance_check(t)[0]]
    print(f"irrelevant examples: {len(IRRELEVANT_EXAMPLES) - len(missed)}/{len(IRRELEVANT_EXAMPLES)} filtered out")
    for t in missed:
        print(f"    - still relevant: {t!r}")
    failures += len(missed)

    if args.db:
        conn = sqlite3.connect(BACKEND / "dataset.db")
        rows = conn.execute("SELECT DISTINCT comment, product FROM comments WHERE owner IS NOT NULL").fetchall()
        flagged = [(c, why) for c, prod in rows for ok, why in [mp.relevance_check(c, prod)] if not ok]
        print(f"dataset.db: {len(flagged)} of {len(rows)} distinct analyzed comments would be filtered out")
        for c, why in flagged:
            print(f"    - {c!r}: {why}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
