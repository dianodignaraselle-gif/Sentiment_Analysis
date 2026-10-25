"""Score the app's own classify() on the held-out test set, old vs new.

    backend/venv/bin/python training/evaluate_app.py [--old-pipeline PATH]

Runs in the app's venv (stdlib only besides the app's own deps), through
backend/model_pipeline.classify - the exact function /api/analyze calls - so
it covers preprocessing, the model and the rule layer together.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT.parent / "backend"
LABELS = ("negative", "neutral", "positive")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def score(y_true, y_pred) -> dict:
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    per = {}
    for lbl in LABELS:
        tp = sum(t == p == lbl for t, p in zip(y_true, y_pred))
        prec = tp / max(1, sum(p == lbl for p in y_pred))
        rec = tp / max(1, sum(t == lbl for t in y_true))
        per[lbl] = (prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    macro = [sum(v[i] for v in per.values()) / 3 for i in range(3)]
    return {"accuracy": acc, "precision": macro[0], "recall": macro[1], "f1": macro[2], "per_class": per}


def run(label: str, module, rows) -> list:
    preds = [r["sentiment"] for r in module.classify([row["text"] for row in rows])]
    assert len(preds) == len(rows)
    m = score([row["label"] for row in rows], preds)
    print(f"{label:38s} acc={m['accuracy']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  "
          + "  ".join(f"{k[:3]}-F1={v[2]:.2f}" for k, v in m["per_class"].items()))
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", default=str(ROOT / "data" / "test.csv"))
    p.add_argument("--old-pipeline", default=str(ROOT.parent / "backups" / "2026-10-16_pre-model-upgrade" / "backend" / "model_pipeline.py"))
    args = p.parse_args()

    with open(args.test, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"test set: {len(rows)} real comments {dict(Counter(r['label'] for r in rows))}\n")

    sys.path.insert(0, str(BACKEND))
    results = {}
    if Path(args.old_pipeline).exists():
        results["old app"] = run("old app (stock model + rules)", load_module("old_pipeline", Path(args.old_pipeline)), rows)

    for rules in ("1", "0"):
        os.environ["SENTIMENT_RULES"] = rules
        new = load_module(f"new_pipeline_{rules}", BACKEND / "model_pipeline.py")
        name = f"new app ({'fine-tuned' if new.IS_FINETUNED else 'stock'} model, rules {'on' if rules == '1' else 'off'})"
        results[name] = run(name, new, rows)

    names = list(results)
    print("\ncomments where the pipelines disagree:")
    for i, row in enumerate(rows):
        preds = [results[n][i] for n in names]
        if len(set(preds)) > 1:
            print(f"  true={row['label']:8s} " + " ".join(f"{n.split('(')[0].strip()}{'' if 'old' in n else ('+R' if 'on' in n else '-R')}={pr[:3]}"
                                                     for n, pr in zip(names, preds)) + f" | {row['text'][:90]}")


if __name__ == "__main__":
    main()
