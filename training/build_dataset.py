"""Build the upgraded train/validation/test sets from the raw sources.

    python training/build_dataset.py

Reads training/raw/, writes training/data/{train,validation,test}.csv plus
training/data/build_stats.json, and lists suspected label errors in
training/data/suspected_label_errors.csv for manual review.

What it fixes in the old (Colab) dataset:
- The 1,467 "foundation_reviews" rows are synthetic: ~225 template sentences
  with a random opener ("Ugh,", "Hmm,", "Ok so"), filler ("Oily skin here.")
  and an emoji, and every emoji/opener maps to exactly one label (😡 is always
  negative, "Hmm," always neutral). The old model could learn those shortcuts
  instead of the sentence. They are stripped here, and copies that only differ
  by shade name ("Shade Beige/Honey/Mocha is way too dark") are collapsed.
- The old split put near-copies of the same sentence in train and test, so
  its 99% test accuracy measured memorization. Here the test set is real,
  human-written comments only, and none of them is used for training.
- Real comments are 87% positive (the TikTok file has 6 negatives), so the
  training set is balanced by capping every class at the smallest one.
- Only foundation reviews are used: the general-product Shopee reviews the
  current model was trained with, and the TikTok file's powder/concealer
  comments, were removed from training/raw/ on 2026-10-17.
- Known label errors are corrected from training/label_corrections.csv.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from preprocessing import clean_text  # noqa: E402

RAW = ROOT / "raw"
OUT = ROOT / "data"
CACHE = ROOT / "cache"
SEED = 42
LABELS = ("negative", "neutral", "positive")

# Flag a row for manual review when the base model is this sure its label is wrong.
CONTRADICTION_THRESHOLD = 0.80

# Real comments: share held out for the test and validation sets.
REAL_TEST_SIZE = 0.40
REAL_VAL_SIZE = 0.15

_SYNTH_OPENER_RE = re.compile(r"^(Honestly|Sis|Grabe|Ugh|Hmm|Tbh|Update:|Ok so)\b,?:?\s*")
_SYNTH_FILLER_RE = re.compile(
    r"(Oily skin here\.|Dry skin here\.|Used it for 3 days\.|Combination skin po ako\.|"
    r"Gamit ko for 2 weeks na\.|Bought during sale\.)(!!)?"
)
_SYNTH_EMOJI = "😡😐😊😞👎😍💯🤔"
_SHADE_RE = re.compile(r"\b(Beige|Caramel|Golden|Honey|Ivory|Light Vanilla|Medium|Mocha|Natural|Porcelain|Sand|Warm Tan)\b")


def _norm_key(text: str) -> str:
    """Duplicate key: case-, punctuation- and spacing-insensitive."""
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def load_real() -> pd.DataFrame:
    xlsx = pd.read_excel(RAW / "tiktok_foundation_sentiment_dataset.xlsx", sheet_name="Dataset")
    xlsx = pd.DataFrame({"text": xlsx["Comment Text"], "label": xlsx["Sentiment Label"], "source": "real_handwritten"})
    tiktok = pd.read_csv(RAW / "tiktok_comments_labeled.csv")
    tiktok = pd.DataFrame({"text": tiktok["comment"], "label": tiktok["sentiment"], "source": "real_tiktok"})
    # Foundation_Dataset_Part2.xlsx is not loaded: 40 of its 41 rows are already in
    # tiktok_comments_labeled.csv and the 41st is a truncated copy of one of them.
    return pd.concat([xlsx, tiktok], ignore_index=True)


def strip_synthetic(text: str) -> str:
    text = _SYNTH_OPENER_RE.sub("", text)
    text = _SYNTH_FILLER_RE.sub("", text)
    for e in _SYNTH_EMOJI:
        text = text.replace(e, "")
    text = _SYNTH_OPENER_RE.sub("", text.strip())  # "Hmm, Ok so ..." stacks two openers
    return re.sub(r"\s+", " ", text).strip()


def load_synthetic() -> pd.DataFrame:
    df = pd.read_csv(RAW / "combined_dataset.csv")
    # The "tiktok_80" rows are the xlsx comments again; load_real() has those.
    df = df[df["source"] == "foundation_reviews"].copy()
    df["text"] = df["text"].astype(str).map(strip_synthetic)
    df = df[df["text"].str.len() > 3]
    df["template"] = df["text"].map(lambda t: _norm_key(_SHADE_RE.sub("<shade>", t)))
    df = df.drop_duplicates("template")
    return pd.DataFrame({"text": df["text"], "label": df["label"], "source": "synthetic_reviews"})


def base_model_scores(texts: list[str]) -> pd.DataFrame:
    """P(negative/neutral/positive) from the stock cardiffnlp model, cached on disk."""
    cache_file = CACHE / "base_model_scores.csv"
    cached = pd.read_csv(cache_file) if cache_file.exists() else pd.DataFrame(columns=["text", *LABELS])
    todo = sorted(set(texts) - set(cached["text"]))
    if todo:
        from transformers import pipeline

        print(f"scoring {len(todo)} texts with the base model (cached afterwards)...")
        local = CACHE / "base-model"  # safetensors copy; see train_sentiment.py BASE_MODEL
        name = str(local) if (local / "config.json").exists() else "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        clf = pipeline("sentiment-analysis", model=name, top_k=None)
        rows = []
        for text, out in zip(todo, clf(todo, truncation=True, max_length=128, batch_size=32)):
            rows.append({"text": text, **{o["label"].lower(): o["score"] for o in out}})
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        CACHE.mkdir(exist_ok=True)
        cached.to_csv(cache_file, index=False)
    return cached.drop_duplicates("text").set_index("text")


def contradicts(label: str, p: pd.Series, threshold: float) -> bool:
    if label == "positive":
        return p["negative"] >= threshold
    if label == "negative":
        return p["positive"] >= threshold
    return max(p["positive"], p["negative"]) >= threshold + 0.1  # neutral: only very polar text


def main() -> None:
    stats: dict = {}
    real, synth = load_real(), load_synthetic()
    stats["loaded"] = {"real": len(real), "synthetic_after_template_dedupe": len(synth)}

    df = pd.concat([real, synth], ignore_index=True)
    df["text"] = df["text"].astype(str).map(clean_text)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"].str.len() > 0]
    assert df["label"].isin(LABELS).all(), df.loc[~df["label"].isin(LABELS), "label"].unique()

    corrections_file = ROOT / "label_corrections.csv"
    if corrections_file.exists():
        corrections = pd.read_csv(corrections_file)
        fix = dict(zip(corrections["text"].map(clean_text), corrections["corrected_label"]))
        hit = df["text"].isin(fix)
        df.loc[hit, "label"] = df.loc[hit, "text"].map(fix)
        stats["label_corrections_applied"] = int(hit.sum())

    # Exact/near-exact duplicates across all sources; real comments win.
    priority = {"real_handwritten": 0, "real_tiktok": 1, "synthetic_reviews": 2}
    df["key"] = df["text"].map(_norm_key)
    df = df.sort_values("source", key=lambda s: s.map(priority), kind="stable")
    before = len(df)
    conflicting = df.groupby("key")["label"].nunique()
    stats["duplicate_keys_with_conflicting_labels"] = int((conflicting > 1).sum())
    df = df.drop_duplicates("key")
    stats["duplicates_removed"] = before - len(df)

    scores = base_model_scores(df["text"].tolist())
    p = scores.loc[df["text"]].reset_index(drop=True)
    df = df.reset_index(drop=True)
    flags = [contradicts(lbl, p.iloc[i], CONTRADICTION_THRESHOLD) for i, lbl in enumerate(df["label"])]
    df["base_model_disagrees"] = flags

    OUT.mkdir(exist_ok=True)
    review = df[df["base_model_disagrees"]]
    review.assign(**{f"p_{k}": p.loc[review.index, k].round(3) for k in LABELS})[
        ["source", "label", "p_negative", "p_neutral", "p_positive", "text"]
    ].to_csv(OUT / "suspected_label_errors.csv", index=False)
    stats["flagged_for_manual_review"] = len(review)

    # Splits: test/validation get real comments only. Synthetic rows only train.
    real = df[df["source"].str.startswith("real_")]
    real_rest, test = train_test_split(real, test_size=REAL_TEST_SIZE, stratify=real["label"], random_state=SEED)
    real_train, real_val = train_test_split(
        real_rest, test_size=REAL_VAL_SIZE / (1 - REAL_TEST_SIZE), stratify=real_rest["label"], random_state=SEED
    )
    synth = df[df["source"] == "synthetic_reviews"]

    train = pd.concat([real_train, synth])
    # Balance: cap every class at the size of the smallest one, dropping
    # synthetic rows before real ones.
    cap = train["label"].value_counts().min()
    train["_prio"] = train["source"].map(priority)
    train = (
        train.sample(frac=1, random_state=SEED)
        .sort_values("_prio", kind="stable")
        .groupby("label", group_keys=False)
        .head(cap)
        .drop(columns="_prio")
    )
    validation = real_val

    cols = ["text", "label", "source"]
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part = part.sample(frac=1, random_state=SEED)[cols]
        part.to_csv(OUT / f"{name}.csv", index=False)
        stats[name] = {
            "rows": len(part),
            "labels": part["label"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
            "sources": part["source"].value_counts().to_dict(),
        }
        print(f"{name:10s} {len(part):5d}  {stats[name]['labels']}  {stats[name]['sources']}")

    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("train", "validation", "test")}, indent=2))


if __name__ == "__main__":
    main()
