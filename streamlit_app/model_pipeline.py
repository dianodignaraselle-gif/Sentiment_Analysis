"""XLM-RoBERTa sentiment pipeline for English / Tagalog / Taglish TikTok comments.

Uses cardiffnlp/twitter-xlm-roberta-base-sentiment, an XLM-RoBERTa checkpoint
fine-tuned for 3-way (negative/neutral/positive) sentiment on multilingual
social-media text. XLM-RoBERTa's pretraining corpus includes Tagalog, and the
fine-tuning task matches our output scheme directly (no star-rating mapping
needed), so it handles English, Tagalog, and code-switched Taglish comments
in a single pipeline call.

The base checkpoint wasn't tuned on foundation-cosmetics vocabulary, so a
small local-market lexicon (shade/finish/wear-time slang in English and
Taglish) nudges only low-confidence calls toward the polarity those terms
imply - it never overrides a confident model prediction.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

from pathlib import Path

BASE_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# Same fine-tuned model as the Flask backend (training/train_sentiment.py);
# SENTIMENT_MODEL overrides, the stock checkpoint is the fallback.
_FINETUNED_DIR = Path(__file__).resolve().parent.parent / "backend" / "models" / "foundation-sentiment-xlmr"
MODEL_NAME = os.getenv("SENTIMENT_MODEL") or (
    str(_FINETUNED_DIR) if (_FINETUNED_DIR / "config.json").exists() else BASE_MODEL_NAME
)
MAX_LENGTH = 128  # must match backend/preprocessing.py, which training uses

# Normalizes whatever label spelling a checkpoint variant uses onto our scheme.
_LABEL_MAP = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}

# Local foundation-cosmetics slang (English + Taglish) the base model wasn't
# fine-tuned on. Checked against the raw comment text, lowercased.
POSITIVE_LEXICON = {
    "sulit", "panalo", "legit", "glowing", "nakakaglow", "poreless",
    "non-cakey", "non cakey", "natural finish", "worth it", "money's worth",
    "bet na bet", "flawless", "second skin", "no filter needed", "pang-araw araw",
    "budol worth it", "malagkit pero sulit", "maayos ang coverage", "amazing", "hindi siya nakaka-breakout", 
    "doesn't make my skin look cakey", "sobrang ganda talaga", "soft sa mukha", "highly pigmented", "walang white cast", 
    "socail ang packaging", "blends well", "blendable", "okay na shade match", "mabango",
}
NEGATIVE_LEXICON = {
    "cakey", "oxidize", "oxidizes", "oxidized", "breakout", "nag-breakout",
    "nakaka-breakout", "pumping", "ashy", "chalky", "sayang", "pangit",
    "flashback", "pabato", "peeling", "patchy", "panot", "hulas", "malagkit",
    "budol", "nakaka-hulas", "pangit ng product", "hindi siya nakaka-glow", "nagkakaroon ng breakouts",
    "nagkakaroon ng acne", "nagiging oily", "nagiging dry", "hindi siya nakaka-poreless", "sira", "damaged", "substandard", "hindi siya long-lasting",
    "hindi siya worth it", "hindi siya sulit", "hindi siya panalo",
    "punit-punit", "missing item", "scammed", "scam", "fake", "hindi siya legit", "hindi siya authentic", "hindi siya original", "hindi siya quality",
    "hindi siya effective", "hindi siya efficient", "hindi nagrereply yung seller", "chaka", "mahapdi", "hindi siya nakaka-natural finish", 
    "hindi siya nakaka-second skin", "hindi siya nakaka-non-cakey", "hindi siya nakaka-poreless", "hindi siya nakaka-glowing",
    "hapdi sa balat", "parang sunog sa mukha ko", "mabaho ang amoy",

}

_WHITESPACE_RE = re.compile(r"\s+")
_HANDLE_RE = re.compile(r"@\w+")
_URL_RE = re.compile(r"https?://\S+")

SENTIMENT_LABELS = ("positive", "neutral", "negative")


@dataclass(frozen=True)
class SentimentResult:
    text: str
    sentiment: str  # "positive" | "neutral" | "negative"
    score: float  # confidence of the winning label, 0-1
    scores: dict = field(default_factory=dict)  # full 3-way distribution
    matched_terms: tuple = ()  # domain lexicon hits found in the text
    low_confidence: bool = False  # model probability for `sentiment` is below LOW_CONFIDENCE_THRESHOLD


def clean_text(text: str) -> str:
    """Normalize raw comment/caption/transcript text before tokenization."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _URL_RE.sub(" ", text)
    text = _HANDLE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_video_text(caption: str = "", transcript: str = "") -> str:
    """Combine a video's caption and transcript into one classifiable string.

    Speech-to-text itself is out of scope here; the scraper is responsible
    for supplying whatever transcript/caption/on-screen text it can obtain
    for a video - this just normalizes it for the classifier.
    """
    combined = " ".join(part for part in (caption, transcript) if part)
    return clean_text(combined)


@lru_cache(maxsize=1)
def get_classifier():
    """Load tokenizer + model once per process and reuse it on every call.

    Cached at module scope (not just per Streamlit rerun) so the ~1GB
    XLM-RoBERTa weights are loaded once regardless of how many times the
    dashboard script re-executes.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        top_k=None,  # return the full 3-way distribution, not just the argmax
    )


# ---------- decision rule ----------
# The base model was fine-tuned on tweets where "neutral" mostly means
# factual/no-opinion text. In product reviews, mixed or lukewarm opinions
# ("decent lang, pwede na pero meh", "not bad, but...") are annotated as
# neutral, and the model spreads its probability for them across all three
# classes instead of committing to neutral. So: a comment is only labeled
# positive/negative when that polarity is clearly dominant; weaker polarity
# counts as neutral. 0.70 was chosen on the small labeled set in
# eval/labeled_comments.csv (0.70-0.75 all gave the same result there) -
# re-tune it with `python evaluate.py --sweep` once more labeled data exists.
POLARITY_THRESHOLD = float(os.getenv("SENTIMENT_POLARITY_THRESHOLD", "0.70"))
# Below this probability for the reported label, the prediction is flagged
# low-confidence instead of being presented as reliable.
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("SENTIMENT_LOW_CONFIDENCE", "0.50"))

# Tokens that negate a slang term that follows within a few words:
# "hindi siya nakaka-breakout", "doesn't make my skin look cakey".
_NEGATORS = {
    "hindi", "di", "'di", "hnd", "not", "no", "never", "without", "wala", "walang",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt", "isn't", "isnt",
    "won't", "wont", "wasn't", "wasnt", "non", "zero",
}
_NEGATION_WINDOW = 5
_TOKEN_RE = re.compile(r"[\w'’-]+", re.UNICODE)


def _lexicon_hits(text: str) -> tuple[list, list]:
    """Domain slang found in the text, with negation flipping its polarity.

    Matches whole words only, keeps the longest overlapping term ("nakaka-
    breakout" rather than also "breakout"), and treats a term preceded by a
    negator within a few words as the opposite polarity, so "hindi siya
    nakaka-breakout" counts as positive rather than negative.
    """
    lowered = text.lower().replace("’", "'")
    terms = [(t, "positive") for t in POSITIVE_LEXICON] + [(t, "negative") for t in NEGATIVE_LEXICON]
    spans = []
    for term, polarity in terms:
        for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?![\w-])", lowered):
            spans.append((m.start(), m.end(), term, polarity))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    kept, last_end = [], -1
    for start, end, term, polarity in spans:
        if start >= last_end:
            kept.append((start, end, term, polarity))
            last_end = end

    pos, neg = [], []
    for start, _end, term, polarity in kept:
        before = _TOKEN_RE.findall(lowered[:start])[-_NEGATION_WINDOW:]
        if any(tok in _NEGATORS for tok in before):
            polarity = "negative" if polarity == "positive" else "positive"
            term = f"not {term}"
        (pos if polarity == "positive" else neg).append(term)
    return pos, neg


def _decide(scores: dict) -> tuple[str, bool]:
    """Pick the label from the model's 3-way distribution; flag low confidence."""
    polar = "positive" if scores["positive"] >= scores["negative"] else "negative"
    sentiment = polar if scores[polar] >= POLARITY_THRESHOLD else "neutral"
    return sentiment, scores[sentiment] < LOW_CONFIDENCE_THRESHOLD


def classify(texts: Iterable[str]) -> list:
    """Classify a batch of comment/caption/transcript strings.

    Returns a list of SentimentResult, one per non-empty input string. Same
    decision rule as backend/model_pipeline.py: the label comes from the
    model's probabilities, weak polarity counts as neutral, and low-probability
    predictions are flagged. Lexicon hits are informational only.
    """
    originals = [t for t in texts if t and t.strip()]
    cleaned = [clean_text(t) for t in originals]
    if not cleaned:
        return []

    clf = get_classifier()
    raw_batch = clf(cleaned, truncation=True, max_length=MAX_LENGTH, batch_size=16)

    results = []
    for original, text, raw in zip(originals, cleaned, raw_batch):
        scores = {}
        for r in raw:
            norm = _LABEL_MAP.get(r["label"].strip().lower(), r["label"].strip().lower())
            scores[norm] = r["score"]
        for label in SENTIMENT_LABELS:
            scores.setdefault(label, 0.0)

        pos_hits, neg_hits = _lexicon_hits(text)
        sentiment, low_confidence = _decide(scores)
        results.append(
            SentimentResult(
                text=original,
                sentiment=sentiment,
                score=round(scores[sentiment], 4),
                scores={k: round(v, 4) for k, v in scores.items()},
                matched_terms=tuple(sorted(set(pos_hits) | set(neg_hits))),
                low_confidence=low_confidence,
            )
        )
    return results


def classify_one(text: str) -> SentimentResult | None:
    results = classify([text])
    return results[0] if results else None
