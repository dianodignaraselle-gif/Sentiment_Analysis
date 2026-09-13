"""BERT-based sentiment classifier.

Uses nlptown/bert-base-multilingual-uncased-sentiment, a BERT model
fine-tuned to output a 1-5 star rating. Stars are mapped onto the
three-way Positive / Neutral / Negative scheme used across the app:
  1-2 stars -> negative, 3 stars -> neutral, 4-5 stars -> positive.

The underlying mBERT encoder is pretrained on 104 languages, including
Tagalog, so it also handles Tagalog and code-switched Taglish comments
(common in Philippine TikTok comment sections) reasonably well despite
only being fine-tuned on English/Dutch/German/French/Italian/Spanish
reviews - verified against a manual Tagalog/Taglish test set before
relying on it here. No separate language routing is needed: the same
pipeline call handles English, Tagalog, and Taglish input as-is.
"""

from functools import lru_cache

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

_STAR_TO_LABEL = {
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}


@lru_cache(maxsize=1)
def _get_pipeline():
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    import torch

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def classify(texts):
    """Classify a list of comment strings.

    Returns a list of dicts: {"comment": str, "sentiment": str, "score": float, "stars": str}
    """
    if not texts:
        return []

    clf = _get_pipeline()
    raw_results = clf(texts, truncation=True, max_length=256)

    results = []
    for text, raw in zip(texts, raw_results):
        label = _STAR_TO_LABEL.get(raw["label"], "neutral")
        results.append(
            {
                "comment": text,
                "sentiment": label,
                "score": round(float(raw["score"]), 4),
                "stars": raw["label"],
            }
        )
    return results
