"""Text preprocessing shared by model training and the running app.

training/train_sentiment.py imports this exact module, so a comment is
cleaned and truncated the same way when the model learns from it as when
the app classifies it. Standard library only, so it also runs on Colab.
"""

import re
import unicodedata

# Token budget for the classifier. Comments are short (median 27 tokens,
# 99th percentile 94 in the training data; 3 of ~1,800 exceed 128), and
# training and inference must use the same limit.
MAX_LENGTH = 128

_WHITESPACE_RE = re.compile(r"\s+")
_HANDLE_RE = re.compile(r"@\w+")
_URL_RE = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    """Normalize raw comment/caption/transcript text before tokenization.

    Emoji and elongated words ("gandaaaa") are kept: both carry sentiment
    and XLM-RoBERTa's tokenizer handles them.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _URL_RE.sub(" ", text)
    text = _HANDLE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
