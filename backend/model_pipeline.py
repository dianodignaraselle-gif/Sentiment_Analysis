"""XLM-RoBERTa sentiment pipeline + Whisper speech-to-text for TikTok video analysis.

The classifier is cardiffnlp/twitter-xlm-roberta-base-sentiment (3-way
negative/neutral/positive, multilingual incl. Tagalog) further fine-tuned
on foundation-review comments by training/train_sentiment.py and loaded from
backend/models/foundation-sentiment-xlmr. On 103 held-out real TikTok
comments it scores 0.951 accuracy / 0.938 macro-F1 vs 0.874 / 0.798 for the
stock checkpoint (backend/models/foundation-sentiment-xlmr/evaluation.md).
If those weights are missing it falls back to the stock checkpoint.

For the stock checkpoint, a rule layer adds evidence to the model's
probabilities: a Taglish beauty lexicon (with negation - "walang hulas"
counts as praise), emoji cues, and question detection (product questions
like "Anong shade?" are neutral). The fine-tuned model learned these cues
from data and scores better without it, so it's off by default there
(SENTIMENT_RULES overrides).

Mixed opinions - a balanced mix of praise and complaints, like "Ang ganda ng
coverage pero nag-oxidize" - are labeled neutral (see "Mixed opinions"
below; SENTIMENT_MIXED_NEUTRAL=0 turns that off).

Whisper (openai-whisper) turns a TikTok creator's spoken audio into text so
it can be classified the same way as a caption or a comment.
"""

from __future__ import annotations

import os
import re
import shutil
import zlib
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import relevance
from preprocessing import MAX_LENGTH, clean_text  # shared with training/train_sentiment.py

BASE_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# Fine-tuned on foundation-review comments by training/train_sentiment.py.
FINETUNED_MODEL_DIR = Path(__file__).parent / "models" / "foundation-sentiment-xlmr"


def _resolve_model_name() -> str:
    """SENTIMENT_MODEL env var > local fine-tuned model > stock checkpoint.

    The fine-tuned weights (~1.1GB) aren't in git, so a fresh clone or a
    host without them still starts, on the stock checkpoint.
    """
    override = os.getenv("SENTIMENT_MODEL")
    if override:
        return override
    if (FINETUNED_MODEL_DIR / "config.json").exists():
        return str(FINETUNED_MODEL_DIR)
    return BASE_MODEL_NAME


MODEL_NAME = _resolve_model_name()
IS_FINETUNED = MODEL_NAME != BASE_MODEL_NAME
WHISPER_MODEL_SIZE = "base"  # good Taglish/English accuracy-vs-speed tradeoff on CPU


def _ensure_ffmpeg_on_path() -> str | None:
    """Make sure a real `ffmpeg` binary is reachable for yt-dlp/Whisper to shell out to.

    Homebrew's current installer no longer supports Intel Macs at all (it
    hard-aborts unless `uname -m` is arm64), so rather than requiring a
    system ffmpeg install, fall back to the static binary the `imageio-ffmpeg`
    pip package bundles - scoped entirely to this venv, no sudo or system
    changes - whenever nothing is already on PATH.
    """
    existing = shutil.which("ffmpeg")
    if existing:
        return existing
    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    # imageio-ffmpeg ships its binary under a version-qualified filename
    # (e.g. ffmpeg-macos-x86_64-v7.1), not literally "ffmpeg" - yt-dlp and
    # Whisper both shell out to the bare "ffmpeg" command name, so symlink
    # it under the name they expect before adding its directory to PATH.
    ffmpeg_link = bundled.parent / "ffmpeg"
    if not ffmpeg_link.exists():
        try:
            ffmpeg_link.symlink_to(bundled)
        except OSError:
            return str(bundled)  # still usable via an explicit ffmpeg_location, just not bare "ffmpeg"

    os.environ["PATH"] = str(bundled.parent) + os.pathsep + os.environ.get("PATH", "")
    return str(ffmpeg_link)


FFMPEG_PATH = _ensure_ffmpeg_on_path()

# Normalizes whatever label spelling a checkpoint variant uses onto our scheme.
_LABEL_MAP = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}

_WHITESPACE_RE = re.compile(r"\s+")

SENTIMENT_LABELS = ("positive", "neutral", "negative")


@dataclass(frozen=True)
class SentimentResult:
    text: str
    sentiment: str  # "positive" | "neutral" | "negative"
    score: float  # confidence of the winning label, 0-1
    scores: dict = field(default_factory=dict)  # full 3-way distribution
    matched_terms: tuple = ()  # domain lexicon hits found in the text


def build_video_text(caption: str = "", transcript: str = "") -> str:
    """Combine a video's caption and Whisper transcript into one classifiable string."""
    combined = " ".join(part for part in (caption, transcript) if part)
    return clean_text(combined)


@lru_cache(maxsize=1)
def get_classifier():
    """Load tokenizer + model once per process and reuse it on every call."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    # int8 quantization saves ~70% memory but flattens this checkpoint's
    # outputs to near-uniform (~0.33-0.45) and drops accuracy from ~87% to
    # ~67% on our Taglish test set, so it's opt-in for memory-capped hosts only.
    if os.getenv("SENTIMENT_QUANTIZE", "0") == "1":
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        top_k=None,  # return the full 3-way distribution, not just the argmax
    )


@lru_cache(maxsize=1)
def get_whisper_model():
    """Load the Whisper speech-to-text model once per process and reuse it."""
    import whisper

    return whisper.load_model(WHISPER_MODEL_SIZE)


# Whisper decoding guards. TikTok audio is often mostly music, where Whisper
# tends to "hallucinate" a plausible sentence and then loop it forever.
WHISPER_NO_SPEECH_THRESHOLD = 0.6  # segment's P(no speech) above this ...
WHISPER_LOGPROB_THRESHOLD = -1.0  # ... with avg log-prob below this = silence/music
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4  # gzip ratio above this = repetitive text

# Whisper labels music-only audio with sound tags ("Music", "[Music]", "♪").
_SOUND_TAGS_ONLY_RE = re.compile(
    r"^(?:[\s\[\]().,!♪♫*-]|music|musika|applause|laughter|silence|instrumental)*$", re.IGNORECASE
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NON_WORD_RE = re.compile(r"[^\w\s]")


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    reliable: bool
    issue: str | None = None  # why the transcript shouldn't be trusted, if it shouldn't


def _compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw)) if raw else 0.0


def _normalize_sentence(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", _NON_WORD_RE.sub("", s.lower())).strip()


def detect_repetition(text: str) -> str | None:
    """Return a human-readable reason if `text` looks like a looping hallucination."""
    sentences = [_normalize_sentence(s) for s in _SENTENCE_SPLIT_RE.split(text)]
    sentences = [s for s in sentences if s]
    if len(sentences) >= 3:
        top, n = Counter(sentences).most_common(1)[0]
        if n >= 3 and n / len(sentences) >= 0.5:
            return f'the same sentence ("{top}") repeats {n} of {len(sentences)} times'
        if len(set(sentences)) / len(sentences) < 0.4:
            return f"only {len(set(sentences))} unique sentences out of {len(sentences)}"

    # Loops without punctuation: a single 4-gram covering half the transcript.
    words = _normalize_sentence(text).split()
    if len(words) >= 20:
        grams = Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
        gram, n = grams.most_common(1)[0]
        if n >= 4 and (n * 4) / len(words) >= 0.5:
            return f'the phrase "{" ".join(gram)}" repeats {n} times'
        if _compression_ratio(text) > WHISPER_COMPRESSION_RATIO_THRESHOLD:
            return "the transcript is highly repetitive"
    return None


def _is_suspect_segment(seg: dict) -> bool:
    if seg.get("compression_ratio", 0.0) > WHISPER_COMPRESSION_RATIO_THRESHOLD:
        return True
    return (
        seg.get("no_speech_prob", 0.0) > WHISPER_NO_SPEECH_THRESHOLD
        and seg.get("avg_logprob", 0.0) < WHISPER_LOGPROB_THRESHOLD
    )


def transcribe_audio_checked(audio_path: str, language: str | None = None) -> TranscriptResult:
    """Transcribe an extracted audio file and judge whether the result is trustworthy.

    `language` is left as None by default so Whisper auto-detects between
    English/Tagalog rather than forcing one - TikTok creators code-switch
    mid-sentence, and Whisper's own language-detection handles that better
    than a fixed setting would.

    condition_on_previous_text=False stops one hallucinated segment from
    being fed back as the prompt for the next (the main cause of loops), and
    segments Whisper itself flags as non-speech or overly repetitive are
    dropped before the remaining text is checked for repetition.
    """
    model = get_whisper_model()
    result = model.transcribe(
        audio_path,
        language=language,
        fp16=False,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
        logprob_threshold=WHISPER_LOGPROB_THRESHOLD,
        compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
    )
    segments = result.get("segments") or []
    if segments:
        text = clean_text(" ".join(s.get("text", "") for s in segments if not _is_suspect_segment(s)))
    else:
        text = clean_text(result.get("text", ""))

    if not text or _SOUND_TAGS_ONLY_RE.match(text):
        return TranscriptResult(text="", reliable=False, issue="no speech detected (likely music only)")
    repetition = detect_repetition(text)
    if repetition:
        return TranscriptResult(text=text, reliable=False, issue=f"likely Whisper hallucination: {repetition}")
    return TranscriptResult(text=text, reliable=True)


def transcribe_audio(audio_path: str, language: str | None = None) -> str:
    """Transcribe an extracted audio file (wav/mp3/m4a) to plaintext."""
    return transcribe_audio_checked(audio_path, language).text


# ---- Rule layer on top of the model -------------------------------------
# The model is good at clear English/Tagalog sentiment but (a) labels plain
# product questions ("Anong shade?", "Hindi ba malagkit?") as negative,
# (b) misses Taglish negation ("walang hulas", "hindi mabigat" = praise), and
# (c) under-weights short beauty slang/emoji ("legit", "🥰🥰"). These rules add
# evidence to the model's probabilities rather than replacing them.

# Single words (matched as whole tokens, after squeezing "gandaaaa" -> "ganda").
POSITIVE_WORDS = {
    "ganda", "maganda", "napakaganda", "pinakamaganda", "pinakamagandang", "gandang",
    "nice", "love", "loved", "lovely", "bet", "gusto", "gustong", "favorite", "fave", "fav",
    "recommended", "recommend", "legit", "sulit", "panalo", "glowing", "glow", "kapit",
    "lasting", "tumagal", "waterproof", "smooth", "flawless", "perfect", "amazing", "galing",
    "magaling", "pretty", "beautiful", "gorgeous", "best", "blends", "poreless", "ayos",
    "goods", "bagay", "nakakaglow", "worth", "maayos",
    "good", "great", "excellent", "satisfied", "kashade", "ka-shade",
    "mura", "affordable",
    "tibay", "matibay", "weightless", "lightweight", "blendable", "natural", "fresh", "hiyang",
    "adik", "naadik", "nakakaadik", "adikan", "addicting", "addicted", "obsessed", "satisfying",
    "papakalma", "kumalma", "superb", "awesome", "wow", "solid", "swabe", "kinis", "nakakakinis",
}
NEGATIVE_WORDS = {
    "pangit", "panget", "ayoko", "ayaw", "sayang", "cakey", "ka-cakey", "oxidize", "oxidized",
    "oxidizes", "nag-oxidize", "breakout", "breakouts", "nag-breakout", "nakaka-breakout",
    "tagyawat", "ashy", "chalky", "patchy", "peeling", "flashback", "hulas", "malagkit",
    "lagkit", "mabigat", "bigat", "makapal", "kapal", "disappoint", "nakaka-disappoint",
    "disappointed", "disappointing", "worst", "hate", "bad", "waste", "scam", "fake", "ew",
    "eww", "yuck", "ugh", "erghh", "ergh", "crack", "nag-crack", "crease", "pabato", "panot", "nakaka-hulas",
    "pricey", "expensive", "madilim", "terrible", "awful",
    "dismayado", "nadismaya", "basag", "sira", "nasira", "nanghihinayang", "hinayang",
    "heavy", "broke", "broken",
    # mostly seen negated as praise: "hindi siya nagme-melt", "di natunaw"
    "melt", "melts", "nagme-melt", "nagmemelt", "natunaw", "natutunaw", "tunaw",
    "umitim", "nangitim", "kumupas", "nangati", "kati", "allergy", "allergic", "namula", "redness",
    "espasol", "mabaho", "baho", "defective", "refund", "nabawasan", "flaky", "itchy",
    "grey", "gray", "nag-grey", "cracked", "separated", "disappointing",
}
# Multi-word phrases, checked before single words; their words are then skipped.
POSITIVE_PHRASES = {
    "holy grail", "worth it", "second skin", "bet na bet", "natural finish", "no filter needed",
    "10/10", "money's worth", "non-cakey", "non cakey", "highly recommended",
    # praise that contains a complaint word
    "hulas proof", "hulas-proof", "anti hulas", "anti-hulas", "expensive looking", "expensive ang atake",
    "mukhang mamahalin", "hirap tanggalin", "hard to remove", "super stay",
    "shade match", "perfect match", 
    "such a steal", "a steal", "full coverage", "5 stars", "five stars", "five star", "5/5",
    "exceeded my expectations", "worth the hype", "match na match",
    "repurchase", "buy again", "bibili ulit", "oorder ulit", "uulit ako",
}
NEGATIVE_PHRASES = {
    "hindi ko na bibilhin", "di ko na bibilhin", "hindi ko na uulitin", "di ko na uulitin",
    "never again", "hindi recommended", "not recommended", "broke me out", "not worth it",
    "hindi worth it", "di worth it", "sayang pera", "waste of money",
    "ang mahal", "medyo mahal", "sobrang mahal", "masyadong mahal", "too expensive", "overpriced",
    "hindi ko ka-shade", "di ko ka-shade", "diko ka-shade", "hindi ko ka shade", "di ko ka shade",
    "diko ka shade", "too dark", "too light", "madilim sa akin", "maitim sa akin",
    "di ko shade match", "hindi shade match", "not a shade match", "no shade match",
    "mukhang mura", "looks cheap", "mali yung", "mali ang", "maling item", "wrong item", "wrong shade",
    # another product is better: "mas maganda YUNG/ANG X" (vs praise "mas maganda pa SA X")
    "mas maganda yung", "mas maganda pa yung", "mas maganda ang", "mas maganda pa ang",
    "mas okay pa yung", "mas okay yung", "mas ok pa yung", "mas better pa yung", "mas better yung",
    "not worth the hype", "di worth the hype", "hindi worth the hype", "matagal dumating",
    "tagal dumating", "ang tagal dumating", "sumakit ulo", "sumakit ang ulo", "masakit sa ulo",
    "hindi tumagal", "di tumagal", "hindi bagay", "di bagay", "hindi hiyang", "di hiyang", "masakit sa ilong",
}
_POLAR_PHRASES = sorted(
    [(p, 1) for p in POSITIVE_PHRASES] + [(p, -1) for p in NEGATIVE_PHRASES], key=lambda x: -len(x[0])
)
NEUTRAL_PHRASES = {
    "okay lang", "ok lang", "oks lang", "pwede na", "puwede na", "meh", "decent", "average",
    "sakto lang", "so-so", "not bad", "ganun ka-special", "wala namang special",
    # lukewarm / so-so
    "keri lang", "keri na", "okay naman", "ok naman", "oks naman", "pwede na rin", "medyo okay",
    "normal lang", "nothing special", "walang special", "no special", "it's okay", "its okay",
    "it's fine", "its fine", "fine lang", "mid lang", "so so", "hindi masama", "di masama",
    "same lang", "parang same", "walang pinagkaiba", "no difference", "depende", "50/50",
    "hit or miss", "neither", "mild lang", "not too bad", "not too good", "not too great", "not too special",
    # not tried yet / still testing
    "di ko pa natry", "hindi ko pa natry", "di ko pa na-try", "hindi ko pa na-try",
    "di ko pa nagagamit", "hindi ko pa nagagamit", "di ko pa nasusubukan", "hindi ko pa nasusubukan",
    "haven't tried", "not yet tried", "kakadating lang", "kararating lang", "just arrived",
    "will update", "update ko", "testing pa", "tinetesting ko pa",
    # will use it later / hoping it works
    "gagamitin ko", "gagamitin ko pa", "i still try", "will try", "i-try ko", "try ko pa",
    "i hope", "hopefully", "sana tama", "sana bagay", "sana hiyang",
    # about the commenter, not the product ("Hindi ako magaling mag make-up")
    "hindi ako magaling", "di ako magaling", "not good at makeup", "i'm not good at", "im not good at",
}
POSITIVE_EMOJI = set("😍🥰❤💕💖💗💓😘🤩😊☺😁😄😃👍👌💯🔥✨🙌👏🤍🧡💛💚💙💜❣")
NEGATIVE_EMOJI = set("😡😠🤬👎🤮🤢💔😞😔😒🙄😤😩😫")

NEGATORS = {"hindi", "di", "without", "'di", "hnd", "hinde", "not", "no", "never", "wala", "walang",
            "wla", "dont", "don't", "doesnt", "doesn't", "isnt", "isn't", "wasnt", "wasn't"}
_NEGATION_WINDOW = 4  # a negator up to 4 words before a term flips its polarity

QUESTION_WORDS = {"ano", "anong", "saan", "san", "magkano", "paano", "pano", "pwede", "puwede",
                  "ilan", "sino", "kailan", "bakit", "alin", "what", "where", "how", "which",
                  "can", "does", "is", "are", "do"}
_QUESTION_PARTICLE_RE = re.compile(r"\b(ba|po ba|poba|bayan|ba yan|diba)\b", re.IGNORECASE)  # not "kaya": usually "so, that's why"

# SENTIMENT_RULES=1/0 forces the rule layer on/off. Default: on for the stock
# checkpoint (which needs it), off for the fine-tuned model, which learned
# these cues from data - see training/evaluate_app.py for the comparison.
USE_RULES = os.getenv("SENTIMENT_RULES", "0" if IS_FINETUNED else "1") == "1"

LEXICON_WEIGHT = 0.4  # added probability mass per net lexicon vote (max 3 votes)
NEUTRAL_WEIGHT = 0.6
QUESTION_WEIGHT = 1.5

_TOKEN_RE = re.compile(r"[a-z0-9'/-]+")
_SQUEEZE_RE = re.compile(r"([a-z])\1{2,}")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*")


def _is_question(sentence: str) -> bool:
    s = sentence.strip()
    if s.endswith("?"):
        return True
    words = _TOKEN_RE.findall(s)
    if words and words[0] in QUESTION_WORDS and len(words) <= 12:
        return True
    return bool(_QUESTION_PARTICLE_RE.search(s))


def _lexicon_votes(text: str) -> tuple[int, int, int, list]:
    """Count (positive, negative, neutral) cues in the non-question sentences."""
    lowered = _SQUEEZE_RE.sub(r"\1", text.lower())
    pos = neg = neu = 0
    terms = []

    sentences = [s for s in _SENTENCE_RE.findall(lowered) if s.strip()] or [lowered]
    statements = [s for s in sentences if not _is_question(s)]

    for sent in statements:
        for phrase in sorted(NEUTRAL_PHRASES, key=len, reverse=True):
            if phrase in sent:
                neu += 1
                terms.append(phrase)
                sent = sent.replace(phrase, " ")  # "not bad" isn't also a negated "bad"
        for phrase, polarity in _POLAR_PHRASES:  # longest first: "hindi worth it" before "worth it"
            if phrase in sent:
                pos, neg = (pos + 1, neg) if polarity > 0 else (pos, neg + 1)
                terms.append(phrase)
                sent = sent.replace(phrase, " ")

        # Negation stops at a comma/semicolon: in "hindi siya nagme-melt,
        # sobrang tibay" the "hindi" doesn't reach "tibay".
        for chunk in re.split(r"[,;]", sent):
            tokens = _TOKEN_RE.findall(chunk)
            for i, tok in enumerate(tokens):
                polarity = 1 if tok in POSITIVE_WORDS else -1 if tok in NEGATIVE_WORDS else 0
                if not polarity:
                    continue
                if any(t in NEGATORS for t in tokens[max(0, i - _NEGATION_WINDOW):i]):
                    polarity = -polarity
                    terms.append(f"not {tok}")
                else:
                    terms.append(tok)
                pos, neg = (pos + 1, neg) if polarity > 0 else (pos, neg + 1)

    # Emoji count once per kind, wherever they appear (incl. in questions).
    if any(ch in POSITIVE_EMOJI for ch in text):
        pos += 1
    if any(ch in NEGATIVE_EMOJI for ch in text):
        neg += 1
    return pos, neg, neu, terms


def _only_questions(text: str) -> bool:
    sentences = [s for s in _SENTENCE_RE.findall(text.lower()) if _TOKEN_RE.search(s)]
    return bool(sentences) and all(_is_question(s) for s in sentences)


def _combine(scores: dict, text: str) -> tuple[dict, list]:
    """Blend the model's distribution with lexicon, negation and question cues."""
    pos, neg, neu, terms = _lexicon_votes(text)
    combined = dict(scores)
    net = max(-3, min(3, pos - neg))
    if net > 0:
        combined["positive"] += LEXICON_WEIGHT * net
    elif net < 0:
        combined["negative"] += LEXICON_WEIGHT * -net
    if neu:
        combined["neutral"] += NEUTRAL_WEIGHT
    # A pure question ("Anong shade?", "Hindi ba malagkit?") is an inquiry,
    # not an opinion - unless it carries its own emoji sentiment.
    has_emoji = any(ch in POSITIVE_EMOJI or ch in NEGATIVE_EMOJI for ch in text)
    if _only_questions(text) and not has_emoji:
        combined["neutral"] += QUESTION_WEIGHT
    total = sum(combined.values()) or 1.0
    return {k: v / total for k, v in combined.items()}, terms


# ---- Mixed opinions -> neutral -------------------------------------------
# A comment that praises one thing and criticizes another ("Ang ganda ng
# coverage pero nag-oxidize") is neutral overall. The model tends to side
# with whichever half sounds stronger, so each clause is scored on its own:
# a clearly positive clause plus a clearly negative one makes the comment
# neutral. SENTIMENT_MIXED_NEUTRAL=0 turns this off.
MIXED_AS_NEUTRAL = os.getenv("SENTIMENT_MIXED_NEUTRAL", "1") == "1"

_CLAUSE_SPLIT_RE = re.compile(
    r"[.!?;\n]+|,?\s+\b(?:but|pero|kaso|kaya lang|although|though|however|yet|whereas|except|"
    r"sadly|unfortunately|sayang lang|ang problema|problem is)\b,?\s*",
    re.IGNORECASE,
)
# Clauses that only state an expectation or a hope ("Akala ko maganda, pero
# ...", "sana maganda rin") aren't an opinion of the product, so they don't
# count as the "other side".
_EXPECTATION_RE = re.compile(
    r"\b(?:akala ko|kala ko|feeling ko|i thought|i expected|expected ko|sabi nila|they said)\b"
    r"|^(?:at first|sana|i hope|hopefully)\b",  # start only: "sana all" is praise slang
    re.IGNORECASE,
)


def _opinion_clauses(text: str) -> tuple[list, bool]:
    """(clauses worth scoring separately, whether an expectation clause was dropped).

    Normally [] unless there are 2+ clauses. After a dropped expectation
    ("Akala ko maganda, pero nag-crack") a single remaining clause is kept:
    it's the actual verdict.
    """
    clauses, dropped_expectation = [], False
    for part in _CLAUSE_SPLIT_RE.split(text):
        part = (part or "").strip(" ,")
        if len(_TOKEN_RE.findall(part.lower())) < 2 or _is_question(part):
            continue
        if _EXPECTATION_RE.search(part):
            dropped_expectation = True
            continue
        clauses.append(part)
    if dropped_expectation:
        return clauses, True
    return (clauses if len(clauses) >= 2 else []), False


LEXICON_CLAUSE_CONFIDENCE = 0.85  # confidence given to a clause decided by the lexicon


def _clause_polarity(clause: str, scores: dict) -> tuple[str | None, float]:
    """("positive"/"negative"/None, confidence) for one clause.

    Only a known praise/complaint expression (negation-aware) makes a clause
    count. The model's score for a fragment cut out of its comment is too
    erratic to decide on its own ("Sobrang dismayado ako" alone scores 0.94
    positive, "nag-crack at ang kapal sa mukha" 0.97 positive); it only sets
    the confidence.
    """
    pos, neg, _neu, _terms = _lexicon_votes(clause)
    if pos == neg:
        return None, 0.0
    polarity = "positive" if pos > neg else "negative"
    return polarity, max(scores[polarity], LEXICON_CLAUSE_CONFIDENCE)


MIXED_BALANCE = 0.5  # weaker side needs more than half as many clauses as the stronger one


def _mixed_confidence(polarities: list) -> float | None:
    """Confidence that the comment is mixed, or None.

    Mixed = both a positive and a negative clause, roughly balanced. One
    caveat in an otherwise glowing review ("...although a bit pricey, very
    worth it") stays positive - that's how the project's own labeled data
    annotates such comments.
    """
    counts = {p: sum(q == p for q, _ in polarities) for p in ("positive", "negative")}
    if not all(counts.values()) or min(counts.values()) <= MIXED_BALANCE * max(counts.values()):
        return None
    return min(max(c for q, c in polarities if q == p) for p in counts)


def _to_scores(raw: list) -> dict:
    scores = {}
    for r in raw:
        norm = _LABEL_MAP.get(r["label"].strip().lower(), r["label"].strip().lower())
        scores[norm] = r["score"]
    for label in SENTIMENT_LABELS:
        scores.setdefault(label, 0.0)
    return scores


# ---- Questions and uncertain calls -> neutral ----------------------------
# Product questions ("Nag-oxidize po ba?", "Hindi ba malagkit?", "FDA
# approved?") mention a problem word, and the model reads that as a
# complaint. A question with no praise/complaint words or emoji around it is
# an inquiry, so it's neutral. "Bakit/why" questions are usually rhetorical
# complaints ("Bakit palagi walang COD?"), so the model decides those.
QUESTIONS_AS_NEUTRAL = os.getenv("SENTIMENT_QUESTIONS_NEUTRAL", "1") == "1"
# A NEUTRAL_PHRASES hit with no praise/complaint word is neutral.
NEUTRAL_PHRASES_AS_NEUTRAL = os.getenv("SENTIMENT_NEUTRAL_PHRASES", "1") == "1"
_RHETORICAL_RE = re.compile(r"^\W*(?:bakit|bat|why)\b", re.IGNORECASE)
# A negative call this unsure, with no complaint word behind it, is neutral.
LOW_CONFIDENCE_NEGATIVE = float(os.getenv("SENTIMENT_LOW_CONFIDENCE_NEGATIVE", "0.55"))
# Short comments whose words are all praise (or all complaint) but which the
# model calls the opposite polarity: trust the words.
SHORT_COMMENT_WORDS = int(os.getenv("SENTIMENT_SHORT_OVERRIDE_WORDS", "4"))
# A neutral call below this confidence yields to 2+ praise words (and no complaint).
UNSURE_NEUTRAL = float(os.getenv("SENTIMENT_UNSURE_NEUTRAL", "0.8"))
# A negative call yields to 2+ praise words with no complaint word.
POSITIVE_WORDS_OVERRIDE = os.getenv("SENTIMENT_POSITIVE_OVERRIDE", "1") == "1"


# "Ang + adjective" is a Tagalog exclamation ("ANG GANDA???" = how pretty!),
# not a question, however many question marks it has.
_EXCLAMATION_RE = re.compile(
    r"\bang\s+(?:ganda|gaganda|pangit|panget|galing|sulit|bango|kinis|gaan|perfect|cute)", re.IGNORECASE
)
_ALL_EMOJI = POSITIVE_EMOJI | NEGATIVE_EMOJI


def _word_votes(text: str) -> tuple[int, int]:
    """Lexicon (positive, negative) votes from words/phrases only - no emoji.

    Emoji are too context-dependent to override the model on their own
    ("I want this 😫" is longing, not a complaint).
    """
    pos, neg, _neu, _terms = _lexicon_votes("".join(ch for ch in text if ch not in _ALL_EMOJI))
    return pos, neg


def _is_inquiry(text: str) -> bool:
    sentences = [s for s in _SENTENCE_RE.findall(text) if _TOKEN_RE.search(s.lower())] or [text]
    questions = [s for s in sentences if _is_question(s)]
    if not questions or any(
        _RHETORICAL_RE.match(q) or (_EXCLAMATION_RE.search(q) and not _QUESTION_PARTICLE_RE.search(q))
        for q in questions
    ):  # "Ang ganda po ba sa oily skin?" is still a question
        return False
    pos, neg, _neu, _terms = _lexicon_votes(text)  # statements + emoji only
    return pos == 0 and neg == 0


def _with_winner(scores: dict, label: str, confidence: float) -> dict:
    """Make `label` win with `confidence`, keeping the others' relative sizes."""
    rest = sum(v for k, v in scores.items() if k != label) or 1.0
    return {k: confidence if k == label else (1 - confidence) * v / rest for k, v in scores.items()}


def _neutral_overrides(scores: dict, text: str) -> dict:
    top = max(scores, key=scores.get)
    if QUESTIONS_AS_NEUTRAL and top != "neutral" and _is_inquiry(text):
        return _with_winner(scores, "neutral", max(scores["neutral"], LEXICON_CLAUSE_CONFIDENCE))
    pos, neg, neu, _terms = _lexicon_votes(text)
    if NEUTRAL_PHRASES_AS_NEUTRAL and top != "neutral" and neu and not pos and not neg:
        # "Keri lang, pwede na", "Di ko pa nagagamit" - lukewarm or not tried yet, not too much to praise or complain about. The model often calls these negative, so override, sobrang basic lang. The lexicon is more reliable here than the model, and the model's score is often low confidence anyway.
        # with no praise/complaint word or emoji anywhere.
        return _with_winner(scores, "neutral", max(scores["neutral"], LEXICON_CLAUSE_CONFIDENCE))
    if top == "neutral" and scores["neutral"] < UNSURE_NEUTRAL and not neu and not _is_inquiry(text):
        # An unsure neutral call on clearly positive wording ("Mas maganda pa sa
        # mahal na brand, sulit na sulit"): the words decide. Positive only -
        # complaint words also appear in retractions ("Binabawi ko yung
        # pagiging patchy na review ko").
        wpos, wneg = _word_votes(text)
        if wpos >= 2 and wneg == 0:
            return _with_winner(scores, "positive", LEXICON_CLAUSE_CONFIDENCE)
        # No negative counterpart: complaint words repeated as information
        # ("perfect match at one swipe but eventually it oxidizes ... saw a
        # vid that oxidizes so I chose lighter") are labeled neutral in the data.
    if POSITIVE_WORDS_OVERRIDE and top == "negative" and not _is_inquiry(text):
        # Clear praise the model reads as a complaint ("Kahit pawisin ako, hindi
        # siya nagme-melt, sobrang tibay"): 2+ praise words, no complaint word.
        wpos, wneg = _word_votes(text)
        if wpos >= 2 and wneg == 0:
            return _with_winner(scores, "positive", LEXICON_CLAUSE_CONFIDENCE)
    if top == "negative" and scores["negative"] < LOW_CONFIDENCE_NEGATIVE and neg == 0:
        return _with_winner(scores, "neutral", max(scores["neutral"], 1 - scores["negative"]))
    if len(_TOKEN_RE.findall(text.lower())) <= SHORT_COMMENT_WORDS and not _is_inquiry(text):
        wpos, wneg = _word_votes(text)
        if top == "negative" and wpos > 0 and wneg == 0:
            return _with_winner(scores, "positive", LEXICON_CLAUSE_CONFIDENCE)
        if top == "positive" and wneg > 0 and wpos == 0:
            return _with_winner(scores, "negative", LEXICON_CLAUSE_CONFIDENCE)
        if top == "neutral" and wpos > 0 and wneg == 0 and not neu and not _only_questions(text):
            # "Looks like second skin": short praise the model calls neutral.
            return _with_winner(scores, "positive", LEXICON_CLAUSE_CONFIDENCE)
    return scores


# ---- Compliments on the creator -> neutral -------------------------------
# "Ganda ng skin mo!!!", "Ang ganda mo ng skin nyo, ang kinis!" praise the
# creator's own skin/face, not the foundation, so they're neutral. They stay
# positive when the comment credits the product ("ang ganda ng skin mo kapag
# gamit mo yang foundation", "nakakakinis siya tingnan", "poreless ka").
PERSONAL_COMPLIMENT_AS_NEUTRAL = os.getenv("SENTIMENT_PERSONAL_COMPLIMENT_NEUTRAL", "1") == "1"
_PERSONAL_TARGET_RE = re.compile(
    r"\b(?:skin|balat|kutis|face|mukha|muka|fes)\s+(?:mo|nyo|niyo|ninyo|ni\s+\w+)\b"
    r"|\b(?:ganda|gandang|kinis|makinis|pretty|cute|blooming|glowing|glowy)\s+(?:mo|nyo|niyo|ninyo|ka)\b"
)
_PRODUCT_CREDIT_RE = re.compile(
    r"\b(?:foundation|pundasyon|product|produkto|cushion|concealer|primer|powder|makeup|make-up|brand"
    r"|gamit|ginamit|gumamit|gamitin|ginagamit|gagamitin|suot|nilagay|apply|inapply|applied"
    r"|using|use|used|wearing|thanks to|because of|dahil sa|nakakakinis|nakakaganda|nakakaglow"
    r"|poreless|coverage|shade|finish|formula|blend|blends|blendable|tint"
    r"|siya|sya|niya|nya|ito|it|this)\b"  # "it/siya" = the product: "tumatagal siya sa skin mo"
)
PERSONAL_COMPLIMENT_MAX_WORDS = 12  # longer comments are reviews, not a quick compliment


# Praise for what the product does to the skin. The model misreads some of
# these ("nakakakinis" looks like "nakakainis" = annoying), so they decide.
_PRODUCT_EFFECT_RE = re.compile(r"\b(?:nakakakinis|nakakaganda|nakakaglow|poreless)\b")
# "Parang si Liza", "kamukha mo si Anne" - the look is compared to a
# celebrity, which is praise.
_LOOKALIKE_RE = re.compile(
    r"^\W*(?:parang|mukhang|kamukha|kahawig)(?:\s+(?:mo|nyo|niyo|niya|nya|ka))?\s+(?:na\s+)?si\s+\w+"
)


def _skin_compliment_label(text: str) -> str | None:
    """"neutral" for a compliment on the creator's own skin/face, "positive"
    when the compliment credits the product, else None (no change)."""
    lowered = _SQUEEZE_RE.sub(r"\1", text.lower())
    personal = bool(_PERSONAL_TARGET_RE.search(lowered))
    wpos, wneg = _word_votes(text)
    if (
        personal
        and not _PRODUCT_CREDIT_RE.search(lowered)
        and wneg == 0
        and len(_TOKEN_RE.findall(lowered)) <= PERSONAL_COMPLIMENT_MAX_WORDS
    ):
        return "neutral"
    if (personal or _PRODUCT_EFFECT_RE.search(lowered)) and not _is_inquiry(text) and wpos > 0 and wneg == 0:
        return "positive"
    if _LOOKALIKE_RE.search(lowered) and wneg == 0 and not _only_questions(text):
        return "positive"
    return None


def relevance_check(text: str, product: str | None = None) -> tuple[bool, str]:
    """(is_relevant, reason): whether a comment is about the foundation product.

    See relevance.py; the sentiment lexicon above counts as product opinion.
    """
    return relevance.check(
        text,
        product,
        opinion_words=POSITIVE_WORDS | NEGATIVE_WORDS,
        opinion_phrases=tuple(POSITIVE_PHRASES | NEGATIVE_PHRASES | NEUTRAL_PHRASES),
        opinion_emoji=_ALL_EMOJI,
    )


def split_relevant(texts: Iterable[str], product: str | None = None) -> tuple[list, list]:
    """Split comments into (relevant texts, [{"comment", "reason"}] skipped ones)."""
    kept, skipped = [], []
    for t in texts:
        ok, reason = relevance_check(t, product)
        if ok:
            kept.append(t)
        else:
            skipped.append({"comment": clean_text(t) or t, "reason": reason})
    return kept, skipped


def classify_raw(texts: Iterable[str]) -> list[SentimentResult]:
    """Classify a batch of strings, returning the detailed SentimentResult form."""
    cleaned = [clean_text(t) for t in texts if t and t.strip()]
    if not cleaned:
        return []

    clf = get_classifier()
    raw_batch = clf(cleaned, truncation=True, max_length=MAX_LENGTH, batch_size=16)

    # Score every multi-clause comment's clauses in one extra batch.
    split = [_opinion_clauses(t) if MIXED_AS_NEUTRAL else ([], False) for t in cleaned]
    clauses = [cs for cs, _ in split]
    flat = [c for cs in clauses for c in cs]
    flat_polarities = iter(
        [_clause_polarity(c, _to_scores(r)) for c, r in zip(flat, clf(flat, truncation=True, max_length=MAX_LENGTH, batch_size=16))]
        if flat else []
    )

    results = []
    for text, raw, text_clauses, (_, after_expectation) in zip(cleaned, raw_batch, clauses, split):
        scores = _to_scores(raw)

        if USE_RULES:
            scores, terms = _combine(scores, text)
        else:
            terms = _lexicon_votes(text)[3]  # still reported as matched_terms

        polarities = [next(flat_polarities) for _ in text_clauses]
        mixed = _mixed_confidence(polarities)
        verdicts = {p for p, _ in polarities if p}
        if mixed is None and after_expectation and len(verdicts) == 1:
            # "Akala ko maganda, pero nag-crack" - the clauses after the expectation decide.
            verdict = verdicts.pop()
            conf = max(c for p, c in polarities if p == verdict)
            scores = {k: conf if k == verdict else (1 - conf) / 2 for k in SENTIMENT_LABELS}
        if mixed is not None:
            polar = (scores["positive"] + scores["negative"]) or 1.0
            scores = {
                "neutral": mixed,
                "positive": (1 - mixed) * scores["positive"] / polar,
                "negative": (1 - mixed) * scores["negative"] / polar,
            }
        else:
            scores = _neutral_overrides(scores, text)
        skin_label = _skin_compliment_label(text) if PERSONAL_COMPLIMENT_AS_NEUTRAL else None
        if skin_label:
            scores = _with_winner(scores, skin_label, max(scores[skin_label], LEXICON_CLAUSE_CONFIDENCE))

        sentiment = max(scores, key=scores.get)
        results.append(
            SentimentResult(
                text=text,
                sentiment=sentiment,
                score=round(scores[sentiment], 4),
                scores={k: round(v, 4) for k, v in scores.items()},
                matched_terms=tuple(sorted(set(terms))),
            )
        )
    return results


def classify_one(text: str) -> SentimentResult | None:
    results = classify_raw([text])
    return results[0] if results else None


def classify(texts):
    """Classify a list of comment strings for the REST API / database layer.

    Returns a list of dicts: {"comment": str, "sentiment": str, "score": float}
    - the shape backend/app.py and database.py already expect.
    """
    results = classify_raw(texts)
    return [
        {"comment": r.text, "sentiment": r.sentiment, "score": r.score}
        for r in results
    ]
