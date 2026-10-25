"""Decides whether a comment is about the foundation product at all.

Comments under a foundation video include plenty that aren't about the
product: questions about the creator's hair or necklace ("ano color ng hair
mo mi?"), personal stories ("di ako pinayagan ni papa bumili"), follow /
engagement spam, laughter and stray emoji. Those aren't positive, neutral or
negative opinions of the product, so the app drops them before the sentiment
model runs - they are never classified, stored, or counted in the charts.

The check is rule-based (no extra model to load) and leans toward keeping a
comment: anything that mentions the product, its use (shade, coverage,
oxidize, oily skin...), buying it, or carries a praise/complaint word stays.
Only comments with a clear off-topic signal, or too little text to be about
anything, are dropped. training/evaluate_relevance.py checks the rules
against the labeled datasets (all product comments) and known off-topic ones.
"""

from __future__ import annotations

import re

from preprocessing import clean_text

# Foundation / base-makeup vocabulary, matched as whole words after
# lowercasing and squeezing elongations ("gandaaa" -> "ganda").
PRODUCT_WORDS = {
    # the product and base makeup that goes with it
    "foundation", "foundations", "pundasyon", "concealer", "primer", "powder", "setting", "spray",
    "base", "bb", "cc", "cushion", "tint", "tinted", "spf", "makeup", "make-up",
    "product", "products", "item", "brand", "formula", "texture", "variant", "variation", "bottle",
    "bote", "pump", "packaging", "swatch", "sample", "samples", "sampol", "brush", "sponge", "puff",
    # shade matching
    "shade", "shades", "shed", "shaed", "kashade", "ka-shade", "undertone", "undertones", "skintone",
    "tone", "morena", "moreno", "mestiza", "tisay", "maputi", "mapuputi", "maputing", "olive",
    "acidic", "medium", "fair", "tan", "natural", "darker", "lighter", "beige", "ivory",
    # skin and how it wears
    "skin", "balat", "face", "mukha", "muka", "oily", "oil", "dry", "combination", "sensitive", "acne",
    "pimple", "pimples", "tagyawat", "eyebags", "pores", "scars", "marks", "pawisin", "pawis", "sweat",
    "coverage", "cover", "covered", "matte", "dewy", "glowy", "finish", "blend", "blending", "blendable",
    "apply", "inaply", "iapply", "oxidize", "oxidizes", "oxidized", "nag-oxidize", "nagooxidize",
    "nag-o-oxidize", "nagoxidize", "cakey", "nagcacakey", "nag-cacakey", "nagca-cakey", "hulas",
    "nahulas", "huhulas", "malagkit", "lagkit", "mabigat", "heavy", "lightweight", "waterproof",
    "water-proof", "transfer", "lasting", "long-lasting", "tumagal", "tinatagal", "kapit", "lapat",
    "white", "cast", "flashback", "gray", "grey", "patchy", "crease", "flawless", "smooth",
    "fda", "price", "presyo", "magkano", "howmuch", "mura", "mahal", "affordable", "pesos", "php",
    "shipping", "seller", "authentic", "restock", "restak", "repurchase", "sulit", "stock",
}
# Buying words count as a product mention, except in a personal story:
# "di ako pinayagan ni papa bumili" is about the commenter's dad.
BUY_WORDS = {
    "order", "mag-order", "oorder", "omorder", "orderin", "buy", "bought", "bili", "bumili",
    "bibili", "nabili", "makabili", "mapapabili", "checkout", "cod", "budol", "mabubudol", "mabudol",
}
PRODUCT_PHRASES = (
    "water proof", "sweat proof", "long lasting", "skin tone", "skin prep", "setting spray",
    "full coverage", "white cast", "cash on delivery", "add to cart", "yellow basket",
    "yellow vasket", "light weight", "how much", "how to order", "shade number", "shade no",
)
# Core words that keep a comment even when it also mentions an off-topic
# item or spam ("Bought this for my gf ... cakey makeup", "Link ng foundation").
CORE_WORDS = {
    "foundation", "pundasyon", "coverage", "undertone", "undertones", "oxidize", "oxidizes",
    "oxidized", "nag-oxidize", "nagoxidize", "cakey", "hulas", "concealer", "primer", "blend",
    "blendable",
    # makeup rubbing off on clothes: "Transfers on everything, ruined my white shirt"
    "transfer", "transfers", "mag-transfer", "nagtransfer", "dikit", "nadikit", "dumikit",
    "nadumihan", "mantsa", "stain", "stains",
}

# Things on the creator that aren't the product: clothes, accessories,
# hair, the background music.
OFF_TOPIC_WORDS = {
    "hair", "buhok", "kilay", "brow", "brows", "eyebrow", "eyebrows", "lips", "labi", "nails",
    "kuko", "necklace", "kwintas", "kwentas", "earrings", "hikaw", "ring", "bracelet", "watch",
    "relo", "specs", "glasses", "salamin", "pants", "shorts", "skirt", "dress", "damit", "outfit",
    "shirt", "blouse", "jacket", "bag", "shoes", "sapatos", "song", "kanta", "music", "background",
}
OFF_TOPIC_PHRASES = ()

# Other cosmetics and skincare - a review of a lotion or a lipstick isn't a
# foundation review, even under a foundation video.
OTHER_PRODUCT_WORDS = {
    # skincare / body care
    "lotion", "lotions", "serum", "serums", "sunscreen", "sunblock", "moisturizer", "moisturiser",
    "toner", "cleanser", "soap", "sabon", "shampoo", "conditioner", "deodorant", "deo", "perfume",
    "pabango", "cologne", "scrub", "exfoliant", "retinol", "gluta", "glutathione", "kojic",
    "rejuv", "rejuvenating", "toothpaste", "vitamins", "capsule", "capsules",
    # eye / lip / cheek makeup
    # not "maskara": in Tagalog it's also "mask" ("mukhang maskara" = cakey)
    "lipstick", "lipsticks", "lippie", "liptint", "lipgloss", "lipbalm", "mascara", "mascarra",
    "eyeliner", "eyeliners", "eyeshadow", "eyeshadows", "lashes", "glitter", "glitters",
    "blush", "blusher", "highlighter", "contour", "bronzer", "cutex",
}
OTHER_PRODUCT_PHRASES = (
    "lip tint", "lip gloss", "lip balm", "lip oil", "lip liner", "eye liner", "eye shadow",
    "eyebrow pencil", "brow pencil", "cheek tint", "nail polish", "face wash", "facial wash",
    "night cream", "day cream", "eye cream", "face cream", "body wash", "sheet mask", "face mask",
    "whitening soap", "sun block",
)
# Naming another product only to compare ("parang tinted sunscreen yung
# texture", "it's like wearing a liquid highlighter") or to pair it with the
# foundation ("ganda nito, sabayan mo ng serum spray") still describes the
# foundation. Bare "like" isn't a comparison ("I like this lotion").
_COMPARISON_RE = re.compile(
    r"\b(?:parang|mukhang|kagaya|kasing|katulad|than|mas|vs|it'?s like|its like|as if|"
    r"sabayan|isabay|sinabayan|pair|paired|partner|ipartner|combine|combined|with|kasama|under|ilalim)"
    r"\s+(?:\S+\s+){0,6}$"
)

# Follow-for-follow / engagement spam from other sellers and affiliates.
_SPAM_RE = re.compile(
    r"\b(?:follow(?:ers?| back)?|fback|f4f|engage(?:d|ment)?|affiliat(?:e|or)s?|support back|"
    r"auto follow|check my|visit my|my (?:latest )?post|subscribe|pa help)\b"
)
# Compliments to the creator, not the product ("ganda mo po", "you're pretty").
_CREATOR_RE = re.compile(
    r"\b(?:(?:ganda|gaganda|pretty|cute|pogi|sexy|kinis) (?:mo|nyo|ninyo|niyo|mu)|"
    r"(?:you(?:'re| are)|ur|you r) (?:so )?(?:pretty|beautiful|gorgeous|cute)|"
    r"(?:i like|love) (?:ur|your) (?!skin|shade|base|foundation|makeup))\b"
)
# How the commenter applies makeup ("ako rin finger din bet ko hahaha") is
# chit-chat about their own habit, not an opinion of the foundation.
APPLICATION_HABIT_WORDS = {
    "finger", "fingers", "fingertip", "fingertips", "daliri", "kamay", "hand", "hands",
}
# Vulgar / sexual comments are dropped unless they're about the product
# ("tangina ang ganda ng foundation" is still a review).
VULGAR_WORDS = {
    "puta", "putang", "putangina", "putanginamo", "tangina", "tanginamo", "tangna", "pakshet",
    "gago", "gaga", "ulol", "ulul", "bobo", "tanga", "inamo", "kantot", "kantutan", "pakantot",
    "iyot", "jakol", "chupa", "tite", "titi", "etits", "burat", "puke", "pekpek", "bilat", "kiffy",
    "tamod", "libog", "malibog", "manyak", "bastos", "suso", "dede", "boobs", "tits", "fuck",
    "fucking", "shit", "bitch", "dick", "pussy", "porn", "nude", "nudes", "horny", "fingering",
}
VULGAR_PHRASES = ("tang ina", "putang ina", "puta ka", "gago ka")
# Chit-chat with no product opinion: "first ma", "first mii", "is that
# Anne Curtis?", greetings and shout-out requests. Only applied when the
# comment has no product word and no praise/complaint word ("is that
# waterproof?" and "first time ko, ang ganda" are kept).
_FIRST_RE = re.compile(r"^\W*(?:first|1st|frist|fisrt|firstt|una|ako una|una ako)\b(?!\s+(?:time|try|use|impression))")
FIRST_MAX_WORDS = 4
_CHITCHAT_RE = re.compile(
    r"\b(?:(?:is that|is this|isn'?t that|is she|is he)\s+(?:si\s+)?[a-z]+|ikaw ba si|sino (?:to|yan|ito|siya|sya|po (?:to|yan))|"
    r"si .{1,30} ba (?:to|yan|ito|siya|sya)|"
    r"ano (?:pangalan|name)|what'?s (?:her|his|your|ur) name|name mo|"
    r"notice me|pa ?shout ?out|shout ?out|watching from|sino nanonood|who'?s watching|fyp)\b"
    # a bare greeting; "hello po nag o-oxidize po ba sya?" is a product question
    r"|^\W*(?:hi|hello|helo|hey|hai|good (?:morning|evening|afternoon))\b(?:\W+\w+){0,3}\W*$"
)
# The commenter's family / love life - a personal story, not a product opinion.
PERSONAL_WORDS = {
    "papa", "mama", "nanay", "tatay", "mommy", "daddy", "lola", "lolo", "jowa", "crush",
    "boyfriend", "girlfriend", "asawa", "anak", "kuya", "bunso",
}

# Short reactions that are still an opinion on the product. The sentiment
# lexicon from model_pipeline is added on top of these (see check()).
REACTION_WORDS = {
    "wow", "omg", "ok", "okay", "oks", "okey", "keri", "pwede", "puwede", "sakto", "meh", "goods",
    "try", "itry", "matry", "ma-try", "must", "want", "gusto", "need", "reco", "recommend",
    "recommended", "honest", "review", "legit", "totoo", "true", "real", "basic", "expectation",
}
_LAUGHTER_RE = re.compile(r"^(?:(?:ha|he|hi|hu|ah|eh)+h?|lol+|lmao+|xd+|charot|chariz)$")
_PLACEHOLDER_RE = re.compile(r"\[[^\]]{1,30}\]")  # TikTok's text emoji: "[Tears of joy]"
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")
_SQUEEZE_RE = re.compile(r"([a-z])\1{2,}")


def _normalize(text: str) -> str:
    text = _PLACEHOLDER_RE.sub(" ", clean_text(text).lower())
    return _SQUEEZE_RE.sub(r"\1", text)


def _has_any(tokens: set, text: str, words: set, phrases=()) -> bool:
    return bool(tokens & words) or any(p in text for p in phrases)


def _other_product(text: str) -> str | None:
    """The first non-foundation product the comment is about, if any
    (skipping ones only named as a comparison)."""
    terms = [re.escape(t) for t in sorted(OTHER_PRODUCT_PHRASES + tuple(OTHER_PRODUCT_WORDS), key=len, reverse=True)]
    for m in re.finditer(rf"\b(?:{'|'.join(terms)})\b", text):
        if not _COMPARISON_RE.search(text[: m.start()]):
            return m.group(0)
    return None


def check(
    text: str,
    product: str | None = None,
    opinion_words: set = frozenset(),
    opinion_phrases=(),
    opinion_emoji: set = frozenset(),
) -> tuple[bool, str]:
    """Return (is_relevant, reason). `product` is the product name the user
    entered, whose words count as product mentions. The opinion_* arguments
    let the caller supply its sentiment lexicon (praise/complaint words)."""
    norm = _normalize(text)
    tokens = _TOKEN_RE.findall(norm)
    token_set = set(tokens)
    words = [t for t in tokens if not t.isdigit()]

    has_emoji = any(ch in opinion_emoji for ch in text)
    if not words:
        if has_emoji:
            return True, "emoji reaction"
        return False, "no words to analyze"
    if all(_LAUGHTER_RE.match(w) for w in words):
        return False, "laughter only"

    product_words = {w for w in _TOKEN_RE.findall((product or "").lower()) if len(w) > 2}
    mentions_product = _has_any(token_set, norm, PRODUCT_WORDS | product_words, PRODUCT_PHRASES)
    core = bool(token_set & (CORE_WORDS | product_words))
    mentions_buying = bool(token_set & BUY_WORDS)

    if _SPAM_RE.search(norm) and not core:
        return False, "follow/engagement spam"
    other = _other_product(norm)
    if other and not core:
        return False, f"about another product ({other}), not the foundation"
    if _has_any(token_set, norm, OFF_TOPIC_WORDS, OFF_TOPIC_PHRASES) and not core:
        return False, "about something other than the foundation"
    if _CREATOR_RE.search(norm) and not mentions_product:
        return False, "compliment to the creator, not the product"
    if token_set & PERSONAL_WORDS and not mentions_product:
        return False, "personal story, not about the product"
    if _has_any(token_set, norm, VULGAR_WORDS, VULGAR_PHRASES) and not mentions_product:
        return False, "vulgar or offensive, not about the product"
    has_opinion = _has_any(token_set, norm, set(opinion_words), opinion_phrases)
    if not mentions_product and not has_opinion:
        if _FIRST_RE.match(norm) and len(words) <= FIRST_MAX_WORDS:
            return False, "\"first\" comment, not about the product"
        if _CHITCHAT_RE.search(norm):
            return False, "chit-chat with the creator, not about the product"
    if token_set & APPLICATION_HABIT_WORDS and not mentions_product:
        return False, "about how the commenter applies makeup, not the product"

    if mentions_product or mentions_buying:
        return True, "mentions the product or its use"
    if _has_any(token_set, norm, REACTION_WORDS | set(opinion_words), opinion_phrases) or has_emoji:
        return True, "reaction to the product"
    if len(words) >= 2:
        # Short sentences under a foundation video with no off-topic signal
        # ("pwede ba sa buntis", "age limit") are usually about the product.
        return True, "no off-topic signal"
    return False, "too short to be about the product"
