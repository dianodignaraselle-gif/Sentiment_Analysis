import re
import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path

DB_PATH = Path(__file__).parent / "dataset.db"

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "for", "with", "this", "that",
    "it", "its", "i", "you", "your", "my", "me", "we", "they", "he", "she",
    "at", "as", "so", "if", "not", "no", "just", "very", "really", "too",
    "than", "then", "there", "here", "have", "has", "had", "will", "would",
    "can", "could", "do", "does", "did", "im", "its", "it's", "after", "few",
    "all", "up", "out", "about", "into", "over", "also", "don't", "doesn't",
    # Tagalog / Taglish function words, so code-switched comments don't
    # flood the word cloud with particles instead of actual opinion words.
    "ang", "mga", "ng", "nang", "sa", "na", "at", "ay", "ito", "nito",
    "niyan", "niyon", "iyon", "iyan",
    "dito", "doon", "diyan", "ako", "ikaw", "ka", "siya", "kami", "tayo",
    "kayo", "sila", "ko", "mo", "niya", "namin", "natin", "ninyo", "nila",
    "akin", "iyo", "kanya", "atin", "inyo", "kanila", "hindi", "oo", "opo",
    "wala", "meron", "mayroon", "din", "rin", "lang", "po", "ho", "naman",
    "kasi", "kaya", "para", "pero", "kung", "yung", "yun", "nga", "ba",
    "raw", "daw", "muna", "pala", "talaga",
}


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comment TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            score REAL NOT NULL,
            product TEXT,
            comment_date TEXT NOT NULL,
            likes INTEGER,
            owner TEXT
        )
        """
    )
    conn.commit()
    # Owners used to be stored with whatever casing/whitespace the user typed
    # at login; the frontend now normalizes to trimmed lowercase before
    # sending requests, so existing rows need the same normalization or
    # they'd become unreachable under the new key.
    conn.execute(
        "UPDATE comments SET owner = TRIM(LOWER(owner)) "
        "WHERE owner IS NOT NULL AND owner != TRIM(LOWER(owner))"
    )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) AS c FROM comments").fetchone()["c"]
    conn.close()
    if count == 0:
        _seed()


def insert_comments(rows):
    """rows: list of dicts with comment, sentiment, score, product, likes(optional), owner(optional)"""
    conn = get_conn()
    today = date.today().isoformat()
    conn.executemany(
        """
        INSERT INTO comments (comment, sentiment, score, product, comment_date, likes, owner)
        VALUES (:comment, :sentiment, :score, :product, :comment_date, :likes, :owner)
        """,
        [
            {
                "comment": r["comment"],
                "sentiment": r["sentiment"],
                "score": r["score"],
                "product": r.get("product") or None,
                "comment_date": r.get("comment_date", today),
                "likes": r.get("likes"),
                "owner": r.get("owner") or None,
            }
            for r in rows
        ],
    )
    conn.commit()
    conn.close()


def get_dataset(page=1, page_size=20, search="", product="", owner=None):
    conn = get_conn()
    where = ["owner = :owner"]
    params = {"owner": owner}
    if search:
        where.append("comment LIKE :search")
        params["search"] = f"%{search}%"
    if product:
        where.append("product = :product")
        params["product"] = product
    clause = f"WHERE {' AND '.join(where)}"

    total = conn.execute(
        f"SELECT COUNT(*) AS c FROM comments {clause}", params
    ).fetchone()["c"]

    offset = (page - 1) * page_size
    params_paged = dict(params, limit=page_size, offset=offset)
    rows = conn.execute(
        f"""
        SELECT id, comment, sentiment, score, product, comment_date, likes
        FROM comments {clause}
        ORDER BY id DESC
        LIMIT :limit OFFSET :offset
        """,
        params_paged,
    ).fetchall()
    conn.close()

    return {
        "rows": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def get_products(owner=None):
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT DISTINCT product FROM comments
        WHERE product IS NOT NULL AND product != '' AND owner = :owner
        ORDER BY product
        """,
        {"owner": owner},
    ).fetchall()
    conn.close()
    return [r["product"] for r in rows]


def get_all_rows(owner=None):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, comment, sentiment, score, product, comment_date, likes FROM comments WHERE owner = :owner ORDER BY id",
        {"owner": owner},
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats(owner=None):
    conn = get_conn()
    rows = conn.execute(
        "SELECT comment, sentiment, product, comment_date FROM comments WHERE owner = :owner",
        {"owner": owner},
    ).fetchall()
    conn.close()

    total = len(rows)
    dist = Counter(r["sentiment"] for r in rows)
    positive = dist.get("positive", 0)
    neutral = dist.get("neutral", 0)
    negative = dist.get("negative", 0)

    def pct(n):
        return round((n / total) * 100, 2) if total else 0.0

    distribution = {
        "positive": {"count": positive, "pct": pct(positive)},
        "neutral": {"count": neutral, "pct": pct(neutral)},
        "negative": {"count": negative, "pct": pct(negative)},
        "total": total,
    }

    by_date = {}
    for r in rows:
        d = r["comment_date"]
        by_date.setdefault(d, Counter())[r["sentiment"]] += 1
    over_time = []
    for d in sorted(by_date.keys()):
        c = by_date[d]
        day_total = sum(c.values())
        over_time.append(
            {
                "date": d,
                "positive": round((c.get("positive", 0) / day_total) * 100, 2),
                "neutral": round((c.get("neutral", 0) / day_total) * 100, 2),
                "negative": round((c.get("negative", 0) / day_total) * 100, 2),
            }
        )

    by_product = {}
    for r in rows:
        p = r["product"] or "Unspecified"
        by_product.setdefault(p, Counter())[r["sentiment"]] += 1
    products = []
    for p, c in by_product.items():
        products.append(
            {
                "product": p,
                "positive": c.get("positive", 0),
                "neutral": c.get("neutral", 0),
                "negative": c.get("negative", 0),
            }
        )

    words = Counter()
    for r in rows:
        for w in re.findall(r"[a-zA-Z']+", r["comment"].lower()):
            if len(w) > 2 and w not in STOPWORDS:
                words[w] += 1
    word_cloud = [{"text": w, "count": c} for w, c in words.most_common(30)]

    return {
        "distribution": distribution,
        "over_time": over_time,
        "by_product": products,
        "word_cloud": word_cloud,
    }


_SEED_COMMENTS = [
    ("Love this foundation! It's lightweight and feels so natural on skin.", "positive", "Maybelline Fit Me", 128),
    ("Oxidizes after a few hours. Not worth the price.", "negative", "Maybelline Fit Me", 45),
    ("Great coverage and long lasting! Definitely recommend.", "positive", "L'Oreal Infallible", 98),
    ("It's okay, not too good but not bad either.", "neutral", "SACE LADY Airy", 23),
    ("Makes my skin look flawless! Will repurchase for sure.", "positive", "Issy & Co. Active", 76),
    ("Perfect for daily use. Doesn't make my skin look cakey.", "positive", "Happy Skin Air Blur", 61),
    ("Broke me out after two days of use, very disappointed.", "negative", "Maybelline Fit Me", 34),
    ("Decent coverage for the price, nothing special though.", "neutral", "L'Oreal Infallible", 19),
    ("Amazing matte finish and it controls oil so well!", "positive", "SACE LADY Airy", 87),
    ("Shade range is limited, hard to find my match.", "negative", "Issy & Co. Active", 41),
    ("Blends easily and gives a natural glow.", "positive", "Happy Skin Air Blur", 55),
    ("Not bad, but I expected better lasting power.", "neutral", "Maybelline Fit Me", 12),
    ("Absolutely favoring this over my old foundation, so good!", "positive", "L'Oreal Infallible", 70),
    ("Too oily for my skin type, caused breakouts.", "negative", "SACE LADY Airy", 29),
    ("Good lightweight formula, wears comfortably all day.", "positive", "Issy & Co. Active", 64),
]


def _seed():
    import random

    random.seed(42)
    rows = []
    for i in range(120):
        comment, sentiment, product, likes = _SEED_COMMENTS[i % len(_SEED_COMMENTS)]
        day = random.randint(1, 30)
        rows.append(
            {
                "comment": comment,
                "sentiment": sentiment,
                "score": round(random.uniform(0.6, 0.99), 4),
                "product": product,
                "comment_date": f"2024-05-{day:02d}",
                "likes": likes + random.randint(-10, 10),
            }
        )
    insert_comments(rows)
