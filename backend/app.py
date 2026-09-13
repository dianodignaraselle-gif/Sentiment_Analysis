import csv
import io
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

import database
import model

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

database.init_db()


# ---------- static frontend ----------

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


# ---------- API ----------

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}
    comments = [c.strip() for c in data.get("comments", []) if c and c.strip()]
    product = (data.get("product") or "").strip() or None
    owner = (data.get("username") or "").strip() or None

    if not comments:
        return jsonify({"error": "No comments provided."}), 400
    if len(comments) > 200:
        return jsonify({"error": "Too many comments in a single request (max 200)."}), 400

    results = model.classify(comments)

    database.insert_comments(
        [
            {
                "comment": r["comment"],
                "sentiment": r["sentiment"],
                "score": r["score"],
                "product": product,
                "likes": None,
                "owner": owner,
            }
            for r in results
        ]
    )

    total = len(results)
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for r in results:
        counts[r["sentiment"]] += 1

    def pct(n):
        return round((n / total) * 100, 2) if total else 0.0

    summary = {
        "total": total,
        "product": product,
        "positive": {"count": counts["positive"], "pct": pct(counts["positive"])},
        "neutral": {"count": counts["neutral"], "pct": pct(counts["neutral"])},
        "negative": {"count": counts["negative"], "pct": pct(counts["negative"])},
    }

    return jsonify({"summary": summary, "results": results})


@app.route("/api/dataset", methods=["GET"])
def dataset():
    page = max(int(request.args.get("page", 1)), 1)
    page_size = min(max(int(request.args.get("page_size", 20)), 1), 200)
    search = request.args.get("search", "")
    product = request.args.get("product", "")
    owner = request.args.get("owner") or None
    return jsonify(database.get_dataset(page, page_size, search, product, owner))


@app.route("/api/dataset/products", methods=["GET"])
def dataset_products():
    owner = request.args.get("owner") or None
    return jsonify({"products": database.get_products(owner)})


@app.route("/api/dataset/export", methods=["GET"])
def export_csv():
    owner = request.args.get("owner") or None
    rows = database.get_all_rows(owner)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "comment", "sentiment", "score", "product", "date", "likes"])
    for r in rows:
        writer.writerow(
            [r["id"], r["comment"], r["sentiment"], r["score"], r["product"] or "", r["comment_date"], r["likes"] or ""]
        )
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="tiktok_sentiment_dataset.csv",
    )


@app.route("/api/stats", methods=["GET"])
def stats():
    owner = request.args.get("owner") or None
    return jsonify(database.get_stats(owner))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
