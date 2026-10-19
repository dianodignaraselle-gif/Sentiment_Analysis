import csv
import io
import os
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load secrets (e.g. MS_TOKEN for TikTokApi) before anything reads os.environ.
load_dotenv(Path(__file__).parent / ".env")

from flask import Flask, jsonify, request, send_file, send_from_directory  # noqa: E402
from flask_cors import CORS  # noqa: E402

import database  # noqa: E402
import model_pipeline as model  # noqa: E402
import scraper  # noqa: E402

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

database.init_db()

# ---------- TikTok link analysis: job registry ----------
# Each job's background poller (a live thread) can only live in this
# process's memory, but the job itself - URL, product, classified comments,
# stopped flag - is persisted in SQLite (database.tiktok_jobs). If the
# in-memory entry is gone (server restart, stale-job pruning), the next poll
# rebuilds it from the database and resumes polling instead of 404ing.
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL_SECONDS = 60 * 30  # stop orphaned pollers after 30 minutes of no polling
_REAL_POLL_INTERVAL = 15.0  # real TikTok scraping: slower, to avoid rate limits
_MOCK_POLL_INTERVAL = 5.0


def _prune_stale_jobs() -> None:
    cutoff = time.time() - _JOB_TTL_SECONDS
    with _JOBS_LOCK:
        stale = [jid for jid, job in _JOBS.items() if job["last_polled_at"] < cutoff]
        for jid in stale:
            if _JOBS[jid]["poller"]:
                _JOBS[jid]["poller"].stop()
            del _JOBS[jid]


def _janitor() -> None:
    while True:
        time.sleep(60)
        _prune_stale_jobs()


threading.Thread(target=_janitor, daemon=True).start()


def _start_poller(source, video_url, seen_ids=None):
    if getattr(source, "comments_error", None):
        return None  # nothing to poll - the error is reported to the page instead
    interval = _MOCK_POLL_INTERVAL if source.comments_simulated else _REAL_POLL_INTERVAL
    poller = scraper.CommentPoller(source, video_url, interval=interval, seen_ids=seen_ids)
    poller.start()
    return poller


def _get_or_restore_job(job_id):
    """Return the live job, rebuilding it from SQLite if this process lost it."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            return job

        record = database.get_tiktok_job(job_id)
        if record is None:
            return None

        comments_error = record["comments_error"]
        poller = None
        if not record["stopped"] and not comments_error:
            source = scraper.get_source(record["video_url"], use_mock=record["comments_simulated"])
            comments_error = getattr(source, "comments_error", None)
            seen = {c["id"] for c in record["comments"] if c.get("id")}
            poller = _start_poller(source, record["video_url"], seen_ids=seen)

        job = {
            "poller": poller,
            "video_url": record["video_url"],
            "product": record["product"],
            "owner": record["owner"],
            "comments": record["comments"],
            "comments_simulated": record["comments_simulated"],
            "comments_error": comments_error,
            "stopped": record["stopped"],
            "last_polled_at": time.time(),
        }
        _JOBS[job_id] = job
        return job


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


def _summarize(results):
    total = len(results)
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for r in results:
        counts[r["sentiment"]] += 1

    def pct(n):
        return round((n / total) * 100, 2) if total else 0.0

    return {
        "total": total,
        "positive": {"count": counts["positive"], "pct": pct(counts["positive"])},
        "neutral": {"count": counts["neutral"], "pct": pct(counts["neutral"])},
        "negative": {"count": counts["negative"], "pct": pct(counts["negative"])},
    }


@app.route("/api/tiktok/analyze", methods=["POST"])
def tiktok_analyze():
    """Kick off both pipelines for a pasted TikTok link.

    Pipeline A (audio -> Whisper -> XLM-RoBERTa) runs synchronously here,
    since TikTok clips are short; Pipeline B (live comment polling) is
    started on a background thread and keeps running after this responds.
    """
    data = request.get_json(force=True) or {}
    video_url = (data.get("url") or "").strip()
    product = (data.get("product") or "").strip() or None
    owner = (data.get("username") or "").strip() or None
    use_mock = bool(data.get("use_mock", False))

    if not video_url:
        return jsonify({"error": "Please paste a TikTok video link."}), 400
    if not scraper.is_tiktok_url(video_url):
        return jsonify({"error": "That doesn't look like a TikTok video link."}), 400

    _prune_stale_jobs()

    source = scraper.get_source(video_url, use_mock=use_mock)
    video_info = source.fetch_video_info(video_url)

    # Only trust the transcript if Whisper produced real, non-looping speech;
    # otherwise classify the caption alone and tell the page why.
    transcript_ok = bool(video_info.transcript) and not video_info.transcript_issue
    caption = video_info.caption or ""
    if transcript_ok:
        analysis_basis = "caption+transcript" if caption.strip() else "transcript"
        transcript_note = None
    else:
        analysis_basis = "caption" if caption.strip() else None
        if video_info.transcript:
            transcript_note = "Transcript unreliable - caption used" if analysis_basis else "Transcript unreliable - no caption to analyze"
        else:
            transcript_note = "No usable transcript - caption used" if analysis_basis else None
    video_text = model.build_video_text(caption, video_info.transcript if transcript_ok else "")
    video_result = model.classify_one(video_text) if video_text else None

    comments_simulated = getattr(source, "comments_simulated", False)
    comments_error = getattr(source, "comments_error", None)
    poller = _start_poller(source, video_url)

    job_id = uuid.uuid4().hex
    database.save_tiktok_job(job_id, video_url, product, owner, comments_simulated, comments_error)
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "poller": poller,
            "video_url": video_url,
            "product": product,
            "owner": owner,
            "comments": [],
            "comments_simulated": comments_simulated,
            "comments_error": comments_error,
            "stopped": False,
            "last_polled_at": time.time(),
        }

    video_payload = {
        "author": video_info.author,
        "caption": video_info.caption,
        "transcript": video_info.transcript,
        "transcript_error": video_info.transcript_error,
        "transcript_issue": video_info.transcript_issue,
        "transcript_reliable": transcript_ok,
        "transcript_note": transcript_note,
        "analysis_basis": analysis_basis,
        "sentiment": video_result.sentiment if video_result else None,
        "score": video_result.score if video_result else None,
        "matched_terms": list(video_result.matched_terms) if video_result else [],
    }

    return jsonify(
        {
            "job_id": job_id,
            "video": video_payload,
            "summary": _summarize([]),
            "results": [],
            "comments_simulated": comments_simulated,
            "comments_error": comments_error,
        }
    )


@app.route("/api/tiktok/updates/<job_id>", methods=["GET"])
def tiktok_updates(job_id):
    """Drain whatever new comments Pipeline B queued, classify them, return the running totals."""
    job = _get_or_restore_job(job_id)
    if job is None:
        return jsonify({"error": "Unknown or expired session. Start a new analysis."}), 404

    job["last_polled_at"] = time.time()
    poller = job["poller"]
    new_comments = poller.drain() if poller else []
    # Whitespace/emoji-only comments are dropped by the classifier, so pair
    # results back up by text rather than by position.
    new_comments = [c for c in new_comments if model.clean_text(c.text)]

    if new_comments:
        classified = model.classify([c.text for c in new_comments])
        new_rows = []
        for comment, result in zip(new_comments, classified):
            row = {
                "id": comment.id,
                "comment": result["comment"],
                "sentiment": result["sentiment"],
                "score": result["score"],
                "author": comment.author,
                "likes": comment.likes,
                "posted_at": comment.posted_at,
            }
            new_rows.append(row)
        job["comments"].extend(new_rows)
        database.update_tiktok_job_comments(job_id, job["comments"])

        database.insert_comments(
            [
                {
                    "comment": r["comment"],
                    "sentiment": r["sentiment"],
                    "score": r["score"],
                    "product": job["product"],
                    "likes": r["likes"],
                    "owner": job["owner"],
                }
                for r in new_rows
            ]
        )

    return jsonify(
        {
            "summary": _summarize(job["comments"]),
            "results": job["comments"],
            "new_count": len(new_comments),
            "is_running": bool(poller and poller.is_running()),
            "stopped": job["stopped"],
            "last_error": poller.get_last_error() if poller else None,
            "comments_error": job["comments_error"],
            "comments_simulated": job["comments_simulated"],
        }
    )


@app.route("/api/tiktok/stop/<job_id>", methods=["POST"])
def tiktok_stop(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job["stopped"] = True
            if job["poller"]:
                job["poller"].stop()
    known = job is not None or database.get_tiktok_job(job_id) is not None
    if known:
        database.mark_tiktok_job_stopped(job_id)
    return jsonify({"stopped": known})


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
    # threaded=True: Pipeline B's polling background threads run per job, but
    # the dev server itself also needs to handle the polling GET requests
    # while a separate /tiktok/analyze POST (audio extraction + Whisper) is
    # still in flight for a different job - that needs more than one worker.
    #
    # use_reloader=False: the debug reloader restarts the whole process on
    # any .py change, which kills every running poller thread mid-analysis.
    # Debug error pages stay on; set FLASK_DEBUG=0 to turn them off.
    debug = os.getenv("FLASK_DEBUG", "1") != "0"
    app.run(debug=debug, use_reloader=False, port=5000, threaded=True)
