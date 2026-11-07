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
import relevance  # noqa: E402
import scraper  # noqa: E402
import slang  # noqa: E402

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

database.init_db()
# Loaded now, so the first analysis doesn't wait for it.
threading.Thread(target=model.get_classifier, daemon=True).start()

# ---------- TikTok link analysis: job registry ----------
# Each job's background poller (a live thread) can only live in this
# process's memory, but the job itself - URL, product, classified comments,
# stopped flag - is persisted in SQLite (database.tiktok_jobs). If the
# in-memory entry is gone (server restart, stale-job pruning), the next poll
# rebuilds it from the database and resumes polling instead of 404ing.
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL_SECONDS = 60 * 30  # stop orphaned pollers after 30 minutes of no polling
_REAL_POLL_INTERVAL = 5.0  # real TikTok scraping: fast enough to feel live; backs off if blocked
_FIRST_READ_WAIT = 6.0  # how long a new analysis waits for comments before the results page opens
_AUDIO_WAIT = 60.0  # how long the audio waits for a video's comments to be classified
_COMMENTS_PER_ROUND = 48  # classified per round, so a big video's numbers fill in instead of stalling
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


def _start_poller(source, video_url, seen_ids=None, known_ids=None):
    if getattr(source, "comments_error", None):
        return None  # nothing to poll - the error is reported to the page instead
    interval = _MOCK_POLL_INTERVAL if source.comments_simulated else _REAL_POLL_INTERVAL
    poller = scraper.CommentPoller(
        source, video_url, interval=interval, seen_ids=seen_ids, known_ids=known_ids
    )
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
            "brand": record.get("brand"),
            "foundation_type": record.get("foundation_type"),
            "video_id": record.get("video_id"),
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
    problem = relevance.product_name_problem(product)
    if problem:
        return jsonify({"error": problem}), 422
    if len(comments) > 200:
        return jsonify({"error": "Too many comments in a single request (max 200)."}), 400

    # Comments that aren't about the product are never classified or stored.
    comments, skipped = model.split_relevant(comments, product, database.video_type_for(product, None, None))
    results = model.classify(comments) if comments else []

    database.insert_comments(
        [
            {
                "comment": r["comment"],
                "sentiment": r["sentiment"],
                "score": r["score"],
                "product": product,
                "likes": None,
                "owner": owner,
                "brand": database.brand_for(r["comment"], product),
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

    return jsonify({"summary": summary, "results": results, "skipped": skipped})


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

    Whether the video is about a foundation is decided first, from its caption
    and comments. Pipeline A (audio -> Whisper -> XLM-RoBERTa) and Pipeline B
    (live comment polling) then run on background threads.
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
    problem = relevance.product_name_problem(product)
    if problem:
        return jsonify({"error": problem}), 422

    _prune_stale_jobs()

    source = scraper.get_source(video_url, use_mock=use_mock)
    video_id = scraper.video_id_for(video_url, source)
    # Trying a link again replaces its earlier run: several pollers reading the
    # same video only get the network rate limited.
    if video_id:
        with _JOBS_LOCK:
            for earlier_job in _JOBS.values():
                if earlier_job.get("video_id") == video_id and earlier_job["owner"] == owner and earlier_job["poller"]:
                    earlier_job["poller"].stop()
    # The caption and comments decide what the video is about, in a second or
    # two. The audio comes after (see _finish_video): a full episode takes many
    # minutes to transcribe, and nobody should wait for that just to be refused.
    fetch_meta = getattr(source, "fetch_video_meta", None)
    video_meta = fetch_meta(video_url) if fetch_meta else None
    video_info = video_meta or source.fetch_video_info(video_url)

    # Only trust the transcript if Whisper produced real, non-looping speech;
    # otherwise classify the caption alone.
    transcript_ok = bool(video_info.transcript) and not video_info.transcript_issue
    caption = video_info.caption or ""
    if transcript_ok:
        analysis_basis = "caption+transcript" if caption.strip() else "transcript"
    else:
        analysis_basis = "caption" if caption.strip() else None
    transcript = video_info.transcript if transcript_ok else ""

    # Only foundation videos are analyzed - not food, dance or other content.
    # When the caption/audio don't say, the first batch of comments decides.
    comments_simulated = getattr(source, "comments_simulated", False)
    comments_error = getattr(source, "comments_error", None)
    is_foundation, topic_reason = relevance.video_topic(caption, transcript)
    # The brand the video is about; comments that don't name one get this.
    brand = database.brand_for(product, caption, transcript)
    brand = None if brand == database.UNKNOWN_BRAND else brand
    # Comments that don't say which type they mean get the video's.
    foundation_type = database.video_type_for(product, caption, transcript)
    first_texts = []
    if is_foundation is None or (is_foundation and not brand):
        peek = getattr(source, "peek_comments", lambda url: source.fetch_new_comments(url, set()))
        try:
            first_texts = [] if comments_error else [c.text for c in peek(video_url)]
        except Exception:
            first_texts = []
    if is_foundation is None:
        is_foundation = relevance.comments_confirm_foundation(first_texts)
        if not is_foundation:
            topic_reason = "neither the caption nor the comments mention a foundation"
    if is_foundation and not brand:
        brand = relevance.brand_from_comments(first_texts)
    if not is_foundation:
        app.logger.warning("Refused %s: %s", video_url, topic_reason)
        close = getattr(source, "close", None)
        if close:
            close()
        return jsonify(
            {"error": f"This video isn't about a foundation product ({topic_reason}), so it wasn't analyzed. "
                      "Only foundation videos can be analyzed."}
        ), 422

    video_text = model.build_video_text(caption, transcript)
    video_result = model.classify_one(video_text) if video_text else None

    # Analyzing a video again keeps what earlier runs picked up, unless it's been
    # found gone from TikTok. A comment missing now is dropped later if it stays
    # missing (see CommentPoller), and comes back if TikTok returns it.
    earlier = [] if comments_simulated else database.earlier_tiktok_comments(video_id, owner)
    counted = database.counted_source_ids(owner, [c["id"] for c in earlier])
    kept = [
        dict(
            c,
            foundation_type=database.type_for(c["comment"]) or foundation_type,
            sentiment=slang.label(c["comment"]) or c["sentiment"],
        )
        for c in earlier
        if c["id"] in counted and model.relevance_check(c["comment"], product, foundation_type)[0]
    ]
    poller = _start_poller(
        source, video_url, seen_ids={c["id"] for c in kept}, known_ids={c["id"] for c in earlier}
    )

    job_id = uuid.uuid4().hex
    database.save_tiktok_job(
        job_id, video_url, product, owner, comments_simulated, comments_error, brand, foundation_type, video_id,
        video_info.author, caption,
    )
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "poller": poller,
            "video_url": video_url,
            "product": product,
            "brand": brand,
            "foundation_type": foundation_type,
            "video_id": video_id,
            "owner": owner,
            "comments": kept,
            "comments_simulated": comments_simulated,
            "comments_error": comments_error,
            "stopped": False,
            "last_polled_at": time.time(),
        }
        job = _JOBS[job_id]
    if kept:
        database.update_tiktok_job_comments(job_id, kept)
        _store_rows(job, kept)

    job["video"] = _video_payload(video_info, video_result, transcript_ok, analysis_basis, brand, foundation_type)
    if video_meta:
        job["video"]["transcript_pending"] = True
        threading.Thread(target=_finish_video, args=(job, source, video_url), daemon=True).start()

    # A short wait, so small videos open with their numbers ready.
    deadline = time.time() + _FIRST_READ_WAIT
    while poller and poller.is_running() and not poller.first_read_done() and time.time() < deadline:
        if not _process_new_comments(job_id, job)[0]:
            time.sleep(0.25)
    _process_new_comments(job_id, job)

    return jsonify(
        {
            "job_id": job_id,
            "video": job["video"],
            "summary": _summarize(job["comments"]),
            "results": job["comments"],
            "comments_simulated": comments_simulated,
            "comments_error": comments_error,
        }
    )


def _video_payload(video_info, video_result, transcript_ok, analysis_basis, brand, foundation_type):
    return {
        "author": video_info.author,
        "brand": brand,
        "foundation_type": foundation_type,
        "caption": video_info.caption,
        "transcript": video_info.transcript,
        "transcript_error": video_info.transcript_error,
        "transcript_issue": video_info.transcript_issue,
        "transcript_reliable": transcript_ok,
        "analysis_basis": analysis_basis,
        "sentiment": video_result.sentiment if video_result else None,
        "score": video_result.score if video_result else None,
        "matched_terms": list(video_result.matched_terms) if video_result else [],
    }


def _finish_video(job, source, video_url):
    """Transcribe the audio of a video already accepted, while its results show.
    Comments go first: only one model runs at a time, and they're what's on screen."""
    poller = job["poller"]
    deadline = time.time() + _AUDIO_WAIT
    while poller and poller.is_running() and not poller.first_read_done() and time.time() < deadline:
        time.sleep(0.5)
    try:
        info = source.fetch_video_info(video_url)
        transcript_ok = bool(info.transcript) and not info.transcript_issue
        transcript = info.transcript if transcript_ok else ""
        video_text = model.build_video_text(info.caption or "", transcript)
        result = model.classify_one(video_text) if video_text else None
        basis = job["video"]["analysis_basis"]
        if transcript_ok:
            basis = "caption+transcript" if (info.caption or "").strip() else "transcript"
        if not job.get("foundation_type"):
            job["foundation_type"] = database.type_for(transcript)
        job["video"] = _video_payload(info, result, transcript_ok, basis, job.get("brand"), job.get("foundation_type"))
    except Exception as exc:
        job["video"] = dict(job["video"], transcript_pending=False, transcript_error=f"Could not transcribe the audio: {exc}")


def _process_new_comments(job_id, job):
    """Classify and store a round of what the poller queued, and drop comments
    deleted from TikTok. Returns (how many new comments were analyzed, still loading)."""
    poller = job["poller"]
    # Read before draining: "done loading" must not hide comments still queued.
    loading = bool(poller and poller.is_running() and poller.is_busy())
    # One round at a time: a poll arriving during a round just reports the numbers so far.
    lock = job.setdefault("process_lock", threading.Lock())
    if not lock.acquire(blocking=False):
        return 0, loading
    try:
        return _process_round(job_id, job, poller), loading
    finally:
        lock.release()


def _store_rows(job, rows):
    database.insert_comments(
        [
            {
                "comment": r["comment"],
                "sentiment": r["sentiment"],
                "score": r["score"],
                "product": job["product"],
                "likes": r["likes"],
                "owner": job["owner"],
                "brand": r["brand"],
                "foundation_type": r["foundation_type"],
                # Analyzing the same link again doesn't store its comments twice.
                "source_id": None if job["comments_simulated"] else r["id"],
                "posted_date": database.posted_date(r["posted_at"]),
                "video_id": job.get("video_id"),
            }
            for r in rows
        ]
    )


def _process_round(job_id, job, poller):
    new_comments = poller.drain(_COMMENTS_PER_ROUND) if poller else []
    # Comments with no text or that aren't about the product are skipped: never
    # classified or stored, but listed so the totals add up to what TikTok shows.
    skipped = job.setdefault("skipped", [])
    relevant = []
    for comment in new_comments:
        ok, reason = (
            model.relevance_check(comment.text, job["product"], job.get("foundation_type"))
            if model.clean_text(comment.text)
            else (False, "only a sticker: a picture with no words to read" if comment.sticker else "no text to analyze")
        )
        if ok:
            relevant.append(comment)
        else:
            shown = comment.text or ("(sticker)" if comment.sticker else "(no text)")
            skipped.append({"id": comment.id, "comment": shown, "reason": reason, "parent_id": comment.parent_id})
    new_comments = relevant

    if new_comments:
        # The model gives a text the same answer every time, so reuse earlier ones.
        texts = [model.clean_text(c.text) for c in new_comments]
        known = database.known_sentiments(texts)
        fresh = [t for t in texts if t not in known or slang.label(t)]
        results = dict(zip(fresh, model.classify(fresh) if fresh else []))
        new_rows = []
        for comment, text in zip(new_comments, texts):
            sentiment, score = (results[text]["sentiment"], results[text]["score"]) if text in results else known[text]
            row = {
                "id": comment.id,
                "comment": text,
                "sentiment": sentiment,
                "score": score,
                "author": comment.author,
                "likes": comment.likes,
                "posted_at": comment.posted_at,
                "parent_id": comment.parent_id,
                "brand": database.brand_for(comment.text, job.get("product"), job.get("brand")),
                "foundation_type": database.type_for(comment.text) or job.get("foundation_type"),
            }
            new_rows.append(row)
        job["comments"].extend(new_rows)
        database.update_tiktok_job_comments(job_id, job["comments"])

        _store_rows(job, new_rows)
        if poller and poller.first_read_done():
            for r in new_rows:
                app.logger.warning("Live comment on %s: %r (%s)", job.get("video_id"), r["comment"][:60], r["sentiment"])

    # Deleted from TikTok: stop showing and counting them.
    removed = poller.drain_removed() if poller else set()
    if removed:
        app.logger.warning("Gone from TikTok on %s: %s", job.get("video_id"), sorted(removed))
        job["skipped"] = [s for s in skipped if s["id"] not in removed]
        job["comments"] = [c for c in job["comments"] if c.get("id") not in removed]
        database.update_tiktok_job_comments(job_id, job["comments"])
        job["removed_count"] = job.get("removed_count", 0) + database.mark_tiktok_comments_removed(job["owner"], removed)

    return len(new_comments)


@app.route("/api/tiktok/updates/<job_id>", methods=["GET"])
def tiktok_updates(job_id):
    """Drain whatever new comments Pipeline B queued, classify them, return the running totals."""
    job = _get_or_restore_job(job_id)
    if job is None:
        return jsonify({"error": "Unknown or expired session. Start a new analysis."}), 404

    job["last_polled_at"] = time.time()
    poller = job["poller"]
    new_count, loading = _process_new_comments(job_id, job)

    return jsonify(
        {
            "summary": _summarize(job["comments"]),
            "results": job["comments"],
            "new_count": new_count,
            # In-memory only: resets if the job is restored after a restart.
            "skipped_count": len(job.get("skipped", [])),
            "skipped": job.get("skipped", []),
            "tiktok_total": poller.reported_total() if poller else None,
            "removed_count": job.get("removed_count", 0),
            "is_running": bool(poller and poller.is_running()),
            "loading": loading,
            "seconds_since_check": poller.seconds_since_check() if poller else None,
            "video": job.get("video"),
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
    video_id = request.args.get("video_id") or None
    return jsonify(database.get_dataset(page, page_size, search, product, owner, video_id))


@app.route("/api/videos", methods=["GET"])
def videos():
    owner = request.args.get("owner") or None
    return jsonify({"videos": database.get_videos(owner)})


@app.route("/api/dataset/products", methods=["GET"])
def dataset_products():
    owner = request.args.get("owner") or None
    return jsonify({"products": database.get_products(owner)})


@app.route("/api/dataset/export", methods=["GET"])
def export_csv():
    owner = request.args.get("owner") or None
    rows = database.get_all_rows(owner, request.args.get("video_id") or None)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "comment", "sentiment", "score", "product", "date", "likes", "deleted_on_tiktok"])
    for r in rows:
        writer.writerow(
            [r["id"], r["comment"], r["sentiment"], r["score"], r["product"] or "", r["comment_date"], r["likes"] or "", r["removed_at"] or ""]
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
    stats = database.get_stats(owner, request.args.get("video_id") or None)
    for word in stats["word_cloud"]:
        word["tone"] = model.word_tone(word["text"])
    return jsonify(stats)


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
