"""Delete already-stored comments that aren't about the foundation product.

New comments are filtered before they're saved (see relevance.py); this
removes the off-topic ones analyzed before that filter existed, from both
the comments table and saved TikTok live sessions. Run it again after
changing relevance.py's word lists.

    backend/venv/bin/python backend/cleanup_irrelevant.py            # back up, then delete
    backend/venv/bin/python backend/cleanup_irrelevant.py --dry-run  # only list them

The database is copied to backups/<date_time>_before-irrelevant-cleanup/ first.
Doesn't load the sentiment model.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import database
from model_pipeline import relevance_check

BACKUPS_DIR = Path(__file__).resolve().parent.parent / "backups"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="list what would be deleted, change nothing")
    args = p.parse_args()

    conn = database.get_conn()
    rows = conn.execute("SELECT id, comment, product FROM comments").fetchall()
    doomed = [(r["id"], r["comment"], why) for r in rows for ok, why in [relevance_check(r["comment"], r["product"])] if not ok]

    jobs = conn.execute("SELECT job_id, product, comments_json FROM tiktok_jobs").fetchall()
    job_updates = []
    for job in jobs:
        comments = json.loads(job["comments_json"] or "[]")
        kept = [c for c in comments if relevance_check(c.get("comment", ""), job["product"])[0]]
        if len(kept) != len(comments):
            job_updates.append((job["job_id"], kept, len(comments) - len(kept)))

    print(f"comments table: {len(doomed)} of {len(rows)} rows are off-topic")
    for _id, comment, why in doomed:
        print(f"    - #{_id} {comment!r}: {why}")
    print(f"TikTok sessions: {sum(n for *_, n in job_updates)} off-topic comments in {len(job_updates)} sessions")

    if args.dry_run or not (doomed or job_updates):
        conn.close()
        return

    # Timestamped so a second run the same day doesn't overwrite the first backup.
    backup_dir = BACKUPS_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}_before-irrelevant-cleanup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "dataset.db"
    shutil.copy2(database.DB_PATH, backup_path)
    print(f"backed up database to {backup_path}")

    conn.executemany("DELETE FROM comments WHERE id = ?", [(d[0],) for d in doomed])
    conn.executemany(
        "UPDATE tiktok_jobs SET comments_json = ? WHERE job_id = ?",
        [(json.dumps(kept), job_id) for job_id, kept, _n in job_updates],
    )
    conn.commit()
    conn.close()
    print(f"deleted {len(doomed)} comments and cleaned {len(job_updates)} TikTok sessions")


if __name__ == "__main__":
    main()
