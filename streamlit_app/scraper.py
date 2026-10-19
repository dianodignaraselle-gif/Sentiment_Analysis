"""Pluggable TikTok video/comment sources plus a non-blocking polling loop.

`CommentSource` is the seam real scraping backends plug into:
`fetch_video_info` gets the creator's own video (caption/transcript),
`fetch_new_comments` gets whatever audience comments are new since the last
check. `MockTikTokSource` implements that interface with a simulated live
feed so the rest of the app - polling, state management, the dashboard - is
fully exercisable without network access.

`TikTokApiSource` is a best-effort real implementation built on the
unofficial `TikTokApi` package. TikTok has no public comments API, so this
relies on reverse-engineered endpoints that can change or get rate-limited
without notice - treat it as a plug-in to swap in when that trade-off is
acceptable, not as a guaranteed-stable backend.
"""

from __future__ import annotations

import itertools
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class VideoInfo:
    url: str
    author: str
    caption: str
    transcript: str


@dataclass
class Comment:
    id: str
    author: str
    text: str
    likes: int
    posted_at: float  # unix timestamp


class CommentSource(Protocol):
    def fetch_video_info(self, video_url: str) -> VideoInfo: ...

    def fetch_new_comments(self, video_url: str, seen_ids: set) -> list: ...


# ---------------------------------------------------------------------------
# Mock source - deterministic-ish simulated feed, no network required
# ---------------------------------------------------------------------------

_MOCK_CAPTION = (
    "Testing this foundation all day, tita-approved na sulit talaga! "
    "#foundation #makeup #review"
)
_MOCK_TRANSCRIPT = (
    "Hi besties! So today sinusubukan ko itong foundation. Grabe, sobrang "
    "lightweight and the coverage is buildable. Let's see kung paano siya "
    "mag-last the whole day without oxidizing."
)

_MOCK_COMMENT_POOL = [
    "Sulit na sulit this one, panalo talaga!",
    "Grabe ka-cakey nito after a few hours, sayang pera ko.",
    "It's okay lang, hindi naman ganun ka-special.",
    "Omg the coverage is amazing and hindi siya nakaka-breakout!",
    "Nag-oxidize after 2 hours, hindi ko na bibilhin ulit.",
    "Natural finish talaga, bet ko siya lagi!",
    "Mura pero quality, worth it!",
    "Chalky sa dry skin, hindi recommended.",
    "Decent lang, pwede na pero meh.",
    "Sobrang glowing ng skin ko after using this, legit!",
    "Pangit yung shade match, limited lang choices.",
    "Average lang, wala namang masabi.",
    "Non-cakey and poreless finish, panalo!",
    "Broke me out after two days, nakaka-disappoint.",
    "Blends so well, parang second skin lang!",
    "Maganda sana yung product but mali yung shade na dumating sa akin.",
    "Hindi siya long-lasting, kailangan i-reapply every 3 hours.",
    "Best foundation I've ever used, highly recommend!",
    "Not worth the hype, medyo overpriced.",
    "Lightweight and breathable, perfect for summer!",
]
_MOCK_AUTHORS = [f"user{n}" for n in range(1, 50)]


class MockTikTokSource:
    """Simulates a creator video plus a slowly arriving live comment feed."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._counter = itertools.count(1)

    def fetch_video_info(self, video_url: str) -> VideoInfo:
        return VideoInfo(
            url=video_url,
            author="demo_creator",
            caption=_MOCK_CAPTION,
            transcript=_MOCK_TRANSCRIPT,
        )

    def fetch_new_comments(self, video_url: str, seen_ids: set) -> list:
        n = self._rng.randint(0, 4)  # simulate a trickle of new comments per poll
        now = time.time()
        batch = []
        for _ in range(n):
            batch.append(
                Comment(
                    id=f"mock-{next(self._counter)}",
                    author=self._rng.choice(_MOCK_AUTHORS),
                    text=self._rng.choice(_MOCK_COMMENT_POOL),
                    likes=self._rng.randint(0, 500),
                    posted_at=now,
                )
            )
        return batch


# ---------------------------------------------------------------------------
# Unofficial live source (best-effort, optional dependency)
# ---------------------------------------------------------------------------

class TikTokApiSource:
    """Best-effort live source using the unofficial `TikTokApi` package.

    Requires `pip install TikTokApi` and `python -m playwright install`.
    Not guaranteed to keep working - there is no official comments API.
    """

    def __init__(self) -> None:
        try:
            from TikTokApi import TikTokApi
        except ImportError as exc:
            raise RuntimeError(
                "TikTokApi is not installed. Run `pip install TikTokApi` and "
                "`python -m playwright install` to enable live scraping, or "
                "use the mock/demo source instead."
            ) from exc
        self._TikTokApi = TikTokApi

    def fetch_video_info(self, video_url: str) -> VideoInfo:
        import asyncio

        async def _fetch():
            async with self._TikTokApi() as api:
                await api.create_sessions(num_sessions=1, sleep_after=3)
                info = await api.video(url=video_url).info()
                return VideoInfo(
                    url=video_url,
                    author=info.get("author", {}).get("uniqueId", "unknown"),
                    caption=info.get("desc", ""),
                    transcript="",  # TikTokApi exposes no ASR transcript
                )

        return asyncio.run(_fetch())

    def fetch_new_comments(self, video_url: str, seen_ids: set) -> list:
        import asyncio

        async def _fetch():
            out = []
            async with self._TikTokApi() as api:
                await api.create_sessions(num_sessions=1, sleep_after=3)
                video = api.video(url=video_url)
                async for c in video.comments(count=30):
                    cid = str(c.id)
                    if cid in seen_ids:
                        continue
                    data = c.as_dict
                    out.append(
                        Comment(
                            id=cid,
                            author=data.get("user", {}).get("unique_id", "unknown"),
                            text=data.get("text", ""),
                            likes=data.get("digg_count", 0),
                            posted_at=time.time(),
                        )
                    )
            return out

        return asyncio.run(_fetch())


def get_source(use_mock: bool = True, seed: int | None = None) -> CommentSource:
    if use_mock:
        return MockTikTokSource(seed=seed)
    return TikTokApiSource()


# ---------------------------------------------------------------------------
# Non-blocking polling loop
# ---------------------------------------------------------------------------

class CommentPoller:
    """Polls a CommentSource on a background thread and queues new comments.

    Runs independently of the Streamlit script's rerun cycle, so the UI
    thread is never blocked on network I/O: the dashboard calls `drain()`
    on each rerun to pick up whatever arrived since the last check.
    """

    def __init__(self, source: CommentSource, video_url: str, interval: float = 5.0) -> None:
        self.source = source
        self.video_url = video_url
        self.interval = interval
        self.queue: "queue.Queue" = queue.Queue()
        self._seen_ids: set = set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def get_last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                new_comments = self.source.fetch_new_comments(self.video_url, self._seen_ids)
                for c in new_comments:
                    if c.id not in self._seen_ids:
                        self._seen_ids.add(c.id)
                        self.queue.put(c)
                with self._lock:
                    self._last_error = None
            except Exception as exc:  # keep polling even if one round fails
                with self._lock:
                    self._last_error = str(exc)
            self._stop_event.wait(self.interval)

    def drain(self) -> list:
        """Pull everything queued since the last drain. Call from the main thread."""
        drained = []
        while True:
            try:
                drained.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return drained
