"""TikTok video/audio extraction plus a non-blocking live-comment polling loop.

Two independent jobs live here, matching the two pipelines app.py kicks off
per analyzed link:

  Pipeline A (one-time): `YtDlpSource.fetch_video_info` downloads the video's
  audio track with yt-dlp, transcribes it with Whisper (model_pipeline.py),
  and returns it alongside the creator's caption.

  Pipeline B (continuous): `CommentPoller` runs a background thread that
  repeatedly calls `fetch_new_comments` on an interval and pushes whatever is
  new onto a thread-safe queue, so the Flask request thread handling the UI's
  poll requests is never the thing doing the (slow, flaky) scraping.

TikTokApi has no official comments API, so `TikTokApiSource` is a best-effort
implementation on top of the unofficial `TikTokApi` package - treat it as a
plug-in to use when that trade-off is acceptable, not as a guaranteed-stable
backend. It needs an `MS_TOKEN` (the `msToken` cookie from tiktok.com) in
backend/.env. If it can't be set up, the real source reports a
`comments_error` instead of silently substituting simulated comments;
`MockTikTokSource` is only used when a caller explicitly asks for it.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import queue
import random
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

TIKTOK_URL_RE = re.compile(r"^https?://(www\.|vm\.|vt\.|m\.)?tiktok\.com/", re.IGNORECASE)


def is_tiktok_url(url: str) -> bool:
    return bool(url and TIKTOK_URL_RE.match(url.strip()))


@dataclass
class VideoInfo:
    url: str
    author: str
    caption: str
    transcript: str
    transcript_error: str | None = None
    transcript_issue: str | None = None  # set when the transcript looks hallucinated/empty


class CommentSourceUnavailable(RuntimeError):
    """The real comment source can't be used (missing package, browser or ms_token)."""


@dataclass
class Comment:
    id: str
    author: str
    text: str
    likes: int
    posted_at: float  # unix timestamp


class CommentSource(Protocol):
    comments_simulated: bool

    def fetch_video_info(self, video_url: str) -> VideoInfo: ...

    def fetch_new_comments(self, video_url: str, seen_ids: set) -> list: ...


# ---------------------------------------------------------------------------
# Direct TikTok web client - no browser, no msToken
# ---------------------------------------------------------------------------
# TikTok's own web page embeds the video's data (caption, author, audio URL)
# as JSON, and its web comment endpoint answers plain requests. curl_cffi
# makes those requests with a real browser's TLS fingerprint. TikTok
# challenges some fingerprints at some times and not others, so several are
# tried in turn and the first one that gets real data is kept.

_FINGERPRINTS = ("safari17_0", "chrome", "chrome124", "safari15_5", "edge101")
_PAGE_DATA_RE = re.compile(r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', re.S)
_VIDEO_ID_RE = re.compile(r"/(?:video|photo)/(\d+)")
_BLOCKED_MESSAGE = (
    "TikTok is blocking requests from your network right now (bot check / rate limit), "
    "so the video page, audio and transcript couldn't be fetched. This is usually "
    "temporary: wait 30-60 minutes or switch networks (e.g. a phone hotspot), then "
    "run a new analysis."
)


class TikTokWebClient:
    """Video data, audio and comments straight from tiktok.com's web endpoints."""

    comments_per_poll = 300  # up to 6 pages of 50 per poll; later polls continue from there

    def __init__(self) -> None:
        from curl_cffi import requests as curl_requests  # raises ImportError if missing

        self._curl = curl_requests
        self._session = None
        self._lock = threading.Lock()
        self._items: dict = {}  # video_url -> itemStruct
        self._cursors: dict = {}  # video_id -> next comment cursor

    def _new_sessions(self):
        if self._session is not None:
            yield self._session
        for fp in _FINGERPRINTS:
            try:
                yield self._curl.Session(impersonate=fp)
            except Exception:  # fingerprint not supported by this curl_cffi version
                continue

    def video_item(self, video_url: str) -> dict:
        """The page's itemStruct: id, desc (caption), author, music, video."""
        with self._lock:
            if video_url in self._items:
                return self._items[video_url]
            last_error = _BLOCKED_MESSAGE
            for session in self._new_sessions():
                try:
                    page = session.get(video_url, timeout=20)
                except Exception as exc:
                    last_error = f"Could not reach TikTok: {exc}"
                    continue
                match = _PAGE_DATA_RE.search(page.text)
                if not match:
                    continue  # bot-check page for this fingerprint; try the next one
                detail = json.loads(match.group(1)).get("__DEFAULT_SCOPE__", {}).get("webapp.video-detail", {})
                item = (detail.get("itemInfo") or {}).get("itemStruct")
                if not item:
                    status = detail.get("statusCode")
                    if status in (10204, 10216, 10222):
                        raise RuntimeError("This video is private or unavailable, so it can't be analyzed.")
                    continue
                self._session = session
                self._items[video_url] = item
                return item
            raise RuntimeError(last_error)

    def download_audio(self, item: dict, folder: str) -> tuple[str | None, str | None]:
        """Save the clip's audio; returns (path, None) or (None, reason)."""
        video = item.get("video") or {}
        music = item.get("music") or {}
        url = video.get("playAddr") or video.get("downloadAddr")
        if not url and music.get("original"):
            url = music.get("playUrl")  # an "original sound" track is the creator's own audio
        if not url:
            return None, (
                "This video uses a non-original sound and TikTok didn't provide its audio track, "
                "so there is no creator speech to transcribe."
            )
        with self._lock:
            response = self._session.get(url, headers={"Referer": "https://www.tiktok.com/"}, timeout=90)
        if response.status_code != 200 or not response.content:
            return None, f"Could not download the video's audio (HTTP {response.status_code})."
        path = Path(folder) / "audio.mp4"
        path.write_bytes(response.content)
        return str(path), None

    def video_id(self, video_url: str) -> str:
        match = _VIDEO_ID_RE.search(video_url)
        return match.group(1) if match else str(self.video_item(video_url)["id"])  # short vt./vm. links

    def fetch_new_comments(self, video_url: str, seen_ids: set) -> list:
        video_id = self.video_id(video_url)
        cursor = self._cursors.get(video_id, 0)
        out = []
        with self._lock:
            sessions = self._new_sessions()
            session = next(sessions)
            fetched = 0
            while fetched < self.comments_per_poll:
                try:
                    data = session.get(
                        "https://www.tiktok.com/api/comment/list/",
                        params={"aid": "1988", "aweme_id": video_id, "count": "50", "cursor": str(cursor)},
                        headers={"Referer": f"https://www.tiktok.com/video/{video_id}"},
                        timeout=20,
                    ).json()
                except Exception:
                    session = next(sessions, None)  # empty/HTML reply = challenged; switch fingerprint
                    if session is None:
                        raise RuntimeError(_BLOCKED_MESSAGE.replace("the video page, audio and transcript", "comments"))
                    continue
                self._session = session
                page = data.get("comments") or []
                fetched += len(page)
                for c in page:
                    cid = str(c.get("cid"))
                    if cid in seen_ids:
                        continue
                    out.append(
                        Comment(
                            id=cid,
                            author=(c.get("user") or {}).get("unique_id", "unknown"),
                            text=c.get("text", ""),
                            likes=c.get("digg_count", 0),
                            posted_at=float(c.get("create_time") or time.time()),
                        )
                    )
                if not data.get("has_more") or not page:
                    cursor = 0  # reached the end; next poll starts over to catch new comments
                    break
                cursor = data.get("cursor", cursor + len(page))
        self._cursors[video_id] = cursor
        return out


class FallbackCommentSource:
    """Direct web client first; the TikTokApi browser session only if that fails."""

    comments_simulated = False

    def __init__(self, primary: "TikTokWebClient | None", fallback: "TikTokApiSource | None") -> None:
        self._primary = primary
        self._fallback = fallback

    def fetch_new_comments(self, video_url: str, seen_ids: set) -> list:
        try:
            if self._primary is None:
                raise RuntimeError("curl_cffi is not installed - run `pip install -r requirements.txt` in backend/.")
            return self._primary.fetch_new_comments(video_url, seen_ids)
        except Exception as primary_error:
            if self._fallback is None:
                raise
            try:
                return self._fallback.fetch_new_comments(video_url, seen_ids)
            except Exception:
                raise primary_error

    def close(self) -> None:
        if self._fallback is not None:
            self._fallback.close()


# ---------------------------------------------------------------------------
# Real source - TikTok web client (or yt-dlp) for audio/metadata, Whisper
# for transcript, web client / TikTokApi for comments
# ---------------------------------------------------------------------------

def _explain_ytdlp_error(exc: Exception) -> str:
    """Turn yt-dlp's raw TikTok extraction errors into something actionable."""
    msg = str(exc)
    if "Unable to extract webpage video data" in msg or "HTTP Error 429" in msg:
        return (
            "TikTok is blocking requests from your network right now (bot check / rate limit), "
            "so the video page, audio and transcript couldn't be fetched. This is usually "
            "temporary: wait 30-60 minutes or switch networks (e.g. a phone hotspot), then "
            "run a new analysis."
        )
    if "private" in msg.lower() or "status code 10216" in msg or "10222" in msg:
        return "This video is private or unavailable, so it can't be analyzed."
    return f"Could not fetch video metadata: {msg}"


def _fetch_oembed(video_url: str) -> dict | None:
    """Caption/author via TikTok's public oEmbed endpoint (no video page needed)."""
    try:
        import requests

        r = requests.get("https://www.tiktok.com/oembed", params={"url": video_url}, timeout=15)
        if r.ok and "json" in (r.headers.get("content-type") or ""):
            return r.json()
    except Exception:
        pass
    return None


class YtDlpSource:
    """Real video analysis via yt-dlp + Whisper, with best-effort comments."""

    def __init__(
        self,
        comment_backend: "CommentSource | None" = None,
        comments_simulated: bool = False,
        comments_error: str | None = None,
        web_client: "TikTokWebClient | None" = None,
    ) -> None:
        self._web = web_client
        # Comment scraping is a separate concern from audio/caption
        # extraction. With no comment backend the video is still analyzed,
        # and `comments_error` explains why there's no comment feed - no
        # simulated comments are substituted behind the user's back.
        self._comment_backend = comment_backend
        self.comments_simulated = comments_simulated
        self.comments_error = comments_error

    def fetch_video_info(self, video_url: str) -> VideoInfo:
        if self._web is not None:
            try:
                return self._fetch_video_info_web(video_url)
            except Exception:
                pass  # fall back to yt-dlp below
        return self._fetch_video_info_ytdlp(video_url)

    def _fetch_video_info_web(self, video_url: str) -> VideoInfo:
        item = self._web.video_item(video_url)
        author = (item.get("author") or {}).get("uniqueId") or "unknown"
        caption = item.get("desc") or ""
        transcript, transcript_error, transcript_issue = "", None, None
        with tempfile.TemporaryDirectory(prefix="tiktok_audio_") as tmpdir:
            try:
                audio_path, transcript_error = self._web.download_audio(item, tmpdir)
                if audio_path:
                    import model_pipeline

                    checked = model_pipeline.transcribe_audio_checked(audio_path)
                    transcript, transcript_issue = checked.text, checked.issue
            except Exception as exc:  # whisper/ffmpeg are best-effort on top of the caption
                transcript_error = f"Could not transcribe the video's audio: {exc}"
        return VideoInfo(
            url=video_url,
            author=author,
            caption=caption,
            transcript=transcript,
            transcript_error=transcript_error,
            transcript_issue=transcript_issue,
        )

    def _fetch_video_info_ytdlp(self, video_url: str) -> VideoInfo:
        try:
            import yt_dlp
        except ImportError:
            return VideoInfo(
                url=video_url,
                author="unknown",
                caption="",
                transcript="",
                transcript_error=(
                    "yt-dlp is not installed - run `pip install -r requirements.txt` "
                    "in backend/ to enable audio extraction."
                ),
            )

        author = "unknown"
        caption = ""
        transcript = ""
        transcript_error = None
        transcript_issue = None

        # Metadata (author/caption) needs only yt-dlp + network access; audio
        # download + extraction additionally needs the `ffmpeg` system binary.
        # Fetching metadata on its own first means a real caption/author still
        # comes through even on a machine without ffmpeg installed.
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
                info = ydl.extract_info(video_url, download=False)
            author = info.get("uploader") or info.get("creator") or "unknown"
            caption = info.get("description") or info.get("title") or ""
        except Exception as exc:
            transcript_error = _explain_ytdlp_error(exc)
            # The video page is blocked, but TikTok's lighter oEmbed endpoint
            # may still return the caption, so caption sentiment can still run.
            oembed = _fetch_oembed(video_url)
            if oembed:
                author = oembed.get("author_unique_id") or oembed.get("author_name") or author
                caption = oembed.get("title") or ""
            return VideoInfo(url=video_url, author=author, caption=caption, transcript=transcript, transcript_error=transcript_error)

        with tempfile.TemporaryDirectory(prefix="tiktok_audio_") as tmpdir:
            import model_pipeline

            audio_template = str(Path(tmpdir) / "audio.%(ext)s")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": audio_template,
                "postprocessors": [
                    {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
                ],
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
            }
            if model_pipeline.FFMPEG_PATH:
                ydl_opts["ffmpeg_location"] = model_pipeline.FFMPEG_PATH
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.extract_info(video_url, download=True)

                audio_path = Path(tmpdir) / "audio.wav"
                if audio_path.exists():
                    checked = model_pipeline.transcribe_audio_checked(str(audio_path))
                    transcript = checked.text
                    transcript_issue = checked.issue
            except Exception as exc:  # ffmpeg/whisper are best-effort on top of the metadata above
                transcript_error = str(exc)

        return VideoInfo(
            url=video_url,
            author=author,
            caption=caption,
            transcript=transcript,
            transcript_error=transcript_error,
            transcript_issue=transcript_issue,
        )

    def fetch_new_comments(self, video_url: str, seen_ids: set) -> list:
        if self._comment_backend is None:
            raise CommentSourceUnavailable(self.comments_error or "No comment source configured.")
        return self._comment_backend.fetch_new_comments(video_url, seen_ids)

    def close(self) -> None:
        close = getattr(self._comment_backend, "close", None)
        if close:
            close()


# ---------------------------------------------------------------------------
# Mock source - deterministic-ish simulated feed, no network required
# ---------------------------------------------------------------------------

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
    "Grabe sobrang hulas pagka-ilang oras, sayang.",
    "Budol worth it! Hindi malagkit sa balat.",
]
_MOCK_AUTHORS = [f"user{n}" for n in range(1, 50)]


class MockTikTokSource:
    """Simulates a slowly arriving live comment feed when no real backend is wired up."""

    comments_simulated = True

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._counter = itertools.count(1)
        # Per-instance prefix so ids don't collide with a restored job's seen ids.
        self._id_prefix = f"mock-{uuid.uuid4().hex[:8]}"

    def fetch_video_info(self, video_url: str) -> VideoInfo:
        return VideoInfo(
            url=video_url,
            author="demo_creator",
            caption="Testing this foundation all day, tita-approved na sulit talaga!",
            transcript="",
            transcript_error="Mock source - no real audio/transcript extracted.",
        )

    def fetch_new_comments(self, video_url: str, seen_ids: set) -> list:
        n = self._rng.randint(0, 3)  # simulate a trickle of new comments per poll
        now = time.time()
        batch = []
        for _ in range(n):
            batch.append(
                Comment(
                    id=f"{self._id_prefix}-{next(self._counter)}",
                    author=self._rng.choice(_MOCK_AUTHORS),
                    text=self._rng.choice(_MOCK_COMMENT_POOL),
                    likes=self._rng.randint(0, 500),
                    posted_at=now,
                )
            )
        return batch


# ---------------------------------------------------------------------------
# Unofficial live comment source (best-effort, optional dependency)
# ---------------------------------------------------------------------------

def _ms_token_from_env() -> str | None:
    token = (os.getenv("MS_TOKEN") or os.getenv("ms_token") or "").strip()
    return token or None


class TikTokApiSource:
    """Best-effort comment source using the unofficial `TikTokApi` package.

    Requires `pip install TikTokApi playwright`, `python -m playwright install
    chromium`, and an `MS_TOKEN` in backend/.env. Not guaranteed to keep
    working - there is no official comments API.

    One Playwright browser session is created lazily and reused across polls
    (launching a browser every few seconds is slow and gets rate-limited).
    Playwright objects are bound to the asyncio loop that created them, so
    all TikTokApi calls run on a private event loop thread.
    """

    comments_simulated = False
    comments_per_poll = 50

    def __init__(self, ms_token: str | None = None) -> None:
        try:
            from TikTokApi import TikTokApi
        except ImportError as exc:
            raise CommentSourceUnavailable(
                "TikTokApi is not installed. In backend/ run: "
                "`pip install -r requirements.txt` then `python -m playwright install chromium`."
            ) from exc
        self._TikTokApi = TikTokApi
        self._ms_token = ms_token or _ms_token_from_env()
        if not self._ms_token:
            raise CommentSourceUnavailable(
                "No TikTok msToken configured. Add MS_TOKEN=<your msToken cookie from tiktok.com> "
                "to backend/.env and restart the server."
            )
        self._browser = os.getenv("TIKTOK_BROWSER", "chromium")
        self._headless = os.getenv("TIKTOK_HEADLESS", "1") != "0"
        self._api = None
        self._video_ids: dict = {}
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

    def _run(self, coro, timeout: float = 120.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _ensure_session(self) -> None:
        if self._api is not None:
            return
        api = self._TikTokApi()
        try:
            await api.create_sessions(
                ms_tokens=[self._ms_token],
                num_sessions=1,
                sleep_after=3,
                browser=self._browser,
                headless=self._headless,
                # TikTok's autoplaying feed keeps the network busy forever, so
                # session setup (which waits for "networkidle") times out
                # unless heavy resources are blocked.
                suppress_resource_load_types=["image", "media", "font", "stylesheet"],
            )
        except Exception as exc:
            try:
                await api.close_sessions()
            except Exception:
                pass
            msg = str(exc)
            if "Executable doesn't exist" in msg or "playwright install" in msg:
                msg = "Playwright browser is missing - run `python -m playwright install chromium` in backend/."
            raise RuntimeError(f"Could not open a TikTok session: {msg}") from exc
        self._api = api

    async def _reset_session(self) -> None:
        api, self._api = self._api, None
        if api is not None:
            try:
                await api.close_sessions()
            except Exception:
                pass

    async def _fetch(self, video_url: str, seen_ids: set) -> list:
        await self._ensure_session()
        video_id = self._video_ids.get(video_url)
        video = self._api.video(id=video_id) if video_id else self._api.video(url=video_url)
        self._video_ids[video_url] = video.id

        out = []
        async for c in video.comments(count=self.comments_per_poll):
            data = c.as_dict
            cid = str(data.get("cid") or c.id)
            if cid in seen_ids:
                continue
            created = data.get("create_time")
            out.append(
                Comment(
                    id=cid,
                    author=(data.get("user") or {}).get("unique_id", "unknown"),
                    text=data.get("text", ""),
                    likes=data.get("digg_count", 0),
                    posted_at=float(created) if created else time.time(),
                )
            )
        return out

    def fetch_new_comments(self, video_url: str, seen_ids: set) -> list:
        try:
            return self._run(self._fetch(video_url, seen_ids))
        except Exception as exc:
            # Drop the (possibly blocked/expired) browser session so the next
            # poll starts fresh instead of failing the same way forever.
            try:
                self._run(self._reset_session(), timeout=30)
            except Exception:
                pass
            name = exc.__class__.__name__
            message = str(exc) or name
            if name == "EmptyResponseException" or "detecting you're a bot" in message:
                message = (
                    "TikTok blocked the request as a bot (empty response). Refresh MS_TOKEN in "
                    "backend/.env with a fresh msToken cookie from a logged-in tiktok.com tab, "
                    "then restart the server."
                )
            elif name == "TimeoutError" and not str(exc):
                message = "TikTok comment fetch timed out (TikTok may be slow or blocking this session)."
            elif "TikTok" not in message:
                message = f"TikTok comment fetch failed: {message}"
            raise RuntimeError(message) from exc

    def close(self) -> None:
        try:
            self._run(self._reset_session(), timeout=30)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


def get_source(video_url: str, use_mock: bool = False) -> CommentSource:
    """Build the source for a link: real (yt-dlp + TikTokApi comments) or explicit mock.

    If TikTokApi can't be set up, the returned source still analyzes the
    video but carries `.comments_error` explaining why there are no comments,
    so the UI can show a clear error instead of fake data.
    """
    if use_mock or not is_tiktok_url(video_url):
        return MockTikTokSource()

    try:
        web = TikTokWebClient()
    except ImportError:
        web = None
    try:
        api = TikTokApiSource()
    except CommentSourceUnavailable as exc:
        if web is None:
            return YtDlpSource(comments_error=str(exc))
        api = None
    return YtDlpSource(comment_backend=FallbackCommentSource(web, api), web_client=web)


# ---------------------------------------------------------------------------
# Non-blocking polling loop (Pipeline B)
# ---------------------------------------------------------------------------

class CommentPoller:
    """Polls a CommentSource on a background thread and queues new comments.

    Runs independently of the Flask request lifecycle, so no single HTTP
    request ever blocks on scraping: the request thread only ever drains
    whatever already arrived in `self.queue`.
    """

    def __init__(self, source: CommentSource, video_url: str, interval: float = 5.0, seen_ids: set | None = None) -> None:
        self.source = source
        self.video_url = video_url
        self.interval = interval
        self.queue: "queue.Queue" = queue.Queue()
        self._seen_ids: set = set(seen_ids or ())
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
        failures = 0
        while not self._stop_event.is_set():
            try:
                new_comments = self.source.fetch_new_comments(self.video_url, self._seen_ids)
                for c in new_comments:
                    if c.id not in self._seen_ids:
                        self._seen_ids.add(c.id)
                        self.queue.put(c)
                with self._lock:
                    self._last_error = None
                failures = 0
            except Exception as exc:  # keep polling even if one round fails
                with self._lock:
                    self._last_error = str(exc) or exc.__class__.__name__
                failures += 1
            # Back off after repeated failures so a blocked session isn't hammered.
            self._stop_event.wait(min(self.interval * (2 ** min(failures, 3)), 120.0))
        close = getattr(self.source, "close", None)
        if close:
            close()

    def drain(self) -> list:
        """Pull everything queued since the last drain. Call from the main thread."""
        drained = []
        while True:
            try:
                drained.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return drained
