"""Real-time TikTok foundation-cosmetics sentiment dashboard (Streamlit).

Dual-processing workflow:
  1. Video analysis: fetch the creator's own video (caption/transcript) for
     the given link and classify it to establish the base video sentiment.
  2. Live comment polling: a background CommentPoller fetches fresh audience
     comments on an interval without blocking this script; each rerun drains
     whatever arrived and classifies it immediately.
The dashboard below is a reactive view over that growing comment pool - every
rerun recomputes percentages, metrics, and charts from current state.
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import model_pipeline
import scraper

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

try:
    import TikTokApi  # noqa: F401
    HAS_TIKTOKAPI = True
except ImportError:
    HAS_TIKTOKAPI = False

# Fixed categorical-by-entity colors (status-style: good / neutral / critical),
# used consistently across every chart and badge so color never carries
# meaning alone - every use is paired with a text label or icon.
SENTIMENT_COLORS = {"positive": "#0ca30c", "neutral": "#898781", "negative": "#d03b3b"}
SENTIMENT_ICONS = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}
SENTIMENT_ORDER = ("positive", "neutral", "negative")

st.set_page_config(
    page_title="TikTok Foundation Sentiment Dashboard",
    page_icon="💄",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading XLM-RoBERTa sentiment model (first run only)…")
def _warm_model() -> bool:
    model_pipeline.get_classifier()
    return True


def _init_state() -> None:
    defaults = {
        "video_info": None,
        "video_result": None,
        "poller": None,
        "comments": [],  # list of dicts: id, author, text, likes, posted_at, sentiment, score, scores
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _classify_comment(comment: scraper.Comment) -> dict:
    result = model_pipeline.classify_one(comment.text)
    return {
        "id": comment.id,
        "author": comment.author,
        "text": comment.text,
        "likes": comment.likes,
        "posted_at": comment.posted_at,
        "sentiment": result.sentiment,
        "score": result.score,
        "scores": result.scores,
        "matched_terms": ", ".join(result.matched_terms),
    }


def _start_analysis(video_url: str, use_mock: bool, interval: float) -> None:
    source = scraper.get_source(use_mock=use_mock)
    video_info = source.fetch_video_info(video_url)
    video_text = model_pipeline.build_video_text(video_info.caption, video_info.transcript)
    video_result = model_pipeline.classify_one(video_text)

    old_poller = st.session_state.get("poller")
    if old_poller is not None:
        old_poller.stop()

    poller = scraper.CommentPoller(source, video_url, interval=interval)
    poller.start()

    st.session_state.video_info = video_info
    st.session_state.video_result = video_result
    st.session_state.poller = poller
    st.session_state.comments = []


def _stop_analysis() -> None:
    poller = st.session_state.get("poller")
    if poller is not None:
        poller.stop()
    st.session_state.poller = None


def _drain_new_comments() -> None:
    poller = st.session_state.get("poller")
    if poller is None:
        return
    for comment in poller.drain():
        st.session_state.comments.append(_classify_comment(comment))


def _sentiment_badge(sentiment: str, score: float) -> str:
    color = SENTIMENT_COLORS[sentiment]
    icon = SENTIMENT_ICONS[sentiment]
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}66;'
        f'border-radius:999px;padding:4px 12px;font-weight:600;">'
        f"{icon} {sentiment.title()} ({score * 100:.0f}%)</span>"
    )


def _render_sidebar() -> tuple[str, bool, float]:
    st.sidebar.header("Video source")
    video_url = st.sidebar.text_input(
        "TikTok video link", placeholder="https://www.tiktok.com/@creator/video/123..."
    )
    use_mock = st.sidebar.checkbox(
        "Use demo/mock data",
        value=not HAS_TIKTOKAPI,
        disabled=not HAS_TIKTOKAPI,
        help=(
            "Live scraping needs the optional `TikTokApi` package (not installed)."
            if not HAS_TIKTOKAPI
            else "Uncheck to poll the real video via the unofficial TikTokApi backend."
        ),
    )
    interval = st.sidebar.slider("Comment poll interval (seconds)", 2, 30, 5)

    col1, col2 = st.sidebar.columns(2)
    start_clicked = col1.button("▶ Start", type="primary", use_container_width=True)
    stop_clicked = col2.button("■ Stop", use_container_width=True)

    if start_clicked and video_url.strip():
        _start_analysis(video_url.strip(), use_mock, interval)
    elif start_clicked:
        st.sidebar.warning("Enter a TikTok video link first.")

    if stop_clicked:
        _stop_analysis()

    poller = st.session_state.get("poller")
    if poller is not None:
        status = "🟢 polling live" if poller.is_running() else "⚪ stopped"
        st.sidebar.caption(f"Status: {status}")
        last_error = poller.get_last_error()
        if last_error:
            st.sidebar.warning(f"Last poll error: {last_error}")

    return video_url, use_mock, interval


def _render_video_overview() -> None:
    st.subheader("Video overview")
    video_info = st.session_state.video_info
    video_result = st.session_state.video_result
    if video_info is None:
        st.info("Enter a video link and click **Start** to analyze it.")
        return

    st.markdown(f"**Creator:** @{video_info.author}")
    with st.expander("Caption", expanded=True):
        st.write(video_info.caption or "_No caption available._")
    with st.expander("Transcript"):
        st.write(video_info.transcript or "_No transcript available._")

    if video_result is not None:
        st.markdown("**Base video sentiment**")
        st.markdown(_sentiment_badge(video_result.sentiment, video_result.score), unsafe_allow_html=True)
        if video_result.matched_terms:
            st.caption(f"Local slang detected: {', '.join(video_result.matched_terms)}")


def _render_metrics(df: pd.DataFrame) -> None:
    st.subheader("Live sentiment metrics")
    total = len(df)
    counts = {s: int((df["sentiment"] == s).sum()) for s in SENTIMENT_ORDER} if total else {s: 0 for s in SENTIMENT_ORDER}

    cols = st.columns(4)
    cols[0].metric("Total comments", total)
    for col, sentiment in zip(cols[1:], SENTIMENT_ORDER):
        pct = (counts[sentiment] / total * 100) if total else 0.0
        col.markdown(
            f'<div style="color:{SENTIMENT_COLORS[sentiment]};font-size:0.85rem;font-weight:600;">'
            f"{SENTIMENT_ICONS[sentiment]} {sentiment.title()}</div>"
            f'<div style="font-size:1.6rem;font-weight:700;">{pct:.1f}%</div>'
            f'<div style="color:#898781;font-size:0.8rem;">{counts[sentiment]} comments</div>',
            unsafe_allow_html=True,
        )


def _render_distribution_chart(df: pd.DataFrame) -> None:
    st.markdown("**Current distribution**")
    if df.empty:
        st.caption("No comments yet.")
        return
    counts = [int((df["sentiment"] == s).sum()) for s in SENTIMENT_ORDER]
    fig = go.Figure(
        go.Pie(
            labels=[s.title() for s in SENTIMENT_ORDER],
            values=counts,
            hole=0.55,
            sort=False,  # keep fixed sentiment order, never resort by value
            marker=dict(colors=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER]),
            textinfo="label+percent",
        )
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def _render_trend_chart(df: pd.DataFrame) -> None:
    st.markdown("**Sentiment mix as comments arrive**")
    if len(df) < 2:
        st.caption("Needs at least 2 comments to show a trend.")
        return
    cum = {s: (df["sentiment"] == s).cumsum() for s in SENTIMENT_ORDER}
    x = list(range(1, len(df) + 1))
    fig = go.Figure()
    for sentiment in SENTIMENT_ORDER:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=cum[sentiment],
                name=sentiment.title(),
                mode="lines",
                stackgroup="one",
                groupnorm="percent",
                line=dict(width=2, color=SENTIMENT_COLORS[sentiment]),
            )
        )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        xaxis_title="Comment arrival order",
        yaxis_title="% of comments so far",
        yaxis=dict(ticksuffix="%"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_feed(df: pd.DataFrame) -> None:
    st.subheader("Raw comment feed")
    if df.empty:
        st.caption("No comments yet. They'll appear here as they're polled.")
        return
    view = df.sort_values("posted_at", ascending=False).copy()
    view["sentiment"] = view["sentiment"].map(lambda s: f"{SENTIMENT_ICONS[s]} {s.title()}")
    view["confidence"] = (view["score"] * 100).round(1).astype(str) + "%"
    view["posted_at"] = pd.to_datetime(view["posted_at"], unit="s").dt.strftime("%H:%M:%S")
    st.dataframe(
        view[["posted_at", "author", "text", "sentiment", "confidence", "likes", "matched_terms"]].rename(
            columns={
                "posted_at": "Time",
                "author": "Author",
                "text": "Comment",
                "sentiment": "Sentiment",
                "confidence": "Confidence",
                "likes": "Likes",
                "matched_terms": "Slang detected",
            }
        ),
        use_container_width=True,
        height=360,
    )


def main() -> None:
    _init_state()
    _warm_model()

    st.title("💄 Real-Time TikTok Foundation Sentiment Dashboard")
    st.caption("XLM-RoBERTa sentiment · English / Tagalog / Taglish · live comment polling")

    _render_sidebar()
    _drain_new_comments()

    df = pd.DataFrame(st.session_state.comments)

    left, right = st.columns([1, 1.4])
    with left:
        _render_video_overview()
    with right:
        _render_metrics(df)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        _render_distribution_chart(df)
    with chart_right:
        _render_trend_chart(df)

    _render_feed(df)

    if st.session_state.get("poller") is not None:
        if HAS_AUTOREFRESH:
            st_autorefresh(interval=3000, key="live_poll_tick")
        else:
            st.button("🔄 Refresh now")


if __name__ == "__main__":
    main()
