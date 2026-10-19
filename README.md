# TikTok Sentiment Analysis of Foundation Cosmetic Products

A web app that classifies TikTok comments about foundation cosmetic products as
**Positive / Neutral / Negative** using XLM-RoBERTa, with pages for input,
results, visualizations, and a browsable dataset.

- **Frontend**: plain HTML / CSS / JavaScript (`frontend/`)
- **Backend**: Flask REST API (`backend/`) that runs XLM-RoBERTa sentiment
  inference and persists analyzed comments to a local SQLite database
- **Model**: [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)
  (multilingual XLM-RoBERTa, covers English, Tagalog and Taglish) further
  fine-tuned on foundation-review comments, stored in
  `backend/models/foundation-sentiment-xlmr/`. See "Model & retraining" below.

## Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### TikTok Link setup (audio transcription)

The "TikTok Link" input on the Analysis page downloads a video's audio with
`yt-dlp`, transcribes it with Whisper, then classifies the transcript — this
needs the `ffmpeg` system binary on top of the pip packages above:

```bash
brew install ffmpeg        # macOS
# apt install ffmpeg       # Debian/Ubuntu
```

The caption, audio and live comments are fetched straight from tiktok.com's
web endpoints with `curl_cffi` (browser TLS fingerprints, no login or token
needed); this is best-effort, since TikTok has no official comments API. If
that is blocked, the app falls back to `yt-dlp` for the video and to the
unofficial `TikTokApi` package for comments, which needs a Playwright browser
and your `msToken` cookie:

```bash
python -m playwright install chromium
cp .env.example .env       # then set MS_TOKEN=... in backend/.env
```

To get `msToken`: log in at https://www.tiktok.com in Chrome, open DevTools →
Application → Cookies → `https://www.tiktok.com`, and copy the `msToken`
value. It expires, so refresh it when the Results page reports TikTok
blocking the request. `.env` is git-ignored — never commit the token.

If comment fetching can't work (no token, package/browser missing, TikTok
blocking), the Results page shows the error instead of fake comments; the
video's caption/transcript sentiment is still analyzed.

This whole feature is built for **local use** (your machine), not the hosted
Render deployment — `yt-dlp`/Whisper/`ffmpeg` and long-running background
polling threads don't fit that deployment's memory/runtime constraints.

## Run

```bash
cd backend
python app.py
```

Then open **http://localhost:5000** in your browser. Flask serves both the
API (`/api/*`) and the static frontend, so there's nothing else to start.

The first request that hits `/api/analyze` will download the XLM-RoBERTa
model (~1.1GB) from Hugging Face and cache it locally — this can take a few
minutes the first time. The first TikTok Link analysis will similarly
download a Whisper model.

## Model & retraining

| On 103 held-out real TikTok comments | accuracy | macro precision | macro recall | macro F1 |
|---|---|---|---|---|
| stock cardiffnlp checkpoint (old app) | 0.874 | 0.786 | 0.834 | 0.798 |
| fine-tuned model (current) | 0.951 | 0.921 | 0.960 | 0.938 |

Full report with confusion matrices: `backend/models/foundation-sentiment-xlmr/evaluation.md`.

```bash
python3 -m venv --system-site-packages training/venv
training/venv/bin/pip install -r training/requirements.txt
training/venv/bin/python training/build_dataset.py     # raw data -> training/data/*.csv
training/venv/bin/python training/train_sentiment.py   # ~15 min on CPU, saves to backend/models/
backend/venv/bin/python training/evaluate_app.py       # old vs new, through the app's classify()
```

- **Data** (`training/raw/`, foundation reviews only): 169 real TikTok
  comments + 80 hand-written comments, and the synthetic `combined_dataset.csv`
  reviews (template openers, fillers and label-revealing emoji stripped,
  shade-name copies collapsed). The test set is real comments only. Label
  fixes go in `training/label_corrections.csv`. (The current model was also
  trained on general Taglish Shopee reviews, since removed from the folder.)
- **Preprocessing** lives in `backend/preprocessing.py` and is imported by
  both the training script and the app, so they always match.
- **Weights aren't in git** (1.1GB). Without them the app falls back to the
  stock checkpoint; `SENTIMENT_MODEL=<path or HF repo>` overrides the model,
  and `SENTIMENT_RULES=1/0` toggles the lexicon rule layer (default: on only
  for the stock checkpoint). Comments that mix praise and complaints in
  roughly equal measure ("Ang ganda ng coverage pero nag-oxidize") are
  labeled **neutral**; `SENTIMENT_MIXED_NEUTRAL=0` turns that off. Product
  questions ("Nag-oxidize po ba?", "FDA approved?") are neutral too
  (`SENTIMENT_QUESTIONS_NEUTRAL=0` turns that off), as are unsure negative
  calls with no complaint word in them. For the hosted Render deployment, upload the
  folder to a Hugging Face model repo and set `SENTIMENT_MODEL` to it.
- **Colab**: upload `training/` and `backend/preprocessing.py` (same layout),
  `pip install transformers==4.46.0 sentencepiece scikit-learn pandas openpyxl`,
  then run the two training commands; a T4 GPU trains in a few minutes.

## Pages

| Page | File | Description |
|---|---|---|
| Home | `frontend/index.html` | Landing page / overview |
| Analysis | `frontend/analysis.html` | Paste comments (+ optional product name) and run BERT sentiment analysis |
| Results | `frontend/results.html` | Overall sentiment donut, breakdown table, sample comments |
| Visualization | `frontend/visualization.html` | Sentiment distribution, trend over time, by-product chart, word cloud |
| Dataset | `frontend/dataset.html` | Searchable, paginated table of all analyzed comments + CSV export |
| About | `frontend/about.html` | System/technology overview |

## API

- `POST /api/analyze` — `{ comments: string[], product?: string }` → runs XLM-RoBERTa, stores results, returns summary + per-comment sentiment
- `POST /api/tiktok/analyze` — `{ url: string, product?: string }` → extracts the video's audio, transcribes it (Whisper), classifies it, and starts a background live-comment poller; returns `{ job_id, video, summary, results }`
- `GET /api/tiktok/updates/<job_id>` — drains and classifies whatever new comments the poller queued, returns running totals: `{ summary, results, new_count, is_running, last_error }`
- `POST /api/tiktok/stop/<job_id>` — stops the background poller for a job
- `GET /api/dataset?page=&page_size=&search=&product=` — paginated dataset
- `GET /api/dataset/products` — distinct product names
- `GET /api/dataset/export` — CSV download of the full dataset
- `GET /api/stats` — aggregate distribution, daily trend, by-product counts, word frequencies

## Notes

- On first run the SQLite database (`backend/dataset.db`) is seeded with ~120
  sample comments so the Visualization and Dataset pages have data to show
  immediately.
- "TikTok Link" analysis jobs (and their background pollers) live in-process
  and are pruned after 30 minutes of inactivity — this is built for a single
  local user, not concurrent multi-user production use.
- TikTok comment scraping has no official API and depends on the unofficial
  `TikTokApi` package; treat the live comment feed as best-effort.
