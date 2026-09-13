# TikTok Sentiment Analysis of Foundation Cosmetic Products

A web app that classifies TikTok comments about foundation cosmetic products as
**Positive / Neutral / Negative** using a BERT model, with pages for input,
results, visualizations, and a browsable dataset.

- **Frontend**: plain HTML / CSS / JavaScript (`frontend/`)
- **Backend**: Flask REST API (`backend/`) that runs BERT sentiment inference
  and persists analyzed comments to a local SQLite database
- **Model**: [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
  (BERT fine-tuned for 1-5 star sentiment; mapped to negative/neutral/positive)

## Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd backend
python app.py
```

Then open **http://localhost:5000** in your browser. Flask serves both the
API (`/api/*`) and the static frontend, so there's nothing else to start.

The first request that hits `/api/analyze` will download the BERT model
(~700MB) from Hugging Face and cache it locally — this can take a few minutes
the first time.

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

- `POST /api/analyze` — `{ comments: string[], product?: string }` → runs BERT, stores results, returns summary + per-comment sentiment
- `GET /api/dataset?page=&page_size=&search=&product=` — paginated dataset
- `GET /api/dataset/products` — distinct product names
- `GET /api/dataset/export` — CSV download of the full dataset
- `GET /api/stats` — aggregate distribution, daily trend, by-product counts, word frequencies

## Notes

- On first run the SQLite database (`backend/dataset.db`) is seeded with ~120
  sample comments so the Visualization and Dataset pages have data to show
  immediately.
- The "TikTok Link" input tab is present in the UI but scraping TikTok
  directly isn't implemented — paste comments instead.
