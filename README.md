# DecisionLens AI — working MVP

A real, runnable implementation of the DecisionLens AI concept: upload data,
get anomaly detection, forecasts, alerts, and natural-language answers —
computed live, not mocked.

**What's real in this build:**
- CSV upload + a bundled synthetic retail dataset (works with zero setup)
- Anomaly detection: z-score per column + Isolation Forest across columns jointly (scikit-learn), grouped by whatever category column exists (e.g. region)
- "Why did X happen" driver analysis: real correlation ranking against the target column
- Forecasting: OLS trend + weekly-seasonality decomposition, with a 95% confidence band
- Natural-language query box: rule-based answer engine that always works, and optionally hands its computed stats to Claude for a more fluent answer if you set `ANTHROPIC_API_KEY`
- Threshold-based live alert feed with an adjustable sensitivity slider

**What's intentionally simplified (see "Honest scope" below) so you get something you can deploy today instead of a pitch deck:**
- Single in-memory dataset per server process (no multi-user database yet)
- No live IoT/API/weather connectors wired in — the ingestion layer is CSV-shaped but the analytics functions don't care where the DataFrame came from
- "GPU acceleration" is not wired in — the code is written so IsolationForest → cuML and pandas → Spark/RAPIDS are drop-in swaps when your data outgrows one machine, but this build runs on CPU

---

## Project structure

```
decisionlens-ai/
├── backend/
│   ├── main.py           FastAPI app — all HTTP routes, also serves the frontend
│   ├── analytics.py      anomaly detection, forecasting, driver analysis (tested, pure functions)
│   ├── nlq.py            natural-language query parsing + optional Claude explainer
│   ├── requirements.txt
│   └── sample_data.csv   synthetic 90-day, 4-region retail dataset with a real injected anomaly
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js            calls the API above — no build step, no framework
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Run it locally (2 minutes)

```bash
cd decisionlens-ai/backend
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — the backend serves the frontend directly.
Click **"Load demo data"** and everything (KPIs, anomalies, forecast, alerts,
NL query) goes live immediately using the bundled retail dataset, which has
a real sales-drop anomaly in the West region caused by low inventory +
a competitor discount — the exact scenario from the product pitch. Try
asking it: *"Why did sales drop today?"*

## Run it with Docker (works the same everywhere)

```bash
cd decisionlens-ai
docker compose up --build
```
Then open http://localhost:8000.

## Deploy it for real

### Option A — Render (simplest, free tier available)
1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com) → New → Web Service → connect the repo.
3. Root directory: `backend`. Build command: `pip install -r requirements.txt`.
   Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Add an environment variable `ANTHROPIC_API_KEY` only if you want LLM-powered answers.
5. Deploy. Render gives you a public URL serving both the API and the dashboard.

### Option B — Railway
1. `railway init` in the project root, or connect the GitHub repo in the dashboard.
2. Railway auto-detects the `Dockerfile` at the repo root and builds it.
3. Set `PORT` (Railway sets this automatically) and optionally `ANTHROPIC_API_KEY`.
4. Deploy — you get a public URL.

### Option C — Any VM / EC2 / a Droplet
```bash
git clone <your-repo>
cd decisionlens-ai
docker compose up -d --build
```
Put it behind nginx + Let's Encrypt (or a load balancer) for HTTPS in production.

### Option D — Split frontend/backend (e.g. Vercel + Render)
The frontend is plain HTML/CSS/JS with no build step, so it can be hosted
anywhere static (Vercel, Netlify, S3+CloudFront). If you split it from the
backend, open `frontend/app.js` and change `API_BASE` to your backend's
public URL, and add that frontend origin to the `allow_origins` list in
`backend/main.py` (currently `"*"` for convenience — tighten this before
going to production).

## Turning on GPU acceleration / your real data sources

Every analytics function in `backend/analytics.py` takes a plain pandas
`DataFrame` and returns a plain dict/list — nothing above it cares how that
DataFrame was produced. To grow this into the full pitch:

1. **Real data sources**: add ingestion functions (DB connectors, a weather
   API client, an IoT MQTT/Kafka consumer, a market-data feed) that each
   produce a DataFrame, and merge/append them into the store in `main.py`.
2. **Scale**: once a single machine's RAM/CPU isn't enough, swap
   `pandas` → `Spark` or `RAPIDS cuDF` for the DataFrame operations, and
   `sklearn.ensemble.IsolationForest` → `cuML`'s GPU-accelerated equivalent.
   The function signatures don't need to change.
3. **Multi-user / persistence**: replace the single in-memory `DataStore`
   in `main.py` with Postgres (metadata, users, alert rules) + a columnar
   store (Parquet on S3, or DuckDB) for the actual rows, keyed by
   user/session/dataset id.
4. **Smarter NL queries**: set `ANTHROPIC_API_KEY` — `nlq.py` already
   calls Claude with the pre-computed stats (never raw data) to produce
   a fluent, decision-ready answer instead of the rule-based template.

## Verifying accuracy

The analytics engine was tested against the bundled dataset before being
wired into the API (not just assumed to work):
- The West-region sales drop is correctly flagged as a top anomaly.
- The driver analysis correctly ranks `competitor_discount` (inverse) and
  `inventory` (direct) as the strongest correlates of the sales drop —
  matching the causal story the dataset was generated with.
- The forecast correctly identifies a downward trend for that region.

You can re-run this check yourself:
```bash
cd backend
python3 -c "
import pandas as pd
from analytics import detect_anomalies, correlation_drivers
df = pd.read_csv('sample_data.csv')
west = df[df.region == 'West']
print(correlation_drivers(west, target='sales'))
"
```

## API reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/load-sample` | load the bundled demo dataset |
| POST | `/api/upload` | upload a CSV/TSV (multipart form, field `file`) |
| GET | `/api/summary` | per-column stats for the KPI cards |
| GET | `/api/insights` | anomaly list (z-score + Isolation Forest) |
| GET | `/api/drivers?target=<col>` | correlation-based driver ranking |
| GET | `/api/forecast?column=<col>&periods=<n>&group=<val>` | trend + seasonal forecast with confidence band |
| GET | `/api/alerts?z_threshold=<f>` | live threshold alerts |
| POST | `/api/query` | body `{"question": "..."}` — natural-language answer |
| GET | `/api/groups` | distinct values of the detected grouping column |
| GET | `/api/health` | liveness + whether a dataset is loaded |

Interactive API docs are auto-generated by FastAPI at `/docs` once the
server is running.
