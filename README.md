# VWCE Health Check

A Streamlit dashboard that monitors the Vanguard FTSE All-World UCITS ETF (VWCE.DE). Built for family and friends who feel the need to monitor their investment and want a quick daily check with a bit more perspective than the daily graph that shows up if you google it.

## Overview

The dashboard is meant for medium-to-long-term investors who invest savings once a month or so. The main graph shows daily prices over a period of several years, along with a 200 day smoothed average.

The most interesting part is the health score, a single number (0–100) that summarizes investment health at a glance. It is meant to reflect the medium-term health of the ticker, without giving much weight to brief fluctuations. At first we put a bunch of complex indicators: moving averages (200 day), momentum across multiple timeframes, Sharpe ratio... Then we threw most of them out leaving only a few percentages:

- 5-year return
- Last year's return
- Distance from 5-year high

They are aggregated with a weighted mean; we chose the weights and metrics based on how much they aligned with our subjective sense of investment "health".

Data comes from Yahoo Finance (`yfinance`). It's free, flaky, but good enough for our needs here.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Prices are cached for 1 hour.

## Architecture

The code is structured to be easy to fork and adapt:

- `streamlit_app.py`: Streamlit UI
- `lib/`: business logic (not Streamlit-specific)
  - `data_fetcher.py`: downloads historical data from Yahoo Finance
  - `metrics.py`: calculates return and drawdown
  - `health_score.py`: computes the health score
- `tests/`: unit and integration tests

The ticker defaults to VWCE.DE but can be changed in the UI, and all of the business logic is ticker-independent.

## For developers

It's MIT, have fun...

Testing: `pytest tests/ -v` (integration tests excluded by default, run with `-m integration` to include)

### Troubleshooting

Most issues can be diagnosed from the Diagnostics panel at the bottom of the page. It shows the ticker, rows downloaded, last date, and library versions.

If you see stale or missing data, try to clear the Streamlit cache (menu → "Clear cache") and reload.

Yahoo Finance is known to be unreliable, so if the problem persists, it's probably their feed. You can inspect the raw data to find out:
```python
from lib import DataFetcher
df = DataFetcher().fetch_historical_data("VWCE.DE", period="5y", interval="1d")
df.tail(10)
```

## Roadmap

Goals:
- [X] Daily health score based on three intuitive metrics
- [X] Cached data (1 hour TTL)
- [X] Diagnostics panel for debugging
- [X] Basic test coverage
- [ ] Multi-ticker portfolio support?

Non-goals:
- Technical indicators 
- Predictions/forecasts
- Alerts/notifications
