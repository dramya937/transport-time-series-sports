
# Time Series Analysis for Transportation — Sports Scheduling

Unlock the value of transportation time series data with a journey into **scheduling analysis**.
This repo shows how to go from a season schedule → **travel legs** → **weekly distance time series** → **flight cost forecasting**.

## What you can do
- Parse a team's season schedule and compute **travel legs** and **distances** (km) using haversine.
- Aggregate to **weekly travel** time series.
- Generate a synthetic but realistic **flight-cost** series tied to distance + seasonality.
- Fit a SARIMAX model with exogenous regressor (weekly km) and produce **12-week forecasts**.
- Visualize weekly distance, cumulative distance, and forecasted costs.
- Swap in **real schedules** later (NBA/NHL/NFL/MLB) — only the `data/schedule.csv` format matters.

## Project Structure
```
transport-time-series-sports/
├── data/
│   ├── cities.csv          # lat/lon/airport for major US/Canada cities
│   └── schedule.csv        # synthetic season for a Dallas-based team
├── src/
│   ├── utils.py            # haversine, travel legs, synthetic cost generator
│   └── analysis.py         # end-to-end pipeline + plots + forecast CSV
├── assets/                 # plots and outputs land here
├── run.py                  # CLI to run the analysis
├── requirements.txt
└── README.md
```

## How to run
```bash
# 1) Install (ideally in a virtualenv)
pip install -r requirements.txt

# 2) Run
python run.py

# Outputs → assets/
#   - weekly_km.png
#   - cumulative_km.png
#   - cost_forecast.png
#   - weekly_cost_forecast.csv
#   - travel_legs.csv
```

## Data format

**`data/schedule.csv`** (you can replace this with a real league schedule later):
```
game_id,date,home_city,away_city,venue_city
1,2024-10-03,Dallas,Chicago,Dallas
2,2024-10-06,Miami,Dallas,Miami
...
```
Only `date` and `venue_city` are required for the legs computation; the rest are for readability.

**`data/cities.csv`** should contain the team's home city plus opponent cities with lat/lon and an airport code:
```
city,lat,lon,airport
Dallas,32.7767,-96.7970,DFW
Chicago,41.8781,-87.6298,ORD
...
```

## Swap in real data
- Export a real schedule to `data/schedule.csv` in the same format.
- If you add new cities, append them to `data/cities.csv` with lat/lon.
- Rerun `python run.py` — the pipeline recomputes legs, weekly distance, and forecasts.

## Modeling notes
- The cost series is **synthetic** but driven by weekly travel distance with seasonal effects and noise.
- We fit a **SARIMAX** model (`(1,0,1)` with seasonal `(1,0,1,52)`) using weekly distance as an **exogenous regressor**.
- You can try: auto-ARIMA, Prophet, gradient-boosted trees on lagged features, or a Bayesian structural time series.

## Ideas to extend
- Add **rest-day** features and back-to-back counts.
- Model **hotel** and **per-diem** costs separately.
- Compare **alternative schedules** (what-if analysis) for miles saved.
- Add **carbon emissions** estimation using distance × emissions factor.
- Per-leg fare curves using **advance-purchase windows**.

---

*Made for portfolio use. Replace the synthetic schedule with a real one to make it league-specific.*
