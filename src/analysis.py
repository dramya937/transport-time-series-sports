
"""
Time Series Analysis & Scheduling Insights for a Sports Team

Steps:
1) Load synthetic schedule and city coords
2) Build travel legs and distances via haversine
3) Aggregate to a time series (total km traveled per week)
4) Generate synthetic flight costs tied to travel distance
5) Fit a SARIMAX model to forecast costs for the next 12 weeks
6) Produce plots & CSV outputs
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils import add_coords, compute_trip_legs, basic_cost_model

DATA_DIR = "data"
ASSETS_DIR = "assets"
BASE_CITY = "Dallas"

def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    df_cities = pd.read_csv(os.path.join(DATA_DIR, "cities.csv"))
    df_sched = pd.read_csv(os.path.join(DATA_DIR, "schedule.csv"))
    df_sched["date"] = pd.to_datetime(df_sched["date"])

    # Compute travel legs and distances
    legs = compute_trip_legs(df_sched, df_cities, base_city=BASE_CITY)
    # Create weekly time series of distance
    legs["week"] = legs["date"].dt.to_period("W").apply(lambda r: r.start_time)
    ts_km_weekly = legs.groupby("week")["km"].sum().sort_index()
    ts_km_weekly.index.name = "week"

    # Synthetic cost series tied to weekly km
    cost_series = basic_cost_model(ts_km_weekly)
    cost_series.name = "flight_cost_usd"

    # Fit SARIMAX with exogenous regressor (weekly km)
    # Make both series aligned and regular (weekly index)
    weekly_index = pd.date_range(ts_km_weekly.index.min(), ts_km_weekly.index.max(), freq="W")
    km = ts_km_weekly.reindex(weekly_index).fillna(0.0)
    cost = cost_series.reindex(weekly_index).interpolate()

    # Train/test split (last 8 weeks as test)
    split = -8 if len(weekly_index) > 10 else -2
    train_cost, test_cost = cost.iloc[:split], cost.iloc[split:]
    train_km, test_km = km.iloc[:split], km.iloc[split:]

    # Fit SARIMAX
    model = SARIMAX(train_cost, order=(1,0,1), seasonal_order=(1,0,1,52),
                    exog=train_km, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    # Forecast horizon = len(test) + 12 weeks ahead
    horizon = len(test_cost) + 12
    exog_future = pd.concat([test_km, pd.Series([0.0]*12, index=pd.date_range(test_km.index[-1]+pd.Timedelta(weeks=1), periods=12, freq="W"))])
    pred = res.get_forecast(steps=horizon, exog=exog_future)
    pred_mean = pred.predicted_mean
    conf = pred.conf_int(alpha=0.2)

    # Save outputs
    out_df = pd.DataFrame({
        "week": pred_mean.index,
        "predicted_cost": pred_mean.values,
        "lower": conf.iloc[:,0].values,
        "upper": conf.iloc[:,1].values
    })
    out_path = os.path.join("assets", "weekly_cost_forecast.csv")
    out_df.to_csv(out_path, index=False)

    # PLOTS
    # Plot 1: Weekly km traveled
    plt.figure()
    plt.plot(km.index, km.values)
    plt.title("Weekly Travel Distance (km)")
    plt.xlabel("Week")
    plt.ylabel("Kilometers")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "weekly_km.png"), dpi=160)
    plt.close()

    
    # Plot 2: Historical cost + forecast (datetime-aware fill_between)
    import matplotlib.dates as mdates
    plt.figure()
    plt.plot(cost.index, cost.values, label="History")
    plt.plot(pred_mean.index, pred_mean.values, label="Forecast")
    x_dates = mdates.date2num(pred_mean.index.to_pydatetime())
    lower = conf.iloc[:,0].values.astype(float)
    upper = conf.iloc[:,1].values.astype(float)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.fill_between(x_dates, lower, upper, alpha=0.2, label="80% CI")
    plt.title("Flight Cost: History + Forecast")
    plt.xlabel("Week")
    plt.ylabel("USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "cost_forecast.png"), dpi=160)
    plt.close()

    # Plot 3: Cumulative season distance traveled
    plt.figure()
    cum_km = km.cumsum()
    plt.plot(cum_km.index, cum_km.values)
    plt.title("Cumulative Travel Distance (Season)")
    plt.xlabel("Week")
    plt.ylabel("Kilometers (cumulative)")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "cumulative_km.png"), dpi=160)
    plt.close()

    # Export legs to CSV
    legs.to_csv(os.path.join(ASSETS_DIR, "travel_legs.csv"), index=False)

    print("Analysis complete. Outputs saved to /assets.")

if __name__ == "__main__":
    main()
