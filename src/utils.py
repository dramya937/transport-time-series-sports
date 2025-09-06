
import math
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance (in km) between two points.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2 +
         math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

def add_coords(df, cities_df, col_name):
    """
    Left-join city coordinates into df for a given column name (e.g., 'venue_city').
    Creates columns: {col_name}_lat, {col_name}_lon, {col_name}_airport
    """
    merge_df = cities_df.rename(columns={
        "city": col_name,
        "lat": f"{col_name}_lat",
        "lon": f"{col_name}_lon",
        "airport": f"{col_name}_airport"
    })
    return df.merge(merge_df, on=col_name, how="left")

def compute_trip_legs(df_games, cities_df, base_city="Dallas"):
    """
    Given games sorted by date, compute the travel legs for the team based in base_city.
    A leg occurs when the venue city changes from the previous game (or from base city for game 1).
    Returns a dataframe of legs with from_city, to_city, km, and date.
    """
    df = df_games.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    # Determine where the team needs to be for each game: venue_city
    # Team always travels to the venue city from the previous venue (or base city initially)
    legs = []
    prev_city = base_city
    prev_date = None

    for _, row in df.iterrows():
        to_city = row["venue_city"]
        if to_city != prev_city:
            # lookup coords
            from_row = cities_df[cities_df["city"] == prev_city].iloc[0]
            to_row = cities_df[cities_df["city"] == to_city].iloc[0]
            km = haversine(from_row["lat"], from_row["lon"], to_row["lat"], to_row["lon"])
            legs.append({
                "date": row["date"],
                "from_city": prev_city,
                "to_city": to_city,
                "km": km
            })
        prev_city = to_city
        prev_date = row["date"]

    # After last game, travel home
    if prev_city != base_city:
        from_row = cities_df[cities_df["city"] == prev_city].iloc[0]
        to_row = cities_df[cities_df["city"] == base_city].iloc[0]
        km = haversine(from_row["lat"], from_row["lon"], to_row["lat"], to_row["lon"])
        legs.append({
            "date": prev_date,
            "from_city": prev_city,
            "to_city": base_city,
            "km": km
        })

    return pd.DataFrame(legs)

def basic_cost_model(km_series, seed=0):
    """
    Simple synthetic flight cost model (USD) driven by distance with seasonality and noise.
    cost_t = 0.12 * km + seasonal_factor + noise
    Seasonal factor: monthly sine/cosine terms to mimic higher winter/holiday costs.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    costs = []
    dates = km_series.index
    for dt, km in zip(dates, km_series.values):
        month = dt.month
        # seasonality (peaks in Dec/Mar)
        seasonal = 40.0 + 10.0 * np.sin(2 * np.pi * (month-1) / 12.0) + 8.0 * np.cos(2 * np.pi * (month-3) / 12.0)
        noise = rng.normal(0, 25.0)
        cost = 0.12 * km + seasonal + noise
        costs.append(max(50.0, cost))  # floor at 50
    return pd.Series(costs, index=dates)
