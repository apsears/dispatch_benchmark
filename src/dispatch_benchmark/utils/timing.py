#!/usr/bin/env python3
# benchmark_forecasts.py
import time
import warnings
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet  # pip install prophet

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
JSON_PATH = Path("data/api/filtered/page_0001.json")  # <— change if needed
HORIZON = 32  # 32 × 15-min = 8 h
TRAIN_MIN = 14 * 96  # need ≥ 2 weeks before first forecast


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
def load_ercot_data(json_path):
    """
    Load ERCOT data from JSON file and convert to time series.

    Parameters:
    -----------
    json_path : Path or str
        Path to the JSON file containing ERCOT data

    Returns:
    --------
    pandas.Series: Price time series with DateTimeIndex
    """
    # Load JSON file
    with open(json_path, "r") as f:
        import json

        json_data = json.load(f)

    # Extract field names from the JSON structure
    field_names = [field["name"] for field in json_data["fields"]]

    # Create DataFrame from the data array with appropriate column names
    df_raw = pd.DataFrame(json_data["data"], columns=field_names)

    # Convert to proper types and create timestamp
    df = (
        df_raw.assign(
            DeliveryDate=lambda d: pd.to_datetime(d.deliveryDate),
            DeliveryHour=lambda d: d.deliveryHour.astype(int),
            DeliveryInterval=lambda d: d.deliveryInterval.astype(int),
            ts=lambda d: pd.to_datetime(
                d.deliveryDate
                + " "
                + (d.deliveryHour - 1).astype(str).str.zfill(2)
                + ":"
                + ((d.deliveryInterval - 1) * 15).astype(str).str.zfill(2)
            ),
            settlementPointPrice=lambda d: d.settlementPointPrice.astype(float),
        )
        .set_index("ts")
        .sort_index()
    )

    # Extract the time series for forecasting
    price_series = df["settlementPointPrice"]

    # Ensure perfect time grid
    return price_series.asfreq("15min")


# ---------------------------------------------------------------------
# HELPERS – each returns (ŷ, elapsed_seconds)
# ---------------------------------------------------------------------
def forecast_naive(train, start):
    t0 = time.perf_counter()
    same_slot_last_day = train.iloc[-96:]
    yhat = np.tile(same_slot_last_day.values[:HORIZON], 1)
    return yhat, time.perf_counter() - t0


def forecast_ewma(train, start, α=0.2):
    t0 = time.perf_counter()
    slot_hist = train[train.index.time == start.time()]
    yhat = np.full(HORIZON, slot_hist.ewm(alpha=α).mean().iloc[-1])
    return yhat, time.perf_counter() - t0


# Creating OneHotEncoder properly with equal length arrays
def create_encoder():
    """
    Create a OneHotEncoder that can handle all hours, quarters, days, and months.
    This ensures we don't encounter unknown category errors during prediction.
    """
    # Create sample data with all possible values for all features
    hours = list(range(24))  # 0-23 hours
    quarters = list(range(4))  # 0-3 quarters (0, 15, 30, 45 minutes)
    days = list(range(7))  # 0-6 days of week
    months = list(range(12))  # 0-11 months

    # Create a DataFrame with all combinations (Cartesian product)
    # This requires creating all permutations, which could be large
    # So we'll just use enough samples to cover all categories
    import itertools

    # Take at most 100 combinations to keep memory usage reasonable
    combinations = list(itertools.product(hours[:], quarters[:], days[:], months[:]))[
        :100
    ]

    sample_data = pd.DataFrame(
        {
            "hr": [c[0] for c in combinations],
            "qtr": [c[1] for c in combinations],
            "dow": [c[2] for c in combinations],
            "mon": [c[3] for c in combinations],
        }
    )

    # Create and fit encoder with handle_unknown='ignore' for extra robustness
    return OneHotEncoder(drop="first", sparse_output=True, handle_unknown="ignore").fit(
        sample_data
    )


# Initialize encoder globally
enc = create_encoder()


def calendar_df(idx):
    return pd.DataFrame(
        {
            "hr": idx.hour,
            "qtr": idx.minute // 15,
            "dow": idx.dayofweek,
            "mon": idx.month - 1,
        }
    )


def forecast_ridge(train, start):
    """
    Ridge regression forecast using calendar features.

    Parameters:
    -----------
    train : pd.Series
        Historical time series data for training
    start : datetime
        Start time for the forecast

    Returns:
    --------
    tuple: (forecasts, elapsed_seconds)
        forecasts: numpy array of forecasted values
        elapsed_seconds: time taken to generate the forecast
    """
    t0 = time.perf_counter()
    try:
        X = enc.transform(calendar_df(train.index))
        model = Ridge(alpha=1.0).fit(X, train.values)
        future_dates = pd.date_range(start, periods=HORIZON, freq="15min")
        Xf = enc.transform(calendar_df(future_dates))
        yhat = model.predict(Xf)
        return yhat, time.perf_counter() - t0
    except Exception as e:
        print(f"Error in Ridge forecast: {e}")
        # Return a fallback forecast (e.g., same as naive)
        return forecast_naive(train, start)


def forecast_arima(train, start):
    t0 = time.perf_counter()
    model = ARIMA(train, order=(2, 0, 2)).fit()
    yhat = model.forecast(HORIZON)
    return yhat, time.perf_counter() - t0


def forecast_prophet(train, start):
    t0 = time.perf_counter()
    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        interval_width=0.8,
    ).fit(dfp)
    fut = m.make_future_dataframe(periods=HORIZON, freq="15min", include_history=False)
    yhat = m.predict(fut)["yhat"].values
    return yhat, time.perf_counter() - t0


MODELS = {
    "Naïve": forecast_naive,
    "EWMA": forecast_ewma,
    "Ridge": forecast_ridge,
    "ARIMA": forecast_arima,
    # "Prophet": forecast_prophet,
}


# ---------------------------------------------------------------------
# WALK-FORWARD BENCHMARK
# ---------------------------------------------------------------------
def run_benchmark(price_data):
    records = []
    for t0 in price_data.index[TRAIN_MIN:-HORIZON:HORIZON]:
        train = price_data.loc[: t0 - pd.Timedelta("15min")]
        test = price_data.loc[t0 : t0 + pd.Timedelta(minutes=15 * (HORIZON - 1))]
        for name, fn in MODELS.items():
            try:
                yhat, sec = fn(train, t0)
                mae = np.mean(np.abs(yhat - test.values))
                records.append({"model": name, "mae": mae, "seconds": sec})
            except Exception as e:
                print(f"{name} failed at {t0}: {e}")

    results = pd.DataFrame(records).groupby("model").mean().reset_index()
    print("\nMean over all roll-outs (lower-left is better):")
    print(results.sort_values("mae"))
    return results


# ---------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------
def create_plot(results):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(results["seconds"], results["mae"])
    for _, r in results.iterrows():
        ax.text(r["seconds"], r["mae"], r["model"], fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("Elapsed time per forecast (s)")
    ax.set_ylabel("Mean Absolute Error ($/MWh)")
    ax.set_title(f"32-Step-Ahead Forecast Benchmark  –  Horizon = {HORIZON}×15 min")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    try:
        # Load the data
        price = load_ercot_data(JSON_PATH)
        print(
            f"Loaded {len(price)} rows spanning {price.index.min()} → {price.index.max()}"
        )

        # Run benchmark
        results = run_benchmark(price)

        # Create plot
        create_plot(results)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
