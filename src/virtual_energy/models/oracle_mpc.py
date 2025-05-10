#!/usr/bin/env python3
"""Notebook 2 – Rolling‑Horizon MPC battery dispatch optimiser (fixed)

Changes v1.1
------------
* **Deduplicate timestamps** before training and forecasting to avoid X/y length
  mismatch (root cause of the `ValueError: inconsistent numbers of samples`).
* Replaced deprecated/slow `Series.append` with `pd.concat`.
* Added explicit checks that X and y have equal length after alignment.
* Minor refactor: `train_forecaster` no longer needs the `node` argument – it
  consumes a *Series* instead, making it reusable for any column.

Run:
```bash
python mpc_dispatch.py --prices data/prices_wide.csv --node ALP_BESS_RN
```
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import argparse
from pathlib import Path
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pydantic import BaseModel, Field, validator

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import pulp
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpContinuous

# Import utility functions
from ercot_utils import tidy
from model_config import BatteryConfig

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def make_features(prices: pd.Series, max_lag: int = 4) -> pd.DataFrame:
    """Lagged prices + calendar dummies."""
    df = pd.DataFrame({"price": prices})
    for l in range(1, max_lag + 1):
        df[f"lag_{l}"] = df["price"].shift(l)

    if isinstance(df.index, pd.DatetimeIndex):
        df["hour"] = df.index.hour
        df["qtr"] = df.index.minute // 15
        df["dow"] = df.index.dayofweek
    return df.drop(columns=["price"])


def train_forecaster(price_series: pd.Series, max_lag: int = 4) -> Ridge:
    """Fit ridge regression with expanding‑window CV on a *Series*."""
    # Ensure unique, monotonic index
    price_series = (
        price_series.dropna()
        .sort_index()
        .loc[~price_series.index.duplicated(keep="last")]
    )

    y = price_series
    X = make_features(price_series, max_lag).dropna()
    y = y.loc[X.index]  # align exactly

    assert len(X) == len(y), "X/y length mismatch after alignment"

    tscv = TimeSeriesSplit(n_splits=5)
    ridge_grid = GridSearchCV(Ridge(), {"alpha": np.logspace(-2, 3, 10)}, cv=tscv)
    ridge_grid.fit(X, y)
    return ridge_grid.best_estimator_


def forecast_one_step(model: Ridge, hist_series: pd.Series, max_lag: int = 4) -> float:
    """Forecast price for *next* interval (k+1)."""
    # Clean up input data
    hist_series = hist_series.dropna().sort_index()

    # Make sure we have enough data for the required lags
    if len(hist_series) < max_lag:
        # Not enough data for forecasting, return mean or last value
        return float(hist_series.mean()) if len(hist_series) > 0 else 0.0

    last_ts = hist_series.index[-1]
    next_ts = last_ts + pd.Timedelta(minutes=15)

    # Add placeholder for next timestamp
    extended = pd.concat([hist_series, pd.Series([np.nan], index=[next_ts])])

    # Create feature matrix
    features_df = make_features(extended, max_lag)

    # Check if the last row (which we need for forecasting) has any NaN values
    if features_df.iloc[-1].isna().any():
        # Fill missing values with the mean of each column
        column_means = features_df.mean()
        features_df = features_df.fillna(column_means)

    # Extract the feature vector for prediction
    feat = features_df.iloc[-1]

    # Make the prediction
    return float(model.predict(feat.to_frame().T)[0])


def solve_lp_horizon(
    prices: np.ndarray, soc0: float, discharged_today: float, cfg: BatteryConfig
):
    H = len(prices)
    Δt = cfg.delta_t
    P_MAX = cfg.p_max_mw
    E_MAX = cfg.e_max_mwh
    η = cfg.eta_chg

    prob = LpProblem("MPC_Dispatch", LpMaximize)
    p_pos = LpVariable.dicts("discharge", range(H), 0, P_MAX, LpContinuous)
    p_neg = LpVariable.dicts("charge", range(H), 0, P_MAX, LpContinuous)
    soc = LpVariable.dicts("soc", range(H), 0, E_MAX, LpContinuous)

    prob += lpSum(prices[h] * (p_pos[h] - p_neg[h]) * Δt for h in range(H))

    for h in range(H):
        delta_e = (η * p_neg[h] - p_pos[h]) * Δt
        prob += soc[h] == (soc0 if h == 0 else soc[h - 1]) + delta_e

    prob += lpSum(p_pos[h] * Δt for h in range(H)) <= E_MAX - discharged_today
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    return p_pos[0].value(), p_neg[0].value()


# ----------------------------------------------------------------------
# MPC driver
# ----------------------------------------------------------------------


def run_mpc(price_df: pd.DataFrame, cfg: BatteryConfig, model: Ridge, node: str):
    Δt = cfg.delta_t
    soc = cfg.e_max_mwh * 0.5
    discharged_today = 0.0
    last_date = price_df["timestamp"].dt.date.iloc[0]

    # After tidy(), the node's price should be in 'SettlementPointPrice' column
    price_column = "SettlementPointPrice"

    # Check if price column exists in the dataframe
    if price_column not in price_df.columns:
        raise ValueError(
            f"Price column '{price_column}' not found in data. Available columns: {price_df.columns.tolist()}"
        )

    price_df = price_df.set_index("timestamp")

    rec = []
    hist_series = pd.Series(dtype=float)

    for ts, row in tqdm(price_df.iterrows(), total=len(price_df), desc="MPC loop"):
        try:
            true_price = row[price_column]
            # rolling history series
            hist_series = pd.concat([hist_series, pd.Series([true_price], index=[ts])])
            hist_series = hist_series.loc[~hist_series.index.duplicated(keep="last")]

            if ts.date() != last_date:
                discharged_today = 0.0
                last_date = ts.date()

            # build remainder‑of‑day horizon
            same_day_mask = price_df.index.date == ts.date()
            cumsum_series = pd.Series(same_day_mask.cumsum(), index=price_df.index)
            horizon_len = (
                same_day_mask.sum() - cumsum_series.loc[ts] + 1
            )  # +1 to include current interval

            # iterative forecasts
            h_prices = []
            fc_hist = hist_series.copy()
            for _ in range(horizon_len):
                fc = forecast_one_step(model, fc_hist)
                h_prices.append(fc)
                fc_hist = pd.concat(
                    [
                        fc_hist,
                        pd.Series(
                            [fc], index=[fc_hist.index[-1] + pd.Timedelta(minutes=15)]
                        ),
                    ]
                )

            p_pos, p_neg = solve_lp_horizon(
                np.array(h_prices), soc, discharged_today, cfg
            )
            mwh = (p_pos - p_neg) * Δt
            rev = mwh * true_price

            soc += (cfg.eta_chg * p_neg - p_pos) * Δt
            discharged_today += p_pos * Δt

            rec.append(
                {
                    "timestamp": ts,
                    "Price": true_price,
                    "p_discharge_MW": p_pos,
                    "p_charge_MW": p_neg,
                    "SoC_MWh": soc,
                    "MWhDeployed": mwh,
                    "Revenue$": rev,
                }
            )
        except KeyError as e:
            print(f"Error processing row at {ts}: {e}")
            continue

    return pd.DataFrame(rec)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/prices_wide.csv")
    ap.add_argument("--node", default="ALP_BESS_RN")
    ap.add_argument(
        "--capacity", type=float, default=200, help="Battery capacity in MWh"
    )
    ap.add_argument(
        "--power", type=float, default=25, help="Max charge/discharge power in MW"
    )
    ap.add_argument(
        "--efficiency",
        type=float,
        default=0.95,
        help="Battery charging efficiency (0-1)",
    )
    ap.add_argument(
        "--list-nodes",
        action="store_true",
        help="List all available nodes in the price data and exit",
    )
    args = ap.parse_args()

    cfg = BatteryConfig(
        delta_t=0.25,
        eta_chg=args.efficiency,
        p_max_mw=args.power,
        e_max_mwh=args.capacity,
    )

    print("Loading price data…")
    raw = pd.read_csv(args.prices)

    if args.list_nodes:
        print(f"Available nodes in {args.prices}:")
        for node in sorted(raw.columns[1:]):  # Skip timestamp column
            print(f"  {node}")
        return

    try:
        prices = tidy(raw, args.node)
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list-nodes to see available options.")
        return

    print("Training forecaster…")
    forecaster = train_forecaster(prices.set_index("timestamp")["SettlementPointPrice"])

    print("Running MPC…")
    dispatch = run_mpc(prices, cfg, forecaster, args.node)
    total = dispatch["Revenue$"].sum()
    print(f"Total revenue: ${total:,.0f}")

    out = Path("mpc_dispatch.csv")
    dispatch.to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
