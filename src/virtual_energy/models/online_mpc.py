#!/usr/bin/env python3
"""
online_mpc.py  –  Non-clairvoyant MPC with pluggable forecast models

Models implemented
------------------
ewma   : EWMA of trailing prices in the *same quarter-hour slot*  (α = 0.2)
ridge  : Ridge-regression on calendar dummies   (fit every step, but fast)
arima  : ARIMA(2,0,2) on the whole history      (slow, demo only)

Extend by adding  forecast_X(train_series, t0, horizon)->np.ndarray
and inserting into the FORECASTERS dict at the bottom.
"""

import argparse, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
import pulp
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Get battery parameters from config
from virtual_energy.config import get_battery_config

battery_config = get_battery_config()
P_MAX = battery_config.p_max_mw
E_MAX = battery_config.e_max_mwh
ETA_CHG = battery_config.eta_chg
DELTA_T = battery_config.delta_t
CYCLE_CAP = E_MAX  # MWh discharged per day


# ──────────────────────────────────────────────────────────────────────────────
# Price loading (same as before)
def load_price_csv(path: Path, node: str | None = None) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if node:
        if node not in df.columns:
            raise ValueError(f"Node '{node}' not in CSV columns.")
        df["SettlementPointPrice"] = df[node]
    return (
        df.set_index("timestamp")["SettlementPointPrice"].sort_index().asfreq("15min")
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Forecast model implementations
# ──────────────────────────────────────────────────────────────────────────────
def forecast_ewma(
    train: pd.Series, t0: pd.Timestamp, horizon: int, α: float = 0.2
) -> np.ndarray:
    slot_hist = train[train.index.time == t0.time()]
    level = slot_hist.ewm(alpha=α).mean().iloc[-1]
    return np.full(horizon, level, dtype=float)


# Ridge (calendar dummies)  – fits in ≈3 ms / step
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

_onehot = OneHotEncoder(drop="first", handle_unknown="ignore").fit(
    pd.DataFrame(
        {
            "hr": range(24),
            "qtr": list(range(4)) * 6,  # 0,1,2,3 for each quarter hour
            "dow": list(range(7)) * (24 // 7)
            + list(range(24 % 7)),  # 0-6 for days of week
            "mon": list(range(12)) * 2,  # 0-11 for each month
        }
    )
)


def _cal_df(idx):
    return pd.DataFrame(
        {
            "hr": idx.hour,
            "qtr": idx.minute // 15,
            "dow": idx.dayofweek,
            "mon": idx.month - 1,
        }
    )


def forecast_ridge(
    train: pd.Series, t0: pd.Timestamp, horizon: int, **kw
) -> np.ndarray:
    X = _onehot.transform(_cal_df(train.index))
    model = Ridge(alpha=1.0, fit_intercept=True).fit(X, train.values)
    fut = pd.date_range(t0, periods=horizon, freq="15min")
    return model.predict(_onehot.transform(_cal_df(fut)))


# ARIMA  – slow (~0.3 s / step)
from statsmodels.tsa.arima.model import ARIMA


def forecast_arima(
    train: pd.Series, t0: pd.Timestamp, horizon: int, **kw
) -> np.ndarray:
    # If we have too few observations for ARIMA, fall back to a simpler method
    if len(train) < 5:  # Need at least a few observations for ARIMA(2,0,2)
        # Fall back to EWMA for very small samples
        return forecast_ewma(train, t0, horizon)

    try:
        # Set frequency explicitly to avoid statsmodels warning
        train_with_freq = train.copy()
        if train_with_freq.index.freq is None:
            train_with_freq.index = pd.DatetimeIndex(
                train_with_freq.index, freq="15min"
            )

        return ARIMA(train_with_freq, order=(2, 0, 2)).fit().forecast(horizon).values
    except (ValueError, np.linalg.LinAlgError) as e:
        # If ARIMA fails, fall back to a simpler method
        print(f"ARIMA forecasting failed: {str(e)}. Falling back to EWMA.")
        return forecast_ewma(train, t0, horizon)


# Registry
FORECASTERS = {
    "ewma": forecast_ewma,
    "ridge": forecast_ridge,
    "arima": forecast_arima,
}


# ──────────────────────────────────────────────────────────────────────────────
# 2.  LP solver (unchanged)
def solve_first_action(price_path, soc0, discharged_today):
    H = len(price_path)
    m = pulp.LpProblem("mpc", pulp.LpMaximize)
    p_pos = {h: pulp.LpVariable(f"d_{h}", 0, P_MAX) for h in range(H)}
    p_neg = {h: pulp.LpVariable(f"c_{h}", 0, P_MAX) for h in range(H)}
    soc = {h: pulp.LpVariable(f"soc_{h}", 0, E_MAX) for h in range(H)}

    m += pulp.lpSum(price_path[h] * (p_pos[h] - p_neg[h]) * DELTA_T for h in range(H))
    for h in range(H):
        Δe = (ETA_CHG * p_neg[h] - p_pos[h]) * DELTA_T
        m += soc[h] == (soc0 if h == 0 else soc[h - 1]) + Δe
    m += (
        pulp.lpSum(p_pos[h] * DELTA_T for h in range(H)) <= CYCLE_CAP - discharged_today
    )
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    return p_pos[0].value(), p_neg[0].value()


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MPC driver
def run_mpc(series: pd.Series, horizon: int, forecaster):
    soc, discharged_today = E_MAX / 2, 0.0
    last_day = series.index[0].date()
    hist_values = []
    hist_indices = []
    rec = []

    for ts, price in tqdm(series.items(), total=len(series), desc="Running MPC"):
        hist_values.append(price)
        hist_indices.append(ts)
        hist = pd.Series(hist_values, index=hist_indices)

        if ts.date() != last_day:
            discharged_today, last_day = 0.0, ts.date()

        h_prices = forecaster(hist, ts, horizon)
        p_pos, p_neg = solve_first_action(h_prices, soc, discharged_today)
        mwh = (p_pos - p_neg) * DELTA_T
        rev = mwh * price

        soc += (ETA_CHG * p_neg - p_pos) * DELTA_T
        discharged_today += p_pos * DELTA_T

        rec.append(
            {
                "timestamp": ts,
                "DeliveryDate": ts.date(),
                "DeliveryHour": ts.hour + 1,
                "DeliveryInterval": ts.minute // 15 + 1,
                "SettlementPointPrice": price,
                "MWhDeployed": mwh,
                "Revenue$": rev,
                "SoC_MWh": soc,
            }
        )
    return pd.DataFrame(rec)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default="data/prices_wide.csv")
    p.add_argument("--node", default="ALP_BESS_RN")
    p.add_argument(
        "--model", choices=FORECASTERS, default="arima", help="Forecast model to use"
    )
    p.add_argument("--horizon", type=int, default=32)
    args = p.parse_args()

    prices = load_price_csv(Path(args.prices), args.node)
    print(f"Loaded {len(prices)} rows.  Forecast model = {args.model}")

    t0 = time.perf_counter()
    disp = run_mpc(prices, args.horizon, FORECASTERS[args.model])
    run_sec = time.perf_counter() - t0

    print(f"Finished in {run_sec:.1f}s   |   Revenue = ${disp['Revenue$'].sum():,.0f}")
    out = Path("dispatch_schedule.csv")
    disp[
        [
            "DeliveryDate",
            "DeliveryHour",
            "DeliveryInterval",
            "SettlementPointPrice",
            "MWhDeployed",
            "Revenue$",
        ]
    ].to_csv(out, index=False)
    print("Wrote", out)
