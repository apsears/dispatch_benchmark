#!/usr/bin/env python3
"""
online_mpc.py  –  Non-clairvoyant MPC with pluggable forecast models

This module implements the online Model Predictive Control (MPC) optimizer
that uses price forecasts to make battery dispatch decisions.
"""

import argparse
import warnings
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pulp
from tqdm import tqdm

# Import forecasters
from virtual_energy.forecasters import ridge, naive

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Battery & solver constants
P_MAX, E_MAX, ETA_CHG = 25, 200, 0.95  # MW, MWh, efficiency
DELTA_T, CYCLE_CAP = 0.25, 200  # h ,  MWh  (per day)


# ──────────────────────────────────────────────────────────────────────────────
# Price loading (same as before)
def load_price_csv(path: Path, node: str | None = None) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if node:
        if node not in df.columns:
            raise ValueError(f"Node '{node}' not in CSV columns.")
        df["SettlementPointPrice"] = df[node]
    return (
        df.set_index("timestamp")["SettlementPointPrice"]
        .sort_index()
        .asfreq("15min")
    )


# ──────────────────────────────────────────────────────────────────────────────
# Forecast models registry
# Import directly from the forecasters module
def forecast_ewma(
    train: pd.Series, t0: pd.Timestamp, horizon: int, α: float = 0.2
) -> np.ndarray:
    slot_hist = train[train.index.time == t0.time()]
    level = slot_hist.ewm(alpha=α).mean().iloc[-1]
    return np.full(horizon, level, dtype=float)


# Registry of available forecasters
FORECASTERS = {
    "ewma": forecast_ewma,
    "ridge": ridge.forecast,
    "naive": naive.forecast,
}


# ──────────────────────────────────────────────────────────────────────────────
# LP solver (unchanged)
def solve_first_action(price_path, soc0, discharged_today):
    H = len(price_path)
    m = pulp.LpProblem("mpc", pulp.LpMaximize)
    p_pos = {h: pulp.LpVariable(f"d_{h}", 0, P_MAX) for h in range(H)}
    p_neg = {h: pulp.LpVariable(f"c_{h}", 0, P_MAX) for h in range(H)}
    soc = {h: pulp.LpVariable(f"soc_{h}", 0, E_MAX) for h in range(H)}

    m += pulp.lpSum(
        price_path[h] * (p_pos[h] - p_neg[h]) * DELTA_T for h in range(H)
    )
    for h in range(H):
        Δe = (ETA_CHG * p_neg[h] - p_pos[h]) * DELTA_T
        m += soc[h] == (soc0 if h == 0 else soc[h - 1]) + Δe
    m += (
        pulp.lpSum(p_pos[h] * DELTA_T for h in range(H))
        <= CYCLE_CAP - discharged_today
    )
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    return p_pos[0].value(), p_neg[0].value()


# ──────────────────────────────────────────────────────────────────────────────
# MPC driver
def run_mpc(series: pd.Series, horizon: int, forecaster):
    soc, discharged_today = E_MAX / 2, 0.0
    last_day = series.index[0].date()
    hist_values = []
    hist_indices = []
    rec = []

    for ts, price in tqdm(
        series.items(), total=len(series), desc="Running MPC"
    ):
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
# CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default="data/prices_wide.csv")
    p.add_argument("--node", default="ALP_BESS_RN")
    p.add_argument(
        "--model",
        choices=FORECASTERS,
        default="ridge",
        help="Forecast model to use",
    )
    p.add_argument("--horizon", type=int, default=32)
    args = p.parse_args()

    prices = load_price_csv(Path(args.prices), args.node)
    print(f"Loaded {len(prices)} rows.  Forecast model = {args.model}")

    t0 = time.perf_counter()
    disp = run_mpc(prices, args.horizon, FORECASTERS[args.model])
    run_sec = time.perf_counter() - t0

    print(
        f"Finished in {run_sec:.1f}s   |   Revenue = ${disp['Revenue$'].sum():,.0f}"
    )
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
