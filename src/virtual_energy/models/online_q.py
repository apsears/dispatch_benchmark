#!/usr/bin/env python3
"""
online_q.py - Q-learning for battery dispatch

State variables:
soc_bin   ∈ {0-50, 50-100, 100-150, 150-200 MWh}  →  0..3
price_bin ∈ {lo, med, hi}  → 0..2
hour      ∈ {0..23}

Actions:
0 : charge  (-25 MW)
1 : hold    (0 MW)
2 : discharge (+25 MW)

Everything is learned *online*, no pre-train pass.  A single --seed value
makes results repeatable.
"""

from __future__ import annotations
import argparse, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
import random

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Get battery parameters from config
from virtual_energy.config import get_battery_config

battery_config = get_battery_config()
P_MAX = battery_config.p_max_mw
E_MAX = battery_config.e_max_mwh
ETA_CHG = battery_config.eta_chg
DT = battery_config.delta_t
CYCLE_CAP = E_MAX  # MWh discharged per day
WINDOW = 96 * 3  # 3-day trailing window for price bins


# -----------------------------------------------------------------------------
def load_prices(csv: Path, node: str | None = None) -> pd.Series:
    df = pd.read_csv(csv, parse_dates=["timestamp"])
    if node:
        if node not in df.columns:
            raise ValueError(f"node '{node}' not found; columns={df.columns}")
        df["SettlementPointPrice"] = df[node]
    return (
        df.set_index("timestamp")["SettlementPointPrice"].sort_index().asfreq("15min")
    )


# state encoding --------------------------------------------------------------
def make_state(price: float, hist: list[float], soc: float) -> int:
    if len(hist) < 20:
        p_bin = 1  # until we have quantiles, call it 'mid'
    else:
        q25, q75 = np.percentile(hist, [25, 75])
        p_bin = 0 if price < q25 else 2 if price > q75 else 1
    s_bin = min(int(soc // 50), 3)
    return p_bin * 10 + s_bin  # 0..23 but only 0,1,2 *10 + 0..3 appear


# Q-learning driver ------------------------------------------------------------
def q_learn(
    series: pd.Series,
    seed: int = 42,
    alpha: float = 0.5,  # learning-rate
    gamma: float = 0.95,  # discount
    eps0: float = 0.2,  # starting ε
    eps_decay: float = 0.999,
):
    rng = np.random.default_rng(seed)

    Q = np.zeros((30, 3))  # 30 > 23 → safe
    eps = eps0

    soc, discharged_today = E_MAX / 2, 0.0
    last_day = series.index[0].date()
    hist: list[float] = []
    records = []

    for ts, price in series.items():
        if ts.date() != last_day:  # reset daily cycle cap
            discharged_today, last_day = 0.0, ts.date()

        state = make_state(price, hist, soc)

        # ε-greedy action selection
        if rng.random() < eps:
            action = rng.integers(0, 3)  # explore
        else:
            action = int(Q[state].argmax())  # exploit

        # translate to power, then clip to constraints
        p_chg = P_MAX if action == 0 else 0
        p_dis = P_MAX if action == 2 else 0
        p_chg = min(p_chg, (E_MAX - soc) / (ETA_CHG * DT))
        p_dis = min(
            p_dis,
            soc / DT,
            (CYCLE_CAP - discharged_today) / DT,
        )

        mwh = (p_dis - p_chg) * DT
        reward = mwh * price

        # environment step
        soc_next = soc + (ETA_CHG * p_chg - p_dis) * DT
        discharged_today += p_dis * DT
        hist.append(price)
        if len(hist) > WINDOW:
            hist.pop(0)

        next_state = make_state(price, hist, soc_next)

        # Q-update
        Q[state, action] += alpha * (
            reward + gamma * Q[next_state].max() - Q[state, action]
        )

        # decay exploration rate
        eps *= eps_decay

        # log
        soc = soc_next
        records.append(
            {
                "DeliveryDate": ts.date(),
                "DeliveryHour": ts.hour + 1,
                "DeliveryInterval": ts.minute // 15 + 1,
                "SettlementPointPrice": price,
                "MWhDeployed": mwh,
                "Revenue$": reward,
                "SoC_MWh": soc,
            }
        )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/prices_wide.csv")
    ap.add_argument("--node", default="ALP_BESS_RN")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    series = load_prices(Path(args.prices), args.node)
    print(
        f"Loaded {len(series)} intervals   "
        f"{series.index.min()} → {series.index.max()}"
    )

    df = q_learn(series, seed=args.seed)
    total_rev = df["Revenue$"].sum()
    print(f"Q-learning revenue (seed {args.seed}) = ${total_rev:,.0f}")

    out = Path("dispatch_schedule.csv")
    df[
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
