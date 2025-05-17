#!/usr/bin/env python3
"""
online_river.py  –  streaming bandit dispatch (River ≥ 0.19, tested 0.22)

Example
-------
pip install -U river
python online_river.py --prices ercot.csv --algo linucb      # contextual
python online_river.py --prices ercot.csv --algo egreedy     # non-contextual
python online_river.py --prices ercot.csv --algo thompson --seed 42  # reproducible
"""

from pathlib import Path
import argparse
import warnings
import random
import numpy as np
import pandas as pd
from river import bandit, preprocessing
from river import proba  # Import for ThompsonSampling distribution

# Get battery parameters from config
from dispatch_benchmark.config import get_battery_config

battery_config = get_battery_config()
P_MAX = battery_config.p_max_mw
E_MAX = battery_config.e_max_mwh
ETA_CHG = battery_config.eta_chg
Δt = battery_config.delta_t
CYCLE_CAP = E_MAX  # MWh discharged per day

random.seed(42)
np.random.seed(42)

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────
def load_prices(csv: Path, node: str | None = None) -> pd.Series:
    df = pd.read_csv(csv, parse_dates=["timestamp"])
    if node:
        df["SettlementPointPrice"] = df[node]
    return (
        df.set_index("timestamp")["SettlementPointPrice"]
        .sort_index()
        .asfreq("15min")
    )


def features(price, mean_p, std_p, soc, hrs_left):
    return {
        "price": price,
        "z_price": (price - mean_p) / std_p if std_p else 0,
        "soc_pct": soc / E_MAX,
        "hrs_left": hrs_left / 24,
    }


# ─────────────────────────────────────────
def get_learner(kind: str, n_arms=3, alpha=0.3, eps=0.1):
    """Return a River bandit instance that works from 0.19 through 0.22."""

    if kind == "linucb":
        try:  # older API (needs n_arms)
            return bandit.LinUCBDisjoint(alpha=alpha, n_arms=n_arms)
        except TypeError:  # newer API (infers arms) or seed param not supported
            try:
                return bandit.LinUCBDisjoint(alpha=alpha)
            except TypeError:
                return bandit.LinUCBDisjoint(alpha=alpha)
    if kind == "egreedy":
        try:  # older API (has n_arms)
            return bandit.EpsilonGreedy(epsilon=eps, n_arms=n_arms)
        except TypeError:  # newer API (no n_arms) or seed param not supported
            try:
                return bandit.EpsilonGreedy(epsilon=eps)
            except TypeError:
                return bandit.EpsilonGreedy(epsilon=eps)
    if kind == "thompson":
        try:  # older API (has n_arms)
            return bandit.ThompsonSampling(n_arms=n_arms)
        except TypeError:  # newer API (no n_arms) or seed param not supported
            try:
                # For continuous rewards, use a Gaussian distribution
                return bandit.ThompsonSampling(reward_obj=proba.Gaussian())
            except TypeError:
                return bandit.ThompsonSampling(reward_obj=proba.Gaussian())
    raise ValueError("algo must be linucb | egreedy | thompson")


# ─────────────────────────────────────────
def run(series: pd.Series, algo: str, alpha: float = 0.3, eps: float = 0.1):
    learner = get_learner(algo, alpha=alpha, eps=eps)
    scaler = preprocessing.StandardScaler()

    soc, discharged_today = E_MAX / 2, 0.0
    last_day = series.index[0].date()
    rec = []

    for ts, price in series.items():
        if ts.date() != last_day:
            discharged_today, last_day = 0.0, ts.date()

        hrs_left = 24 - (ts.hour + ts.minute / 60)
        ctx_raw = features(
            price,
            scaler.means.get("price", 0),
            np.sqrt(scaler.vars.get("price", 1)),
            soc,
            hrs_left,
        )
        scaler.learn_one(ctx_raw)
        ctx = scaler.transform_one(ctx_raw)

        # ----- arm selection (robust to signature) -----
        arms = [0, 1, 2]  # 0 charge | 1 hold | 2 discharge
        try:
            arm = learner.pull(arms, ctx)  # contextual signature
        except TypeError:
            try:
                arm = learner.pull(arms, context=ctx)
            except TypeError:
                arm = learner.pull(arms)  # context-free

        p_chg = P_MAX if arm == 0 else 0
        p_dis = P_MAX if arm == 2 else 0
        p_chg = min(p_chg, (E_MAX - soc) / (ETA_CHG * Δt))
        p_dis = min(p_dis, soc / Δt, (CYCLE_CAP - discharged_today) / Δt)

        mwh = (p_dis - p_chg) * Δt
        reward = mwh * price

        # ----- update learner -----
        try:
            learner.update(
                arm, reward
            )  # standard update format for River 0.22.0
        except TypeError:
            try:
                # Try the contextual update format
                learner.update(arm, ctx, reward)
            except (TypeError, ZeroDivisionError):
                try:
                    # Try reversed parameter order
                    learner.update(ctx, arm, reward)
                except (TypeError, ZeroDivisionError):
                    # Fallback method
                    learner.update(arm, reward)

        # battery state
        soc += (ETA_CHG * p_chg - p_dis) * Δt
        discharged_today += p_dis * Δt

        rec.append(
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

    return pd.DataFrame(rec)


# ─────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/prices_wide.csv")
    ap.add_argument("--node", default="ALP_BESS_RN")
    ap.add_argument(
        "--algo", choices=["linucb", "egreedy", "thompson"], default="egreedy"
    )
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--eps", type=float, default=0.1)
    args = ap.parse_args()

    price_series = load_prices(Path(args.prices), args.node)
    print(
        f"Loaded {len(price_series)} rows   "
        f"{price_series.index.min()} → {price_series.index.max()}"
    )

    df = run(price_series, args.algo, args.alpha, args.eps)
    rev = df["Revenue$"].sum()
    print(f"{args.algo} revenue = ${rev:,.0f}")

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
