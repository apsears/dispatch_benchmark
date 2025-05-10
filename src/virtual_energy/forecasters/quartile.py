#!/usr/bin/env python3
"""
Percentile-based price forecasting model.

This model uses historical percentiles to identify price thresholds for
battery charging and discharging decisions. It doesn't actually forecast
future prices, but rather determines thresholds for price-based decisions.

Strategy
--------
* Keep a rolling history window (default 7 days = 7×96 = 672 intervals).
* At each 15-min step:
    - If price ≤ P_low  (x-th percentile)  → **charge**  at 25 MW
    - If price ≥ P_high (100-x percentile) → **discharge** at 25 MW
    - Else                                 → hold
* Enforce:
    •  SoC ∈ [0, 200] MWh
    •  Daily discharge energy ≤ 200 MWh
    •  Max |power| = 25 MW per interval

CLI
---
python online_quartile.py --prices ercot.csv --pct 10 --window 672
"""

from pathlib import Path
import argparse, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Battery constants
P_MAX = 25  # MW   (per 15-min)
E_MAX = 200  # MWh
ETA_CHG = 0.95
DELTA_T = 0.25  # h
CYCLE_CAP = 200  # MWh discharged per day


# ──────────────────────────────────────────────────────────────────────────────
def load_prices(csv: Path, node: str | None = None) -> pd.Series:
    df = pd.read_csv(csv, parse_dates=["timestamp"])
    if node:
        if node not in df.columns:
            raise ValueError(f"Node '{node}' not found; use one of {df.columns}")
        df["SettlementPointPrice"] = df[node]
    return (
        df.set_index("timestamp")["SettlementPointPrice"].sort_index().asfreq("15min")
    )


# ──────────────────────────────────────────────────────────────────────────────
def calculate_thresholds(
    price_history: pd.Series, pct: float = 10.0
) -> tuple[float, float]:
    """
    Calculate price thresholds for battery dispatch decisions.
    
    Args:
        price_history: Historical price series
        pct: Percentile threshold (we use pct and 100-pct)
        
    Returns:
        Tuple of (lower_threshold, upper_threshold)
    """
    if len(price_history) < 30:  # safety: need ≥30 samples
        return np.inf, np.inf
    
    p_low = np.percentile(price_history, pct)
    p_high = np.percentile(price_history, 100 - pct)
    
    return p_low, p_high


# ──────────────────────────────────────────────────────────────────────────────
def quartile_dispatch(
    series: pd.Series, pct: float = 10.0, window: int = 672
) -> pd.DataFrame:
    """
    Return dispatch DataFrame using percentile rule.
    
    Args:
        series: Price series
        pct: Percentile threshold
        window: Rolling window length in intervals
        
    Returns:
        DataFrame with dispatch schedule
    """
    soc, discharged_today = E_MAX / 2, 0.0
    last_day = series.index[0].date()
    hist = pd.Series(dtype=float)
    rec = []

    for ts, price in series.items():
        hist = pd.concat([hist, pd.Series([price], index=[ts])]).tail(window)

        # new calendar day → reset cycle counter
        if ts.date() != last_day:
            discharged_today, last_day = 0.0, ts.date()

        # compute thresholds
        p_low, p_high = calculate_thresholds(hist, pct)

        # ----- Decision logic -----
        p_dis, p_chg = 0.0, 0.0
        if price <= p_low and soc < E_MAX:  # CHARGE
            p_chg = min(P_MAX, (E_MAX - soc) / (ETA_CHG * DELTA_T))
        elif price >= p_high and soc > 0 and discharged_today < CYCLE_CAP:  # DISCHARGE
            p_dis = min(P_MAX, soc / DELTA_T, (CYCLE_CAP - discharged_today) / DELTA_T)
        # else HOLD

        mwh = (p_dis - p_chg) * DELTA_T
        revenue = mwh * price

        # update state
        soc += (ETA_CHG * p_chg - p_dis) * DELTA_T
        discharged_today += p_dis * DELTA_T

        rec.append(
            {
                "timestamp": ts,
                "DeliveryDate": ts.date(),
                "DeliveryHour": ts.hour + 1,
                "DeliveryInterval": ts.minute // 15 + 1,
                "SettlementPointPrice": price,
                "MWhDeployed": mwh,
                "Revenue$": revenue,
                "SoC_MWh": soc,
            }
        )

    return pd.DataFrame(rec)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prices",
        default="data/prices_wide.csv",
        help="CSV with 'timestamp' and price columns",
    )
    ap.add_argument(
        "--node",
        default="ALP_BESS_RN",
        help="Column containing the nodal price (optional)",
    )
    ap.add_argument(
        "--pct",
        type=float,
        default=45.0,
        help="Lower percentile threshold x (use x%% and 100-x%%)",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=672,
        help="Rolling window length in intervals (15-min steps)",
    )
    args = ap.parse_args()

    prices = load_prices(Path(args.prices), args.node)
    print(f"Loaded {len(prices)} rows  {prices.index.min()} → {prices.index.max()}")

    dispatch = quartile_dispatch(prices, args.pct, args.window)
    total_rev = dispatch["Revenue$"].sum()
    print(f"Percentile model x={args.pct:.1f} | Total revenue = ${total_rev:,.0f}")

    out = Path("dispatch_schedule.csv")
    dispatch[
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
