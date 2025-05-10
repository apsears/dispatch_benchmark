#!/usr/bin/env python3
"""Notebook 1 – Oracle battery dispatch optimiser

This script implements an omniscient predictor (oracle) for battery dispatch optimization.
It assumes perfect knowledge of future prices to compute the optimal dispatch strategy.

Usage
-----
$ python oracle.py --prices data/prices_wide.csv --node ALP_BESS_RN

Outputs `optimised_dispatch.csv` with optimal dispatch and revenue.

Additional functionality
-----------------------
Process settlement zip files into a wide format CSV:
$ python oracle.py --process-zips --source-dir data/source/ --output-file data/prices_wide.csv

This reads all zip files containing settlement CSVs and consolidates them into a single
wide-format CSV with one column per node and timestamps as the index.
"""

# ----------------------------------------------------------------------
# Imports & pip‑install guard
# ----------------------------------------------------------------------
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import pulp
from pulp import (
    LpProblem,
    LpMaximize,
    LpVariable,
    lpSum,
    LpStatus,
    value,
    LpContinuous,
)

# Import utility functions
from ercot_utils import process_settlement_zips, tidy
from model_config import BatteryConfig

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def create_and_solve_model(
    prices: pd.DataFrame, cfg: BatteryConfig
) -> (LpProblem, pd.DataFrame):
    """Create and solve the LP optimization model using perfect price knowledge."""
    T = len(prices)  # number of 15-min intervals
    Δt = cfg.delta_t  # hours per interval
    eta_chg = cfg.eta_chg
    P_MAX_MW = cfg.p_max_mw
    E_MAX_MWh = cfg.e_max_mwh

    # Create the optimization problem
    prob = LpProblem("Oracle_Dispatch", LpMaximize)

    p_pos = LpVariable.dicts("discharge_MW", range(T), 0, P_MAX_MW, LpContinuous)
    p_neg = LpVariable.dicts("charge_MW", range(T), 0, P_MAX_MW, LpContinuous)
    soc = LpVariable.dicts("soc_MWh", range(T), 0, E_MAX_MWh, LpContinuous)

    # Objective: maximise revenue (price × net-power × hours)
    prob += lpSum(
        prices.SettlementPointPrice[t] * (p_pos[t] - p_neg[t]) * Δt for t in range(T)
    )

    # State-of-charge recursion
    for t in range(T):
        if t == 0:
            prob += soc[t] == E_MAX_MWh * 0.5 + (eta_chg * p_neg[t] - p_pos[t]) * Δt
        else:
            prob += soc[t] == soc[t - 1] + (eta_chg * p_neg[t] - p_pos[t]) * Δt

    # One-cycle-per-day constraint (maximum energy discharged per day)
    day_groups = prices.timestamp.dt.floor("D").factorize()[0]
    for d in set(day_groups):
        idx = [i for i, g in enumerate(day_groups) if g == d]
        prob += lpSum(p_pos[i] * Δt for i in idx) <= E_MAX_MWh

    # Solve the model
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Create results dataframe
    dispatch = prices.copy()
    dispatch["p_discharge_MW"] = [p_pos[t].value() for t in range(T)]
    dispatch["p_charge_MW"] = [p_neg[t].value() for t in range(T)]
    dispatch["SoC_MWh"] = [soc[t].value() for t in range(T)]
    dispatch["MWhDeployed"] = [
        (p_pos[t].value() - p_neg[t].value()) * Δt for t in range(T)
    ]
    dispatch["Revenue$"] = dispatch["MWhDeployed"] * dispatch["SettlementPointPrice"]

    return prob, dispatch


# ----------------------------------------------------------------------
# CLI entry‑point
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prices",
        help="CSV of raw price data",
        default="data/prices_wide.csv",
    )
    parser.add_argument(
        "--node", default="ALP_BESS_RN", help="SettlementPointName to filter"
    )
    parser.add_argument(
        "--process-zips",
        action="store_true",
        help="Process settlement zip files in data/source/ to wide format CSV",
    )
    parser.add_argument(
        "--source-dir",
        default="data/source/",
        help="Directory containing settlement zip files",
    )
    parser.add_argument(
        "--output-file",
        default="data/prices_wide.csv",
        help="Path to save the output wide-format CSV",
    )
    parser.add_argument(
        "--capacity", type=float, default=200, help="Battery capacity in MWh"
    )
    parser.add_argument(
        "--power", type=float, default=25, help="Max charge/discharge power in MW"
    )
    parser.add_argument(
        "--efficiency",
        type=float,
        default=0.95,
        help="Battery charging efficiency (0-1)",
    )
    args = parser.parse_args()

    # If processing zip files is requested, do that and exit
    if args.process_zips:
        process_settlement_zips(args.source_dir, args.output_file)
        return

    # Create battery config from command line arguments
    cfg = BatteryConfig(
        delta_t=0.25,
        eta_chg=args.efficiency,
        p_max_mw=args.power,
        e_max_mwh=args.capacity,
    )

    # ------------------------------------------------------------------
    # Data prep
    # ------------------------------------------------------------------
    print(f"Loading price data from {args.prices} …")
    if args.prices.endswith(".csv"):
        # Check if this is a wide format CSV with timestamp as index
        raw = pd.read_csv(args.prices)
        if "timestamp" in raw.columns:
            print("Loading data with timestamp as column...")
        else:
            # Try reading with timestamp as index
            print("Attempting to load with timestamp as index...")
            raw = pd.read_csv(args.prices, index_col=0, parse_dates=True)
            if not isinstance(raw.index, pd.DatetimeIndex):
                print("Warning: First column might not be a valid timestamp index")
    else:
        raise ValueError(f"Unsupported file format: {args.prices}")

    print(f"Processing data for node: {args.node}")
    prices = tidy(raw, args.node)

    # ------------------------------------------------------------------
    # Oracle optimization (assumes perfect knowledge of future prices)
    # ------------------------------------------------------------------
    print("Running Oracle optimization with perfect price knowledge...")
    prob, dispatch = create_and_solve_model(prices, cfg)

    total_rev = dispatch["Revenue$"].sum()
    print(f"Status: {LpStatus[prob.status]}")
    print(f"Total revenue (optimal): ${total_rev:,.0f}")

    out_path = Path("optimised_dispatch.csv")
    dispatch.to_csv(out_path, index=False)
    print(f"Wrote {out_path} (rows = {len(dispatch)})")


if __name__ == "__main__":
    main()
