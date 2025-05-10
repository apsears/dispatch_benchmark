print("Loading conftest.py...")
"""
Pytest fixtures for the ERCOT virtualization tests.
"""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from virtual_energy.models.config import BatteryConfig


@pytest.fixture
def ercot_1day_csv():
    """Fixture that returns the path to the ERCOT 1-day sample dataset."""
    return os.path.join(os.path.dirname(__file__), "data", "ercot_1day_sample.csv")


@pytest.fixture
def nyiso_1day_csv():
    """Fixture that returns the path to the NYISO 1-day sample dataset."""
    return os.path.join(os.path.dirname(__file__), "data", "nyiso_1day_sample.csv")


@pytest.fixture
def ercot_sample_df(ercot_1day_csv):
    """Fixture that loads the ERCOT sample dataset."""
    return pd.read_csv(ercot_1day_csv)


@pytest.fixture
def nyiso_sample_df(nyiso_1day_csv):
    """Fixture that loads the NYISO sample dataset."""
    return pd.read_csv(nyiso_1day_csv)


@pytest.fixture
def random_prices():
    """Fixture that generates 96 random price intervals."""
    # Generate 96 price points (24 hours x 4 intervals)
    np.random.seed(42)  # For reproducibility
    timestamps = pd.date_range(start="2024-01-01", periods=96, freq="15min")
    prices = 20 + 10 * np.sin(np.pi * np.arange(96) / 48) + np.random.normal(0, 3, 96)

    return pd.DataFrame({"timestamp": timestamps, "SettlementPointPrice": prices})


@pytest.fixture
def three_day_prices():
    """Fixture that generates 3 days of price data for testing time series splits."""
    # Generate 3 days of price data (3 days x 24 hours x 4 intervals = 288 points)
    np.random.seed(42)  # For reproducibility
    timestamps = pd.date_range(start="2024-01-01", periods=288, freq="15min")

    # Create a daily pattern with some random noise
    hour_of_day = timestamps.hour + timestamps.minute / 60
    prices = 20 + 10 * np.sin(np.pi * hour_of_day / 12) + np.random.normal(0, 3, 288)

    return pd.DataFrame({"timestamp": timestamps, "SettlementPointPrice": prices})


@pytest.fixture
def battery_config():
    """Fixture that returns a standard battery configuration."""
    return BatteryConfig(
        delta_t=0.25,
        eta_chg=0.95,
        p_max_mw=25,
        e_max_mwh=200,
    )
