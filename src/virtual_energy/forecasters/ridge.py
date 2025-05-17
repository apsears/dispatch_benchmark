#!/usr/bin/env python3
"""
Ridge regression forecasting model using calendar features.

This model uses ridge regression with one-hot encoded calendar features
(hour, quarter, day of week, month) to predict electricity prices.
It's fast and effective for forecasting price patterns that follow
time-of-day and seasonal patterns.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from typing import Optional

# Create OneHotEncoder for calendar features
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

# Cache to store the trained model
_model_cache = {
    "model": None,
    "last_train_size": 0,
    "update_frequency": 96,  # Update model every 96 steps (daily)
    "counter": 0,
    "window_size": 672,  # Default window size: 7 days (7×96=672 intervals)
}


def _cal_df(idx):
    """Create a DataFrame of calendar features from a DatetimeIndex."""
    return pd.DataFrame(
        {
            "hr": idx.hour,
            "qtr": idx.minute // 15,
            "dow": idx.dayofweek,
            "mon": idx.month - 1,
        }
    )


def forecast(
    train: pd.Series,
    t0: pd.Timestamp,
    horizon: int,
    alpha: float = 1.0,
    window_size: Optional[int] = None,
    update_freq: Optional[int] = None,
) -> np.ndarray:
    """
    Forecast prices using ridge regression on calendar features.

    Args:
        train: Historical price series
        t0: Timestamp to start forecasting from
        horizon: Number of periods to forecast
        alpha: Regularization strength
        window_size: Number of recent observations to use (default: 672, or 7 days)
        update_freq: How often to update model (in steps, default: 96, or daily)

    Returns:
        Array of forecasted prices
    """
    global _model_cache

    # Update cache settings if provided
    if window_size is not None:
        _model_cache["window_size"] = window_size
    if update_freq is not None:
        _model_cache["update_frequency"] = update_freq

    # Use rolling window instead of full history
    window = _model_cache["window_size"]
    if len(train) > window:
        train = train.iloc[-window:]

    # Only retrain the model periodically or when data size has changed significantly
    should_update = (
        _model_cache["model"] is None
        or _model_cache["counter"] >= _model_cache["update_frequency"]
        or abs(len(train) - _model_cache["last_train_size"]) > window // 10
    )

    if should_update:
        X = _onehot.transform(_cal_df(train.index))
        _model_cache["model"] = Ridge(alpha=alpha, fit_intercept=True).fit(
            X, train.values
        )
        _model_cache["last_train_size"] = len(train)
        _model_cache["counter"] = 0
    else:
        _model_cache["counter"] += 1

    # Generate predictions using cached model
    fut = pd.date_range(t0, periods=horizon, freq="15min")
    return _model_cache["model"].predict(_onehot.transform(_cal_df(fut)))
