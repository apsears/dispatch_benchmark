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
    train: pd.Series, t0: pd.Timestamp, horizon: int, alpha: float = 1.0
) -> np.ndarray:
    """
    Forecast prices using ridge regression on calendar features.
    
    Args:
        train: Historical price series
        t0: Timestamp to start forecasting from
        horizon: Number of periods to forecast
        alpha: Regularization strength
        
    Returns:
        Array of forecasted prices
    """
    X = _onehot.transform(_cal_df(train.index))
    model = Ridge(alpha=alpha, fit_intercept=True).fit(X, train.values)
    fut = pd.date_range(t0, periods=horizon, freq="15min")
    return model.predict(_onehot.transform(_cal_df(fut))) 