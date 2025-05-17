#!/usr/bin/env python3
"""
Naive price forecasting model.

This simple model predicts that the future price will be equal to the last observed price.
It serves as a baseline to compare against more sophisticated forecasting methods.
"""

import numpy as np
import pandas as pd


class NaivePrice:
    """Forecast k+1 price = last observed price."""

    def fit(self, y):
        """No fitting required for naive model."""
        pass

    def predict(self, y_hist):
        """Predict future price equal to the last observed price.

        Args:
            y_hist: Historical price series

        Returns:
            Last observed price
        """
        return y_hist.iloc[-1]


def forecast(
    train: pd.Series, t0: pd.Timestamp, horizon: int, **kw
) -> np.ndarray:
    """
    Naive price forecast - uses last observed price as prediction for all future periods.

    Args:
        train: Historical price series
        t0: Current timestamp
        horizon: Number of periods to forecast
        **kw: Ignored additional parameters

    Returns:
        Array of forecasted prices (all equal to last observed price)
    """
    last_price = train.iloc[-1]
    return np.full(horizon, last_price, dtype=float)
