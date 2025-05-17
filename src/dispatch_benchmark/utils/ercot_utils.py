#!/usr/bin/env python3
"""
Utility functions for working with ERCOT data
"""

import pandas as pd


def tidy(df, node):
    """
    Convert a wide format dataframe to a tidy format for a specific settlement point.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with "timestamp" column and columns for each settlement point
    node : str
        Settlement point to extract
        
    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns "timestamp" and "SettlementPointPrice"
    """
    # Extract the settlement point prices for the specified node
    if node not in df.columns:
        raise ValueError(f"Node '{node}' not found in price data")
        
    tidy_df = df[["timestamp", node]].copy()
    tidy_df.rename(columns={node: "SettlementPointPrice"}, inplace=True)
    
    return tidy_df 