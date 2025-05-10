"""
Utilities for time series analysis and forecasting.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union

def time_series_split(
    df: pd.DataFrame,
    n_splits: int = 3,
    test_size: Union[int, float] = 0.2,
    gap: int = 0
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward time series splits for cross-validation.
    
    Args:
        df: DataFrame with time series data
        n_splits: Number of train/test splits to create
        test_size: Size of test set (int=number of samples, float=proportion)
        gap: Number of samples to exclude between train and test sets
        
    Returns:
        List of (train_df, test_df) tuples for each split
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    n_samples = len(df)
    
    # Convert test_size to number of samples if it's a proportion
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size as proportion must be between 0 and 1")
        test_size = int(n_samples * test_size)
    
    # Calculate the fold increment size
    if n_splits <= 1:
        raise ValueError("n_splits must be at least 2")
    
    increment = (n_samples - test_size - (n_splits - 1) * (test_size + gap)) // (n_splits - 1)
    
    if increment <= 0:
        raise ValueError(
            f"Cannot create {n_splits} splits with test_size={test_size} and gap={gap}. "
            f"Try reducing n_splits, test_size, or gap."
        )
    
    # Create splits
    splits = []
    for i in range(n_splits):
        test_end = n_samples - i * (test_size + gap + increment)
        test_start = test_end - test_size
        train_end = test_start - gap
        
        if train_end <= 0:
            # Not enough data for this split
            break
            
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        splits.append((train_df, test_df))
    
    # Reverse the splits so earlier splits come first
    return splits[::-1] 