"""
Test the data loading functionality.
"""

import pandas as pd
import numpy as np
from virtual_energy.models.benchmark import load_prices

def test_loader_roundtrip(ercot_1day_csv):
    """Test that the data loader can load and transform ERCOT data correctly."""
    # Load the ERCOT data
    df = load_prices(ercot_1day_csv)
    
    # Check that we have the expected number of rows
    # 24 hours x 4 intervals = 96 data points per node
    # We have 3 nodes in our sample data
    expected_nodes = 3
    
    # Verify we have the timestamp column + node columns
    assert len(df.columns) == expected_nodes + 1
    assert 'timestamp' in df.columns
    
    # Check the node columns
    node_columns = [col for col in df.columns if col != 'timestamp']
    assert len(node_columns) == expected_nodes
    assert 'HB_HOUSTON' in node_columns
    assert 'HB_NORTH' in node_columns
    assert 'HB_WEST' in node_columns
    
    # Check we have 96 rows (24 hours x 4 intervals)
    assert len(df) == 96
    
    # Check there are no NaN values
    assert not df.isna().any().any()
    
    # Check that we can load a single node
    df_houston = load_prices(ercot_1day_csv, 'HB_HOUSTON')
    assert len(df_houston) == 96
    assert 'timestamp' in df_houston.columns
    assert 'SettlementPointPrice' in df_houston.columns
    assert not df_houston.isna().any().any() 