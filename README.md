# Virtual Energy

This project provides utilities for downloading electricity price data and benchmarking battery dispatch models. The code primarily targets ERCOT but also supports NYISO. We model a battery system with maximum energy capacity (`e_max_mwh`), maximum power capacity (`p_max_mw`), and round-trip efficiency (`eta_chg`). The battery operates in 15-minute intervals (`delta_t`).

e_max_mwh = 200  
p_max_mw = 25  
eta_chg = 0.95  
delta_t = 0.25  # 15-minute intervals  
initial_soc_frac = 0.0  

## ERCOT Data Pipeline

ERCOT Settlement Point Price data can be downloaded using `download_ercot.py` or the `process-ercot` entry point. Key command line options include:

```python
def parse_args():
    current_year = datetime.date.today().year
    prev_year = current_year - 1
    
    parser = argparse.ArgumentParser(description='Download, process, and combine ERCOT Settlement Point Price data')
    parser.add_argument('--year', type=int, default=prev_year,
                        help=f'Year to process (YYYY) - defaults to previous year ({prev_year})')
    parser.add_argument('--market', choices=['rtm', 'dam'], default='rtm',
                        help='Market data: real-time (rtm) or day-ahead (dam) (default: rtm)')
    parser.add_argument('--output-dir', default='data/ercot',
                        help='Base directory for output (default: data/ercot)')
    parser.add_argument('--max-workers', type=int, default=5,
                        help='Maximum number of concurrent downloads (default: 5)')
    parser.add_argument('--nodes', nargs='+',
                        help='Optional list of specific settlement point nodes to include')
    parser.add_argument('--username',
                        help='ERCOT API username')
    parser.add_argument('--password',
                        help='ERCOT API password')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip the download step and only process existing files')
    parser.add_argument('--wide-format', action='store_true',
                        help='Output data in wide format instead of tidy format')
    return parser.parse_args()
```

The script retrieves monthly JSON reports using the ERCOT Public API and combines them into CSV files. `combine_and_format_data` can output tidy or wide formats:

```python
def combine_and_format_data(dfs, year, market_type, output_dir, wide_format=False):
    """Combine DataFrames and output in tidy or wide format.
    
    Args:
        dfs: List of DataFrames to combine
        year: Year of data
        market_type: Market type (rtm or dam)
        output_dir: Directory to save output
        wide_format: Whether to output in wide format (default: False for tidy format)
        
    Returns:
        Path to the output file
    """
    if not dfs:
        print("No data to combine.")
        return None
    
    # Concatenate all dataframes
    print("Combining all data...")
    all_data = pd.concat(dfs, ignore_index=True)
    
    # Find the actual column names in the data
    settlement_point_col = next((col for col in all_data.columns if col.lower() == 'settlementpoint'), None)
    price_col = next((col for col in all_data.columns if col.lower() == 'settlementpointprice'), None)
    
    if not settlement_point_col or not price_col:
        print("Error: Required columns not found in data")
        return None
    
    # Drop any duplicate timestamps for the same node (take the last value)
    all_data = all_data.drop_duplicates(subset=["timestamp", settlement_point_col], keep="last")
    
    # Rename columns to standardize names
    all_data = all_data.rename(columns={
        settlement_point_col: "node",
        price_col: "price",
        "timestamp": "timestamp"
    })
    
    # Ensure the data is sorted
    all_data = all_data.sort_values(["timestamp", "node"])
    
    # Define market name for output file
    market_name = "RealTime" if market_type == "rtm" else "DayAhead"
    
    if wide_format:
        # Pivot to make each node a column (wide format)
        print("Pivoting data to wide format (this may take a while for many nodes)...")
        pivoted = all_data.pivot_table(
            index="timestamp", 
            columns="node", 
            values="price",
            aggfunc="mean"  # If there are still duplicates, take the mean
        )
        
        # Ensure the index is sorted by timestamp
        pivoted = pivoted.sort_index()
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"{year}_{market_name}_SPP_wide.csv")
        pivoted.to_csv(output_path)
    else:
        # Tidy format (default) - already in this format after combining
        output_path = os.path.join(output_dir, f"{year}_{market_name}_SPP.csv")
        all_data.to_csv(output_path, index=False)
    
    # Print statistics
    num_nodes = all_data['node'].nunique()
    num_intervals = all_data['timestamp'].nunique()
    expected_intervals = 365 * 24 * 4  # 15-min data for a normal year
    if year % 4 == 0:  # Leap year
        expected_intervals = 366 * 24 * 4
    
    print(f"\nSummary for {year} {market_name} Market:")
    print(f"Total settlement points (nodes): {num_nodes}")
    print(f"Total time intervals: {num_intervals}")
    print(f"Expected intervals for complete year: {expected_intervals} (15-min data)")
    print(f"Data completeness: {num_intervals / expected_intervals:.1%}")
    print(f"Format: {'Wide' if wide_format else 'Tidy'}")
    
    return output_path
```

The output files are placed under `data/ercot/` by default. Tidy format contains `timestamp`, `node`, and `price` columns, while wide format has one column per node.

## Benchmarking

Benchmark scripts live in the `benchmarks/` directory. `run_all_benchmarks.sh` executes tests for both ERCOT and NYISO, while `run_benchmark.sh` focuses on ERCOT only. The core script `comprehensive_benchmark.py` benchmarks several dispatch models:

- Oracle LP (perfect foresight)
- Online MPC with ridge regression
- Online MPC with naive forecasting
- Online Quartile dispatch (10%, 25%, 45%)

Results and summary statistics are saved under `results/`.
