#!/usr/bin/env python3

import argparse
import datetime
import os
import zipfile
import glob
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser(description='Download, process, and combine NYISO LBMP data')
    parser.add_argument('--year', type=int, required=True,
                        help='Year to process (YYYY)')
    parser.add_argument('--market', choices=['da', 'rt'], default='da',
                        help='Market data: day-ahead (da) or realtime (rt) (default: da)')
    parser.add_argument('--format', choices=['csv', 'html', 'pdf'], default='csv',
                        help='File format to download (default: csv)')
    parser.add_argument('--output-dir', default='data/nyiso',
                        help='Directory to save downloaded and processed files (default: data/nyiso)')
    parser.add_argument('--max-workers', type=int, default=5,
                        help='Maximum number of concurrent downloads (default: 5)')
    parser.add_argument('--wide-format', action='store_true',
                        help='Output data in wide format instead of tidy format')
    return parser.parse_args()

def download_file(url, output_path):
    """Download a file from the given URL and save it to the specified path."""
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"Downloaded: {url} -> {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return False

def get_months_for_year(year):
    """Generate list of months for a given year."""
    current_date = datetime.date.today()
    months = []
    
    for month in range(1, 13):
        # First day of the month
        month_date = datetime.date(year, month, 1)
        
        # Skip future months
        if month_date > current_date:
            continue
            
        # Skip current month if it's not complete
        if month_date.year == current_date.year and month_date.month == current_date.month:
            continue
            
        months.append(month_date)
    
    return months

def generate_monthly_urls(months, file_format, market_type):
    """Generate URLs for each month."""
    market_segment = "damlbmp" if market_type == "da" else "realtime"
    
    for month_date in months:
        first_day_str = month_date.strftime('%Y%m01')
        month_display = month_date.strftime('%b %Y')
        
        # Example: https://mis.nyiso.com/public/csv/damlbmp/20240101damlbmp_zone_csv.zip
        url = f"https://mis.nyiso.com/public/{file_format}/{market_segment}/{first_day_str}{market_segment}_zone_{file_format}.zip"
        filename = f"{first_day_str}{market_segment}_zone_{file_format}.zip"
        
        yield url, filename, month_display

def unzip_file(zip_path, extract_dir):
    """Extract the contents of a zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            print(f"Extracted: {zip_path} -> {extract_dir}")
            
            # Return the names of files extracted
            return zip_ref.namelist()
    except zipfile.BadZipFile as e:
        print(f"Error extracting {zip_path}: {e}")
        return []

def process_csv_file(csv_path):
    """
    Read a CSV file and return the processed DataFrame in tidy format.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Convert Time Stamp to datetime
        df['Time Stamp'] = pd.to_datetime(df['Time Stamp'])
        
        # Select only the needed columns and rename them to standardize
        tidy_df = df[['Time Stamp', 'Name', 'LBMP ($/MWHr)']].copy()
        tidy_df = tidy_df.rename(columns={
            'Time Stamp': 'timestamp',
            'Name': 'zone',
            'LBMP ($/MWHr)': 'price'
        })
        
        print(f"Processed {csv_path} - {len(tidy_df)} records")
        
        return tidy_df
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def pivot_csv_for_wide_format(df):
    """
    Convert a tidy DataFrame to wide format with zones as columns.
    """
    # Create a date column for grouping at daily level
    df['date'] = df['timestamp'].dt.date
    
    # Pivot the data: each unique zone becomes a column
    pivot_df = df.pivot_table(
        index='date',
        columns='zone',
        values='price',
        aggfunc='mean'  # Average price for each day
    ).reset_index()
    
    return pivot_df

def combine_processed_data(processed_dfs, year, market_type, output_dir, wide_format=False):
    """Combine all processed DataFrames into a single yearly file."""
    if not processed_dfs:
        print("No data to combine.")
        return None, None
    
    try:
        # Concatenate all DataFrames
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        
        # Sort by timestamp and zone
        combined_df = combined_df.sort_values(['timestamp', 'zone'])
        
        # Save the combined data
        market_name = "DayAhead" if market_type == "da" else "RealTime"
        
        if wide_format:
            # Convert to wide format before saving
            wide_df = pivot_csv_for_wide_format(combined_df)
            output_path = os.path.join(output_dir, f"{year}_{market_name}_LBMP_wide.csv")
            wide_df.to_csv(output_path, index=False)
            final_df = wide_df
        else:
            # Save in tidy format (default)
            output_path = os.path.join(output_dir, f"{year}_{market_name}_LBMP.csv")
            combined_df.to_csv(output_path, index=False)
            final_df = combined_df
        
        print(f"Combined data saved to {output_path}")
        
        # Calculate statistics based on the format
        if wide_format:
            print(f"Data shape: {final_df.shape[0]} rows (days) x {final_df.shape[1]} columns")
            print(f"Number of unique zones: {final_df.shape[1] - 1}")  # Subtract 1 for the date column
        else:
            total_timestamps = combined_df['timestamp'].nunique()
            total_zones = combined_df['zone'].nunique()
            print(f"Data shape: {len(combined_df)} rows x 3 columns")
            print(f"Number of unique timestamps: {total_timestamps}")
            print(f"Number of unique zones: {total_zones}")
        
        return final_df, market_name
        
    except Exception as e:
        print(f"Error combining data: {e}")
        return None, None

def main(args=None):
    if args is None:
        args = parse_args()
    
    year = args.year
    market_type = args.market
    file_format = args.format
    wide_format = getattr(args, 'wide_format', False)  # Make wide_format optional for backward compatibility
    
    # Ensure it's a valid year
    current_year = datetime.date.today().year
    if year < 2000 or year > current_year:
        print(f"Error: Year {year} is out of range. Must be between 2000 and {current_year}.")
        return
    
    # Get list of months for the requested year
    months = get_months_for_year(year)
    
    if not months:
        print(f"No complete months found for year {year}.")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a directory for extracted files
    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    
    print(f"Processing {args.market} market data for {len(months)} months in {year}")
    
    # Step 1: Download ZIP files
    download_tasks = []
    for url, filename, month_display in generate_monthly_urls(months, file_format, market_type):
        output_path = output_dir / filename
        print(f"Adding download task: {month_display}")
        download_tasks.append((url, str(output_path)))
    
    if not download_tasks:
        print("No downloads to process.")
        return
    
    print(f"Starting {len(download_tasks)} downloads...")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(executor.map(lambda x: download_file(*x), download_tasks))
    
    # Step 2: Extract ZIP files
    csv_files = []
    for url, output_path in download_tasks:
        if os.path.exists(output_path):
            # Extract the files
            extracted_files = unzip_file(output_path, extract_dir)
            
            # Collect the CSV files
            for extracted_file in extracted_files:
                if extracted_file.endswith('.csv'):
                    csv_files.append(str(extract_dir / extracted_file))
    
    if not csv_files:
        print("No CSV files found in the downloaded archives.")
        return
    
    # Step 3: Process each CSV file
    processed_dfs = []
    for csv_file in csv_files:
        df = process_csv_file(csv_file)
        if df is not None:
            processed_dfs.append(df)
    
    # Step 4: Combine all processed data
    final_df, market_name = combine_processed_data(processed_dfs, year, market_type, output_dir, wide_format=wide_format)
    
    # Print summary
    if final_df is not None:
        print("\nSummary:")
        print(f"Year: {year}")
        print(f"Market: {market_type}")
        
        if wide_format:
            total_days = final_df.shape[0]
            total_zones = final_df.shape[1] - 1  # Subtract 1 for the date column
            print(f"Total days of data: {total_days}")
            print(f"Total zones: {total_zones}")
            print(f"Format: Wide")
            print(f"Output file: {output_dir}/{year}_{market_name}_LBMP_wide.csv")
        else:
            total_timestamps = final_df['timestamp'].nunique()
            total_zones = final_df['zone'].nunique()
            print(f"Total timestamps: {total_timestamps}")
            print(f"Total zones: {total_zones}")
            print(f"Format: Tidy")
            print(f"Output file: {output_dir}/{year}_{market_name}_LBMP.csv")
        
        return output_dir / f"{year}_{market_name}_LBMP{'_wide' if wide_format else ''}.csv"
    
    return None

def load_prices(start, end, nodes=None, freq="1D") -> pd.DataFrame:
    """
    Returns tidy DataFrame with NYISO price data for the specified period and nodes.
    
    Parameters
    ----------
    start : str or datetime-like
        Start date for the data. Can be a string in format 'YYYY-MM-DD' or a datetime object.
    end : str or datetime-like
        End date for the data. Can be a string in format 'YYYY-MM-DD' or a datetime object.
    nodes : list or None
        Optional list of specific zones to include.
        If None, all available zones will be included.
    freq : str
        Frequency of the data. NYISO data is typically daily ('1D') by default for 
        day-ahead market or hourly ('1H') for real-time market.
        
    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns: timestamp, node, price
    """
    # Convert string dates to datetime if needed
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    # Get the list of years to process
    years = range(start_date.year, end_date.year + 1)
    
    # Determine market type from frequency
    # Hourly or sub-hourly is likely real-time, daily is likely day-ahead
    market_type = 'rt' if freq.endswith(('T', 'H', 'min')) else 'da'
    
    # Determine base directory for data files
    base_dir = os.getenv("NYISO_DATA_DIR", "data/nyiso")
    
    # List to hold dataframes for each year
    all_dfs = []
    
    for year in years:
        # Construct filename based on market type
        market_name = "RealTime" if market_type == "rt" else "DayAhead"
        yearly_file = os.path.join(base_dir, f"{year}_{market_name}_LBMP.csv")
        wide_file = os.path.join(base_dir, f"{year}_{market_name}_LBMP_wide.csv")
        
        # Try to load the tidy format file first
        if os.path.exists(yearly_file):
            print(f"Loading tidy data from {yearly_file}")
            df = pd.read_csv(yearly_file)
            
            # Ensure column names are standardized
            if 'timestamp' not in df.columns and 'Time Stamp' in df.columns:
                df = df.rename(columns={'Time Stamp': 'timestamp'})
            if 'node' not in df.columns and 'zone' in df.columns:
                df = df.rename(columns={'zone': 'node'})
            if 'price' not in df.columns and 'LBMP ($/MWHr)' in df.columns:
                df = df.rename(columns={'LBMP ($/MWHr)': 'price'})
            
            # Ensure the timestamp column is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add to collection
            all_dfs.append(df)
            
        # If tidy format not available, try wide format
        elif os.path.exists(wide_file):
            print(f"Loading wide data from {wide_file}")
            wide_df = pd.read_csv(wide_file)
            
            # Determine the timestamp column
            timestamp_col = None
            if 'date' in wide_df.columns:
                timestamp_col = 'date'
            elif 'Date' in wide_df.columns:
                timestamp_col = 'Date'
            elif 'timestamp' in wide_df.columns:
                timestamp_col = 'timestamp'
            
            if timestamp_col:
                # Convert from wide to tidy format
                tidy_df = wide_df.melt(
                    id_vars=[timestamp_col],
                    var_name='node',
                    value_name='price'
                )
                
                # Standardize column names
                tidy_df = tidy_df.rename(columns={timestamp_col: 'timestamp'})
                
                # Ensure the timestamp column is datetime
                tidy_df['timestamp'] = pd.to_datetime(tidy_df['timestamp'])
                
                all_dfs.append(tidy_df)
            else:
                print(f"Could not find timestamp column in {wide_file}, skipping")
        else:
            print(f"Warning: No data file found for {year} {market_name}")
            
            # Check if extracted CSV files exist that could be processed
            extract_dir = os.path.join(base_dir, "extracted")
            if os.path.exists(extract_dir):
                csv_files = glob.glob(os.path.join(extract_dir, f"*{year}*.csv"))
                if csv_files:
                    print(f"Found {len(csv_files)} CSV files to process for {year}")
                    
                    year_dfs = []
                    for csv_file in csv_files:
                        # Process each CSV directly (simplified version of what's in the main function)
                        try:
                            # Read the CSV file
                            raw_df = pd.read_csv(csv_file)
                            
                            # Convert columns to the standard format
                            if 'Time Stamp' in raw_df.columns and 'Name' in raw_df.columns and 'LBMP ($/MWHr)' in raw_df.columns:
                                df = raw_df[['Time Stamp', 'Name', 'LBMP ($/MWHr)']].copy()
                                df = df.rename(columns={
                                    'Time Stamp': 'timestamp',
                                    'Name': 'node',
                                    'LBMP ($/MWHr)': 'price'
                                })
                                
                                # Ensure timestamp is datetime
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                
                                year_dfs.append(df)
                            else:
                                print(f"Expected columns not found in {csv_file}, skipping")
                                
                        except Exception as e:
                            print(f"Error processing {csv_file}: {e}")
                    
                    if year_dfs:
                        combined_df = pd.concat(year_dfs, ignore_index=True)
                        all_dfs.append(combined_df)
                    else:
                        print(f"No data processed from extracted files for {year}")
    
    # Combine all dataframes
    if not all_dfs:
        print("No data available for the specified period")
        return pd.DataFrame(columns=['timestamp', 'node', 'price'])
    
    result_df = pd.concat(all_dfs, ignore_index=True)
    
    # Filter by date range
    result_df = result_df[(result_df['timestamp'] >= start_date) & 
                          (result_df['timestamp'] <= end_date)]
    
    # Filter by nodes if specified
    if nodes:
        result_df = result_df[result_df['node'].isin(nodes)]
    
    # Sort the result
    result_df = result_df.sort_values(['timestamp', 'node'])
    
    # Resample if needed to match requested frequency
    target_freq = freq
    original_freq = infer_frequency(result_df)
    
    if target_freq != original_freq:
        print(f"Resampling from {original_freq} to {target_freq}")
        # Group by node and resample
        result_df = (result_df.groupby('node')
                    .apply(lambda x: x.set_index('timestamp')
                           .resample(target_freq)
                           .mean()
                           .reset_index())
                    .reset_index(drop=True))
    
    # Ensure we have the right columns and order
    result_df = result_df[['timestamp', 'node', 'price']]
    
    return result_df

def infer_frequency(df):
    """Infer the frequency of a DataFrame with timestamp column"""
    if len(df) <= 1:
        return "1D"  # Default to daily if insufficient data
    
    # Get unique timestamps for one node to check frequency
    if 'node' in df.columns:
        sample_node = df['node'].iloc[0]
        timestamps = df[df['node'] == sample_node]['timestamp'].sort_values().reset_index(drop=True)
    else:
        timestamps = df['timestamp'].sort_values().reset_index(drop=True)
    
    if len(timestamps) <= 1:
        return "1D"
    
    # Calculate the most common time difference
    time_diffs = timestamps.diff().dropna()
    
    if time_diffs.empty:
        return "1D"
    
    # Convert to seconds and find the most common difference
    seconds = time_diffs.dt.total_seconds()
    most_common_seconds = seconds.value_counts().idxmax()
    
    # Map to pandas frequency strings
    if most_common_seconds == 60:
        return "1min"
    elif most_common_seconds == 300:
        return "5min"
    elif most_common_seconds == 900:
        return "15T"
    elif most_common_seconds == 3600:
        return "1H"
    elif most_common_seconds == 86400:
        return "1D"
    else:
        # Return the frequency in seconds if it doesn't match common intervals
        return f"{int(most_common_seconds)}S"

if __name__ == "__main__":
    main() 