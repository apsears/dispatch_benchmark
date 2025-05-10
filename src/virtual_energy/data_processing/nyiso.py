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

def pivot_and_process_csv(csv_path, output_dir):
    """
    Read a CSV file, pivot the data so each unique 'Name' becomes a column,
    and return the processed DataFrame.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Convert Time Stamp to datetime
        df['Time Stamp'] = pd.to_datetime(df['Time Stamp'])
        
        # Extract date (without time) for grouping
        df['Date'] = df['Time Stamp'].dt.date
        
        # Pivot the data: each unique Name becomes a column
        # We'll use LBMP ($/MWHr) as the value
        pivot_df = df.pivot_table(
            index='Date',
            columns='Name',
            values='LBMP ($/MWHr)',
            aggfunc='mean'  # Average price for each day
        ).reset_index()
        
        # Save the pivoted data temporarily to see progress
        temp_output_path = os.path.join(output_dir, f"pivoted_{os.path.basename(csv_path)}")
        pivot_df.to_csv(temp_output_path, index=False)
        
        print(f"Pivoted {csv_path} -> {temp_output_path}")
        
        return pivot_df
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def combine_pivoted_data(pivoted_dfs, year, market_type, output_dir):
    """Combine all pivoted DataFrames into a single yearly file."""
    if not pivoted_dfs:
        print("No data to combine.")
        return None, None
    
    try:
        # Concatenate all DataFrames
        combined_df = pd.concat(pivoted_dfs, ignore_index=True)
        
        # Sort by date
        combined_df = combined_df.sort_values('Date')
        
        # Save the combined data
        market_name = "DayAhead" if market_type == "da" else "RealTime"
        output_path = os.path.join(output_dir, f"{year}_{market_name}_LBMP.csv")
        combined_df.to_csv(output_path, index=False)
        
        print(f"Combined data saved to {output_path}")
        print(f"Data shape: {combined_df.shape[0]} rows x {combined_df.shape[1]} columns")
        print(f"Number of unique zones: {combined_df.shape[1] - 1}")  # Subtract 1 for the Date column
        
        return combined_df, market_name
        
    except Exception as e:
        print(f"Error combining data: {e}")
        return None, None

def main():
    args = parse_args()
    
    year = args.year
    market_type = args.market
    file_format = args.format
    
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
    pivoted_dfs = []
    for csv_file in csv_files:
        pivot_df = pivot_and_process_csv(csv_file, output_dir)
        if pivot_df is not None:
            pivoted_dfs.append(pivot_df)
    
    # Step 4: Combine all pivoted data
    combined_df, market_name = combine_pivoted_data(pivoted_dfs, year, market_type, output_dir)
    
    # Print summary
    if combined_df is not None:
        total_rows = combined_df.shape[0]
        total_zones = combined_df.shape[1] - 1  # Subtract 1 for the Date column
        
        print("\nSummary:")
        print(f"Year: {year}")
        print(f"Market: {market_type}")
        print(f"Total days of data: {total_rows}")
        print(f"Total zones: {total_zones}")
        print(f"Output file: {output_dir}/{year}_{market_name}_LBMP.csv")

if __name__ == "__main__":
    main() 