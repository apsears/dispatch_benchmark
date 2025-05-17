"""ERCOT data processing module for retrieving and processing settlement point price data."""

import argparse
import datetime
import os
import zipfile
import glob
import pandas as pd
import numpy as np
import requests
import json
import getpass
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get ERCOT API credentials from environment variables
CLIENT_ID = os.getenv("ERCOT_CLIENT_ID")
SUB_KEY = os.getenv("ERCOT_SUB_KEY")

def parse_args():
    parser = argparse.ArgumentParser(description='Download, process, and combine ERCOT Settlement Point Price data')
    parser.add_argument('--year', type=int, default=2024,
                        help='Year to process (YYYY)')
    parser.add_argument('--market', choices=['rtm', 'dam'], default='dam',
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

def get_id_token(username, password):
    """
    Get the ID token using Resource Owner Password Credentials flow.
    
    Parameters
    ----------
    username : str
        ERCOT username
    password : str
        ERCOT password
        
    Returns
    -------
    str
        ID token for API authorization
    """
    auth_url = (
        "https://ercotb2c.b2clogin.com/"
        "ercotb2c.onmicrosoft.com/"
        "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    )
    
    payload = {
        "username": username,
        "password": password,
        "grant_type": "password",
        "scope": f"openid {CLIENT_ID} offline_access",
        "client_id": CLIENT_ID,
        "response_type": "id_token",
    }
    
    try:
        r = requests.post(auth_url, data=payload, timeout=15)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        sys.exit(f"Token request failed: {e}")
        
    id_token = r.json().get("id_token")
    if not id_token:
        sys.exit(f"No id_token in response: {r.text}")
        
    return id_token

def fetch_spp_prices_api(
    id_token, 
    subscription_key, 
    year, 
    month, 
    output_dir, 
    market_type="rtm",
    settlement_point=None,
    page_size=1000000,
    sleep=5.0,
    max_retries=3
):
    """
    Download ERCOT Settlement Point Prices using the official API.
    
    Parameters
    ----------
    id_token : str
        Azure AD B2C ID token obtained from the ROPC flow.
    subscription_key : str
        `Ocp-Apim-Subscription-Key` for the ERCOT Public API product.
    year : int
        Year to download
    month : int
        Month to download (1-12)
    output_dir : str
        Directory to save downloaded data
    market_type : str
        'rtm' for Real-Time or 'dam' for Day-Ahead
    settlement_point : str, optional
        If given, only fetch data for this node / hub / zone.
    page_size : int, default 10000
        Number of records per API call
    sleep : float, default 5.0s
        Time to wait between API calls
    max_retries : int, default 3
        Maximum number of retries for failed requests
        
    Returns
    -------
    str
        Path to the output file
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct file names
    month_str = f"{month:02d}"
    file_stem = f"spp_{market_type}_{year}{month_str}"
    json_file = os.path.join(output_dir, f"{file_stem}.json")
    
    # Check if the file already exists
    if os.path.exists(json_file):
        print(f"File {json_file} already exists, skipping download.")
        return json_file
    
    # Set up API URLs based on market_type
    if market_type == "rtm":
        api = "https://api.ercot.com/api/public-reports/np6-905-cd/spp_node_zone_hub"
    else:  # dam
        api = "https://api.ercot.com/api/public-reports/np6-788-cd/dam_spp_node_zone_hub"
    
    # Calculate first and last day of month
    first_day = datetime.date(year, month, 1)
    if month == 12:
        last_day = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        last_day = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    
    start_date = first_day.strftime("%Y-%m-%d")
    end_date = last_day.strftime("%Y-%m-%d")
    
    # Set up request parameters
    params = {
        "deliveryDateFrom": start_date,
        "deliveryDateTo": end_date,
        "size": page_size,
    }
    
    if settlement_point:
        params["settlementPoint"] = settlement_point
    
    headers = {
        "Authorization": f"Bearer {id_token}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Accept": "application/json",
    }
    
    print(f"Downloading {market_type.upper()} data for {year}-{month_str} via API...")
    print(f"Request parameters: {params}")
    
    def make_request_with_retry(url, params, headers, retries_left=max_retries):
        """Helper function to make requests with retry logic"""
        try:
            print(f"Making request to: {url}")
            print(f"With params: {params}")
            
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            
            # Print response status and headers for debugging
            print(f"Response status: {resp.status_code}")
            print(f"Response headers: {dict(resp.headers)}")
            
            # Handle empty responses (API might return empty data but status code 200)
            if resp.status_code == 200 and not resp.text.strip():
                print("Warning: Received empty response from API with status code 200")
                if retries_left > 0:
                    print(f"Retrying in {sleep} seconds...")
                    time.sleep(sleep)
                    return make_request_with_retry(url, params, headers, retries_left - 1)
                else:
                    raise requests.exceptions.RequestException("Maximum retries reached with empty responses")
            
            resp.raise_for_status()
            
            # Try parsing the response to ensure it's valid JSON
            try:
                result = resp.json()
                if not result or (isinstance(result, dict) and not result.get('data') and not result.get('_meta')):
                    print("Warning: Received empty data in JSON response")
                    if retries_left > 0:
                        print(f"Retrying in {sleep} seconds...")
                        time.sleep(sleep)
                        return make_request_with_retry(url, params, headers, retries_left - 1)
            except json.JSONDecodeError:
                if retries_left > 0:
                    print(f"Error parsing JSON response. Retrying in {sleep} seconds...")
                    time.sleep(sleep)
                    return make_request_with_retry(url, params, headers, retries_left - 1)
                else:
                    raise requests.exceptions.RequestException("Failed to parse JSON response after maximum retries")
            
            return resp
        except requests.exceptions.RequestException as e:
            if retries_left > 0:
                retry_after = sleep * 2
                
                # Check if we have a response with a status code to determine retry behavior
                if hasattr(e, 'response') and e.response is not None:
                    if e.response.status_code == 429:
                        # Handle rate limiting with Retry-After header
                        retry_after = int(e.response.headers.get('Retry-After', sleep * 2))
                        print(f"Rate limited (429). Retrying after {retry_after} seconds...")
                    elif e.response.status_code >= 500:
                        # Server errors (5xx) - wait longer
                        retry_after = sleep * 3
                        print(f"Server error ({e.response.status_code}). Retrying after {retry_after} seconds...")
                    else:
                        # Other errors - use standard delay
                        print(f"Request failed with {e.response.status_code}. Retrying after {retry_after} seconds...")
                else:
                    # Connection errors, timeouts, etc.
                    print(f"Request failed: {str(e)}. Retrying after {retry_after} seconds...")
                
                time.sleep(retry_after)
                return make_request_with_retry(url, params, headers, retries_left - 1)
            else:
                print(f"Maximum retries reached. Last error: {str(e)}")
                raise e
    
    try:
        # First page to get total count
        try:
            resp = make_request_with_retry(api, {**params, "page": 1}, headers)
            body = resp.json()
            
            # DEBUG: Print a summary of the response
            meta = body.get("_meta", {})
            print(f"Response metadata: totalRecords={meta.get('totalRecords')}, totalPages={meta.get('totalPages')}, currentPage={meta.get('currentPage')}")
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        
        # Extract total pages
        total_pages = body.get("_meta", {}).get("totalPages", 0)
        total_records = body.get("_meta", {}).get("totalRecords", 0)
        
        # Force checking at least page 1 even if totalRecords=0
        # This is a workaround for API inconsistencies
        if total_records == 0:
            print("API reports 0 records, but will try to fetch page 1 anyway...")
            # Try fetching with explicit page parameter in case the API is returning metadata incorrectly
            try:
                alt_resp = make_request_with_retry(api, {**params, "page": 1}, headers)
                alt_body = alt_resp.json()
                alt_data = alt_body.get("data", [])
                
                if alt_data:
                    print(f"Found {len(alt_data)} records on page 1 despite API reporting 0 total records")
                    body = alt_body  # Use this response instead
                    total_records = len(alt_data)
                    total_pages = 1  # Assume at least 1 page
            except requests.exceptions.RequestException:
                print("Failed to get alternative page 1 data, continuing with original response")
        
        print(f"Found {total_records:,d} records across {total_pages} pages.")
        
        # Initialize all_data with the first page
        all_data = body.get("data", [])
        
        # If API reports 0 records but we found data, process it
        if not all_data and "data" in body:
            print("API response contains 'data' field but it's empty.")
        
        # Download remaining pages
        if total_pages > 1:
            for page in tqdm(range(2, total_pages + 1), desc="Downloading pages"):
                print(f"Waiting {sleep} seconds before next request...")
                time.sleep(sleep)  # Be nice to the API
                
                try:
                    resp = make_request_with_retry(api, {**params, "page": page}, headers)
                    page_data = resp.json().get("data", [])
                    print(f"Page {page}: Retrieved {len(page_data)} records")
                    all_data.extend(page_data)
                except requests.exceptions.RequestException as e:
                    print(f"Failed to fetch page {page}: {e}")
                    # Continue with what we have so far
                    continue
        
        # Try additional pages beyond what the API reports (ERCOT API might have pagination issues)
        # Only do this if totalRecords is 0 but we expect data
        if total_records == 0 and not all_data:
            print("API reports 0 records and no data found. Trying additional page numbers as a last resort...")
            for test_page in range(1, 5):  # Try pages 1-4
                if test_page != 1:  # Already tried page 1
                    print(f"Trying page {test_page} (not reported by API)...")
                    try:
                        time.sleep(sleep)
                        resp = make_request_with_retry(api, {**params, "page": test_page}, headers)
                        test_data = resp.json().get("data", [])
                        if test_data:
                            print(f"Found {len(test_data)} records on page {test_page}")
                            all_data.extend(test_data)
                    except requests.exceptions.RequestException:
                        print(f"Failed to get page {test_page}")
        
        # Save the combined data to a JSON file
        result = {
            "_meta": body.get("_meta", {}),
            "report": body.get("report", {}),
            "fields": body.get("fields", []),
            "data": all_data
        }
        
        with open(json_file, "w") as f:
            json.dump(result, f)
            
        print(f"Saved {len(all_data)} records to {json_file}")
        
        # If we still have no data after all attempts, log a warning
        if not all_data:
            print("WARNING: No data was retrieved despite all attempts. Check your API credentials and parameters.")
            
        return json_file
    
    except Exception as e:
        print(f"API request failed: {e}")
        import traceback
        traceback.print_exc()

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
            
        months.append(month)
    
    return months

def process_json_file(json_file, nodes=None):
    """Process an ERCOT API JSON file.
    
    Args:
        json_file: Path to the JSON file
        nodes: Optional list of specific settlement point nodes to include
        
    Returns:
        DataFrame with processed data
    """
    if not os.path.exists(json_file):
        print(f"JSON file {json_file} not found")
        return None
    
    try:
        print(f"Processing {json_file}...")
        
        # Load the JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract field names
        field_names = [field.get('name', f"field_{i}").lower() 
                       for i, field in enumerate(data.get('fields', []))]
        
        # Create DataFrame from raw data
        rows = data.get('data', [])
        if not rows:
            print(f"No data found in {json_file}")
            return None
        
        df = pd.DataFrame(rows, columns=field_names)
        
        # Filter by settlement point if requested
        if nodes and 'settlementpoint' in df.columns:
            df = df[df['settlementpoint'].isin(nodes)]
        
        # Create timestamp column
        df['timestamp'] = pd.to_datetime(df['deliverydate']) + \
                          pd.to_timedelta(df['deliveryhour'] - 1, unit='h') + \
                          pd.to_timedelta((df['deliveryinterval'] - 1) * 15, unit='m')
        
        # Select only needed columns
        processed = df[['settlementpoint', 'timestamp', 'settlementpointprice']].copy()
        
        return processed
        
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None

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

# Rename the old function to maintain compatibility with any existing code
def pivot_and_combine_data(dfs, year, market_type, output_dir):
    """Legacy function that outputs data in wide format."""
    return combine_and_format_data(dfs, year, market_type, output_dir, wide_format=True)

def load_prices(start, end, nodes=None, freq="15T") -> pd.DataFrame:
    """
    Returns tidy DataFrame with ERCOT price data for the specified period and nodes.
    
    Parameters
    ----------
    start : str or datetime-like
        Start date for the data. Can be a string in format 'YYYY-MM-DD' or a datetime object.
    end : str or datetime-like
        End date for the data. Can be a string in format 'YYYY-MM-DD' or a datetime object.
    nodes : list or None
        Optional list of specific settlement point nodes to include.
        If None, all available nodes will be included.
    freq : str
        Frequency of the data. ERCOT RTM prices are at '15T' (15-minute) frequency by default.
        
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
    
    # Get market type from frequency
    # RTM (real-time market) has 15-minute frequency, DAM (day-ahead market) has hourly frequency
    if freq.endswith('T') and int(freq[:-1]) < 60:
        market_type = 'rtm'  # Real-time market (15-minute intervals)
    else:
        market_type = 'dam'  # Day-ahead market (hourly intervals)
    
    # Determine base directory for data files
    base_dir = os.getenv("ERCOT_DATA_DIR", "data/ercot")
    
    # List to hold dataframes for each year
    all_dfs = []
    
    for year in years:
        # Construct filename based on market type
        market_name = "RealTime" if market_type == "rtm" else "DayAhead"
        yearly_file = os.path.join(base_dir, f"{year}_{market_name}_SPP.csv")
        wide_file = os.path.join(base_dir, f"{year}_{market_name}_SPP_wide.csv")
        
        # Try to load the tidy format file first
        if os.path.exists(yearly_file):
            print(f"Loading tidy data from {yearly_file}")
            df = pd.read_csv(yearly_file)
            
            # Ensure the timestamp column is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add to collection
            all_dfs.append(df)
            
        # If tidy format not available, try wide format
        elif os.path.exists(wide_file):
            print(f"Loading wide data from {wide_file}")
            wide_df = pd.read_csv(wide_file)
            
            # Convert from wide to tidy format
            if 'timestamp' in wide_df.columns:
                index_col = 'timestamp'
            else:
                # Some wide files might use the timestamp as index
                wide_df = wide_df.reset_index()
                if 'index' in wide_df.columns:
                    wide_df = wide_df.rename(columns={'index': 'timestamp'})
                    index_col = 'timestamp'
                else:
                    # Older files might have used 'Date' instead
                    index_col = next((col for col in wide_df.columns if col.lower() in ['date', 'time', 'datetime']), None)
                    if index_col:
                        wide_df = wide_df.rename(columns={index_col: 'timestamp'})
                        index_col = 'timestamp'
            
            # Melt the wide format to tidy format
            if index_col:
                id_vars = [index_col]
                tidy_df = wide_df.melt(
                    id_vars=id_vars,
                    var_name='node',
                    value_name='price'
                )
                
                # Ensure the timestamp column is datetime
                tidy_df['timestamp'] = pd.to_datetime(tidy_df['timestamp'])
                
                all_dfs.append(tidy_df)
            else:
                print(f"Could not find timestamp column in {wide_file}, skipping")
        else:
            print(f"Warning: No data file found for {year} {market_name}")
            
            # Check if we should try to download the data
            if os.path.exists(os.path.join(base_dir, 'source')):
                print(f"Attempting to process from source data for {year}")
                
                # Try to find and process JSON source files
                source_dir = os.path.join(base_dir, 'source')
                json_files = []
                
                for month in range(1, 13):
                    month_str = f"{month:02d}"
                    json_file = os.path.join(source_dir, f"spp_{market_type}_{year}{month_str}.json")
                    if os.path.exists(json_file):
                        json_files.append(json_file)
                
                if json_files:
                    # Process each JSON file
                    year_dfs = []
                    for json_file in json_files:
                        df = process_json_file(json_file, nodes)
                        if df is not None:
                            year_dfs.append(df)
                    
                    if year_dfs:
                        # Combine the dataframes
                        combined_df = pd.concat(year_dfs, ignore_index=True)
                        
                        # Rename columns to standardize
                        settlement_point_col = next((col for col in combined_df.columns if col.lower() == 'settlementpoint'), 'settlementpoint')
                        price_col = next((col for col in combined_df.columns if col.lower() == 'settlementpointprice'), 'settlementpointprice')
                        
                        combined_df = combined_df.rename(columns={
                            settlement_point_col: 'node',
                            price_col: 'price',
                            'timestamp': 'timestamp'
                        })
                        
                        # Ensure the timestamp column is datetime
                        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
                        
                        all_dfs.append(combined_df)
                    else:
                        print(f"No data processed from source files for {year}")
    
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
    
    # Filter timestamps to match the specified frequency if needed
    if freq != "15T" and market_type == 'rtm':
        # For RTM data, we need to resample if a different frequency is requested
        # Group by node and resample
        result_df = (result_df.groupby('node')
                     .apply(lambda x: x.set_index('timestamp')
                            .resample(freq)
                            .mean()
                            .reset_index())
                     .reset_index(drop=True))
    
    # Ensure we have the right columns and order
    result_df = result_df[['timestamp', 'node', 'price']]
    
    return result_df

def main(args=None):
    if args is None:
        args = parse_args()
    
    # Set up directories
    base_dir = Path(args.output_dir)
    source_dir = base_dir / "source"
    final_dir = base_dir
    
    # Make sure directories exist
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    # Get months to process
    year = args.year
    market_type = args.market
    current_year = datetime.date.today().year
    
    if year < 2010 or year > current_year:
        print(f"Error: Year {year} is out of range. Must be between 2010 and {current_year}.")
        return
    
    months = get_months_for_year(year)
    
    if not months:
        print(f"No complete months found for year {year}.")
        return
    
    # Step 1: Download data for each month (if not skipping)
    data_files = []
    
    if not args.skip_download:
        # Check if credentials are available
        if not CLIENT_ID or not SUB_KEY:
            print("Warning: ERCOT API credentials not found in environment variables.")
            print("Please create a .env file with ERCOT_CLIENT_ID and ERCOT_SUB_KEY.")
            return
            
        # Get API authentication if needed
        if not args.username:
            args.username = input("ERCOT username: ")
        if not args.password:
            args.password = getpass.getpass("ERCOT password: ")
            
        print("Getting authentication token...")
        id_token = get_id_token(args.username, args.password)
        print("Successfully obtained authentication token")
        
        # Download via API - one month at a time with delays
        print(f"\nWill download {len(months)} months of data with delays between requests...\n")
        for i, month in enumerate(months):
            print(f"\nDownloading month {i+1} of {len(months)}: {year}-{month:02d}")
            
            # Add a longer delay between months to avoid rate limiting
            if i > 0:
                delay = 30  # 30 seconds between monthly downloads
                print(f"Waiting {delay} seconds before starting next month download...")
                time.sleep(delay)
            
            data_file = fetch_spp_prices_api(
                id_token=id_token,
                subscription_key=SUB_KEY,
                year=year,
                month=month,
                output_dir=source_dir,
                market_type=market_type,
                settlement_point=None if not args.nodes else args.nodes[0],
                sleep=5.0,  # 5 seconds between page requests
                max_retries=3
            )
            if data_file:
                data_files.append(data_file)
                print(f"Successfully downloaded {data_file}")
            else:
                print(f"Failed to download data for {year}-{month:02d}")
    else:
        # Find existing JSON files
        for month in months:
            month_str = f"{month:02d}"
            json_file = os.path.join(source_dir, f"spp_{market_type}_{year}{month_str}.json")
            if os.path.exists(json_file):
                data_files.append(json_file)
    
    if not data_files:
        print("No data files were downloaded or found.")
        return
    
    # Step 2: Process each data file
    all_dfs = []
    
    for data_file in data_files:
        df = process_json_file(data_file, args.nodes)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        print("No data was extracted from the data files.")
        return
    
    # Step 3: Combine all data and format as requested
    output_path = combine_and_format_data(all_dfs, year, market_type, final_dir, wide_format=args.wide_format)
    
    if output_path:
        print(f"\nProcessing complete. Output file: {output_path}")
        return output_path
    else:
        print("\nProcessing failed.")
        return None

if __name__ == "__main__":
    main() 