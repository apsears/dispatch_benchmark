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
    parser.add_argument('--year', type=int, required=True,
                        help='Year to process (YYYY)')
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
    page_size=10000,
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

def pivot_and_combine_data(dfs, year, market_type, output_dir):
    """Pivot and combine DataFrames to create a wide format with nodes as columns.
    
    Args:
        dfs: List of DataFrames to combine
        year: Year of data
        market_type: Market type (rtm or dam)
        output_dir: Directory to save output
        
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
    
    # Pivot to make each node a column
    print("Pivoting data (this may take a while for many nodes)...")
    pivoted = all_data.pivot_table(
        index="timestamp", 
        columns=settlement_point_col, 
        values=price_col,
        aggfunc="mean"  # If there are still duplicates, take the mean
    )
    
    # Ensure the index is sorted by timestamp
    pivoted = pivoted.sort_index()
    
    # Save to CSV
    market_name = "RealTime" if market_type == "rtm" else "DayAhead"
    output_path = os.path.join(output_dir, f"{year}_{market_name}_SPP.csv")
    
    print(f"Saving combined data to {output_path}...")
    pivoted.to_csv(output_path)
    
    # Print statistics
    num_nodes = pivoted.shape[1]
    num_intervals = pivoted.shape[0]
    expected_intervals = 365 * 24 * 4  # 15-min data for a normal year
    if year % 4 == 0:  # Leap year
        expected_intervals = 366 * 24 * 4
    
    print(f"\nSummary for {year} {market_name} Market:")
    print(f"Total settlement points (nodes): {num_nodes}")
    print(f"Total time intervals: {num_intervals}")
    print(f"Expected intervals for complete year: {expected_intervals} (15-min data)")
    print(f"Data completeness: {num_intervals / expected_intervals:.1%}")
    
    return output_path

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
    
    # Step 3: Pivot and combine all data
    output_path = pivot_and_combine_data(all_dfs, year, market_type, final_dir)
    
    if output_path:
        print(f"\nProcessing complete. Output file: {output_path}")
        return output_path
    else:
        print("\nProcessing failed.")
        return None 