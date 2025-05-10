# NYISO and ERCOT Data Processor

This repository contains scripts for downloading, processing, and analyzing NYISO and ERCOT electricity market data, as well as tools for battery storage optimization using price forecasting.

## Scripts Overview

1. `download_nyiso.py` - For downloading monthly NYISO LBMP data archives
2. `process_nyiso.py` - For downloading, processing, pivoting and concatenating NYISO data
3. `process_ercot.py` - For downloading, processing, pivoting and concatenating ERCOT data via the API
4. `get_data.py` - For downloading ERCOT data using the official authenticated API

## ERCOT API Credentials

The ERCOT API scripts now use environment variables loaded from a `.env` file for API credentials:

```
# ERCOT API Credentials
ERCOT_CLIENT_ID=your_client_id_here
ERCOT_SUB_KEY=your_subscription_key_here
```

A `.env` file has been created with default credentials. If you need to use your own credentials, simply edit this file.

## ERCOT Data Processor

The `process_ercot.py` script performs the following tasks:

1. Downloads ERCOT Settlement Point Price (SPP) data for a specified year using the ERCOT API
2. Processes the JSON data received from the API
3. Pivots the data so each settlement point (node) becomes a column
4. Combines all monthly data into a single yearly CSV file

### Requirements

- Python 3
- pandas
- numpy
- requests
- python-dotenv (for loading environment variables)

Install requirements:
```
pip install pandas numpy requests tqdm python-dotenv
```

### Authentication

The script uses the ERCOT authenticated API, which requires:
- ERCOT account credentials (username/password)
- Client ID and Subscription Key (stored in .env file)

### Usage

```
python process_ercot.py --year YYYY [options]
```

### Arguments

- `--year`: Year to process, required (YYYY)
- `--market`: Market data to download (choices: rtm, dam; default: rtm)
  - `rtm`: Real-Time Market
  - `dam`: Day-Ahead Market
- `--output-dir`: Base directory for output (default: data/ercot)
- `--max-workers`: Maximum number of concurrent downloads (default: 5)
- `--nodes`: Optional list of specific settlement point nodes to include
- `--username`: ERCOT API username
- `--password`: ERCOT API password
- `--skip-download`: Skip downloading and only process existing files

### Examples

Process Real-Time Market data for 2023:
```
python process_ercot.py --year 2023
```

Process Day-Ahead Market data for 2022:
```
python process_ercot.py --year 2022 --market dam
```

Filter for specific nodes only:
```
python process_ercot.py --year 2023 --nodes HB_HOUSTON HB_NORTH HB_SOUTH
```

Only process files that have already been downloaded:
```
python process_ercot.py --year 2023 --skip-download
```

### Output

The script follows the directory structure you requested:
- Raw downloaded files are saved to `data/ercot/source/`
- Final pivoted yearly files are saved to `data/ercot/`

The final CSV will have:
- A `timestamp` index at 15-minute intervals
- One column for each ERCOT settlement point (node)
- Price values in $/MWh

## Using get_data.py

The `get_data.py` script is designed for direct access to the ERCOT API:

```
python get_data.py
```

It will prompt for:
1. ERCOT username and password
2. Start and end dates for data retrieval
3. Optional settlement point filter

The script downloads all data pages and saves them as JSON files in `data/api/` for further processing.

## NYISO Data Processor

For details on the NYISO scripts, see their respective sections above.

# ERCOT Data Tools

This repository contains scripts for downloading and processing ERCOT (Electric Reliability Council of Texas) Settlement Point Price data.

## Main Scripts

### fixed_ercot_api.py

Download Settlement Point Price data from the ERCOT API and optionally pivot it into a CSV format.

**Usage:**
```
# Download a specific month
python fixed_ercot_api.py --username YOUR_USERNAME --year 2025 --month 4 --market rtm

# Download all available months for a year
python fixed_ercot_api.py --username YOUR_USERNAME --year 2025 --market rtm

# Just process existing files (no download)
python fixed_ercot_api.py --year 2025 --market rtm --skip-download

# Download only (skip pivoting)
python fixed_ercot_api.py --username YOUR_USERNAME --year 2025 --market rtm --download-only
```

**Key Features:**
- Downloads data for a specific year and month or all months in a year
- Supports both Real-Time Market (rtm) and Day-Ahead Market (dam)
- Configurable page size and maximum pages to download
- Handles API pagination automatically
- Skips empty files and shows record counts while downloading
- Automatically pivots downloaded data into CSV format
- Verifies existing files to avoid unnecessary downloads

**Options:**
- `--username`: ERCOT username (will prompt if not provided)
- `--year`: Year to download (default: 2023)
- `--month`: Month to download (1-12); 0 for all months in year (default: 0)
- `--market`: Market type: rtm or dam (default: rtm)
- `--output-dir`: Directory to save downloaded data (default: data/ercot/source)
- `--settlement-point`: Optional filter for a specific settlement point
- `--page-size`: Number of records per API request (default: 1000000)
- `--max-pages`: Maximum number of pages to download per month (default: 0 = all pages)
- `--sleep`: Sleep time between requests in seconds (default: 3.0)
- `--skip-download`: Skip downloading and just process existing files
- `--download-only`: Only download data, skip pivoting step
- `--pivot-output-dir`: Directory to save pivoted CSV files (default: data/ercot)

### pivot_ercot_data.py

Standalone script to convert downloaded ERCOT JSON data to a pivoted CSV format with timestamps as rows and settlement points as columns.

**Usage:**
```
python pivot_ercot_data.py --input-files data/ercot/source/spp_rtm_202504.json
```

**Key Features:**
- Processes one or more JSON files
- Combines and pivots data for easy analysis
- Handles duplicate timestamps
- Creates a clean CSV with settlement point prices

**Options:**
- `--input-files`: One or more JSON files to process (required)
- `--output-dir`: Directory to save output CSV files (default: data/ercot)
- `--prefix`: Optional prefix for output filename

## Workflow Examples

### Download and Process All Data for a Year:
```
python fixed_ercot_api.py --username YOUR_USERNAME --year 2025 --market rtm
```
This will:
1. Download all available months for 2025
2. Process the downloaded JSON files
3. Create a pivoted CSV file at: `data/ercot/spp_rtm_2025_pivoted.csv`

### Download a Specific Month:
```
python fixed_ercot_api.py --username YOUR_USERNAME --year 2025 --month 4 --market rtm
```
This will create: `data/ercot/spp_rtm_202504_pivoted.csv`

### Process Already Downloaded Data:
```
python fixed_ercot_api.py --year 2025 --market rtm --skip-download
```
This will process existing JSON files without downloading.

## Battery Storage Optimization

This repository includes battery dispatch optimization models that maximize revenue from price arbitrage:

1. `oracle_lp.py` - Linear programming model with perfect price foresight (theoretical upper bound)
2. `online_mpc.py` - Model Predictive Control using forecasted prices
3. `quartile.py` - Percentile-based heuristic dispatch strategy

### Price Forecasting

The battery optimization models use several price forecasters to predict future prices:

- **Naive forecaster**: Simple baseline that predicts future prices will equal the last observed price
- **Ridge regression**: Machine learning model using lagged prices as features
- **EWMA**: Exponentially Weighted Moving Average time series model
- **Quartile**: Percentile-based model using historical price distributions

### Benchmarking

Compare different forecasting approaches using the included scripts:

- `compare_forecasters.py`: Side-by-side comparison of multiple forecasters
- `naive_benchmark.py`: Demonstrates the value of sophisticated forecasting compared to the naive baseline

Example:
```bash
python compare_forecasters.py
python naive_benchmark.py
```

For more details on forecasters, see the [Forecasters README](src/virtual_energy/forecasters/README.md).