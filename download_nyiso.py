#!/usr/bin/env python3
"""
Download and process NYISO LBMP (Location-Based Marginal Price) data.
This script uses functionality from the virtual_energy.io.nyiso module.
"""

import argparse
import datetime
import os
from pathlib import Path

# Try to load dotenv for environment variables
from dotenv import load_dotenv

load_dotenv()

# Import nyiso module functionality
from src.virtual_energy.io.nyiso import (
    get_months_for_year,
    download_file,
    generate_monthly_urls,
    unzip_file,
    process_csv_file,
    combine_processed_data,
    main as nyiso_main,
)


def parse_args():
    current_year = datetime.date.today().year
    prev_year = current_year - 1

    parser = argparse.ArgumentParser(
        description="Download, process, and combine NYISO LBMP data"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=prev_year,
        help=f"Year to process (YYYY) - defaults to previous year ({prev_year})",
    )
    parser.add_argument(
        "--market",
        choices=["da", "rt"],
        default="rt",
        help="Market data: day-ahead (da) or realtime (rt) (default: rt)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "html", "pdf"],
        default="csv",
        help="File format to download (default: csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/nyiso",
        help="Directory to save downloaded and processed files (default: data/nyiso)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent downloads (default: 5)",
    )
    parser.add_argument(
        "--wide-format",
        action="store_true",
        help="Output data in wide format (zones as columns) instead of tidy format",
    )
    return parser.parse_args()


def main():
    """Main function that passes arguments to the nyiso module's main function"""
    args = parse_args()
    return nyiso_main(args)


if __name__ == "__main__":
    main()
