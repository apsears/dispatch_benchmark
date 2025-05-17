#!/usr/bin/env python3
"""
Download and process ERCOT Settlement Point Price data.
This script uses functionality from the dispatch_benchmark.io.ercot module.
"""

import argparse
import datetime
import os

# Try to load dotenv for environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "dotenv module not found. Install with 'uv pip install python-dotenv'"
    )

# Import ercot module functionality
from src.dispatch_benchmark.io.ercot import main as ercot_main

# Get ERCOT API credentials from environment variables
CLIENT_ID = os.getenv("ERCOT_CLIENT_ID")
SUB_KEY = os.getenv("ERCOT_SUB_KEY")


def parse_args():
    current_year = datetime.date.today().year
    prev_year = current_year - 1

    parser = argparse.ArgumentParser(
        description="Download, process, and combine ERCOT Settlement Point Price data"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=prev_year,
        help=f"Year to process (YYYY) - defaults to previous year ({prev_year})",
    )
    parser.add_argument(
        "--market",
        choices=["rtm", "dam"],
        default="rtm",
        help="Market data: real-time (rtm) or day-ahead (dam) (default: rtm)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/ercot",
        help="Base directory for output (default: data/ercot)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent downloads (default: 5)",
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        help="Optional list of specific settlement point nodes to include",
    )
    parser.add_argument("--username", help="ERCOT API username")
    parser.add_argument("--password", help="ERCOT API password")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step and only process existing files",
    )
    parser.add_argument(
        "--wide-format",
        action="store_true",
        help="Output data in wide format instead of tidy format",
    )
    return parser.parse_args()


def main():
    """Main function that passes arguments to the ercot module's main function"""
    args = parse_args()
    return ercot_main(args)


if __name__ == "__main__":
    main()
