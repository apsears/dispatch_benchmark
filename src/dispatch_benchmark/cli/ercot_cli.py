#!/usr/bin/env python3
"""Command-line interface for ERCOT data processing."""

import argparse
from dispatch_benchmark.io.ercot import main as ercot_main
from dispatch_benchmark.models.benchmark import run_benchmark
import yaml
from pathlib import Path

# Import our configuration system
from dispatch_benchmark.config import get_benchmark_config, get_config


def backtest_command(args):
    """Run backtesting for specified nodes and date range."""
    # Process nodes from CLI or YAML file
    nodes = []
    if args.nodes:
        nodes = args.nodes.split(",")
    elif args.nodes_file:
        config_path = Path(args.nodes_file)
        if config_path.exists():
            with open(config_path, "r") as f:
                if (
                    config_path.suffix.lower() == ".yaml"
                    or config_path.suffix.lower() == ".yml"
                ):
                    config = yaml.safe_load(f)
                    nodes = config.get("nodes", [])
                else:
                    # Assume plain text file with one node per line
                    nodes = [line.strip() for line in f if line.strip()]

    # Run the benchmark
    run_benchmark(
        prices_path=args.prices,
        start_date=args.start,
        end_date=args.end,
        nodes=nodes,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        max_nodes=args.max_nodes,
    )


def config_command(args):
    """Display the current configuration."""
    config = get_config()

    if args.section:
        # Display a specific section
        if args.section in config:
            section_config = config[args.section]
            print(f"Configuration for [{args.section}]:")
            for key, value in section_config.items():
                print(f"  {key} = {value}")
        else:
            print(f"Section [{args.section}] not found in configuration.")
            print("Available sections:")
            for section in config.keys():
                print(f"  {section}")
    else:
        # Display the full configuration
        print("Current configuration:")
        for section, section_config in config.items():
            print(f"[{section}]")
            if isinstance(section_config, dict):
                for key, value in section_config.items():
                    print(f"  {key} = {value}")
            else:
                print(f"  {section_config}")
            print()


def main():
    """Entry point for the ERCOT CLI command."""
    # Get benchmark config from our configuration system
    benchmark_config = get_benchmark_config()

    parser = argparse.ArgumentParser(description="ERCOT data tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add the original ERCOT command as a subcommand
    ercot_parser = subparsers.add_parser(
        "ercot", help="Run ERCOT data processing"
    )

    # Add backtest command
    backtest_parser = subparsers.add_parser(
        "backtest", help="Run backtesting on historical data"
    )
    backtest_parser.add_argument(
        "--nodes",
        help="Comma-separated list of nodes to benchmark (e.g., 'HB_HOUSTON,HB_NORTH')",
    )
    backtest_parser.add_argument(
        "--nodes-file",
        help="Path to a YAML or text file containing nodes to benchmark",
    )
    backtest_parser.add_argument(
        "--start", help="Start date for backtesting (YYYY-MM or YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--end", help="End date for backtesting (YYYY-MM or YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--prices",
        default="concatenated_all_data.csv",
        help="Path to the prices CSV file",
    )
    backtest_parser.add_argument(
        "--output-dir",
        default=benchmark_config.get("output_dir", "benchmark_results"),
        help="Directory to save results",
    )
    backtest_parser.add_argument(
        "--n-jobs",
        type=int,
        default=benchmark_config.get("n_jobs", -1),
        help="Number of processes to use (default: from config or all cores)",
    )
    backtest_parser.add_argument(
        "--max-nodes",
        type=int,
        default=benchmark_config.get("max_nodes", 100),
        help="Maximum number of settlement points to use",
    )
    backtest_parser.set_defaults(func=backtest_command)

    # Add config command to display configuration
    config_parser = subparsers.add_parser(
        "config", help="Display configuration settings"
    )
    config_parser.add_argument(
        "--section", help="Specific configuration section to display"
    )
    config_parser.set_defaults(func=config_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "ercot":
        ercot_main()
    elif hasattr(args, "func"):
        args.func(args)


if __name__ == "__main__":
    main()
