#!/usr/bin/env python3
"""
Benchmark results summary script for virtual energy storage.

This script reads benchmark results and produces summary statistics
across all nodes, including model comparisons, rankings, and visualization.
It also handles different data formats from ERCOT and NYISO.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tabulate import tabulate


def load_results(results_dir, iso=None):
    """Load and combine all benchmark results from individual JSON files.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing benchmark results
    iso : str, optional
        ISO name to load specific results for. If None, attempt to load from the provided directory.
        If 'BOTH', look for subdirectories named 'ercot' and 'nyiso'.
    """
    results_path = Path(results_dir)
    
    # If iso is BOTH, check for subdirectories
    if iso == "BOTH":
        # Initialize results for both ISOs
        all_results = []
        
        # Try to load ERCOT results
        ercot_dir = results_path / "ercot"
        if ercot_dir.exists():
            ercot_results = load_results(ercot_dir, "ERCOT")
            if ercot_results is not None:
                # Add ISO identifier column
                ercot_results['iso'] = "ERCOT"
                all_results.append(ercot_results)
                
        # Try to load NYISO results
        nyiso_dir = results_path / "nyiso"
        if nyiso_dir.exists():
            nyiso_results = load_results(nyiso_dir, "NYISO")
            if nyiso_results is not None:
                # Add ISO identifier column
                nyiso_results['iso'] = "NYISO"
                all_results.append(nyiso_results)
        
        if not all_results:
            print("No results found for any ISO")
            return None
            
        # Combine results
        return pd.concat(all_results, ignore_index=True)
        
    # Try to load the combined results file if it exists
    combined_file = results_path / "combined_results.json"
    if combined_file.exists():
        print(f"Loading combined results from {combined_file}")
        with open(combined_file, 'r') as f:
            raw_results = json.load(f)
        
        # Convert to pandas DataFrame for easier analysis
        all_results = []
        for node, model_results in raw_results.items():
            for result in model_results:
                result['node'] = node
                all_results.append(result)
        
        results_df = pd.DataFrame(all_results)
        
        # Add ISO identifier if provided
        if iso:
            results_df['iso'] = iso
            
        return results_df
    
    # If no combined file, look for individual result files
    print(f"No combined results file found. Looking for individual files in {results_dir}")
    all_results = []
    
    # Find all JSON files in the directory
    json_files = list(results_path.glob("*_results.json"))
    if not json_files:
        print(f"No result files found in {results_dir}")
        return None
    
    for json_file in json_files:
        node = json_file.stem.replace("_results", "")
        try:
            with open(json_file, 'r') as f:
                node_results = json.load(f)
            
            for result in node_results:
                result['node'] = node
                all_results.append(result)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if not all_results:
        print("No valid results found")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # Add ISO identifier if provided
    if iso:
        results_df['iso'] = iso
        
    return results_df


def calculate_metrics(results_df, iso=None):
    """Calculate performance metrics across all nodes and models.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame containing benchmark results
    iso : str, optional
        ISO name to filter results by
    """
    if results_df is None or results_df.empty:
        return None
    
    # Filter by ISO if specified
    if iso and 'iso' in results_df.columns:
        results_df = results_df[results_df['iso'] == iso]
        if results_df.empty:
            print(f"No results found for ISO: {iso}")
            return None
    
    # Ensure model names are consistent
    model_mapping = {
        'oracle_LP': 'oracle_lp',
        'online_mpc_ridge': 'online_mpc_ridge',
        'online_mpc_naive': 'online_mpc_naive', 
        'online_quartile_p10': 'online_quartile_p10',
        'online_quartile_p25': 'online_quartile_p25',
        'online_quartile_p45': 'online_quartile_p45'
    }
    results_df['model'] = results_df['model'].map(lambda x: model_mapping.get(x, x))
    
    # Group by node and model to get average performance
    groupby_cols = ['node', 'model']
    if 'iso' in results_df.columns:
        groupby_cols = ['iso', 'node', 'model']
        
    grouped = results_df.groupby(groupby_cols).agg({
        'revenue': 'mean',
        'runtime_seconds': 'mean'
    }).reset_index()
    
    # Create a pivot table to compare models side by side
    pivot_index = ['node']
    if 'iso' in results_df.columns:
        pivot_index = ['iso', 'node']
        
    pivot_revenue = grouped.pivot(index=pivot_index, columns='model', values='revenue')
    pivot_runtime = grouped.pivot(index=pivot_index, columns='model', values='runtime_seconds')
    
    # Calculate performance relative to oracle (theoretical maximum)
    relative_df = pd.DataFrame(index=pivot_revenue.index)
    
    # Only process if oracle_lp exists in the results
    if 'oracle_lp' in pivot_revenue.columns:
        for model in pivot_revenue.columns:
            if model != 'oracle_lp':
                # Calculate percentage of oracle revenue
                relative_df[f'{model}_vs_oracle'] = (pivot_revenue[model] / pivot_revenue['oracle_lp']) * 100
        
        # Calculate difference between ridge and naive if both exist
        if 'online_mpc_ridge' in pivot_revenue.columns and 'online_mpc_naive' in pivot_revenue.columns:
            relative_df['ridge_vs_naive'] = ((pivot_revenue['online_mpc_ridge'] - pivot_revenue['online_mpc_naive']) 
                                            / pivot_revenue['online_mpc_naive']) * 100
    
    # Calculate overall statistics
    if 'iso' in grouped.columns:
        summary_stats = {}
        
        # Calculate metrics for each ISO separately
        for iso_name, iso_group in grouped.groupby('iso'):
            iso_pivot_revenue = pivot_revenue.loc[iso_name]
            iso_relative_df = relative_df.loc[iso_name] if not relative_df.empty else pd.DataFrame()
            
            summary_stats[iso_name] = {
                'revenue_means': iso_group.groupby('model')['revenue'].mean(),
                'revenue_stdev': iso_group.groupby('model')['revenue'].std(),
                'runtime_means': iso_group.groupby('model')['runtime_seconds'].mean(),
                'relative_means': {col: iso_relative_df[col].mean() for col in iso_relative_df.columns},
                'relative_stdev': {col: iso_relative_df[col].std() for col in iso_relative_df.columns}
            }
            
        # Also calculate overall metrics for all ISOs combined
        summary_stats['OVERALL'] = {
            'revenue_means': grouped.groupby('model')['revenue'].mean(),
            'revenue_stdev': grouped.groupby('model')['revenue'].std(),
            'runtime_means': grouped.groupby('model')['runtime_seconds'].mean(),
            'relative_means': {col: relative_df[col].mean() for col in relative_df.columns},
            'relative_stdev': {col: relative_df[col].std() for col in relative_df.columns}
        }
    else:
        summary_stats = {
            'revenue_means': grouped.groupby('model')['revenue'].mean(),
            'revenue_stdev': grouped.groupby('model')['revenue'].std(),
            'runtime_means': grouped.groupby('model')['runtime_seconds'].mean(),
            'relative_means': {col: relative_df[col].mean() for col in relative_df.columns},
            'relative_stdev': {col: relative_df[col].std() for col in relative_df.columns}
        }
    
    return {
        'detailed': grouped,
        'pivot_revenue': pivot_revenue,
        'pivot_runtime': pivot_runtime,
        'relative': relative_df,
        'summary': summary_stats
    }


def print_summary(metrics, iso=None):
    """Print summary statistics in a readable format."""
    if metrics is None:
        print("No metrics available to summarize")
        return
    
    print("\n===== BENCHMARK SUMMARY =====")
    
    # If we have multiple ISOs, handle accordingly
    if isinstance(metrics['summary'], dict) and any(key in metrics['summary'] for key in ['ERCOT', 'NYISO', 'OVERALL']):
        # Determine which ISO(s) to print
        isos_to_print = []
        if iso and iso != 'BOTH':
            if iso in metrics['summary']:
                isos_to_print = [iso]
            else:
                print(f"No data available for ISO: {iso}")
                return
        else:
            isos_to_print = [k for k in metrics['summary'].keys() if k != 'OVERALL']
            if 'OVERALL' in metrics['summary'] and len(isos_to_print) > 1:
                isos_to_print.append('OVERALL')
        
        # Print summary for each ISO
        for current_iso in isos_to_print:
            print(f"\n----- {current_iso} Summary -----")
            
            # 1. Overall revenue performance by model
            print("\nAverage Revenue by Model:")
            revenue_df = pd.DataFrame({
                'Mean Revenue ($)': metrics['summary'][current_iso]['revenue_means'],
                'Std Dev': metrics['summary'][current_iso]['revenue_stdev']
            }).sort_values('Mean Revenue ($)', ascending=False)
            
            print(tabulate(revenue_df, headers='keys', floatfmt='.2f', tablefmt='psql'))
            
            # 2. Relative performance compared to oracle
            if metrics['summary'][current_iso]['relative_means']:
                print("\nRelative Performance (% of optimal):")
                rel_data = []
                
                for model, pct in metrics['summary'][current_iso]['relative_means'].items():
                    if 'vs_oracle' in model:
                        model_name = model.replace('_vs_oracle', '')
                        rel_data.append([
                            model_name, 
                            pct,
                            metrics['summary'][current_iso]['relative_stdev'][model]
                        ])
                
                rel_df = pd.DataFrame(rel_data, columns=['Model', 'Mean %', 'Std Dev'])
                rel_df = rel_df.sort_values('Mean %', ascending=False)
                
                print(tabulate(rel_df, headers='keys', floatfmt='.2f', tablefmt='psql'))
            
            # 3. Ridge vs. Naive comparison
            if 'ridge_vs_naive' in metrics['summary'][current_iso]['relative_means']:
                ridge_vs_naive = metrics['summary'][current_iso]['relative_means']['ridge_vs_naive']
                ridge_vs_naive_std = metrics['summary'][current_iso]['relative_stdev']['ridge_vs_naive']
                
                print(f"\nRidge vs. Naive Forecaster: {ridge_vs_naive:.2f}% ± {ridge_vs_naive_std:.2f}%")
                
                # Count how often ridge beats naive
                if current_iso == 'OVERALL':
                    # For overall, filter the relative dataframe by each ISO
                    for iso_name in [k for k in metrics['summary'].keys() if k != 'OVERALL']:
                        if iso_name in metrics['relative'].index.get_level_values(0):
                            iso_relative_df = metrics['relative'].loc[iso_name]
                            ridge_wins = (iso_relative_df['ridge_vs_naive'] > 0).sum()
                            total_nodes = len(iso_relative_df)
                            print(f"{iso_name}: Ridge outperforms Naive in {ridge_wins} out of {total_nodes} nodes ({ridge_wins/total_nodes*100:.1f}%)")
                else:
                    # For specific ISO, get just that ISO's data
                    if current_iso in metrics['relative'].index.get_level_values(0):
                        iso_relative_df = metrics['relative'].loc[current_iso]
                        ridge_wins = (iso_relative_df['ridge_vs_naive'] > 0).sum()
                        total_nodes = len(iso_relative_df)
                        print(f"Ridge outperforms Naive in {ridge_wins} out of {total_nodes} nodes ({ridge_wins/total_nodes*100:.1f}%)")
            
            # 4. Runtime comparison
            print("\nAverage Runtime by Model (seconds):")
            runtime_df = pd.DataFrame({
                'Mean Runtime': metrics['summary'][current_iso]['runtime_means']
            }).sort_values('Mean Runtime')
            
            print(tabulate(runtime_df, headers='keys', floatfmt='.2f', tablefmt='psql'))
    else:
        # Original single-ISO format
        # 1. Overall revenue performance by model
        print("\nAverage Revenue by Model:")
        revenue_df = pd.DataFrame({
            'Mean Revenue ($)': metrics['summary']['revenue_means'],
            'Std Dev': metrics['summary']['revenue_stdev']
        }).sort_values('Mean Revenue ($)', ascending=False)
        
        print(tabulate(revenue_df, headers='keys', floatfmt='.2f', tablefmt='psql'))
        
        # 2. Relative performance compared to oracle
        if metrics['summary']['relative_means']:
            print("\nRelative Performance (% of optimal):")
            rel_data = []
            
            for model, pct in metrics['summary']['relative_means'].items():
                if 'vs_oracle' in model:
                    model_name = model.replace('_vs_oracle', '')
                    rel_data.append([
                        model_name, 
                        pct,
                        metrics['summary']['relative_stdev'][model]
                    ])
            
            rel_df = pd.DataFrame(rel_data, columns=['Model', 'Mean %', 'Std Dev'])
            rel_df = rel_df.sort_values('Mean %', ascending=False)
            
            print(tabulate(rel_df, headers='keys', floatfmt='.2f', tablefmt='psql'))
        
        # 3. Ridge vs. Naive comparison
        if 'ridge_vs_naive' in metrics['summary']['relative_means']:
            ridge_vs_naive = metrics['summary']['relative_means']['ridge_vs_naive']
            ridge_vs_naive_std = metrics['summary']['relative_stdev']['ridge_vs_naive']
            
            print(f"\nRidge vs. Naive Forecaster: {ridge_vs_naive:.2f}% ± {ridge_vs_naive_std:.2f}%")
            
            # Count how often ridge beats naive
            relative_df = metrics['relative']
            ridge_wins = (relative_df['ridge_vs_naive'] > 0).sum()
            total_nodes = len(relative_df)
            
            print(f"Ridge outperforms Naive in {ridge_wins} out of {total_nodes} nodes ({ridge_wins/total_nodes*100:.1f}%)")
        
        # 4. Runtime comparison
        print("\nAverage Runtime by Model (seconds):")
        runtime_df = pd.DataFrame({
            'Mean Runtime': metrics['summary']['runtime_means']
        }).sort_values('Mean Runtime')
        
        print(tabulate(runtime_df, headers='keys', floatfmt='.2f', tablefmt='psql'))


def visualize_results(metrics, output_dir, iso=None):
    """Create visualization plots from the metrics."""
    if metrics is None:
        print("No metrics available to visualize")
        return
    
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Handle multi-ISO metrics
    if isinstance(metrics['summary'], dict) and any(key in metrics['summary'] for key in ['ERCOT', 'NYISO', 'OVERALL']):
        # Determine which ISO to visualize
        if iso and iso != 'BOTH' and iso in metrics['summary']:
            isos_to_visualize = [iso]
        else:
            isos_to_visualize = [k for k in metrics['summary'].keys()]
        
        for current_iso in isos_to_visualize:
            # Create ISO-specific output directory
            iso_output = output_path / current_iso.lower()
            os.makedirs(iso_output, exist_ok=True)
            
            # 1. Average revenue by model (bar chart)
            plt.figure(figsize=(10, 6))
            summary = metrics['summary'][current_iso]['revenue_means'].sort_values(ascending=False)
            summary.plot(kind='bar', yerr=metrics['summary'][current_iso]['revenue_stdev'])
            plt.title(f'{current_iso} - Average Revenue by Model')
            plt.ylabel('Revenue ($)')
            plt.xlabel('Model')
            plt.tight_layout()
            plt.savefig(iso_output / 'revenue_by_model.png')
            plt.close()
            
            # 2. Relative performance compared to oracle (bar chart)
            if metrics['summary'][current_iso]['relative_means']:
                plt.figure(figsize=(10, 6))
                rel_data = {}
                
                for model, pct in metrics['summary'][current_iso]['relative_means'].items():
                    if 'vs_oracle' in model:
                        model_name = model.replace('_vs_oracle', '')
                        rel_data[model_name] = pct
                
                rel_series = pd.Series(rel_data).sort_values(ascending=False)
                rel_std = pd.Series({k.replace('_vs_oracle', ''): v 
                                for k, v in metrics['summary'][current_iso]['relative_stdev'].items() 
                                if 'vs_oracle' in k})
                
                rel_series.plot(kind='bar', yerr=rel_std)
                plt.title(f'{current_iso} - Model Performance (% of Optimal)')
                plt.ylabel('Percentage of Optimal (%)')
                plt.xlabel('Model')
                plt.axhline(y=100, color='r', linestyle='--', alpha=0.3)  # Reference line at 100%
                plt.tight_layout()
                plt.savefig(iso_output / 'relative_performance.png')
                plt.close()
            
            # 3. Execution time by model (bar chart)
            plt.figure(figsize=(10, 6))
            metrics['summary'][current_iso]['runtime_means'].sort_values().plot(kind='bar')
            plt.title(f'{current_iso} - Average Runtime by Model')
            plt.ylabel('Time (seconds)')
            plt.xlabel('Model')
            plt.tight_layout()
            plt.savefig(iso_output / 'runtime_by_model.png')
            plt.close()
            
        print(f"Visualizations saved to {output_path}")
        return
    
    # Original single-ISO visualization
    # 1. Average revenue by model (bar chart)
    plt.figure(figsize=(10, 6))
    summary = metrics['summary']['revenue_means'].sort_values(ascending=False)
    summary.plot(kind='bar', yerr=metrics['summary']['revenue_stdev'])
    plt.title('Average Revenue by Model')
    plt.ylabel('Revenue ($)')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(output_path / 'revenue_by_model.png')
    plt.close()
    
    # 2. Relative performance compared to oracle (bar chart)
    if metrics['summary']['relative_means']:
        plt.figure(figsize=(10, 6))
        rel_data = {}
        
        for model, pct in metrics['summary']['relative_means'].items():
            if 'vs_oracle' in model:
                model_name = model.replace('_vs_oracle', '')
                rel_data[model_name] = pct
        
        rel_series = pd.Series(rel_data).sort_values(ascending=False)
        rel_std = pd.Series({k.replace('_vs_oracle', ''): v 
                           for k, v in metrics['summary']['relative_stdev'].items() 
                           if 'vs_oracle' in k})
        
        rel_series.plot(kind='bar', yerr=rel_std)
        plt.title('Model Performance (% of Optimal)')
        plt.ylabel('Percentage of Optimal (%)')
        plt.xlabel('Model')
        plt.axhline(y=100, color='r', linestyle='--', alpha=0.3)  # Reference line at 100%
        plt.tight_layout()
        plt.savefig(output_path / 'relative_performance.png')
        plt.close()
    
    # 3. Execution time by model (bar chart)
    plt.figure(figsize=(10, 6))
    metrics['summary']['runtime_means'].sort_values().plot(kind='bar')
    plt.title('Average Runtime by Model')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(output_path / 'runtime_by_model.png')
    plt.close()
    
    print(f"Visualizations saved to {output_path}")


def create_combined_csv(metrics, output_dir):
    """Create a combined CSV file with all results for further analysis."""
    if metrics is None:
        return
    
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save the detailed results
    metrics['detailed'].to_csv(output_path / 'detailed_results.csv', index=False)
    
    # Save the pivot tables
    metrics['pivot_revenue'].to_csv(output_path / 'revenue_by_node_model.csv')
    metrics['pivot_runtime'].to_csv(output_path / 'runtime_by_node_model.csv')
    
    # Save the relative performance metrics
    if not metrics['relative'].empty:
        metrics['relative'].to_csv(output_path / 'relative_performance.csv')
    
    print(f"CSV files saved to {output_path}")


def run_benchmark_for_iso(iso_name, data_file, output_dir, max_nodes=10, n_jobs=4):
    """Run the benchmark for a specific ISO."""
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return False
    
    # Create ISO-specific output directory
    iso_output_dir = os.path.join(output_dir, iso_name.lower())
    
    # Determine data format based on ISO
    if iso_name.upper() == "ERCOT":
        data_format = "tidy"  # node, timestamp, price format
        print(f"Using ERCOT format (tidy) for {data_file}")
    elif iso_name.upper() == "NYISO":
        data_format = "tidy"  # zone, timestamp, price format
        print(f"Using NYISO format (tidy) for {data_file}")
    else:
        print(f"Unknown ISO: {iso_name}, assuming tidy format")
        data_format = "tidy"
    
    # Run the benchmark using the comprehensive_benchmark script
    cmd = f"""
    source .venv/bin/activate
    python3 comprehensive_benchmark.py \\
        --prices-path {data_file} \\
        --data-format {data_format} \\
        --output-dir {iso_output_dir} \\
        --max-nodes {max_nodes} \\
        --n-jobs {n_jobs}
    """
    
    print(f"Running benchmark for {iso_name}...")
    result = os.system(cmd)
    
    return result == 0


def main():
    parser = argparse.ArgumentParser(
        description="Summarize benchmark results and optionally run benchmarks"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmark_results",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_summary",
        help="Directory to save summary results and visualizations",
    )
    parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Run benchmarks before summarizing",
    )
    parser.add_argument(
        "--iso",
        choices=["ERCOT", "NYISO", "BOTH"],
        default="BOTH",
        help="Which ISO to run benchmarks for",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=10,
        help="Maximum number of nodes to benchmark per ISO",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel jobs for benchmarking",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Also create subdirectories for ISO-specific outputs if needed
    if args.iso in ["ERCOT", "BOTH"]:
        os.makedirs(os.path.join(args.output_dir, "ercot_summary"), exist_ok=True)
    if args.iso in ["NYISO", "BOTH"]:
        os.makedirs(os.path.join(args.output_dir, "nyiso_summary"), exist_ok=True)
    if args.iso == "BOTH":
        os.makedirs(os.path.join(args.output_dir, "combined"), exist_ok=True)
    
    # Run benchmarks if requested
    if args.run_benchmarks:
        if args.iso in ["ERCOT", "BOTH"]:
            ercot_success = run_benchmark_for_iso(
                "ERCOT", 
                "data/ercot/2024_RealTime_SPP.csv",
                args.output_dir,
                args.max_nodes,
                args.n_jobs
            )
            if not ercot_success:
                print("Warning: ERCOT benchmark failed")
        
        if args.iso in ["NYISO", "BOTH"]:
            nyiso_success = run_benchmark_for_iso(
                "NYISO", 
                "data/nyiso/2024_DayAhead_LBMP.csv",
                args.output_dir,
                args.max_nodes,
                args.n_jobs
            )
            if not nyiso_success:
                print("Warning: NYISO benchmark failed")
    
    # Process results
    if args.iso == "BOTH":
        # Load results from both ISOs
        results_df = load_results(args.results_dir, "BOTH")
        if results_df is not None:
            metrics = calculate_metrics(results_df)
            print("\n=== COMBINED BENCHMARK RESULTS ===")
            print_summary(metrics)
            
            # Save combined CSVs
            combined_output_dir = os.path.join(args.output_dir, "combined")
            os.makedirs(combined_output_dir, exist_ok=True)
            create_combined_csv(metrics, combined_output_dir)
            visualize_results(metrics, args.output_dir, "BOTH")
    else:
        # Process for a specific ISO
        if args.iso == "ERCOT":
            # Load and analyze ERCOT results
            ercot_results = load_results(args.results_dir, "ERCOT")
            if ercot_results is not None:
                ercot_metrics = calculate_metrics(ercot_results)
                print("\n=== ERCOT BENCHMARK RESULTS ===")
                print_summary(ercot_metrics, "ERCOT")
                create_combined_csv(ercot_metrics, os.path.join(args.output_dir, "ercot_summary"))
                visualize_results(ercot_metrics, args.output_dir, "ERCOT")
            else:
                print("No ERCOT results found")
        
        elif args.iso == "NYISO":
            # Load and analyze NYISO results
            nyiso_results = load_results(args.results_dir, "NYISO")
            if nyiso_results is not None:
                nyiso_metrics = calculate_metrics(nyiso_results)
                print("\n=== NYISO BENCHMARK RESULTS ===")
                print_summary(nyiso_metrics, "NYISO")
                create_combined_csv(nyiso_metrics, os.path.join(args.output_dir, "nyiso_summary"))
                visualize_results(nyiso_metrics, args.output_dir, "NYISO")
            else:
                print("No NYISO results found")
        
    # If we have results from both ISOs, provide a comparison
    if args.iso == "BOTH" and 'metrics' in locals() and isinstance(metrics['summary'], dict) and 'ERCOT' in metrics['summary'] and 'NYISO' in metrics['summary']:
        print("\n=== ERCOT vs NYISO COMPARISON ===")
        
        # Compare oracle performances
        ercot_oracle = metrics['summary']['ERCOT']['revenue_means'].get('oracle_lp', np.nan)
        nyiso_oracle = metrics['summary']['NYISO']['revenue_means'].get('oracle_lp', np.nan)
        
        if not np.isnan(ercot_oracle) and not np.isnan(nyiso_oracle):
            print(f"Average Oracle Revenue: ERCOT ${ercot_oracle:.2f} vs NYISO ${nyiso_oracle:.2f}")
            print(f"NYISO/ERCOT Revenue Ratio: {nyiso_oracle/ercot_oracle:.2f}x")
        
        # Compare ridge forecaster performance relative to oracle
        ercot_ridge = metrics['summary']['ERCOT']['relative_means'].get('online_mpc_ridge_vs_oracle', np.nan)
        nyiso_ridge = metrics['summary']['NYISO']['relative_means'].get('online_mpc_ridge_vs_oracle', np.nan)
        
        if not np.isnan(ercot_ridge) and not np.isnan(nyiso_ridge):
            print(f"Ridge vs Oracle: ERCOT {ercot_ridge:.2f}% vs NYISO {nyiso_ridge:.2f}%")
        
        # Compare ridge vs naive improvement
        ercot_vs_naive = metrics['summary']['ERCOT']['relative_means'].get('ridge_vs_naive', np.nan)
        nyiso_vs_naive = metrics['summary']['NYISO']['relative_means'].get('ridge_vs_naive', np.nan)
        
        if not np.isnan(ercot_vs_naive) and not np.isnan(nyiso_vs_naive):
            print(f"Ridge improvement over Naive: ERCOT {ercot_vs_naive:.2f}% vs NYISO {nyiso_vs_naive:.2f}%")


if __name__ == "__main__":
    main() 