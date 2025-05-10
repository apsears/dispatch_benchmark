"""
Tests for the CLI functionality with a single node.
"""

import os
import json
import subprocess
import sys
import pytest
import tempfile
from pathlib import Path


def test_cli_single_node(ercot_1day_csv):
    """Test that the CLI can run a benchmark on a single node."""
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Run the CLI command
        cmd = [
            sys.executable,  # Use the same Python interpreter as the test
            "-m",
            "virtual_energy.cli.ercot_cli",
            "backtest",
            "--nodes",
            "HB_HOUSTON",
            "--prices",
            ercot_1day_csv,
            "--output-dir",
            str(output_dir),
        ]

        # Set up environment with correct PYTHONPATH
        env = os.environ.copy()
        src_path = Path(__file__).parent.parent / "src"
        env["PYTHONPATH"] = str(src_path)

        # Using a basic setup with the ERCOT CLI
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )

        # Check that the command ran successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"

        # Check that the output contains the expected results
        assert "Results saved to" in result.stdout

        # Check that the output directory contains result files
        result_files = list(output_dir.glob("*.json"))
        assert len(result_files) > 0, "No JSON result files were found"

        # For debugging
        print(f"Output directory contents: {list(output_dir.glob('*'))}")
        print(f"Command output: {result.stdout}")

        # Check the revenue results
        node_result_file = output_dir / "HB_HOUSTON_results.json"
        if node_result_file.exists():
            with open(node_result_file, "r") as f:
                model_results = json.load(f)

            # The file contains a list of model results
            assert isinstance(
                model_results, list
            ), "Expected results to be a list of model results"
            assert len(model_results) > 0, "No model results found in the output file"

            # Check that each model result has a revenue field
            for model_result in model_results:
                # Check for expected keys in the results
                assert "model" in model_result, "No model name in results"
                assert "revenue" in model_result, "No revenue data in results"
                assert isinstance(
                    model_result["revenue"], (int, float)
                ), "Revenue value is not a number"
