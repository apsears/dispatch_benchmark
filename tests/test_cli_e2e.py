"""
End-to-end integration test for the CLI functionality.
"""

import os
import json
import subprocess
import sys
import pathlib
import pytest

CLI = [
    sys.executable,  # path to the same python
    "-m",
    "virtual_energy.cli.ercot_cli",
    "backtest",
]


def test_backtest_single_node(tmp_path):
    """Test that the CLI can run a backtest on a single node."""

    # Get the path to the sample data
    sample_csv = pathlib.Path("tests/data/ercot_1day_sample.csv").resolve()

    # Ensure the sample data exists
    assert sample_csv.exists(), f"Sample data not found at {sample_csv}"

    # Create output directory
    output_dir = tmp_path

    # CLI command
    cmd = [
        sys.executable,  # Use the same Python interpreter as the test
        "-m",
        "virtual_energy.cli.ercot_cli",
        "backtest",
        "--prices",
        str(sample_csv),
        "--nodes",
        "HB_HOUSTON",
        "--output-dir",
        str(output_dir),
    ]

    # Set up environment with correct PYTHONPATH
    env = os.environ.copy()
    src_path = pathlib.Path(__file__).parent.parent / "src"
    env["PYTHONPATH"] = str(src_path)

    # Run the command
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )

    # Check that the output contains success message (looking for output that indicates results were saved)
    assert "Results saved to" in result.stdout

    # Check that the output files were created
    output_files = list(output_dir.glob("*.json"))
    assert len(output_files) > 0, "No JSON output files were created"

    # result file should exist
    result_file = output_dir / "HB_HOUSTON_results.json"
    assert result_file.exists()

    # validate essential fields
    data = json.loads(result_file.read_text())

    # Data is a list of model results
    assert isinstance(data, list)
    assert len(data) > 0

    # Check that each result has at least a model key and revenue
    for model_result in data:
        assert "model" in model_result
        assert "revenue" in model_result
        assert model_result["revenue"] != 0.0

    # Check we have results for the expected node (HB_HOUSTON)
    assert result_file.name.startswith("HB_HOUSTON")
