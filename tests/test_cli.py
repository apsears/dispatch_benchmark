"""
Tests for the CLI functionality.
"""

import os
import subprocess
import sys
import tempfile
import pytest
from pathlib import Path


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory(prefix="test_output") as temp_dir:
        yield Path(temp_dir)


def test_cli_single_node(ercot_1day_csv, temp_output_dir):
    """Test that the CLI can run a benchmark on a single node."""
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
        str(temp_output_dir),
    ]

    # Set up environment with correct PYTHONPATH
    env = os.environ.copy()
    src_path = Path(__file__).parent.parent / "src"
    env["PYTHONPATH"] = str(src_path)

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Check that the command completed successfully
    assert result.returncode == 0, f"Command failed with output: {result.stderr}"

    # Check that the output directory contains the expected files
    result_dir = temp_output_dir
    print(f"Result directory contents: {list(result_dir.glob('*'))}")

    # The results directory should have at least one JSON file
    json_files = list(result_dir.glob("*.json"))
    assert len(json_files) > 0, "No JSON files were generated"

    # Check that the JSON files contain the expected data
    # Here we just check they're non-empty and have the right extension
    for json_file in json_files:
        assert json_file.stat().st_size > 0, f"JSON file {json_file} is empty"
