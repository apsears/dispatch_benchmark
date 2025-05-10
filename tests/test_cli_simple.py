"""
Basic tests for the CLI functionality.
"""

import subprocess
import os
import sys
import pytest
from pathlib import Path


def test_cli_help():
    """Test that the CLI can display help information."""
    cmd = [
        sys.executable,  # Use the same Python interpreter as the test
        "-m",
        "virtual_energy.cli.ercot_cli",
    ]

    # Set up environment with correct PYTHONPATH
    env = os.environ.copy()
    src_path = Path(__file__).parent.parent / "src"
    env["PYTHONPATH"] = str(src_path)

    # Run the command with the helper that sets PYTHONPATH
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Check that the command completed successfully
    assert result.returncode == 0

    # Check that the output contains expected help text
    assert "ERCOT data tools" in result.stdout
