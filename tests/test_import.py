#!/usr/bin/env python3
"""
Test script to verify that imports from the new package structure work correctly.
"""

import pytest
import importlib
import sys

# List of modules to test
MODULES_TO_TEST = [
    # IO modules
    "dispatch_benchmark.io.ercot",
    "dispatch_benchmark.io.nyiso",
    # Forecasters
    "dispatch_benchmark.forecasters.ridge",
    "dispatch_benchmark.forecasters.quartile",
    # Optimisers
    "dispatch_benchmark.optimisers.battery_config",
    "dispatch_benchmark.optimisers.oracle_lp",
    "dispatch_benchmark.optimisers.online_mpc",
    # Utils
    "dispatch_benchmark.utils.time_series",
]


def test_import():
    """Test that all essential modules can be imported."""
    failed_modules = []

    for module_path in MODULES_TO_TEST:
        try:
            module = importlib.import_module(module_path)
            # Successfully imported
        except Exception as e:
            failed_modules.append(f"{module_path}: {str(e)}")

    # If any modules failed to import, report them and fail the test
    if failed_modules:
        pytest.fail(
            "Failed to import the following modules:\n"
            + "\n".join(failed_modules)
        )


if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")

    success_count = 0

    for module in MODULES_TO_TEST:
        try:
            importlib.import_module(module)
            print(f"✓ Successfully imported {module}")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to import {module}: {e}")

    print(
        f"\nSummary: {success_count}/{len(MODULES_TO_TEST)} modules imported successfully"
    )
    sys.exit(0 if success_count == len(MODULES_TO_TEST) else 1)
