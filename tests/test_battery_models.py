"""
Test battery dispatch models to ensure they follow physical constraints.
"""

import pandas as pd
import numpy as np
from virtual_energy.optimisers.oracle_lp import create_and_solve_model

def test_soc_invariant(random_prices, battery_config):
    """Test that the battery SOC never goes below 0 or above battery capacity."""
    # Run oracle LP on random prices
    _, dispatch = create_and_solve_model(random_prices, battery_config)  # No longer returns solver_status
    
    # Get the SOC values
    soc = dispatch["SoC_MWh"]  # Update to match the column name in the returned dataframe
    
    # Check that SOC never goes below 0
    assert soc.min() >= 0, f"SOC went negative: minimum value {soc.min()}"
    
    # Check that SOC never exceeds battery capacity
    assert soc.max() <= battery_config.e_max_mwh, f"SOC exceeded battery capacity: maximum value {soc.max()}"
    
    # Check that power limits are respected
    # The column names have changed from P_MW to p_discharge_MW and p_charge_MW
    assert dispatch["p_discharge_MW"].max() <= battery_config.p_max_mw, "Discharging power exceeded limits"
    assert dispatch["p_charge_MW"].max() <= battery_config.p_max_mw, "Charging power exceeded limits"
    
    # Verify the total energy deployed is calculated correctly
    if "MWhDeployed" in dispatch.columns:
        # Calculate net power (discharge - charge)
        net_power = dispatch["p_discharge_MW"] - dispatch["p_charge_MW"]
        # Convert to energy using delta_t
        expected_mwh = net_power * battery_config.delta_t
        # Compare with the reported MWhDeployed
        mwh_diff = (dispatch["MWhDeployed"] - expected_mwh).abs()
        assert mwh_diff.max() < 1e-6, f"MWh deployed calculation incorrect, max diff: {mwh_diff.max()}" 