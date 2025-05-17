"""
Smoke test for oracle_lp to ensure state of charge remains within specified bounds.
"""

from dispatch_benchmark.optimisers.oracle_lp import create_and_solve_model


def test_soc_invariant(random_prices, battery_config):
    """
    Test that state of charge (SoC) from oracle_lp stays within valid bounds.
    Using 96 intervals of random prices, test that:
    - SoC never goes below 0
    - SoC never exceeds battery capacity (e_max_mwh)
    """
    # Create a random price profile for testing
    df = random_prices.copy()

    # Run the oracle_lp optimizer
    _, dispatch = create_and_solve_model(df, battery_config)

    # Check that SoC is within bounds
    assert (
        dispatch["SoC_MWh"].min() >= 0
    ), f"SoC goes below 0: {dispatch['SoC_MWh'].min()}"
    assert (
        dispatch["SoC_MWh"].max() <= battery_config.e_max_mwh
    ), f"SoC exceeds max capacity of {battery_config.e_max_mwh}: {dispatch['SoC_MWh'].max()}"

    # Additional check: SoC should change based on charge/discharge
    # Calculate SoC differences between consecutive timesteps
    soc_diff = dispatch["SoC_MWh"].diff().dropna()

    # Check if we have at least some non-zero SoC differences
    # (i.e., the battery is actually charging/discharging)
    assert (
        soc_diff != 0
    ).any(), "SoC doesn't change - battery never charges or discharges"

    # Verify charge and discharge power don't exceed maximum power
    assert (
        dispatch["p_discharge_MW"].max() <= battery_config.p_max_mw
    ), f"Discharge power exceeds max: {dispatch['p_discharge_MW'].max()} > {battery_config.p_max_mw}"
    assert (
        dispatch["p_charge_MW"].max() <= battery_config.p_max_mw
    ), f"Charge power exceeds max: {dispatch['p_charge_MW'].max()} > {battery_config.p_max_mw}"
