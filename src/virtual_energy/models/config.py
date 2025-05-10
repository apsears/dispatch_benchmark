#!/usr/bin/env python3
"""
Battery Configuration Model for ERCOT

This module defines the configuration model for battery storage optimization used
across the ERCOT optimization scripts.
"""

from pydantic import BaseModel, Field, validator


class BatteryConfig(BaseModel):
    """Configuration parameters for battery storage optimization."""

    delta_t: float = Field(
        default=0.25,
        description="Time interval in hours (e.g., 0.25 for 15-min intervals)",
    )

    eta_chg: float = Field(
        default=0.95, gt=0, le=1.0, description="Battery charging efficiency (0-1)"
    )

    p_max_mw: float = Field(
        default=25, gt=0, description="Maximum charge/discharge power in MW"
    )

    e_max_mwh: float = Field(
        default=200, gt=0, description="Maximum energy storage capacity in MWh"
    )

    e_cycle_24h: float = Field(
        default=200, gt=0, description="Maximum energy discharge per 24h"
    )

    @validator("eta_chg")
    def efficiency_must_be_reasonable(cls, v):
        if v > 1.0:
            raise ValueError("Efficiency cannot exceed 100%")
        return v

    def as_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return self.dict()
