"""
Data loading and preprocessing for category data.

Handles CSV loading, seasonality adjustments, and derived metric calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

import config


def load_category_data(
    file_path: str = config.CATEGORY_DATA_FILE,
    seasonality_multipliers: Optional[Dict[int, float]] = None,
    valid_completion_rate: Optional[float] = None
) -> pd.DataFrame:
    """
    Load category data and calculate derived metrics.
    
    Derived metrics:
    - required_exposed: How many people must see this category to hit target (= target / IR)
    - value_density: Efficiency metric for knapsack-style allocation
    - stability_score: IR * Length - high value = predictable category
    
    Args:
        file_path: Path to category data CSV
        seasonality_multipliers: Dict of {category_id: multiplier} for IR adjustments
        valid_completion_rate: Fraction of completions that pass quality checks (0-1).
                               Lower values require more exposures. Default from config.
    """
    if seasonality_multipliers is None:
        seasonality_multipliers = config.SEASONALITY_MULTIPLIERS
    if valid_completion_rate is None:
        valid_completion_rate = config.VALID_COMPLETION_RATE

    df = pd.read_csv(file_path)
    df = df.rename(columns={"category_length_seconds": "survey_length_seconds"})

    # Apply seasonality: effective_IR = base_IR * multiplier
    df["effective_incidence_rate"] = df["incidence_rate"]
    for category_id, multiplier in seasonality_multipliers.items():
        mask = df["category_id"] == category_id
        df.loc[mask, "effective_incidence_rate"] *= multiplier
    df["effective_incidence_rate"] = df["effective_incidence_rate"].clip(0.0, 1.0)

    # Required exposures to hit target (the "demand" for this category)
    # Low IR categories have high demand - these are the bottlenecks
    # Adjust for data quality: if only X% of completions are valid, need more exposures
    adjusted_target = config.TARGET_QUALIFIED_COMPLETIONS / valid_completion_rate
    df["required_exposed"] = np.ceil(adjusted_target / df["effective_incidence_rate"]).astype(int)

    # Value density = demand / cost - used for efficiency-first allocation
    df["value_density"] = df["required_exposed"] / (
        df["effective_incidence_rate"] * df["survey_length_seconds"]
    )
    
    # Stability score - high IR * long survey = predictable, stable category
    df["stability_score"] = df["effective_incidence_rate"] * df["survey_length_seconds"]

    return df


def get_category_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for debugging."""
    print(f"\nCategories: {len(df)}")
    print(f"IR Range: {df['effective_incidence_rate'].min():.1%} - {df['effective_incidence_rate'].max():.1%}")
    print(f"Length Range: {df['survey_length_seconds'].min():.0f}s - {df['survey_length_seconds'].max():.0f}s")
    print(f"Total Required Exposed: {df['required_exposed'].sum():,}")
