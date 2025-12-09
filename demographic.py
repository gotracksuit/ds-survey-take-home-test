# demographics.py

import pandas as pd
import itertools


def load_demographics(genders, ages, regions):
    """
    Creates demographic cells with population shares.

    Returns:
        cells_df: DataFrame with columns
            - cell_id
            - gender
            - age
            - region
            - population_share
    """
    rows = []

    for (gender, gender_weight), (age, age_weight), (region, region_weight) in itertools.product(
        genders.items(), ages.items(), regions.items()
    ):
        rows.append({
            "gender": gender,
            "age": age,
            "region": region,
            "population_share": gender_weight * age_weight * region_weight
        })

    cells_df = pd.DataFrame(rows)
    cells_df["cell_id"] = cells_df.index.astype(str)

    # Normalize just to be safe
    cells_df["population_share"] /= cells_df["population_share"].sum()

    return cells_df