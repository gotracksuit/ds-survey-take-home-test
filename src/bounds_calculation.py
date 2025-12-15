import pandas as pd

TARGET_QUALIFIED = 200
TIME_BUDGET = 480

def calculate_bounds(df):
    """Calculate theoretical bounds for the survey allocation problem"""
    # Naive upper bound: Survey each category independently
    naive_upper_bound = (TARGET_QUALIFIED / df['incidence_rate']).sum()

    # Theoretical minimum based on work constraint
    total_work_seconds = (TARGET_QUALIFIED * df['category_length_seconds']).sum()
    work_based_minimum = total_work_seconds / TIME_BUDGET

    # Theoretical minimum based on exposure constraint
    exposure_based_minimum = (TARGET_QUALIFIED / df['incidence_rate']).max()

    # Overall theoretical minimum
    theoretical_minimum = max(work_based_minimum, exposure_based_minimum)

    gap = naive_upper_bound - theoretical_minimum
    ratio = naive_upper_bound / theoretical_minimum

    return {
        'naive_upper_bound': int(naive_upper_bound),
        'theoretical_minimum': int(theoretical_minimum),
        'work_based_minimum': int(work_based_minimum),
        'exposure_based_minimum': int(exposure_based_minimum),
        'gap': int(gap),
        'ratio': round(ratio, 2),
    }

if __name__ == '__main__':
    df = pd.read_csv('data/fake_category_data.csv')
    print(f"Loaded {len(df)} categories\n")

    bounds = calculate_bounds(df)

    print(f"NAIVE UPPER BOUND: {bounds['naive_upper_bound']:,} respondents")
    print(f"  (Survey each category independently)\n")

    print(f"THEORETICAL MINIMUM: {bounds['theoretical_minimum']:,} respondents")
    print(f"  - Work constraint: {bounds['work_based_minimum']:,} respondents")
    print(f"  - Exposure constraint: {bounds['exposure_based_minimum']:,} respondents\n")

    print(f"Gap: {bounds['gap']:,} respondents ({bounds['ratio']}x difference)")
