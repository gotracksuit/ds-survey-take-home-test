import numpy as np
import pandas as pd
from .data_loader import TARGET_QUALIFIED, TIME_BUDGET

def random_allocation(categories, respondents, seed=42, safety_margin=0.75):
    """
    Random: Randomly assign categories to each respondent
    respecting time budget constraint

    Args:
        safety_margin: Fraction of TIME_BUDGET to use (accounts for variance in actual times)
    """
    np.random.seed(seed)
    n_resp = len(respondents)
    n_cats = len(categories)

    cats = categories.copy().reset_index(drop=True)
    cats['expected_time'] = cats['category_length_seconds'] * cats['incidence_rate']
    expected_times = cats['expected_time'].values

    effective_budget = TIME_BUDGET * safety_margin

    allocation = np.zeros((n_resp, n_cats), dtype=bool)

    for i in range(n_resp):
        # Randomly shuffle categories for each respondent
        cat_order = np.random.permutation(n_cats)
        time_used = 0

        for j in cat_order:
            if time_used + expected_times[j] <= effective_budget:
                allocation[i, j] = True
                time_used += expected_times[j]

    return allocation, cats['category_id'].values

def priority_greedy_allocation(categories, respondents, seed=42, safety_margin=0.75):
    """
    Priority Greedy: Focus on hardest categories first
    Hardest = lowest incidence rate (need most exposures)
    Fill one category completely before moving to next

    Args:
        safety_margin: Fraction of TIME_BUDGET to use (accounts for variance in actual times)
    """
    np.random.seed(seed)
    n_resp = len(respondents)
    n_cats = len(categories)

    cats = categories.copy().reset_index(drop=True)
    cats['expected_time'] = cats['category_length_seconds'] * cats['incidence_rate']
    cats['required_exposures'] = np.ceil(TARGET_QUALIFIED / cats['incidence_rate'] * 1.15).astype(int)

    # Sort by hardest first (lowest incidence = most exposures needed)
    cats = cats.sort_values('incidence_rate').reset_index(drop=True)

    effective_budget = TIME_BUDGET * safety_margin

    allocation = np.zeros((n_resp, n_cats), dtype=bool)
    expected_times = cats['expected_time'].values
    required_exp = cats['required_exposures'].values
    exposures = np.zeros(n_cats, dtype=int)
    respondent_times = np.zeros(n_resp)

    # Shuffle respondents for demographic balance
    respondent_order = np.random.permutation(n_resp)

    # Fill categories one at a time by priority
    for j in range(n_cats):
        target = required_exp[j]

        # Find respondents who can fit this category
        for i in respondent_order:
            if exposures[j] >= target:
                break

            if respondent_times[i] + expected_times[j] <= effective_budget:
                allocation[i, j] = True
                respondent_times[i] += expected_times[j]
                exposures[j] += 1

    # Second pass: fill remaining time for all respondents
    for i in range(n_resp):
        for j in range(n_cats):
            if not allocation[i, j] and respondent_times[i] + expected_times[j] <= effective_budget:
                allocation[i, j] = True
                respondent_times[i] += expected_times[j]

    return allocation, cats['category_id'].values

def demographic_aware_allocation(categories, respondents, seed=42, safety_margin=0.75):
    """
    Demographic Aware: Match categories to respondents based on targeting
    Only assign categories to demographically appropriate respondents
    Prioritizes hardest categories first (lowest incidence)

    Args:
        safety_margin: Fraction of TIME_BUDGET to use (accounts for variance in actual times)
    """
    np.random.seed(seed)
    n_resp = len(respondents)
    n_cats = len(categories)

    cats = categories.copy().reset_index(drop=True)
    cats['expected_time'] = cats['category_length_seconds'] * cats['incidence_rate']
    cats['required_exposures'] = np.ceil(TARGET_QUALIFIED / cats['incidence_rate'] * 1.15).astype(int)

    # Sort by hardest first
    cats = cats.sort_values('incidence_rate').reset_index(drop=True)

    effective_budget = TIME_BUDGET * safety_margin

    allocation = np.zeros((n_resp, n_cats), dtype=bool)
    expected_times = cats['expected_time'].values
    required_exp = cats['required_exposures'].values
    exposures = np.zeros(n_cats, dtype=int)
    respondent_times = np.zeros(n_resp)

    # Shuffle respondents
    respondent_order = np.random.permutation(n_resp)

    # Precompute demographic data as arrays for fast lookups
    resp_gender = respondents['gender'].str.lower().values
    resp_age = respondents['age_group'].values

    cat_gender = cats['gender_flag'].fillna('na').str.lower().values
    cat_age = cats['age_flag'].fillna('na').values

    # Helper function to check if respondent matches category targeting
    def matches_targeting(resp_idx, cat_idx):
        # Check gender match
        if cat_gender[cat_idx] != 'na':
            if resp_gender[resp_idx] != cat_gender[cat_idx]:
                return False

        # Check age match
        if cat_age[cat_idx] != 'na':
            if resp_age[resp_idx] != cat_age[cat_idx]:
                return False

        return True

    # First pass: Fill categories to target, respecting demographics
    for j in range(n_cats):
        target = required_exp[j]

        for i in respondent_order:
            if exposures[j] >= target:
                break

            if matches_targeting(i, j):
                if respondent_times[i] + expected_times[j] <= effective_budget:
                    allocation[i, j] = True
                    respondent_times[i] += expected_times[j]
                    exposures[j] += 1

    # Second pass: fill remaining time
    for i in range(n_resp):
        for j in range(n_cats):
            if not allocation[i, j] and matches_targeting(i, j):
                if respondent_times[i] + expected_times[j] <= effective_budget:
                    allocation[i, j] = True
                    respondent_times[i] += expected_times[j]

    return allocation, cats['category_id'].values
