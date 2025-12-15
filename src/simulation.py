import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from .data_loader import TARGET_QUALIFIED, generate_respondents
from .allocation_policies import random_allocation, priority_greedy_allocation, demographic_aware_allocation

def simulate_qualifications(allocation, category_ids, categories_df, respondents, seed=42):
    """Simulate which respondents qualify for each category"""
    np.random.seed(seed)
    n_resp, n_cats = allocation.shape

    cat_data = categories_df.set_index('category_id')
    incidence = np.array([cat_data.loc[cid, 'incidence_rate'] for cid in category_ids])
    lengths = np.array([cat_data.loc[cid, 'category_length_seconds'] for cid in category_ids])

    qualifications = np.zeros((n_resp, n_cats), dtype=bool)
    times = np.zeros(n_resp)

    for i in range(n_resp):
        for j in range(n_cats):
            if allocation[i, j]:
                if np.random.random() < incidence[j]:
                    qualifications[i, j] = True
                    times[i] += lengths[j]

    qualified_per_cat = qualifications.sum(axis=0)
    return qualified_per_cat, times

def run_incremental_simulation(policy_name, categories, max_respondents=15000,
                                step_size=500, seed=42):
    """
    Run simulation incrementally, tracking progress over time

    Returns:
        DataFrame with columns: n_respondents, pct_categories_met, mean_time
    """
    results = []

    # Generate all respondents upfront
    all_respondents = generate_respondents(max_respondents, seed=seed)

    policies = {
        'random': random_allocation,
        'priority_greedy': priority_greedy_allocation,
        'demographic_aware': demographic_aware_allocation,
    }

    policy_func = policies[policy_name]

    for n in tqdm(range(step_size, max_respondents + 1, step_size), desc=f"Running {policy_name}"):
        respondents = all_respondents.iloc[:n]
        allocation, cat_ids = policy_func(categories, respondents, seed=seed)

        # Simulate qualifications
        qualified_per_cat, times = simulate_qualifications(
            allocation, cat_ids, categories, respondents, seed=seed
        )

        # Calculate metrics
        pct_met = (qualified_per_cat >= TARGET_QUALIFIED).mean() * 100
        mean_time = times.mean()

        results.append({
            'n_respondents': n,
            'pct_categories_met': pct_met,
            'mean_time': mean_time,
            'min_qualified': qualified_per_cat.min(),
        })

    return pd.DataFrame(results)

def run_multi_simulation(policy_name, categories, n_respondents=8000,
                         n_iterations=100, seed_start=42):
    """
    Run multiple simulations and track runtime

    Returns:
        DataFrame with runtime stats
    """
    runtimes = []
    metrics = []

    policies = {
        'random': random_allocation,
        'priority_greedy': priority_greedy_allocation,
        'demographic_aware': demographic_aware_allocation,
    }

    policy_func = policies[policy_name]

    for i in tqdm(range(n_iterations), desc=f"Running {policy_name} ({n_iterations} iterations)"):
        seed = seed_start + i
        respondents = generate_respondents(n_respondents, seed=seed)

        # Time the allocation
        start_time = time.time()
        allocation, cat_ids = policy_func(categories, respondents, seed=seed)
        runtime = time.time() - start_time

        # Simulate qualifications
        qualified_per_cat, times = simulate_qualifications(
            allocation, cat_ids, categories, respondents, seed=seed
        )

        runtimes.append(runtime)
        metrics.append({
            'iteration': i,
            'runtime': runtime,
            'pct_categories_met': (qualified_per_cat >= TARGET_QUALIFIED).mean() * 100,
            'mean_time': times.mean(),
            'min_qualified': qualified_per_cat.min(),
        })

    return pd.DataFrame(metrics)
