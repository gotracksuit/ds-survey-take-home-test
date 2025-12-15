"""
Experiment: Find minimum number of respondents needed for each allocation policy

This is the key performance measure: minimize total respondents while meeting all constraints.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.data_loader import load_categories, generate_respondents, TARGET_QUALIFIED
from src.allocation_policies import (
    random_allocation,
    priority_greedy_allocation,
    demographic_aware_allocation
)
from src.simulation import simulate_qualifications

def test_policy_at_n(policy_func, policy_name, categories, n, seed=42, num_trials=5):
    """
    Test if a policy meets all constraints at a given N
    Run multiple trials with different seeds to account for variance

    Returns:
        success_rate: Fraction of trials that met all constraints
        mean_qualified: Mean of minimum qualified across trials
        mean_time: Mean survey time across trials
    """
    successes = 0
    min_qualified_list = []
    mean_time_list = []

    for trial in range(num_trials):
        trial_seed = seed + trial
        respondents = generate_respondents(n, seed=trial_seed)
        alloc, cat_ids = policy_func(categories, respondents, seed=trial_seed)
        qualified, times = simulate_qualifications(alloc, cat_ids, categories, respondents, seed=trial_seed)

        min_qualified = qualified.min()
        mean_time = times.mean()
        pct_over_budget = (times > 480).mean() * 100

        min_qualified_list.append(min_qualified)
        mean_time_list.append(mean_time)

        # Success = all categories get 200+ AND mean time <= 480
        if min_qualified >= TARGET_QUALIFIED and mean_time <= 480:
            successes += 1

    success_rate = successes / num_trials
    return success_rate, np.mean(min_qualified_list), np.mean(mean_time_list)

def find_minimum_n(policy_func, policy_name, categories, seed=42, num_trials=5):
    """
    Binary search to find minimum N that achieves 100% success rate

    Returns:
        min_n: Minimum number of respondents needed
        stats: Dictionary with performance statistics
    """
    print(f"\n{'='*70}")
    print(f"Finding minimum N for: {policy_name}")
    print(f"{'='*70}")

    # Binary search bounds
    # We know from previous analysis that ~3,500-4,000 is around the range
    n_min = 2000
    n_max = 15000

    best_n = None

    while n_min <= n_max:
        n_mid = (n_min + n_max) // 2

        print(f"\nTesting N = {n_mid:,}...", end=" ", flush=True)
        success_rate, mean_min_qual, mean_time = test_policy_at_n(
            policy_func, policy_name, categories, n_mid, seed, num_trials
        )

        print(f"Success: {success_rate*100:.0f}%, Min Qualified: {mean_min_qual:.0f}, Mean Time: {mean_time:.1f}s")

        if success_rate == 1.0:
            # All trials succeeded, try smaller N
            best_n = n_mid
            n_max = n_mid - 100
        else:
            # Some trials failed, need larger N
            n_min = n_mid + 100

    if best_n is None:
        print(f"❌ Could not find valid N in range [2000, 15000]")
        return None, {}

    # Run final verification at best_n with more trials
    print(f"\n✅ Minimum N found: {best_n:,}")
    print(f"Verifying with {num_trials*2} trials...", end=" ", flush=True)

    success_rate, mean_min_qual, mean_time = test_policy_at_n(
        policy_func, policy_name, categories, best_n, seed, num_trials*2
    )

    print(f"Success: {success_rate*100:.0f}%")

    stats = {
        'min_n': best_n,
        'success_rate': success_rate,
        'mean_min_qualified': mean_min_qual,
        'mean_time': mean_time
    }

    return best_n, stats

def main():
    """Run the experiment"""

    print("\n" + "="*70)
    print("EXPERIMENT: Finding Minimum Respondents for Each Policy")
    print("="*70)
    print("\nObjective: Minimize total respondents while ensuring:")
    print("  - Each category gets ≥200 qualified respondents")
    print("  - Mean survey time ≤480 seconds")
    print("  - Success rate = 100% across multiple trials")
    print()

    # Load data
    categories = load_categories(enriched=True)
    print(f"Loaded {len(categories)} categories")

    # Define policies to test
    policies = {
        'Random': random_allocation,
        'Priority Greedy': priority_greedy_allocation,
        'Demographic Aware': demographic_aware_allocation
    }

    # Run experiment for each policy
    results = {}

    for policy_name, policy_func in policies.items():
        min_n, stats = find_minimum_n(policy_func, policy_name, categories, seed=42, num_trials=5)
        results[policy_name] = {'min_n': min_n, **stats}

    # Print final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('min_n')

    print("\n" + results_df.to_string())

    # Determine winner
    print("\n" + "="*70)
    winner = results_df['min_n'].idxmin()
    winner_n = results_df.loc[winner, 'min_n']

    print(f"🏆 WINNER: {winner}")
    print(f"   Minimum Respondents: {winner_n:,}")
    print(f"   Mean Time: {results_df.loc[winner, 'mean_time']:.1f}s")
    print(f"   Mean Min Qualified: {results_df.loc[winner, 'mean_min_qualified']:.0f}")
    print("="*70)

    # Show efficiency gains
    print("\nEfficiency vs Random:")
    random_n = results_df.loc['Random', 'min_n']
    for policy in results_df.index:
        if policy != 'Random':
            policy_n = results_df.loc[policy, 'min_n']
            reduction = random_n - policy_n
            pct_reduction = (reduction / random_n) * 100
            print(f"  {policy}: {reduction:,} fewer respondents ({pct_reduction:.1f}% reduction)")

if __name__ == "__main__":
    main()
