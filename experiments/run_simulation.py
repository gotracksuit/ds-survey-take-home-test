#!/usr/bin/env python
"""
Run survey allocation simulation from command line

Usage:
    poetry run python experiments/run_simulation.py --policy priority_greedy --n 8000
    poetry run python experiments/run_simulation.py --policy demographic_aware --n 10000 --iterations 5
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from tqdm import tqdm
from src.data_loader import load_categories, generate_respondents
from src.allocation_policies import random_allocation, priority_greedy_allocation, demographic_aware_allocation
from src.simulation import simulate_qualifications

POLICIES = {
    'random': random_allocation,
    'priority_greedy': priority_greedy_allocation,
    'demographic_aware': demographic_aware_allocation,
}

def main():
    parser = argparse.ArgumentParser(description='Run survey allocation simulation')
    parser.add_argument('--policy', choices=list(POLICIES.keys()),
                       default='demographic_aware',
                       help='Allocation policy to use')
    parser.add_argument('--n', type=int, default=8000,
                       help='Number of respondents')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of times to run (for averaging)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    print("=" * 70)
    print(f"SURVEY ALLOCATION SIMULATION")
    print("=" * 70)
    print(f"Policy: {args.policy}")
    print(f"Respondents: {args.n:,}")
    print(f"Iterations: {args.iterations}")
    print(f"Seed: {args.seed}")
    print()

    categories = load_categories(enriched=True)
    policy_func = POLICIES[args.policy]

    all_results = []

    for i in tqdm(range(args.iterations), desc="Running simulations", disable=args.iterations == 1):
        seed = args.seed + i
        respondents = generate_respondents(args.n, seed=seed)

        # Run allocation
        alloc, cat_ids = policy_func(categories, respondents, seed=seed)

        # Simulate qualifications
        qualified, times = simulate_qualifications(alloc, cat_ids, categories, respondents, seed=seed)

        all_results.append({
            'min_qualified': qualified.min(),
            'max_qualified': qualified.max(),
            'mean_qualified': qualified.mean(),
            'mean_time': times.mean(),
            'max_time': times.max(),
            'pct_met': (qualified >= 200).mean() * 100,
            'categories_below': (qualified < 200).sum(),
        })

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if args.iterations == 1:
        r = all_results[0]
        print(f"Min Qualified:        {r['min_qualified']:.0f}")
        print(f"Max Qualified:        {r['max_qualified']:.0f}")
        print(f"Mean Qualified:       {r['mean_qualified']:.1f}")
        print(f"Mean Time:            {r['mean_time']:.1f}s")
        print(f"Max Time:             {r['max_time']:.1f}s")
        print(f"% Categories ≥ 200:   {r['pct_met']:.1f}%")
        print(f"Categories < 200:     {r['categories_below']}")
    else:
        # Average across iterations
        import numpy as np
        print("Averaged across {} iterations:".format(args.iterations))
        print(f"Min Qualified:        {np.mean([r['min_qualified'] for r in all_results]):.1f} ± {np.std([r['min_qualified'] for r in all_results]):.1f}")
        print(f"Mean Time:            {np.mean([r['mean_time'] for r in all_results]):.1f} ± {np.std([r['mean_time'] for r in all_results]):.1f}s")
        print(f"% Categories ≥ 200:   {np.mean([r['pct_met'] for r in all_results]):.1f}%")

    print()

if __name__ == '__main__':
    main()
