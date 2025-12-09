# monte_carlo_validation_fixed.py

import numpy as np
import pandas as pd

def lp_monte_carlo_validate(
    categories,
    allocation,
    cells,
    cell_shares,
    target_qualified,
    max_mean_time,
    n_simulations=1000,
    random_seed=42,
):
    """
    Monte Carlo validation of LP allocation under stochastic qualification.

    Parameters
    ----------
    categories : pd.DataFrame
        Must include incidence_rate and category_length_seconds
    allocation : np.ndarray
        Shape (num_categories, num_demographics)
    cells : list
        Demographic cell names
    cell_shares : np.ndarray
        National population shares for each cell
    target_qualified : int
        Target qualified completes per category
    max_mean_time : float
        Maximum allowed mean interview time (seconds)
    n_simulations : int
        Number of Monte Carlo runs
    random_seed : int
        RNG seed
    """
    # monte_carlo_validation_summary.py

    rng = np.random.default_rng(random_seed)
    num_categories, num_cells = allocation.shape

    qualified_results = np.zeros((n_simulations, num_categories))
    mean_times = np.zeros(n_simulations)
    demo_exposures = np.zeros((n_simulations, num_cells))

    incidence = categories["incidence_rate"].values
    lengths = categories["category_length_seconds"].values

    for sim in range(n_simulations):
        total_time = 0.0
        for i in range(num_categories):
            for j in range(num_cells):
                n_exposed = int(allocation[i, j])
                if n_exposed == 0:
                    continue
                qualifies = rng.binomial(1, incidence[i], size=n_exposed)
                n_qualified = qualifies.sum()
                qualified_results[sim, i] += n_qualified
                total_time += n_qualified * lengths[i]
        total_respondents = allocation.sum()
        mean_times[sim] = total_time / total_respondents
        demo_exposures[sim] = allocation.sum(axis=0) / total_respondents

    # --- Summary statistics ---
    qualified_mean = qualified_results.mean(axis=0)
    qualified_median = np.median(qualified_results, axis=0)
    qualified_5th = np.percentile(qualified_results, 5, axis=0)
    qualified_95th = np.percentile(qualified_results, 95, axis=0)
    failure_prob = (qualified_results < target_qualified).mean(axis=0) * 100

    mean_time_avg = mean_times.mean()
    demo_mean = demo_exposures.mean(axis=0)

    # --- Print report ---
    print("\n--- Monte Carlo Validation Summary ---")
    print(f"\nOverall mean survey time: {mean_time_avg:.2f}s (Max allowed: {max_mean_time}s)\n")

    print("Qualified Completes per Category (summary):")
    for i, cat in enumerate(categories["category_name"]):
        print(f"{cat}")
        print(f"  Mean qualified   : {qualified_mean[i]:.1f}")
        print(f"  Median           : {qualified_median[i]:.1f}")
        print(f"  5th percentile   : {qualified_5th[i]:.1f}")
        print(f"  95th percentile  : {qualified_95th[i]:.1f}")
        print(f"  Failure prob (%) : {failure_prob[i]:.1f}\n")

    print("Demographic Exposure Shares (mean over simulations):")
    for j, cell in enumerate(cells):
        print(f"  {cell}: {demo_mean[j]:.4f} (Target: {cell_shares[j]:.4f})")

    # --- Optional: Aggregate summary ---
    total_failures = (failure_prob > 5).sum()
    print(f"\nNumber of categories with >5% failure probability: {total_failures} / {num_categories}")

    return {
        "qualified_mean": qualified_mean,
        "qualified_median": qualified_median,
        "qualified_5th": qualified_5th,
        "qualified_95th": qualified_95th,
        "failure_prob": failure_prob,
        "mean_time_avg": mean_time_avg,
        "demo_mean": demo_mean,
    }
