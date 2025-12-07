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

    rng = np.random.default_rng(random_seed)
    num_categories, num_cells = allocation.shape

    qualified_results = np.zeros((n_simulations, num_categories))
    mean_times = np.zeros(n_simulations)
    demo_exposures = np.zeros((n_simulations, num_cells))

    incidence = categories["incidence_rate"].values
    lengths = categories["category_length_seconds"].values

    for sim in range(n_simulations):
        total_time = 0.0
        # simulate each respondent as Bernoulli trials
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

    # --- Aggregated results ---
    qualified_mean = qualified_results.mean(axis=0)
    mean_time_avg = mean_times.mean()
    demo_mean = demo_exposures.mean(axis=0)

    #
    print("\n--- Monte Carlo Validation Results ---")
    print("\nQualified Completes per Category (Mean over simulations):")
    for i, cat in enumerate(categories["category_name"]):
        print(f"{cat}: {qualified_mean[i]:.1f} (Target: {target_qualified})")       
    print(f"\nMean Interview Time: {mean_time_avg:.2f}s (Max Allowed: {max_mean_time}s)")
    print("\nDemographic Exposure Shares (Mean over simulations):")
    for j, cell in enumerate(cells):
        print(f"{cell}: {demo_mean[j]:.4f} (Target: {cell_shares[j]:.4f})") 
    