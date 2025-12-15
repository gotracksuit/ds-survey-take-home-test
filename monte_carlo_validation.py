import numpy as np
import pandas as pd

def lp_monte_carlo_validate(
    categories: pd.DataFrame,
    allocation: np.ndarray,
    cells: list,
    cell_shares: np.ndarray,
    target_qualified: int,
    max_mean_time: float,
    n_simulations: int = 1000,
    demographic_tolerance: float = 0.02,
    random_seed: int = 42,
):
    """
    Monte Carlo validation of survey allocation under stochastic qualification.

    This simulation evaluates:
    - Probability of meeting qualified targets per category
    - Distribution of achieved qualified completes
    - Mean survey time per respondent
    - Demographic exposure deviation with tolerance-based risk

    Parameters
    ----------
    categories : pd.DataFrame
        Must include:
        - incidence_rate
        - category_length_seconds
        - category_name
    allocation : np.ndarray
        Expected respondent allocation (categories × demographic cells)
    cells : list
        Demographic cell names
    cell_shares : np.ndarray
        Target national demographic proportions
    target_qualified : int
        Monthly qualified target per category
    max_mean_time : float
        Maximum allowed mean interview time (seconds)
    n_simulations : int
        Number of Monte Carlo runs
    demographic_tolerance : float
        Allowed absolute deviation from demographic targets (e.g., 0.02 = ±2%)
    random_seed : int
        RNG seed
    """

    rng = np.random.default_rng(random_seed)

    num_categories, num_cells = allocation.shape

    # --- Storage ---
    qualified_results = np.zeros((n_simulations, num_categories))
    mean_times = np.zeros(n_simulations)
    demo_exposures = np.zeros((n_simulations, num_cells))
    demo_violation = np.zeros((n_simulations, num_cells), dtype=bool)

    incidence = categories["incidence_rate"].values
    lengths = categories["category_length_seconds"].values
    total_respondents = allocation.sum()

    # --- Monte Carlo simulation ---
    for sim in range(n_simulations):
        total_time = 0.0

        for i in range(num_categories):
            for j in range(num_cells):
                n_exposed = int(allocation[i, j])
                if n_exposed == 0:
                    continue

                # Bernoulli trials for qualification
                qualifies = rng.binomial(
                    n=1,
                    p=incidence[i],
                    size=n_exposed
                )
                n_qualified = qualifies.sum()

                qualified_results[sim, i] += n_qualified
                total_time += n_qualified * lengths[i]

        mean_times[sim] = total_time / total_respondents
        demo_exposures[sim] = allocation.sum(axis=0) / total_respondents

        demo_violation[sim] = (
            np.abs(demo_exposures[sim] - cell_shares)
            > demographic_tolerance
        )

    # --- Qualified summary statistics ---
    qualified_mean = qualified_results.mean(axis=0)
    qualified_median = np.median(qualified_results, axis=0)
    qualified_5th = np.percentile(qualified_results, 5, axis=0)
    qualified_95th = np.percentile(qualified_results, 95, axis=0)
    failure_prob = (qualified_results < target_qualified).mean(axis=0) * 100

    # --- Time & demographics ---
    mean_time_avg = mean_times.mean()
    demo_mean = demo_exposures.mean(axis=0)
    demo_failure_prob = demo_violation.mean(axis=0) * 100
    overall_demo_risk = demo_violation.any(axis=1).mean() * 100

    # --- Print report ---
    print("\n--- Monte Carlo Validation Summary ---\n")

    print(
        f"Mean survey time: {mean_time_avg:.2f}s "
        f"(Max allowed: {max_mean_time:.2f}s)"
    )

    if mean_time_avg <= max_mean_time:
        print("✅ Mean survey time constraint satisfied\n")
    else:
        print("❌ Mean survey time constraint violated\n")
    '''
    print("Qualified Completes per Category:")
    for i, cat in enumerate(categories["category_name"]):
        print(f"\n{cat}")
        print(f"  Mean qualified   : {qualified_mean[i]:.1f}")
        print(f"  Median           : {qualified_median[i]:.1f}")
        print(f"  5th percentile   : {qualified_5th[i]:.1f}")
        print(f"  95th percentile  : {qualified_95th[i]:.1f}")
        print(f"  Failure prob (%) : {failure_prob[i]:.1f}")
    '''
    print("\nDemographic Exposure Validation:")
    print(f"(Tolerance: ±{demographic_tolerance:.3f})")

    for j, cell in enumerate(cells):
        print(
            f"  {cell}: "
            f"mean={demo_mean[j]:.4f}, "
            f"target={cell_shares[j]:.4f}, "
            f"failure_prob={demo_failure_prob[j]:.1f}%"
        )

    print(
        f"\nProbability at least one demographic "
        f"exceeds tolerance in a run: "
        f"{overall_demo_risk:.1f}%"
    )

    high_risk_categories = (failure_prob > 5).sum()
    print(
        f"\nCategories with >5% probability of missing target: "
        f"{high_risk_categories} / {num_categories}"
    )

    # --- Return structured results ---
    return {
        "qualified_mean": qualified_mean,
        "qualified_median": qualified_median,
        "qualified_5th": qualified_5th,
        "qualified_95th": qualified_95th,
        "failure_prob": failure_prob,
        "mean_time_avg": mean_time_avg,
        "demo_mean": demo_mean,
        "demo_failure_prob": demo_failure_prob,
        "overall_demo_risk": overall_demo_risk,
    }
