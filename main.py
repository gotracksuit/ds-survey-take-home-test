import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TARGET_QUALIFIED,
    MAX_MEAN_TIME,
    GENDERS,
    AGES,
    REGIONS,
    CATEGORIES_CSV
)
from demographic import load_demographics
from optimizer import lp_allocation, CategoryAllocatorGreedy
from validation import validate_greedy, validate_lp
from monte_carlo_validation import lp_monte_carlo_validate

sns.set_style("whitegrid")


def plot_results(categories, allocation, mc_results):
    """
    Visualize LP expectation vs Monte Carlo simulation.
    Includes scatter plot and histogram/density plot.

    Parameters
    ----------
    categories : pd.DataFrame
        Must include incidence_rate and category_length_seconds
    allocation : np.ndarray
        Shape (num_categories, num_demographics)
    mc_results : np.ndarray
        Shape (n_simulations, num_categories)   

    returns : None
    """

    # LP expected qualified
    lp_expected = allocation.sum(axis=1) * categories["incidence_rate"].values

    # Monte Carlo mean and std
    mc_mean = mc_results.mean(axis=0)
    mc_std = mc_results.std(axis=0)

    # --- Scatter plot: LP vs Monte Carlo ---
    plt.figure(figsize=(8,6))
    plt.errorbar(lp_expected, mc_mean, yerr=mc_std, fmt='o', capsize=5)
    plt.plot([0, max(lp_expected)], [0, max(lp_expected)], 'k--', label='Scatter plot-Perfect Match')
    plt.xlabel("LP Expected Qualified")
    plt.ylabel("Monte Carlo Mean Qualified")
    plt.title("LP vs Monte Carlo Comparison (with Std Dev)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Histogram/density plot per category ---
    plt.figure(figsize=(10,6))
    for i, cat in enumerate(categories["category_name"]):
        sns.histplot(mc_results[:, i], kde=True, label=cat, stat="density", alpha=0.6)
        plt.axvline(lp_expected[i], color='black', linestyle='--', alpha=0.7)
    plt.xlabel("Number of Qualified Respondents")
    plt.title("Monte Carlo Distribution per Category vs LP Expectation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    categories = pd.read_csv(CATEGORIES_CSV)

    demo_df = load_demographics(GENDERS, AGES, REGIONS)

    # Solve LP allocation
    allocation_lp = lp_allocation(
        categories,
        demo_df["cell_id"].tolist(),
        demo_df["population_share"].values,
        TARGET_QUALIFIED,
        MAX_MEAN_TIME
    )

    print("\n================= LP validation =================")
    validate_lp(categories,
                allocation_lp,
                TARGET_QUALIFIED,
                MAX_MEAN_TIME,
                demo_df["cell_id"].tolist(),
                demo_df["population_share"].values,
                tolerance=0.005)

    print("\n================= LP Monte Carlo Validation =================")
    # 2️⃣ LP Monte Carlo validation
    lp_monte_carlo_validate(categories,
                            np.ceil(allocation_lp).astype(int),
                            demo_df["cell_id"].tolist(),
                            demo_df["population_share"].values,
                            TARGET_QUALIFIED, 
                            MAX_MEAN_TIME,
                            n_simulations=1000,
                            random_seed=42)
    
    
    # Initialize and Run greedy Allocator
    print("\n================= Greedy Allocation =================")
    allocator = CategoryAllocatorGreedy(categories, demo_df, TARGET_QUALIFIED, MAX_MEAN_TIME)
    final_slots = allocator.run_allocation()

    validate_greedy(final_slots, categories, TARGET_QUALIFIED, MAX_MEAN_TIME)


if __name__ == "__main__":
    main()