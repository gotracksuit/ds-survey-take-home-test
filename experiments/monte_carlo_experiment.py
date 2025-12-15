"""
Monte Carlo Experiment: Run multiple simulations for each allocation policy
and visualize constraint satisfaction vs sample size.

Creates a single clean chart showing how each policy performs across different sample sizes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

from src.data_loader import load_categories, generate_respondents, TIME_BUDGET, TARGET_QUALIFIED
from src.allocation_policies import (
    random_allocation,
    priority_greedy_allocation,
    demographic_aware_allocation
)
from src.simulation import simulate_qualifications

def run_monte_carlo(policy_func, policy_name, categories, n_values, n_sims=50, seed=42):
    """
    Run Monte Carlo simulation for a policy across different sample sizes

    Args:
        policy_func: Allocation policy function
        policy_name: Name of the policy
        categories: Category data
        n_values: List of N values to test
        n_sims: Number of simulation runs
        seed: Random seed

    Returns:
        DataFrame with results
    """
    results = []
    max_n = max(n_values)

    print(f"\n{'='*70}")
    print(f"Running Monte Carlo for: {policy_name}")
    print(f"{'='*70}")
    print(f"Simulations: {n_sims}, N range: {min(n_values):,} - {max_n:,}")

    for sim in tqdm(range(n_sims), desc=f"{policy_name}"):
        sim_seed = seed + sim
        # Generate all respondents once
        all_respondents = generate_respondents(max_n, seed=sim_seed)

        for n in n_values:
            respondents = all_respondents.iloc[:n]
            alloc, cat_ids = policy_func(categories, respondents, seed=sim_seed)
            qualified, times = simulate_qualifications(
                alloc, cat_ids, categories, respondents, seed=sim_seed
            )

            results.append({
                'sim': sim,
                'n_respondents': n,
                'pct_categories_met': (qualified >= TARGET_QUALIFIED).mean() * 100,
                'mean_time': times.mean(),
                'pct_over_budget': (times > TIME_BUDGET).mean() * 100,
                'min_qualified': qualified.min(),
                'total_assignments': alloc.sum()
            })

    return pd.DataFrame(results)

def create_plot(all_results, n_values):
    """
    Create constraint satisfaction vs sample size chart

    Args:
        all_results: Dictionary of {policy_name: results_df}
        n_values: List of N values
    """
    policy_colors = {
        'Random': '#E74C3C',
        'Priority Greedy': '#3498DB',
        'Demographic Aware': '#2ECC71'
    }

    print("\n" + "="*70)
    print("Creating Visualization")
    print("="*70)

    fig = go.Figure()

    for policy_name, results in all_results.items():
        color = policy_colors[policy_name]

        # Mean line with markers
        mean_data = results.groupby('n_respondents')['pct_categories_met'].mean().reset_index()
        std_data = results.groupby('n_respondents')['pct_categories_met'].std().reset_index()

        fig.add_trace(go.Scatter(
            x=mean_data['n_respondents'],
            y=mean_data['pct_categories_met'],
            mode='lines+markers',
            line=dict(color=color, width=3),
            marker=dict(size=8),
            name=policy_name,
            error_y=dict(
                type='data',
                array=std_data['pct_categories_met'],
                visible=True
            )
        ))

    # Add target line
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text="Target: 100% Categories Met",
        annotation_position="right"
    )

    fig.update_layout(
        title='Constraint Satisfaction vs Sample Size',
        xaxis_title="Number of Respondents",
        yaxis_title="% Categories Meeting Target (≥200 Qualified)",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        font=dict(size=12),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )

    fig.write_html('results/monte_carlo_constraint_satisfaction.html')
    print("\n✅ Saved: results/monte_carlo_constraint_satisfaction.html")

def main():
    """Run the Monte Carlo experiment"""

    print("\n" + "="*70)
    print("MONTE CARLO EXPERIMENT")
    print("="*70)

    # Configuration
    n_min = 2000
    n_max = 10000
    step_size = 500
    n_sims = 30

    n_values = list(range(n_min, n_max + 1, step_size))

    print(f"\nConfiguration:")
    print(f"  N range: {n_min:,} - {n_max:,} (step {step_size})")
    print(f"  Simulations: {n_sims} per policy")
    print(f"  Total runs: {len(n_values) * n_sims} per policy")

    # Load data
    categories = load_categories(enriched=True)
    print(f"\nLoaded {len(categories)} categories")

    # Define policies
    policies = {
        'Random': random_allocation,
        'Priority Greedy': priority_greedy_allocation,
        'Demographic Aware': demographic_aware_allocation
    }

    # Run Monte Carlo for each policy
    all_results = {}

    for policy_name, policy_func in policies.items():
        results_df = run_monte_carlo(
            policy_func, policy_name, categories, n_values, n_sims, seed=42
        )
        all_results[policy_name] = results_df

    # Create visualization
    create_plot(all_results, n_values)

    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print("\nOpen results/monte_carlo_constraint_satisfaction.html to view the chart")
    print()

if __name__ == "__main__":
    main()
