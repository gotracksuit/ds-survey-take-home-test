"""
Main entry point for the Survey Allocation Optimiser (Bundle Mode).

Generates static "survey versions" (bundles) where each respondent sees 
a fixed set of category qualifiers.

Workflow:
1. Load and preprocess category data (with optional seasonality adjustments)
2. Create bundle plans (A, B, C strategies)
3. Size bundles using Bonferroni-adjusted confidence for TARGET_CONFIDENCE
4. Run Monte Carlo validation on each plan
5. Select winner (lowest N among those meeting TARGET_CONFIDENCE)
6. Export winning plan to bundle_plan.csv
"""

import argparse
import sys
from typing import List, Optional, Dict

import config
from data_loader import load_category_data, get_category_summary
from bundle_engine import (
    BundlePlan, MonteCarloResult,
    get_all_bundle_plans, run_monte_carlo_for_plan,
    export_bundle_plan_to_csv, export_bundle_summary_to_csv,
    run_simulation_for_bundle_plan
)
from reporting import (
    print_bundle_plan_summary, print_bundle_simulation_result,
    print_bundle_plan_comparison_table, print_monte_carlo_bundle_summary,
    print_category_exposure_demographics, print_bundle_details,
    generate_bundle_comparison_dataframe
)


def print_seasonality_adjustments(
    seasonality_name: str,
    seasonality_multipliers: Dict[int, float],
    category_data
) -> None:
    """Print summary of seasonality adjustments being applied."""
    print(f"\n{'='*70}")
    print(f"SEASONALITY SCENARIO: {seasonality_name.upper()}")
    print(f"{'='*70}")
    print(f"Adjusting {len(seasonality_multipliers)} categories for next month's IR changes:")
    print(f"\n{'Category':<45} {'Base IR':>10} {'Adjusted':>10} {'Change':>10}")
    print("-" * 75)
    
    lookup = category_data.set_index("category_id")
    
    # Sort by impact (largest change first)
    sorted_adjustments = sorted(
        seasonality_multipliers.items(),
        key=lambda x: abs(x[1] - 1.0),
        reverse=True
    )
    
    for cat_id, multiplier in sorted_adjustments:
        if cat_id not in lookup.index:
            continue
        cat = lookup.loc[cat_id]
        base_ir = cat["incidence_rate"]
        adjusted_ir = min(base_ir * multiplier, 1.0)
        change_pct = (multiplier - 1.0) * 100
        direction = "↑" if multiplier > 1.0 else "↓"
        print(f"{cat['category_name'][:45]:<45} {base_ir:>10.1%} {adjusted_ir:>10.1%} "
              f"{direction}{abs(change_pct):>8.0f}%")
    
    print(f"{'='*70}")


def run_bundle_optimiser(
    verbose: bool = False,
    export_all: bool = False,
    num_monte_carlo_runs: Optional[int] = None,
    use_quota_sampling: bool = False,
    demographic_tolerance: float = 0.05,
    seasonality: Optional[str] = None,
    valid_completion_rate: float = 1.0
) -> Optional[BundlePlan]:
    """
    Run the bundle-based optimisation workflow.
    
    Steps:
    1. Load category data (with optional seasonality adjustments)
    2. Create bundle plans (A, B, C strategies)
    3. Size bundles using Bonferroni-adjusted confidence
    4. Run Monte Carlo validation on each plan
    5. Select winner (lowest N among those meeting TARGET_CONFIDENCE)
    6. Export winning plan to bundle_plan.csv
    
    Args:
        verbose: Print detailed progress
        export_all: Export all comparison data
        num_monte_carlo_runs: Override default number of MC runs
        use_quota_sampling: Force exact demographic match within bundles
        demographic_tolerance: Max allowed demographic deviation (default 5%)
        seasonality: Name of seasonality scenario to apply (e.g., "winter", "summer")
        valid_completion_rate: Fraction of completions passing quality checks (0-1).
                               E.g., 0.9 means 10% of completions are randomly invalidated.
    
    Returns:
        The winning BundlePlan, or None if no plan meets requirements
    """
    mc_runs = num_monte_carlo_runs if num_monte_carlo_runs else config.MONTE_CARLO_RUNS
    
    # Determine seasonality multipliers
    seasonality_multipliers: Dict[int, float] = {}
    if seasonality:
        if seasonality not in config.SEASONALITY_SCENARIOS:
            available = ", ".join(config.SEASONALITY_SCENARIOS.keys())
            print(f"\n❌ Unknown seasonality scenario: '{seasonality}'")
            print(f"   Available scenarios: {available}")
            return None
        seasonality_multipliers = config.SEASONALITY_SCENARIOS[seasonality]
    
    print(f"\n{'='*70}")
    print("SURVEY ALLOCATION OPTIMISER")
    print(f"{'='*70}")
    print(f"Target: {config.TARGET_QUALIFIED_COMPLETIONS} completions/category")
    print(f"Target Confidence: {config.TARGET_CONFIDENCE:.0%}")
    print(f"Max Mean Time: {config.MAX_MEAN_INTERVIEW_TIME_SECONDS}s")
    print(f"Max Qualifiers per Bundle: {config.MAX_CATEGORY_QUALIFIERS_PER_RESPONDENT}")
    print(f"Monte Carlo Runs: {mc_runs}")
    print(f"Quota Sampling: {'Enabled' if use_quota_sampling else 'Disabled'}")
    print(f"Demographic Tolerance: ±{demographic_tolerance:.1%}")
    print(f"Valid Completion Rate: {valid_completion_rate:.0%}")
    print(f"Seasonality: {seasonality.upper() if seasonality else 'None'}")
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    
    # First load without seasonality to show base rates
    df_base = load_category_data(seasonality_multipliers={})
    
    # Then load with seasonality and valid completion rate
    df = load_category_data(
        seasonality_multipliers=seasonality_multipliers,
        valid_completion_rate=valid_completion_rate
    )
    
    if verbose:
        get_category_summary(df)
    print(f"      {len(df)} categories loaded")
    
    # Show seasonality adjustments
    if seasonality and seasonality_multipliers:
        print_seasonality_adjustments(seasonality, seasonality_multipliers, df_base)
    
    # Step 2: Create bundle plans
    print("\n[2/6] Creating bundle plans...")
    plans: List[BundlePlan] = get_all_bundle_plans(
        df, rng_seed=config.RANDOM_SEED, valid_completion_rate=valid_completion_rate
    )
    
    for p in plans:
        print(f"      - {p.name}: {p.num_bundles} bundles, {p.total_planned_respondents:,} planned N")
        if verbose:
            print_bundle_plan_summary(p)
    
    # Step 3: Run Monte Carlo validation for each plan
    print("\n[3/6] Monte Carlo validation...")
    mc_results: List[MonteCarloResult] = []
    
    for i, plan in enumerate(plans, 1):
        print(f"\n      [{i}/{len(plans)}] {plan.name}")
        mc = run_monte_carlo_for_plan(
            plan, 
            num_runs=mc_runs, 
            base_seed=config.RANDOM_SEED,
            verbose=verbose,
            demographic_tolerance=demographic_tolerance,
            use_quota_sampling=use_quota_sampling,
            valid_completion_rate=valid_completion_rate
        )
        mc_results.append(mc)
        print(f"           P(all constraints) = {mc.success_rate_all_constraints:.1%} "
              f"{'✓' if mc.success_rate_all_constraints >= config.TARGET_CONFIDENCE else '✗'}")
    
    # Step 4: Compare and select winner
    print("\n[4/6] Comparing strategies...")
    print_bundle_plan_comparison_table(mc_results)
    
    # Select winner: lowest N among those meeting target confidence
    valid_plans = [
        (plans[i], mc_results[i]) 
        for i in range(len(plans))
        if mc_results[i].success_rate_all_constraints >= config.TARGET_CONFIDENCE
    ]
    
    winner_plan: Optional[BundlePlan] = None
    winner_mc: Optional[MonteCarloResult] = None
    
    if valid_plans:
        winner_plan, winner_mc = min(valid_plans, key=lambda x: x[0].total_planned_respondents)
    else:
        print(f"\n⚠️  No plan met P(all) >= {config.TARGET_CONFIDENCE:.0%}")
        # Fall back to best available
        best_idx = max(range(len(mc_results)), 
                       key=lambda i: mc_results[i].success_rate_all_constraints)
        winner_plan = plans[best_idx]
        winner_mc = mc_results[best_idx]
        print(f"    Using best available: {winner_plan.name} "
              f"(P(all)={winner_mc.success_rate_all_constraints:.1%})")
    
    # Step 5: Detailed analysis of winner
    print("\n[5/6] Detailed analysis of selected plan...")
    if winner_plan and winner_mc:
        print_monte_carlo_bundle_summary(winner_mc)
        
        if verbose:
            print_bundle_details(winner_plan, verbose=True)
        
        # Run one simulation for detailed demographics analysis
        print("\n      Running detailed simulation for demographics check...")
        detailed_result = run_simulation_for_bundle_plan(
            winner_plan, random_seed=config.RANDOM_SEED, verbose=verbose,
            demographic_tolerance=demographic_tolerance,
            use_quota_sampling=use_quota_sampling,
            valid_completion_rate=valid_completion_rate
        )
        print_bundle_simulation_result(detailed_result)
        print_category_exposure_demographics(detailed_result, df, tolerance=demographic_tolerance)
    
    # Step 6: Export
    print("\n[6/6] Exporting...")
    output_suffix = f"_{seasonality}" if seasonality else ""
    if winner_plan:
        export_bundle_plan_to_csv(winner_plan, f"results/bundle_plan{output_suffix}.csv")
        export_bundle_summary_to_csv(winner_plan, f"results/bundle_summary{output_suffix}.csv")
    
    if export_all:
        comparison_df = generate_bundle_comparison_dataframe(mc_results)
        comparison_df.to_csv(f"results/bundle_comparison{output_suffix}.csv", index=False)
        print(f"Bundle comparison exported to: results/bundle_comparison{output_suffix}.csv")
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    
    return winner_plan


def list_seasonality_scenarios() -> None:
    """Print available seasonality scenarios and their details."""
    print(f"\n{'='*70}")
    print("AVAILABLE SEASONALITY SCENARIOS")
    print(f"{'='*70}")
    
    for name, multipliers in config.SEASONALITY_SCENARIOS.items():
        increases = sum(1 for m in multipliers.values() if m > 1.0)
        decreases = sum(1 for m in multipliers.values() if m < 1.0)
        print(f"\n  {name}:")
        print(f"    Categories affected: {len(multipliers)}")
        print(f"    IR increases: {increases}, IR decreases: {decreases}")
    
    print(f"\n{'='*70}")
    print("Use --seasonality <name> to apply a scenario")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Survey Allocation Optimiser - Bundle Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                         # Run with defaults
  python main.py --seasonality winter    # Apply winter IR adjustments
  python main.py --valid-completion 0.9  # Simulate 10%% data quality loss
  python main.py --list-seasonality      # Show available scenarios
  python main.py -v --quota-sampling     # Verbose with exact demographics
  python main.py --seasonality winter --valid-completion 0.85  # Combined
        """
    )
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Show detailed progress")
    parser.add_argument("--export-all", action="store_true", 
                        help="Export all results to CSV")
    parser.add_argument("--monte-carlo-runs", type=int, default=None,
                        help=f"Number of Monte Carlo runs (default: {config.MONTE_CARLO_RUNS})")
    parser.add_argument("--quota-sampling", action="store_true",
                        help="Use quota sampling to guarantee exact demographic match")
    parser.add_argument("--demographic-tolerance", type=float, default=0.05,
                        help="Max demographic deviation allowed, e.g., 0.05 for 5%% (default: 5%%)")
    parser.add_argument("--seasonality", type=str, default=None,
                        help="Seasonality scenario to apply (e.g., winter, summer, holiday, new_year, back_to_school)")
    parser.add_argument("--list-seasonality", action="store_true",
                        help="List available seasonality scenarios and exit")
    parser.add_argument("--valid-completion", type=float, default=1.0,
                        help="Valid completion rate (0-1). Simulates data quality issues. "
                             "E.g., 0.9 means 10%% of completions fail quality checks (default: 1.0)")
    args = parser.parse_args()

    # Handle --list-seasonality
    if args.list_seasonality:
        list_seasonality_scenarios()
        sys.exit(0)

    winner_plan = run_bundle_optimiser(
        verbose=args.verbose,
        export_all=args.export_all,
        num_monte_carlo_runs=args.monte_carlo_runs,
        use_quota_sampling=args.quota_sampling,
        demographic_tolerance=args.demographic_tolerance,
        seasonality=args.seasonality,
        valid_completion_rate=args.valid_completion
    )
    
    if winner_plan:
        # Quick check if plan meets confidence
        mc = run_monte_carlo_for_plan(
            winner_plan, 
            num_runs=args.monte_carlo_runs or config.MONTE_CARLO_RUNS,
            use_quota_sampling=args.quota_sampling,
            demographic_tolerance=args.demographic_tolerance,
            valid_completion_rate=args.valid_completion
        )
        if mc.success_rate_all_constraints >= config.TARGET_CONFIDENCE:
            print(f"\n✅ Best Bundle Plan: {winner_plan.name}")
            print(f"   Total Planned N: {winner_plan.total_planned_respondents:,}")
            print(f"   P(all constraints): {mc.success_rate_all_constraints:.1%}")
            if args.seasonality:
                print(f"   Seasonality: {args.seasonality}")
            if args.valid_completion < 1.0:
                print(f"   Valid Completion Rate: {args.valid_completion:.0%}")
            sys.exit(0)
        else:
            print(f"\n⚠️  Selected plan does not meet {config.TARGET_CONFIDENCE:.0%} confidence")
            print(f"   P(all constraints): {mc.success_rate_all_constraints:.1%}")
            sys.exit(1)
    else:
        print("\n❌ No bundle plan could be created")
        sys.exit(1)


if __name__ == "__main__":
    main()
