"""Reporting and output formatting for bundle simulation results."""

import pandas as pd
import numpy as np
from typing import List, Dict, TYPE_CHECKING
from collections import Counter

import config

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from bundle_engine import BundlePlan, BundleSimulationResult, MonteCarloResult


def print_bundle_plan_summary(plan: "BundlePlan") -> None:
    """Print detailed summary of a bundle plan before simulation."""
    print(f"\n{'='*70}")
    print(f"BUNDLE PLAN: {plan.name}")
    print(f"{'='*70}")
    print(f"Description: {plan.description}")
    print(f"\nTotal Bundles: {plan.num_bundles}")
    print(f"Total Categories: {plan.total_categories}")
    print(f"Total Planned Respondents: {plan.total_planned_respondents:,}")
    
    print(f"\n{'Bundle':<10} {'N Planned':>12} {'Exp. Time':>12} {'# Cats':>10}")
    print("-" * 50)
    for bundle in plan.bundles:
        print(f"{bundle.bundle_id:<10} {bundle.planned_respondents:>12,} "
              f"{bundle.expected_time:>12.1f}s {bundle.num_categories:>10}")
    print("-" * 50)
    print(f"{'TOTAL':<10} {plan.total_planned_respondents:>12,}")
    print(f"{'='*70}")


def print_bundle_simulation_result(result: "BundleSimulationResult") -> None:
    """Print detailed summary of a bundle simulation result."""
    print(f"\n{'='*70}")
    print(f"SIMULATION RESULT: {result.plan_name}")
    print(f"{'='*70}")
    
    target = config.TARGET_QUALIFIED_COMPLETIONS
    max_mean = config.MAX_MEAN_INTERVIEW_TIME_SECONDS
    
    # Summary metrics
    print(f"\nTotal Respondents: {result.total_respondents:,}")
    print(f"Min Completions: {result.min_category_completions} (target: {target}) "
          f"{'✓' if result.all_quotas_met else '✗'}")
    print(f"Mean Time: {result.mean_interview_time:.1f}s (max: {max_mean}s) "
          f"{'✓' if result.mean_time_ok else '✗'}")
    print(f"Q95 Time: {result.q95_interview_time:.1f}s")
    print(f"Demographics OK: {'✓' if result.demographics_ok else '✗'}")
    
    # Overall status
    if result.all_constraints_met:
        print(f"\n✓ ALL CONSTRAINTS MET")
    else:
        print(f"\n✗ CONSTRAINTS NOT MET:")
        if not result.all_quotas_met:
            print(f"   - Quotas: some categories below {target}")
        if not result.mean_time_ok:
            print(f"   - Mean time exceeds {max_mean}s")
        if not result.demographics_ok:
            print(f"   - Demographics out of tolerance")
    print(f"{'='*70}")


def print_bundle_plan_comparison_table(
    monte_carlo_results: List["MonteCarloResult"]
) -> None:
    """Print side-by-side comparison of bundle plans based on Monte Carlo validation."""
    print(f"\n{'='*110}")
    print("BUNDLE PLAN COMPARISON (Monte Carlo Validated)")
    print(f"{'='*110}")
    
    headers = ["Plan", "Total N", "Mean Time", "Q95 Time", "P(Quota)", "P(Time)", "P(Demo)", "P(All)", "Status"]
    print(f"{headers[0]:<35} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10} "
          f"{headers[4]:>10} {headers[5]:>10} {headers[6]:>10} {headers[7]:>10} {headers[8]:>8}")
    print("-" * 110)
    
    target_conf = config.TARGET_CONFIDENCE
    
    for mc in monte_carlo_results:
        meets_conf = mc.success_rate_all_constraints >= target_conf
        status = "✓ PASS" if meets_conf else "✗ FAIL"
        print(f"{mc.plan_name[:35]:<35} {mc.total_respondents_mean:>10,.0f} "
              f"{mc.mean_time_avg:>10.1f} {mc.q95_time_avg:>10.1f} "
              f"{mc.success_rate_all_quotas:>10.1%} {mc.success_rate_mean_time:>10.1%} "
              f"{mc.success_rate_demographics:>10.1%} {mc.success_rate_all_constraints:>10.1%} {status:>8}")
    
    print("-" * 110)
    
    # Identify winner
    valid = [mc for mc in monte_carlo_results if mc.success_rate_all_constraints >= target_conf]
    if valid:
        winner = min(valid, key=lambda mc: mc.total_respondents_mean)
        print(f"\n🏆 WINNER: {winner.plan_name} ({winner.total_respondents_mean:,.0f} respondents, "
              f"P(all)={winner.success_rate_all_constraints:.1%})")
    else:
        print(f"\n⚠️  No plan met P(all constraints) >= {target_conf:.0%}!")
        # Show best anyway
        if monte_carlo_results:
            best = max(monte_carlo_results, key=lambda mc: mc.success_rate_all_constraints)
            print(f"   Closest: {best.plan_name} (P(all)={best.success_rate_all_constraints:.1%})")
    print(f"{'='*110}")


def print_monte_carlo_bundle_summary(mc_result: "MonteCarloResult") -> None:
    """Print detailed Monte Carlo summary for a bundle plan."""
    print(f"\n{'='*70}")
    print(f"MONTE CARLO VALIDATION: {mc_result.plan_name}")
    print(f"{'='*70}")
    print(f"Runs: {mc_result.num_runs}")
    print(f"Target Confidence: {config.TARGET_CONFIDENCE:.0%}")
    print("-" * 70)
    
    # Success rates
    print(f"\nSuccess Rates:")
    print(f"  All Quotas Met:     {mc_result.success_rate_all_quotas:>8.1%}")
    print(f"  Mean Time OK:       {mc_result.success_rate_mean_time:>8.1%}")
    print(f"  Demographics OK:    {mc_result.success_rate_demographics:>8.1%}")
    print(f"  ALL Constraints:    {mc_result.success_rate_all_constraints:>8.1%} "
          f"{'✓' if mc_result.success_rate_all_constraints >= config.TARGET_CONFIDENCE else '✗'}")
    
    # Statistics
    print(f"\nStatistics (Mean ± Std):")
    print(f"  Total Respondents:  {mc_result.total_respondents_mean:>10,.0f} ± {mc_result.total_respondents_std:>8,.1f}")
    print(f"  Mean Interview Time:{mc_result.mean_time_avg:>10.1f}s ± {mc_result.mean_time_std:>8.2f}s")
    print(f"  Q95 Interview Time: {mc_result.q95_time_avg:>10.1f}s")
    print(f"  Min Completions:    {mc_result.min_completions_avg:>10.0f} ± {mc_result.min_completions_std:>8.1f}")
    print(f"{'='*70}")


def print_category_exposure_demographics(
    result: "BundleSimulationResult",
    category_data: pd.DataFrame,
    top_n: int = 5,
    tolerance: float = 0.05
) -> None:
    """
    Print per-category exposure demographic validation.
    
    Shows the worst-performing categories by demographic deviation to help
    identify any systematic representativeness issues.
    """
    print(f"\n{'='*80}")
    print("PER-CATEGORY EXPOSURE DEMOGRAPHIC VALIDATION")
    print(f"{'='*80}")
    print(f"Tolerance: ±{tolerance:.0%} (max absolute deviation from expected)")
    
    lookup = category_data.set_index("category_id")
    
    # Calculate deviations for each category/demo combination
    deviations: List[Dict] = []
    
    for cat_id, demo_counters in result.category_exposure_demographics.items():
        for demo_type, counter in demo_counters.items():
            total = sum(counter.values())
            if total == 0:
                continue
            
            expected_dist = config.DEMOGRAPHICS_DISTRIBUTION.get(demo_type, {})
            for category, expected_pct in expected_dist.items():
                actual_count = counter.get(category, 0)
                actual_pct = actual_count / total
                dev = actual_pct - expected_pct  # Keep sign for direction
                deviations.append({
                    "category_id": cat_id,
                    "category_name": lookup.loc[cat_id, "category_name"],
                    "demo_type": demo_type,
                    "demo_value": category,
                    "expected": expected_pct,
                    "actual": actual_pct,
                    "deviation": dev,
                    "abs_deviation": abs(dev),
                    "exposed_n": total
                })
    
    if not deviations:
        print("\nNo exposure data available.")
        return
    
    # Show worst deviations by demographic type
    for demo_type in config.DEMOGRAPHICS_DISTRIBUTION.keys():
        print(f"\n{demo_type.upper()} - Worst {top_n} Deviations:")
        print(f"{'Category':<40} {'Value':<15} {'Expected':>10} {'Actual':>10} {'Dev':>10} {'N':>8}")
        print("-" * 95)
        
        type_devs = [d for d in deviations if d["demo_type"] == demo_type]
        worst = sorted(type_devs, key=lambda x: x["abs_deviation"], reverse=True)[:top_n]
        
        for d in worst:
            status = "⚠" if d["abs_deviation"] > tolerance else "✓"
            print(f"{d['category_name'][:40]:<40} {d['demo_value']:<15} "
                  f"{d['expected']:>10.1%} {d['actual']:>10.1%} "
                  f"{d['deviation']:>+10.1%} {d['exposed_n']:>8,} {status}")
    
    # Overall summary
    max_dev = max(d["abs_deviation"] for d in deviations)
    n_violations = sum(1 for d in deviations if d["abs_deviation"] > tolerance)
    print(f"\nSummary:")
    print(f"  Max absolute deviation: {max_dev:.1%}")
    print(f"  Violations (>{tolerance:.0%}): {n_violations}/{len(deviations)} category-demo combinations")
    print(f"  Status: {'✓ PASS' if n_violations == 0 else '⚠ REVIEW NEEDED'}")
    print(f"{'='*80}")


def print_bundle_details(plan: "BundlePlan", verbose: bool = False) -> None:
    """Print detailed breakdown of bundle contents."""
    print(f"\n{'='*80}")
    print(f"BUNDLE DETAILS: {plan.name}")
    print(f"{'='*80}")
    
    lookup = plan.category_data.set_index("category_id")
    
    for bundle in plan.bundles:
        print(f"\n--- Bundle {bundle.bundle_id} ---")
        print(f"Planned Respondents: {bundle.planned_respondents:,}")
        print(f"Expected Time: {bundle.expected_time:.1f}s")
        print(f"Categories ({bundle.num_categories}):")
        
        for cat_id in bundle.categories:
            cat = lookup.loc[cat_id]
            ir = cat["effective_incidence_rate"]
            length = cat["survey_length_seconds"]
            name = cat["category_name"]
            if verbose:
                print(f"  [{cat_id:>3}] {name[:45]:<45} IR={ir:.1%} Len={length:.0f}s")
            else:
                print(f"  - {name}")
    
    print(f"\n{'='*80}")


def generate_bundle_comparison_dataframe(
    monte_carlo_results: List["MonteCarloResult"]
) -> pd.DataFrame:
    """Generate DataFrame comparing bundle plans for export."""
    rows = [{
        "plan_name": mc.plan_name,
        "total_respondents_mean": mc.total_respondents_mean,
        "total_respondents_std": mc.total_respondents_std,
        "mean_time_avg": mc.mean_time_avg,
        "mean_time_std": mc.mean_time_std,
        "q95_time_avg": mc.q95_time_avg,
        "success_rate_all_quotas": mc.success_rate_all_quotas,
        "success_rate_mean_time": mc.success_rate_mean_time,
        "success_rate_demographics": mc.success_rate_demographics,
        "success_rate_all_constraints": mc.success_rate_all_constraints,
        "min_completions_avg": mc.min_completions_avg,
        "meets_target_confidence": mc.success_rate_all_constraints >= config.TARGET_CONFIDENCE
    } for mc in monte_carlo_results]
    return pd.DataFrame(rows)
