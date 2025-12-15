"""
Demographic Impact Analysis

Measures whether the demographic constraints actually impact allocation efficiency.
Compares performance with and without demographic targeting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_categories, generate_respondents, TARGET_QUALIFIED
from src.allocation_policies import priority_greedy_allocation, demographic_aware_allocation
from src.simulation import simulate_qualifications

def run_policy_comparison(categories, n_respondents, seed=42):
    """Run both policies and return results"""
    respondents = generate_respondents(n_respondents, seed=seed)

    # Priority Greedy (ignores demographics)
    alloc_pg, cat_ids_pg = priority_greedy_allocation(categories, respondents, seed=seed)
    qual_pg, times_pg = simulate_qualifications(alloc_pg, cat_ids_pg, categories, respondents, seed=seed)

    # Demographic Aware (respects demographics)
    alloc_da, cat_ids_da = demographic_aware_allocation(categories, respondents, seed=seed)
    qual_da, times_da = simulate_qualifications(alloc_da, cat_ids_da, categories, respondents, seed=seed)

    return {
        'pg': {'alloc': alloc_pg, 'cat_ids': cat_ids_pg, 'qualified': qual_pg, 'times': times_pg},
        'da': {'alloc': alloc_da, 'cat_ids': cat_ids_da, 'qualified': qual_da, 'times': times_da}
    }

def analyze_constraint_impact_per_category(categories, results):
    """
    Analyze impact on individual categories based on their constraints
    """
    print("\n" + "="*70)
    print("CONSTRAINT IMPACT PER CATEGORY")
    print("="*70)

    pg_qual = results['pg']['qualified']
    da_qual = results['da']['qualified']
    cat_ids = results['pg']['cat_ids']

    # Create analysis dataframe
    analysis = []
    for i, cat_id in enumerate(cat_ids):
        cat = categories[categories['category_id'] == cat_id].iloc[0]
        analysis.append({
            'category_name': cat['category_name'],
            'gender_flag': cat['gender_flag'],
            'age_flag': cat['age_flag'],
            'incidence_rate': cat['incidence_rate'],
            'pg_qualified': pg_qual[i],
            'da_qualified': da_qual[i],
            'qualified_diff': da_qual[i] - pg_qual[i],
            'qualified_pct_change': ((da_qual[i] - pg_qual[i]) / pg_qual[i] * 100) if pg_qual[i] > 0 else 0
        })

    df = pd.DataFrame(analysis)

    # Categorize by constraint type
    df['has_gender_constraint'] = df['gender_flag'] != 'na'
    df['has_age_constraint'] = df['age_flag'] != 'na'
    df['constraint_type'] = 'None'
    df.loc[df['has_gender_constraint'] & ~df['has_age_constraint'], 'constraint_type'] = 'Gender Only'
    df.loc[~df['has_gender_constraint'] & df['has_age_constraint'], 'constraint_type'] = 'Age Only'
    df.loc[df['has_gender_constraint'] & df['has_age_constraint'], 'constraint_type'] = 'Both'

    # Summary by constraint type
    print("\nImpact by Constraint Type:")
    print(f"{'Constraint Type':<20} {'Count':<8} {'Avg Qualified Diff':<20} {'Avg % Change':<15}")
    print("-" * 70)

    for constraint_type in ['None', 'Gender Only', 'Age Only', 'Both']:
        subset = df[df['constraint_type'] == constraint_type]
        if len(subset) > 0:
            avg_diff = subset['qualified_diff'].mean()
            avg_pct = subset['qualified_pct_change'].mean()
            print(f"{constraint_type:<20} {len(subset):<8} {avg_diff:<20.1f} {avg_pct:<15.2f}%")

    # Categories most hurt by constraints
    print("\nCategories MOST HURT by demographic constraints (DA got fewer qualified):")
    worst = df[df['qualified_diff'] < -5].sort_values('qualified_diff').head(10)
    if len(worst) > 0:
        for _, row in worst.iterrows():
            print(f"  {row['category_name'][:45]:45} ({row['gender_flag']}, {row['age_flag']:5}) | "
                  f"PG: {row['pg_qualified']:3.0f} → DA: {row['da_qualified']:3.0f} ({row['qualified_pct_change']:+.1f}%)")
    else:
        print("  None found")

    # Categories helped by constraints
    print("\nCategories HELPED by demographic constraints (DA got more qualified):")
    best = df[df['qualified_diff'] > 5].sort_values('qualified_diff', ascending=False).head(10)
    if len(best) > 0:
        for _, row in best.iterrows():
            print(f"  {row['category_name'][:45]:45} ({row['gender_flag']}, {row['age_flag']:5}) | "
                  f"PG: {row['pg_qualified']:3.0f} → DA: {row['da_qualified']:3.0f} ({row['qualified_pct_change']:+.1f}%)")
    else:
        print("  None found")

    return df

def analyze_pool_depletion(categories, results, respondents):
    """
    Analyze if demographic constraints cause pool depletion
    """
    print("\n" + "="*70)
    print("POOL DEPLETION ANALYSIS")
    print("="*70)

    # Count respondents by demographic
    gender_counts = respondents['gender'].value_counts()
    age_counts = respondents['age_group'].value_counts()

    print(f"\nRespondent Pool Composition:")
    print(f"  Total: {len(respondents):,}")
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count:,} ({count/len(respondents)*100:.1f}%)")
    for age, count in age_counts.items():
        print(f"  {age}: {count:,} ({count/len(respondents)*100:.1f}%)")

    # Calculate demand for each demographic segment
    print(f"\nDemand Analysis (to reach {TARGET_QUALIFIED} qualified per category):")

    # Female-only categories
    female_cats = categories[categories['gender_flag'] == 'female']
    if len(female_cats) > 0:
        total_female_demand = 0
        for _, cat in female_cats.iterrows():
            expected_needed = TARGET_QUALIFIED / (cat['incidence_rate'] * 0.75)  # with safety margin
            total_female_demand += expected_needed

        female_pool = gender_counts.get('Female', 0)
        print(f"\n  Female-only categories: {len(female_cats)}")
        print(f"    Total demand: {total_female_demand:,.0f} exposures")
        print(f"    Available pool: {female_pool:,} respondents")
        print(f"    Pressure ratio: {total_female_demand/female_pool:.2f}x (each female needs {total_female_demand/female_pool:.1f} exposures)")

        if total_female_demand / female_pool > 8:  # Assuming ~8 categories per person budget
            print(f"    ⚠️  HIGH PRESSURE - May need more female respondents!")

    # Male-only categories
    male_cats = categories[categories['gender_flag'] == 'male']
    if len(male_cats) > 0:
        total_male_demand = 0
        for _, cat in male_cats.iterrows():
            expected_needed = TARGET_QUALIFIED / (cat['incidence_rate'] * 0.75)
            total_male_demand += expected_needed

        male_pool = gender_counts.get('Male', 0)
        print(f"\n  Male-only categories: {len(male_cats)}")
        print(f"    Total demand: {total_male_demand:,.0f} exposures")
        print(f"    Available pool: {male_pool:,} respondents")
        print(f"    Pressure ratio: {total_male_demand/male_pool:.2f}x (each male needs {total_male_demand/male_pool:.1f} exposures)")

        if total_male_demand / male_pool > 8:
            print(f"    ⚠️  HIGH PRESSURE - May need more male respondents!")

    # 65+ categories
    senior_cats = categories[categories['age_flag'] == '65+']
    if len(senior_cats) > 0:
        total_senior_demand = 0
        for _, cat in senior_cats.iterrows():
            expected_needed = TARGET_QUALIFIED / (cat['incidence_rate'] * 0.75)
            total_senior_demand += expected_needed

        senior_pool = age_counts.get('65+', 0)
        print(f"\n  65+ only categories: {len(senior_cats)}")
        print(f"    Total demand: {total_senior_demand:,.0f} exposures")
        print(f"    Available pool: {senior_pool:,} respondents")
        print(f"    Pressure ratio: {total_senior_demand/senior_pool:.2f}x (each senior needs {total_senior_demand/senior_pool:.1f} exposures)")

        if total_senior_demand / senior_pool > 8:
            print(f"    ⚠️  HIGH PRESSURE - May need more 65+ respondents!")

def find_minimum_n_both_policies(categories, max_n=20000, step=500):
    """
    Find minimum N needed for each policy to meet all constraints
    """
    print("\n" + "="*70)
    print("MINIMUM RESPONDENTS COMPARISON")
    print("="*70)
    print("\nFinding minimum N where each policy meets all constraints...")
    print("(All categories ≥ 200 qualified, mean time ≤ 480s)\n")

    results = {'Priority Greedy': None, 'Demographic Aware': None}

    for n in range(2000, max_n + 1, step):
        respondents = generate_respondents(n, seed=42)

        # Test Priority Greedy
        if results['Priority Greedy'] is None:
            alloc_pg, cat_ids_pg = priority_greedy_allocation(categories, respondents, seed=42)
            qual_pg, times_pg = simulate_qualifications(alloc_pg, cat_ids_pg, categories, respondents, seed=42)

            if qual_pg.min() >= TARGET_QUALIFIED and times_pg.mean() <= 480:
                results['Priority Greedy'] = n
                print(f"✓ Priority Greedy: N = {n:,}")

        # Test Demographic Aware
        if results['Demographic Aware'] is None:
            alloc_da, cat_ids_da = demographic_aware_allocation(categories, respondents, seed=42)
            qual_da, times_da = simulate_qualifications(alloc_da, cat_ids_da, categories, respondents, seed=42)

            if qual_da.min() >= TARGET_QUALIFIED and times_da.mean() <= 480:
                results['Demographic Aware'] = n
                print(f"✓ Demographic Aware: N = {n:,}")

        # Break if both found
        if results['Priority Greedy'] is not None and results['Demographic Aware'] is not None:
            break

    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)

    if results['Priority Greedy']:
        print(f"  Priority Greedy (ignores demographics): {results['Priority Greedy']:,} respondents")
    else:
        print(f"  Priority Greedy: Could not meet constraints with N ≤ {max_n:,}")

    if results['Demographic Aware']:
        print(f"  Demographic Aware (respects demographics): {results['Demographic Aware']:,} respondents")
    else:
        print(f"  Demographic Aware: Could not meet constraints with N ≤ {max_n:,}")

    if results['Priority Greedy'] and results['Demographic Aware']:
        diff = results['Demographic Aware'] - results['Priority Greedy']
        pct = (diff / results['Priority Greedy']) * 100

        print(f"\n  Difference: {diff:+,} respondents ({pct:+.1f}%)")

        if diff > 0:
            print(f"\n  ⚠️  Demographic constraints INCREASE respondents needed by {abs(diff):,} ({abs(pct):.1f}%)")
            print(f"      This means the constraints make the problem HARDER")
        elif diff < 0:
            print(f"\n  ✓  Demographic constraints DECREASE respondents needed by {abs(diff):,} ({abs(pct):.1f}%)")
            print(f"      This means the constraints actually HELP efficiency")
        else:
            print(f"\n  →  Demographic constraints have NO IMPACT on minimum N")

    return results

def create_visualizations(category_analysis_df):
    """Create visualizations of constraint impact"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Qualified difference by constraint type
    ax = axes[0, 0]
    constraint_order = ['None', 'Gender Only', 'Age Only', 'Both']
    data_to_plot = [category_analysis_df[category_analysis_df['constraint_type'] == ct]['qualified_diff']
                    for ct in constraint_order]

    bp = ax.boxplot(data_to_plot, labels=constraint_order, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Change in Qualified (DA - PG)')
    ax.set_xlabel('Constraint Type')
    ax.set_title('Impact of Demographic Constraints on Qualified Respondents', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 2. Scatter: Incidence vs Impact
    ax = axes[0, 1]
    colors = {'None': 'gray', 'Gender Only': 'blue', 'Age Only': 'green', 'Both': 'red'}
    for constraint_type, color in colors.items():
        subset = category_analysis_df[category_analysis_df['constraint_type'] == constraint_type]
        ax.scatter(subset['incidence_rate'], subset['qualified_pct_change'],
                  alpha=0.6, label=constraint_type, color=color, s=50)

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Incidence Rate')
    ax.set_ylabel('% Change in Qualified (DA vs PG)')
    ax.set_title('Incidence Rate vs Constraint Impact', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Count of categories by constraint type
    ax = axes[1, 0]
    constraint_counts = category_analysis_df['constraint_type'].value_counts()[constraint_order]
    bars = ax.bar(constraint_order, constraint_counts.values, color=['gray', 'blue', 'green', 'red'])
    ax.set_ylabel('Number of Categories')
    ax.set_xlabel('Constraint Type')
    ax.set_title('Distribution of Constraint Types', fontweight='bold')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    # 4. Categories meeting target: PG vs DA
    ax = axes[1, 1]
    pg_meeting = (category_analysis_df['pg_qualified'] >= TARGET_QUALIFIED).sum()
    da_meeting = (category_analysis_df['da_qualified'] >= TARGET_QUALIFIED).sum()

    bars = ax.bar(['Priority Greedy', 'Demographic Aware'],
                  [pg_meeting, da_meeting],
                  color=['steelblue', 'coral'])
    ax.axhline(y=len(category_analysis_df), color='green', linestyle='--',
               label=f'Target (all {len(category_analysis_df)} categories)', linewidth=2)
    ax.set_ylabel('Number of Categories Meeting Target')
    ax.set_title(f'Categories with ≥{TARGET_QUALIFIED} Qualified', fontweight='bold')
    ax.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/demographic_impact_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: results/demographic_impact_analysis.png")

def main():
    print("\n" + "="*70)
    print("DEMOGRAPHIC CONSTRAINT IMPACT ANALYSIS")
    print("="*70)
    print("\nThis analysis measures whether your demographic constraints")
    print("actually impact allocation efficiency.\n")

    # Load enriched data with demographics
    categories = load_categories(enriched=True)
    print(f"Loaded {len(categories)} categories with demographic constraints")

    # Show constraint distribution
    gender_dist = categories['gender_flag'].value_counts()
    age_dist = categories['age_flag'].value_counts()
    print(f"\nConstraint Distribution:")
    print(f"  Gender: {dict(gender_dist)}")
    print(f"  Age: {dict(age_dist)}")

    # Run comparison at a specific N
    print(f"\n{'='*70}")
    print("TESTING AT N=8000")
    print(f"{'='*70}")

    n_test = 8000
    respondents = generate_respondents(n_test, seed=42)
    results = run_policy_comparison(categories, n_test, seed=42)

    print(f"\nOverall Comparison (N={n_test:,}):")
    print(f"  Priority Greedy:")
    print(f"    Total assignments: {results['pg']['alloc'].sum():,}")
    print(f"    Min qualified: {results['pg']['qualified'].min()}")
    print(f"    Mean time: {results['pg']['times'].mean():.1f}s")
    print(f"    Categories meeting target: {(results['pg']['qualified'] >= TARGET_QUALIFIED).sum()}/{len(categories)}")

    print(f"\n  Demographic Aware:")
    print(f"    Total assignments: {results['da']['alloc'].sum():,}")
    print(f"    Min qualified: {results['da']['qualified'].min()}")
    print(f"    Mean time: {results['da']['times'].mean():.1f}s")
    print(f"    Categories meeting target: {(results['da']['qualified'] >= TARGET_QUALIFIED).sum()}/{len(categories)}")

    # Per-category analysis
    category_df = analyze_constraint_impact_per_category(categories, results)

    # Pool depletion analysis
    analyze_pool_depletion(categories, results, respondents)

    # Find minimum N for each policy
    min_n_results = find_minimum_n_both_policies(categories)

    # Create visualizations
    create_visualizations(category_df)

    # Final verdict
    print("\n" + "="*70)
    print("VERDICT: DO DEMOGRAPHIC CONSTRAINTS MATTER?")
    print("="*70)

    if min_n_results['Priority Greedy'] and min_n_results['Demographic Aware']:
        diff = min_n_results['Demographic Aware'] - min_n_results['Priority Greedy']

        if abs(diff) < 100:
            print("\n❓ MINIMAL IMPACT")
            print(f"   The demographic constraints change minimum N by only {abs(diff)} respondents")
            print("   → Your constraints exist but don't significantly affect efficiency")
        elif diff > 0:
            print(f"\n⚠️  CONSTRAINTS MAKE IT HARDER")
            print(f"   Demographic constraints increase minimum N by {diff:,} respondents ({(diff/min_n_results['Priority Greedy']*100):.1f}%)")
            print("   → Constraints restrict the respondent pool and reduce allocation flexibility")
        else:
            print(f"\n✓  CONSTRAINTS HELP EFFICIENCY")
            print(f"   Demographic constraints decrease minimum N by {abs(diff):,} respondents")
            print("   → This is unexpected but shows constraints might focus allocation better")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
