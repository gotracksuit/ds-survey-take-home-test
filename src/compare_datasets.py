"""
Compare fake vs enriched category data to identify differences in incident rates
by gender and age.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load datasets
fake_df = pd.read_csv('data/fake_category_data.csv')
enriched_df = pd.read_csv('data/fake_category_data_enriched.csv')

print("="*80)
print("DATASET COMPARISON ANALYSIS")
print("="*80)
print(f"\nFake dataset shape: {fake_df.shape}")
print(f"Enriched dataset shape: {enriched_df.shape}")
print(f"\nNew columns in enriched: {set(enriched_df.columns) - set(fake_df.columns)}")

# ============================================================================
# 1. GENDER ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("GENDER ANALYSIS")
print("="*80)

gender_stats = enriched_df.groupby('gender_flag').agg({
    'incidence_rate': ['count', 'mean', 'median', 'std', 'min', 'max']
}).round(4)

print("\nIncident Rate Statistics by Gender:")
print(gender_stats)

# Statistical test: ANOVA for gender differences
gender_groups = [
    enriched_df[enriched_df['gender_flag'] == 'male']['incidence_rate'],
    enriched_df[enriched_df['gender_flag'] == 'female']['incidence_rate'],
    enriched_df[enriched_df['gender_flag'] == 'na']['incidence_rate']
]

f_stat, p_value = stats.f_oneway(*gender_groups)
print(f"\nANOVA Test (Gender):")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Significant difference: {'YES' if p_value < 0.05 else 'NO'}")

# Pairwise comparisons
print("\nPairwise t-tests (Gender):")
gender_pairs = [('male', 'female'), ('male', 'na'), ('female', 'na')]
for g1, g2 in gender_pairs:
    group1 = enriched_df[enriched_df['gender_flag'] == g1]['incidence_rate']
    group2 = enriched_df[enriched_df['gender_flag'] == g2]['incidence_rate']
    t_stat, p_val = stats.ttest_ind(group1, group2)
    print(f"  {g1} vs {g2}: t={t_stat:.4f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")

# ============================================================================
# 2. AGE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("AGE ANALYSIS")
print("="*80)

age_stats = enriched_df.groupby('age_flag').agg({
    'incidence_rate': ['count', 'mean', 'median', 'std', 'min', 'max']
}).round(4)

print("\nIncident Rate Statistics by Age:")
print(age_stats)

# Statistical test: ANOVA for age differences
age_groups = [
    enriched_df[enriched_df['age_flag'] == '18-64']['incidence_rate'],
    enriched_df[enriched_df['age_flag'] == '65+']['incidence_rate'],
    enriched_df[enriched_df['age_flag'] == 'na']['incidence_rate']
]

f_stat, p_value = stats.f_oneway(*age_groups)
print(f"\nANOVA Test (Age):")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Significant difference: {'YES' if p_value < 0.05 else 'NO'}")

# Pairwise comparisons
print("\nPairwise t-tests (Age):")
age_pairs = [('18-64', '65+'), ('18-64', 'na'), ('65+', 'na')]
for a1, a2 in age_pairs:
    group1 = enriched_df[enriched_df['age_flag'] == a1]['incidence_rate']
    group2 = enriched_df[enriched_df['age_flag'] == a2]['incidence_rate']
    t_stat, p_val = stats.ttest_ind(group1, group2)
    print(f"  {a1} vs {a2}: t={t_stat:.4f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")

# ============================================================================
# 3. INTERACTION EFFECTS
# ============================================================================
print("\n" + "="*80)
print("GENDER x AGE INTERACTION")
print("="*80)

interaction_stats = enriched_df.groupby(['gender_flag', 'age_flag']).agg({
    'incidence_rate': ['count', 'mean', 'median']
}).round(4)

print("\nIncident Rate by Gender x Age:")
print(interaction_stats)

# ============================================================================
# 4. TOP CATEGORIES BY DEMOGRAPHIC
# ============================================================================
print("\n" + "="*80)
print("TOP 5 HIGHEST INCIDENT RATE CATEGORIES BY DEMOGRAPHIC")
print("="*80)

for gender in ['male', 'female', 'na']:
    print(f"\n{gender.upper()} Categories:")
    top_cats = enriched_df[enriched_df['gender_flag'] == gender].nlargest(5, 'incidence_rate')[
        ['category_name', 'incidence_rate', 'age_flag']
    ]
    for idx, row in top_cats.iterrows():
        print(f"  - {row['category_name']}: {row['incidence_rate']:.4f} (Age: {row['age_flag']})")

print("\n65+ Age Group:")
age_65_plus = enriched_df[enriched_df['age_flag'] == '65+'][
    ['category_name', 'incidence_rate', 'gender_flag']
]
print(f"  Total categories: {len(age_65_plus)}")
if len(age_65_plus) > 0:
    for idx, row in age_65_plus.iterrows():
        print(f"  - {row['category_name']}: {row['incidence_rate']:.4f} (Gender: {row['gender_flag']})")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("Generating visualizations...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Gender comparison
ax1 = axes[0, 0]
enriched_df.boxplot(column='incidence_rate', by='gender_flag', ax=ax1)
ax1.set_title('Incident Rate Distribution by Gender', fontsize=14, fontweight='bold')
ax1.set_xlabel('Gender Flag')
ax1.set_ylabel('Incident Rate')
plt.sca(ax1)
plt.xticks(rotation=0)

# Plot 2: Age comparison
ax2 = axes[0, 1]
enriched_df.boxplot(column='incidence_rate', by='age_flag', ax=ax2)
ax2.set_title('Incident Rate Distribution by Age', fontsize=14, fontweight='bold')
ax2.set_xlabel('Age Flag')
ax2.set_ylabel('Incident Rate')
plt.sca(ax2)
plt.xticks(rotation=0)

# Plot 3: Gender x Age interaction
ax3 = axes[1, 0]
gender_age_means = enriched_df.groupby(['gender_flag', 'age_flag'])['incidence_rate'].mean().reset_index()
pivot_data = gender_age_means.pivot(index='age_flag', columns='gender_flag', values='incidence_rate')
pivot_data.plot(kind='bar', ax=ax3, rot=0)
ax3.set_title('Mean Incident Rate by Gender x Age', fontsize=14, fontweight='bold')
ax3.set_xlabel('Age Flag')
ax3.set_ylabel('Mean Incident Rate')
ax3.legend(title='Gender')

# Plot 4: Category length vs incident rate colored by gender
ax4 = axes[1, 1]
for gender in ['male', 'female', 'na']:
    subset = enriched_df[enriched_df['gender_flag'] == gender]
    ax4.scatter(subset['category_length_seconds'], subset['incidence_rate'],
               label=gender, alpha=0.6, s=50)
ax4.set_title('Incident Rate vs Category Length (by Gender)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Category Length (seconds)')
ax4.set_ylabel('Incident Rate')
ax4.legend(title='Gender')

plt.tight_layout()
plt.savefig('results/demographic_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved visualization to: results/demographic_comparison.png")

# ============================================================================
# 6. KEY INSIGHTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Calculate some key metrics
male_mean = enriched_df[enriched_df['gender_flag'] == 'male']['incidence_rate'].mean()
female_mean = enriched_df[enriched_df['gender_flag'] == 'female']['incidence_rate'].mean()
na_gender_mean = enriched_df[enriched_df['gender_flag'] == 'na']['incidence_rate'].mean()

age_18_64_mean = enriched_df[enriched_df['age_flag'] == '18-64']['incidence_rate'].mean()
age_65_plus_mean = enriched_df[enriched_df['age_flag'] == '65+']['incidence_rate'].mean()
na_age_mean = enriched_df[enriched_df['age_flag'] == 'na']['incidence_rate'].mean()

print("\n1. GENDER INSIGHTS:")
print(f"   - Male categories avg incident rate: {male_mean:.4f}")
print(f"   - Female categories avg incident rate: {female_mean:.4f}")
print(f"   - Gender-neutral categories avg incident rate: {na_gender_mean:.4f}")
diff_pct = abs(male_mean - female_mean) / min(male_mean, female_mean) * 100
print(f"   - Difference between male/female: {diff_pct:.1f}%")

print("\n2. AGE INSIGHTS:")
print(f"   - 18-64 categories avg incident rate: {age_18_64_mean:.4f}")
print(f"   - 65+ categories avg incident rate: {age_65_plus_mean:.4f}")
print(f"   - Age-neutral categories avg incident rate: {na_age_mean:.4f}")

print("\n3. ENRICHMENT VALUE:")
print(f"   - Categories with gender flag: {len(enriched_df[enriched_df['gender_flag'] != 'na'])} ({len(enriched_df[enriched_df['gender_flag'] != 'na'])/len(enriched_df)*100:.1f}%)")
print(f"   - Categories with age flag: {len(enriched_df[enriched_df['age_flag'] != 'na'])} ({len(enriched_df[enriched_df['age_flag'] != 'na'])/len(enriched_df)*100:.1f}%)")
print(f"   - Categories with both flags: {len(enriched_df[(enriched_df['gender_flag'] != 'na') & (enriched_df['age_flag'] != 'na')])} ({len(enriched_df[(enriched_df['gender_flag'] != 'na') & (enriched_df['age_flag'] != 'na')])/len(enriched_df)*100:.1f}%)")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
