# How to Run Each Script

This document explains how to run each script in the project and what they do.

## Project Structure

```
ds-survey-take-home-test/
├── data/                           # Data files
│   ├── fake_category_data.csv           # Original dataset (no demographics)
│   └── fake_category_data_enriched.csv  # Dataset with gender/age flags
├── src/                            # Core modules
│   ├── allocation_policies.py           # 3 allocation algorithms
│   ├── data_loader.py                   # Load data & generate respondents
│   ├── simulation.py                    # Simulate survey qualifications
│   └── compare_datasets.py              # Compare enriched vs original data
├── experiments/                    # Experiment scripts
│   ├── run_simulation.py                # CLI tool for running allocations
│   ├── experiment.py                    # Find minimum N for each policy
│   ├── monte_carlo_experiment.py        # Monte Carlo with visualizations
│   ├── demographic_impact_analysis.py   # Analyze demographic constraint impact
│   └── ...
└── results/                        # Output directory
```

## Prerequisites

All scripts use Poetry for dependency management. Make sure you have dependencies installed:

```bash
poetry install
```

---

## Core Modules (src/)

### 1. `src/compare_datasets.py` - Dataset Comparison

**What it does:**
- Compares the original vs enriched datasets
- Analyzes incident rates by gender and age
- Statistical tests (ANOVA, t-tests) for significance
- Creates visualizations showing demographic differences

**How to run:**
```bash
poetry run python src/compare_datasets.py
```

**Output:**
- Console: Statistical analysis summary
- `results/demographic_comparison.png`: 4-panel visualization

**Key findings:**
- Shows if gender/age flags correlate with incident rates
- Identifies which demographics have higher/lower rates
- Tests if differences are statistically significant

---

## Experiment Scripts (experiments/)

### 2. `experiments/run_simulation.py` - Quick Simulation CLI

**What it does:**
- Command-line tool for running a single simulation
- Tests one allocation policy at a time
- Quick way to test different parameters

**How to run:**

Basic usage (default: demographic_aware, N=8000):
```bash
poetry run python experiments/run_simulation.py
```

With specific policy:
```bash
poetry run python experiments/run_simulation.py --policy random --n 5000
```

With multiple iterations (for averaging):
```bash
poetry run python experiments/run_simulation.py --policy priority_greedy --n 8000 --iterations 10
```

**Options:**
- `--policy`: `random`, `priority_greedy`, or `demographic_aware`
- `--n`: Number of respondents (default: 8000)
- `--iterations`: Number of runs to average (default: 1)
- `--seed`: Random seed (default: 42)

**Output:**
```
======================================================================
SURVEY ALLOCATION SIMULATION
======================================================================
Policy: demographic_aware
Respondents: 8,000
Iterations: 1

======================================================================
RESULTS
======================================================================
Min Qualified:        195
Max Qualified:        312
Mean Qualified:       248.5
Mean Time:            387.2s
Max Time:             479.8s
% Categories ≥ 200:   96.1%
Categories < 200:     3
```

**Use cases:**
- Quick test of a policy at specific N
- Debugging allocation algorithms
- Parameter tuning

---

### 3. `experiments/experiment.py` - Find Minimum Respondents

**What it does:**
- Binary search to find minimum N where each policy meets ALL constraints
- Constraints: ≥200 qualified per category AND mean time ≤480s
- Compares all 3 policies to determine winner
- Multiple trials per N to account for randomness

**How to run:**
```bash
poetry run python experiments/experiment.py
```

**Output:**
```
======================================================================
Finding minimum N for: Random
======================================================================
Testing N = 8,500... Success: 100%, Min Qualified: 212, Mean Time: 456.3s
Testing N = 7,500... Success: 80%, Min Qualified: 198, Mean Time: 432.1s
...
✅ Minimum N found: 8,000

======================================================================
FINAL RESULTS
======================================================================
                          min_n  success_rate  mean_min_qualified  mean_time
Demographic Aware          3800           1.0               205.2      442.3
Priority Greedy            4100           1.0               208.7      451.8
Random                     8200           1.0               201.3      467.9

🏆 WINNER: Demographic Aware
   Minimum Respondents: 3,800
```

**Key metrics:**
- **min_n**: Minimum respondents needed
- **success_rate**: Fraction of trials that met constraints
- **mean_min_qualified**: Average of lowest qualified category across trials
- **mean_time**: Average survey time

**Use cases:**
- Determine optimal sample size
- Compare policy efficiency
- Cost estimation (fewer respondents = lower cost)

---

### 4. `experiments/monte_carlo_experiment.py` - Monte Carlo Analysis

**What it does:**
- Runs 30 simulations per policy across N range (2,000 - 10,000)
- Creates a clean interactive chart showing constraint satisfaction vs sample size
- Shows mean performance with error bars (standard deviation)

**How to run:**
```bash
poetry run python experiments/monte_carlo_experiment.py
```

**Configuration (edit in script if needed):**
```python
n_min = 2000      # Minimum respondents to test
n_max = 10000     # Maximum respondents to test
step_size = 500   # Step between N values
n_sims = 30       # Number of simulations per N
```

**Output:**
- `results/monte_carlo_constraint_satisfaction.html`
  - Interactive Plotly chart
  - Shows % of categories meeting ≥200 qualified target
  - Mean line with error bars for each policy
  - Target line at 100%

**Runtime:** ~3-5 minutes

**Use cases:**
- Visualize how each policy performs across sample sizes
- Identify the minimum N where policies reach 100% success
- Compare policy reliability (look at error bar sizes)
- See which policy converges to 100% fastest

---

### 5. `experiments/demographic_impact_analysis.py` - Demographic Constraint Analysis

**What it does:**
- Compares Priority Greedy (ignores demographics) vs Demographic Aware (respects demographics)
- Measures whether demographic constraints help or hurt efficiency
- Identifies which categories are most affected by constraints
- Pool depletion analysis (do we have enough of each demographic?)

**How to run:**
```bash
poetry run python experiments/demographic_impact_analysis.py
```

**Output:**

1. **Console analysis:**
   - Constraint impact per category
   - Categories most hurt/helped by constraints
   - Pool composition and demand analysis
   - Minimum N comparison
   - Final verdict: Do constraints matter?

2. **Visualization:**
   - `results/demographic_impact_analysis.png`
     - 4 panels showing constraint effects
     - Box plots by constraint type
     - Scatter plots of incidence vs impact

**Example output:**
```
======================================================================
CONSTRAINT IMPACT PER CATEGORY
======================================================================

Impact by Constraint Type:
Constraint Type      Count    Avg Qualified Diff   Avg % Change
----------------------------------------------------------------------
None                 54       +12.3                +2.4%
Gender Only          3        -8.7                 -3.1%
Age Only             26       -2.1                 -0.8%
Both                 15       -5.4                 -2.0%

Categories MOST HURT by demographic constraints:
  Men's Online Health Providers (male, 18-64) | PG: 218 → DA: 203 (-6.9%)
  Period Products (female, 18-64)            | PG: 225 → DA: 214 (-4.9%)

======================================================================
VERDICT: DO DEMOGRAPHIC CONSTRAINTS MATTER?
======================================================================

⚠️  CONSTRAINTS MAKE IT HARDER
   Demographic constraints increase minimum N by 300 respondents (7.3%)
   → Constraints restrict the respondent pool and reduce allocation flexibility
```

**Use cases:**
- Validate that enriched demographic data is useful
- Identify problematic constraints (e.g., 65+ pool depletion)
- Justify demographic targeting to stakeholders
- Optimize constraint design

---

## Allocation Policies Explained

All experiments test 3 allocation policies (defined in `src/allocation_policies.py`):

### 1. **Random Allocation**
- Randomly assigns categories to respondents
- Only constraint: time budget (≤480s)
- Baseline / worst-case performance

### 2. **Priority Greedy Allocation**
- Prioritizes hardest categories first (lowest incidence rate)
- Fills each category to target before moving to next
- Ignores demographic constraints
- Better than random

### 3. **Demographic Aware Allocation**
- Like Priority Greedy BUT respects demographic targeting
- Only assigns male categories to male respondents, etc.
- Uses enriched data (gender_flag, age_flag)
- Best performance (usually)

---

## Typical Workflow

### Phase 1: Data Validation
```bash
# Compare enriched vs original data
poetry run python src/compare_datasets.py
```
**Goal:** Verify demographic flags are meaningful

### Phase 2: Quick Tests
```bash
# Test each policy at N=8000
poetry run python experiments/run_simulation.py --policy random --n 8000
poetry run python experiments/run_simulation.py --policy priority_greedy --n 8000
poetry run python experiments/run_simulation.py --policy demographic_aware --n 8000
```
**Goal:** Sanity check all policies work

### Phase 3: Find Optimal N
```bash
# Find minimum N for each policy
poetry run python experiments/experiment.py
```
**Goal:** Determine sample size requirements

### Phase 4: Deep Analysis
```bash
# Monte Carlo visualization (takes ~3-5 min)
poetry run python experiments/monte_carlo_experiment.py

# Demographic constraint impact
poetry run python experiments/demographic_impact_analysis.py
```
**Goal:** Visualizations and detailed insights

---

## Common Parameters

All scripts share these concepts:

- **N**: Number of respondents (sample size)
- **Target Qualified**: 200 qualified respondents per category
- **Time Budget**: 480 seconds max average survey time
- **Safety Margin**: 0.75 (use only 75% of budget to account for variance)
- **Seed**: Random seed for reproducibility (default: 42)

---

## Troubleshooting

### "No module named X"
```bash
poetry install
```

### Scripts run but no output files
Check that `results/` directory exists:
```bash
mkdir -p results
```

### Monte Carlo takes too long
Already optimized to ~3-5 minutes. If needed, reduce parameters in the script:
```python
n_sims = 20   # Instead of 30
n_max = 8000  # Instead of 10000
```

### Different results each run
Set consistent seed:
```bash
poetry run python experiments/run_simulation.py --seed 42
```

---

## Quick Reference

| Script | Runtime | Output Type | Use Case |
|--------|---------|-------------|----------|
| `compare_datasets.py` | ~1s | Stats + PNG | Validate demographic data |
| `run_simulation.py` | ~1s | Console | Quick policy test |
| `experiment.py` | ~30s | Console table | Find minimum N |
| `monte_carlo_experiment.py` | ~3-5min | Interactive HTML | Visualize performance curves |
| `demographic_impact_analysis.py` | ~2min | Console + PNG | Constraint impact analysis |

---

## Key Questions Each Script Answers

### compare_datasets.py
❓ *"Do the demographic flags correlate with incident rates?"*
✅ Yes! Gender-neutral categories have highest rates (0.53), male lowest (0.29)

### run_simulation.py
❓ *"How does this policy perform at N=8000?"*
✅ Quick answer: 96% categories met, mean time 387s

### experiment.py
❓ *"What's the minimum sample size we need?"*
✅ Demographic Aware needs only 3,800 respondents vs 8,200 for Random

### monte_carlo_experiment.py
❓ *"How do policies perform across different sample sizes?"*
✅ Interactive chart shows constraint satisfaction curves with error bars across 30 runs

### demographic_impact_analysis.py
❓ *"Do demographic constraints help or hurt efficiency?"*
✅ Detailed category-level analysis + verdict on constraint usefulness
