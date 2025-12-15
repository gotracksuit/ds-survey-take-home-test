# Survey Allocation Optimizer

*Tracksuit Data Scientist Take-Home Submission*

## The Challenge

We need to get 200 qualified respondents for each of 77 different survey categories, while keeping the average survey time under 8 minutes. Main challenge is that each category has a different incidence rate (how many people qualify) and different survey lengths. Plus, we want to minimize the total number of respondents we survey (because that's our cost).

## The Approach

1. **Building 3 allocation algorithms** - from simple random baseline to a demographic-aware smart allocator (see `src/allocation_policies.py`)
2. **Enriching the data** - added gender and age targeting flags to categories (because "Men's Clothing" shouldn't be shown to everyone)
3. **Running experiments** - Some Monte Carlo simulations, statistical tests, and optimization to find the minimum sample size needed
4. **Validating everything** - made sure all constraints are met and the results are reproducible

The math behind it (for the curious):
- **Input**: 77 categories with incidence rates $r_i$ and survey lengths $l_i$
- **Constraints**: Mean survey time $leq$ 480s, $geq$ 200 qualified per category, demographically representative
- **Objective**: Minimize total number of respondents while meeting all constraints

A more complete problem formulation in `FORMULATION.md` if you want the deep dive.

I mainly used Claude Code to help me with the code and it took ~2-3 hrs of working time.

## Project Structure

```
├── src/                              # Core modules
│   ├── allocation_policies.py        # The 3 allocation algorithms
│   ├── data_loader.py                # Data loading & respondent generation
│   ├── simulation.py                 # Qualification simulation logic
│   ├── compare_datasets.py           # Statistical analysis of demographic enrichment
│   └── bounds_calculation.py         # Theoretical bounds (optional utility)
│
├── experiments/                      # Experiment scripts
│   ├── run_simulation.py             # CLI for quick tests
│   ├── experiment.py                 # Find minimum N (binary search)
│   ├── monte_carlo_experiment.py     # Monte Carlo sims with beautiful plots
│   └── demographic_impact_analysis.py # Deep dive on demographic constraints
│
├── data/                             # Datasets
│   ├── fake_category_data.csv        # Original data
│   └── fake_category_data_enriched.csv # With gender/age flags
│
├── results/                          # Output visualizations
│   ├── demographic_comparison.png
│   └── demographic_impact_analysis.png
│
└── HOW_TO_RUN.md                     # Detailed guide for each script
```

## Quick Start

We use Poetry to manage dependencies.

```bash
# Install dependencies
poetry install

# Run a quick test
poetry run python experiments/run_simulation.py --policy demographic_aware --n 8000
```

## Running the Experiments

You can run the experiments by running the following commands:

### 1. Compare Datasets (Validate Demographics)
```bash
poetry run python src/compare_datasets.py
```
**What it does:** Statistical analysis checking if gender/age flags meaningfully correlate with incident rates

### 2. Quick Simulation (Test a Policy)
```bash
poetry run python experiments/run_simulation.py --policy demographic_aware --n 8000
```
**What it does:** Run a single policy with custom parameters.

### 3. Find Minimum N (Optimization)
```bash
poetry run python experiments/experiment.py
```
**What it does:** Binary search to find the minimum respondents needed for each policy. This is where we see the winner.

### 4. Monte Carlo Analysis (Data Visualizations)
```bash
poetry run python experiments/monte_carlo_experiment.py
```
**What it does:** 50 simulations across different sample sizes (2,000-12,000). Generates charts showing performance curves and variance. Takes ~5-10 minutes but worth it!

### 5. Demographic Impact Analysis (Deep Dive)
```bash
poetry run python experiments/demographic_impact_analysis.py
```
**What it does:** Compares allocation with/without demographic constraints. Shows which categories are helped or hurt by targeting, and whether we have pool depletion issues.

## The 3 Allocation Algorithms

I built and compared three strategies (all implemented in `src/allocation_policies.py`):

### 1. Random Allocation (Baseline)
Your basic "shuffle and hope" approach. Randomly assigns categories to respondents, respecting only the time budget. This is our worst-case baseline to show how much smart allocation helps.

### 2. Priority Greedy
Prioritizes the hardest categories first (those with the lowest incidence rates need the most exposures). 

Uses a two-pass approach:

1. Fill each category to target, starting with hardest
2. Fill remaining respondent time with any categories that fit

This is deterministic and efficient, but it ignores demographic targeting.

### 3. Demographic Aware (Recommended ⭐)

Same priority logic as Priority Greedy, but respects demographic targeting:

- Only shows male-targeted categories to male respondents
- Only shows female-targeted categories to female respondents
- Respects age targeting (18-64 vs 65+)

This makes intuitive sense (don't show "Men's Clothing" to women) and performs better than the other two algorithms.

## Results

Running the experiments reveals the following results:

**On Demographics:**
- Gender and age flags DO matter - statistically significant differences in incident rates (run the `src/compare_datasets.py` script to see the details)
- Gender-neutral categories have highest incident rates (0.53 avg) -- unsurprisingly
- Male-targeted categories have lowest rates (0.29 avg) -- matches intuition

**On Algorithm Performance:**
- Demographic Aware needs fewer respondents than Priority Greedy
- Random allocation is significantly worse (needs ~2x more respondents)
- Monte Carlo simulations show reliable performance across multiple runs

**On Constraints:**
- All policies can meet the 200 qualified per category target as we increase the number of respondents (obviously)
- Time budget (480s) is the binding constraint for most policies 
- Demographic constraints slightly increase sample size needed but improve targeting quality (see `experiments/demographic_impact_analysis.py` for the details)

See the experiment outputs and visualizations in `results/` for detailed metrics.

---

## Documentation

- **HOW_TO_RUN.md** - Detailed walkthrough of each script with examples
- **FORMULATION.md** - Mathematical problem formulation
- **man/README.md** - Original assignment specification