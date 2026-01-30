# Survey Bundle Planner

A bundle-based survey allocation system that minimises respondent costs while meeting qualified sample targets with statistical confidence.

## Requirements

- Python 3.11+
- Install dependencies: `pip install -r requirements.txt`

## Objective

Minimise total surveyed respondents (cost) while still meeting every customer's required qualified sample size and protecting respondent attention/quality.

## Required Outcomes

- Each category must achieve ~200 qualified respondents per month (one customer = one category)

## Constraints

| Constraint | Description |
|------------|-------------|
| **Mean interview time** | Must be < 480 seconds (8 minutes) per respondent |
| **Per-category exposure representativeness** | For each category, respondents shown that category's qualifier must match national population distribution (gender, age, region) |
| **Qualifier time** | Assumed to be 0 seconds (simplification), but practical limit exists on qualifiers per respondent (respondent burden) |

## Policy Type (Static Planning)

- Survey design must be decided **before the month begins**
- Define survey structures/versions (bundles) and category allocations upfront
- Execution should **not** rely on mid-month adaptive restructuring
- Be a **pre-computed plan**


## Validation

Simulation-based evidence (using `fake_category_data.csv`) that the fixed monthly plan meets:
- ~200 qualified completes per category
- Mean time < 8 minutes
- Per-category exposure demographics are nationally representative
- Report success probabilities/rates across simulation runs

## Optional Design Extensions

| Feature | Description |
|---------|-------------|
| **Tail risk constraint** | Q95 interview time <= 720s (12 min) - prevents long-tail of burdened respondents |
| **Valid response rate** | Enforce minimum valid response rate based on business rules (e.g., exclude low-quality completes) |
| **Max qualifiers per respondent** | Max <= 10 qualifiers to reduce screening fatigue and maintain survey flow |
| **Incidence rate seasonality** | Allow IR to vary by month/season (e.g., Suncare lower in winter) via seasonality model |

---

## Policy Strategies

The optimiser creates three bundle strategies and selects the one with lowest respondent count that meets all constraints:

| Strategy | Description |
|----------|-------------|
| **Plan A: Bottleneck First** | Prioritises low-IR categories (hardest to fill) by sorting by required exposures |
| **Plan B: Efficiency First** | Prioritises high value-density categories (best demand/cost ratio) |
| **Plan C: Balanced Interleaving** | Alternates between bottleneck and stable categories for variance reduction |

Each strategy partitions all categories into bundles (survey versions) where:
- Each bundle respects max qualifiers (10) and expected time budget (480s)
- Bundle sizes are computed using Bonferroni-adjusted confidence to ensure P(all categories meet target) >= 95%

---

## How to Use

```bash
# Basic run (random demographics sampling)
python main.py

# Quota sampling - guarantees exact demographic match per bundle
python main.py --quota-sampling

# Custom demographic tolerance (default: 5%)
python main.py --demographic-tolerance 0.05

# Apply seasonality scenario
python main.py --seasonality winter

# Simulate data quality issues (90% valid completions)
python main.py --valid-completion 0.9

# More Monte Carlo runs for higher statistical precision
python main.py --monte-carlo-runs 20

# Combined options
python main.py --quota-sampling --seasonality winter
```

---

## Example Results

### Default (random sampling)
```bash
python main.py
```
**Result:** No plan meets 95% confidence due to demographic variance in small bundles. Consider using `--quota-sampling`.

### With quota sampling
```bash
python main.py --quota-sampling
```
**Result:** Plan B passes all constraints with lowest N = **8,341** respondents
- Output: `results/bundle_plan.csv`, `results/bundle_summary.csv`

### With seasonality (winter)
```bash
python main.py --quota-sampling --seasonality winter
```
**Result:** Plan B passes all constraints with lowest N = **8,261** respondents
- Output: `results/bundle_plan_winter.csv`, `results/bundle_summary_winter.csv`
- Suncare IR drops 50%, Coffee IR increases 25%, etc.

---

## Output Files

All outputs are saved to the `results/` folder:

| File | Description |
|------|-------------|
| `results/bundle_plan.csv` | Detailed plan with (bundle_id, category_id, planned_respondents, IR, survey_length) |
| `results/bundle_summary.csv` | One row per bundle with category list and respondent count |

---

## Project Structure

```
├── main.py           # Entry point and CLI
├── bundle_engine.py  # Bundle creation, sizing, simulation, Monte Carlo
├── data_loader.py    # Load category data with seasonality
├── reporting.py      # Output formatting and validation reports
├── config.py         # Configuration parameters
├── requirements.txt  # Python dependencies
├── fake_category_data.csv  # Input category data
├── results/          # Output folder for bundle plans
└── CHALLENGE.md      # Original challenge description
```
