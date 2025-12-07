# Tracksuit Survey Allocation Take-Home

**Author:** [Nastaran]  
**Date:** [8/12/2025]  

---

## Overview

This repository provides a solution to the Tracksuit Data Scientist take-home task [problem.md](problem.md).  
The goal is to allocate survey categories to respondents while:

1. Achieving ≥ 200 qualified respondents per category.
2. Ensuring mean survey time per respondent ≤ 480 seconds (8 minutes).
3. Maintaining demographic representativeness across gender, age, and region.

Solution Architecture: Hybrid Planning and Execution
For this task, I approached the problem using a combination of **Linear Programming (LP), greedy allocation, and Monte Carlo validation**. The LP optimizes the expected number of respondents per category-demographic cell to meet the target qualified completes while minimizing total respondents, ensuring expected demographic representativeness and average survey time constraints. The greedy heuristic provides a simpler, sequential allocation alternative that is easier to implement and works with integer respondents but may slightly over- or under-allocate. Monte Carlo validation then simulates the stochastic nature of qualification, showing how often the LP allocation actually meets the targets and survey time in practice. Together, this approach allows me to justify the allocation mathematically, check robustness under randomness, and demonstrate a practical allocation strategy suitable for real-world deployment.

The solution includes:

- **LP optimizer** for expected-value allocation.
- **Monte Carlo validation** for stochastic outcomes.
- **Greedy allocator** for comparison.
- **Validation module** to check all constraints.

---

## Files / Modules

- `main.py`  
  Orchestrates the workflow: load data, run optimizer, validate results, and run Monte Carlo validation.

- `optimizer/`  
  - `optimizer.py`: solves the problem in two ways:
                        - Eexpected-value Linear Program allocation
                        - Sequential greedy heuristic

- `validation/`  
  - checks if optimizers meets qualified targets, mean survey time, and demographic representativeness.

- `monte_carlo_validation/`  
  - Performs stochastic simulations using Bernoulli trials per category.

- `demographic/`  
  - `load_demographics.py`: generates demographic cells and population shares.

- `config.py`  
  Stores constants such as `TARGET_QUALIFIED`, `MAX_MEAN_TIME`, and demographic distributions.

---

## Assumptions

1. Incidence rates are assumed constant across demographics.  
2. Each respondent may answer multiple categories; the **total survey time** should not exceed 480 seconds.  
3. Demographic representativeness is **relaxed** for simplicity; exposure roughly follows national shares.  
4. Category qualifier time is negligible (0 seconds).  
5. LP allocation is treated as **expected-value**; Monte Carlo validates stochastic outcomes.

---

## Workflow

1. Load category data (`fake_category_data.csv`) and demographic cells.  
2. Solve expected-value LP allocation to **minimize total respondents**:
   - Ensure expected qualified completes ≥ `TARGET_QUALIFIED`.
   - Optionally include demographic weights (relaxed).
   - Ensure mean survey time ≤ `MAX_MEAN_TIME`.
3. Validate allocation using `validate_solution.py`:
   - **Proof 1:** Qualified Target (≥200).  
   - **Proof 2:** Mean Time Constraint (≤480s).  
   - **Proof 3:** Demographic Representativeness (within tolerance).
4. Run `monte_carlo_validate.py` to simulate stochastic qualification:
   - Check mean qualified ± std deviation per category.
   - Validate expected mean survey time.
   - Assess probability of meeting all constraints.
5. Run Greedy allocation
6. Validate greedy allocation

---

## Interpretation of Results

- LP allocation ensures that each category meets its target qualified respondents while respecting the mean survey time constraint.  
- Monte Carlo provides stochastic proof that targets are met with high probability.  
- The greedy allocator provides a practical alternative for comparison and can be used in production when dynamic, real-time allocation of respondents is required.  
- **Relaxed demographic assumption:** exact quota per cell is not strictly enforced, but exposure roughly matches national distribution.


---

## Installation

This project uses **Poetry** for dependency management.

1. Install Poetry (if not already installed):

```bash
# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

    Install dependencies:


poetry install


## How to Run

1. Configure demographic shares in `config.py`.  
2. Ensure `fake_category_data.csv` is available in the working directory.  
3. Run `main.py`:

```bash
poetry install
poetry run python main.py
