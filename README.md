## Overview

Tracksuit’s value depends on one thing: **reliable measurement at sustainable cost**.
To honour customer quotas, protect data quality, maintain respondent engagement, and preserve demographic integrity, the survey engine must decide—thousands of times per month—**which respondent should see which category qualifier**.

This repository contains an end-to-end simulation and allocation algorithm that:

* Delivers ~200 qualified respondents per category per month
* Keeps the average interview length below the 480-second contractual and quality threshold
* Maintains demographic representativeness (gender, age, region)
* Minimises total respondents required (the core cost driver)

The solution is implemented as a simulation environment plus an allocation policy. The aim is not theoretical optimality but a **pragmatic, interpretable system** that performs well under uncertainty and mirrors the constraints of a real routing engine.

All code is in `allocator.ipynb`, which can be executed to reproduce the full analysis.

---

## Method 

### Data

The solution uses the provided `fake_category_data.csv`, which includes:

* `category_id`
* `category_name`
* `incidence_rate` (probability of qualification)
* `category_length_seconds` (LOI)

Additional fields are assigned:

* target quotas (≈200)
* target demographic distributions (global or category-specific)
* simple eligibility rules (e.g., IVF, Baby, Retiree, Car Rental)

This produces a realistic catalogue of categories for simulation.

---

### Synthetic Respondent Stream

Respondents are simulated with attributes:

* `gender` ∈ {male, female}
* `age_band` ∈ {18–34, 35–54, 55+}
* `region` ∈ {north, central, south}

For the prototype, demographics are sampled **uniformly** to isolate allocator behaviour without confounding from real panel supply.
In production, this would be replaced with empirical panel distributions.

---

### Allocation Algorithm

The allocator operates **online**, making routing decisions for each respondent in real time.

Each respondent receives a 480-second time budget. While time remains:

1. Identify eligible categories (gender/age/region rules).
2. Filter categories with remaining quota.
3. Filter categories whose LOI fits within the remaining time.
4. Score remaining categories via a multi-objective priority function:

```
priority = quota_urgency × incidence × time_fit × demographic_weight
```

Where:

* **quota_urgency** prioritises categories behind target
* **incidence** approximates expected value of showing the qualifier
* **time_fit** enforces the LOI constraint
* **demographic_weight** nudges exposure toward target demographic distributions

The algorithm selects the highest-priority category, simulates qualification (Bernoulli), and if the respondent qualifies, deducts LOI and records a completed response.

This loop continues until time is exhausted or no viable categories remain.
The result is an interpretable policy that mirrors real survey-routing behaviour.

---

## Validation

### Quota Delivery

* With N ≈ 5,000 respondents, all categories reliably hit their quota.
* Errors remain near zero across Monte-Carlo runs.

### Interview Length Constraint

* No respondent exceeds the 480-second limit (hard constraint).
* Mean LOI remains well below this threshold.

### Demographic Balance

For each category and each dimension (gender, age, region):

* L1 divergence is computed between screened and target distributions.
* Gender and age divergence are low; region divergence is higher due to uniform supply (33/33/34) vs target (40/20/40).
* Demonstrates demographic weighting works and illustrates the cost/benefit of fairness.

### Robustness to Incidence Mis-specification

Using incidence scaling factors {0.8, 1.0, 1.2}:

* At N ≈ 5,000, success probability remains above 90%.
* Confirms allocator stability under realistic incidence uncertainty.

---

## Baseline Comparison

A naïve allocator is implemented for benchmarking:

* Randomly selects an eligible category with remaining quota
* Ignores incidence, urgency, and demographic balance

Monte-Carlo evaluation shows:

* The greedy allocator requires **~20–25% fewer respondents** to match the naïve success probability
* Demonstrates the “outsized performance” requested in the brief

---

## Design Rationale & Trade-offs

### Why a Greedy Online Policy?

Tracksuit’s routing environment is online and stochastic. A greedy policy:

* Is easy to interpret and debug
* Adapts naturally to quota gaps in real time
* Performs well empirically under uncertainty
* Avoids the opacity and operational overhead of ILPs or RL agents

The aim was **pragmatic reliability**, not mathematical abstraction.

---

### Simplifications for the Prototype

These decisions keep the prototype focused on the allocation logic rather than full ecosystem complexity.

| Area                   | Simplification                        | Consequence                                          |
| ---------------------- | ------------------------------------- | ---------------------------------------------------- |
| Respondent supply      | Uniform sampling                      | Avoids confounding; does not reflect real panel skew |
| Completion times       | Deterministic                         | Real respondents vary; conservative approximation    |
| Completion probability | 100% conditional on qualifying & time | Ignores abandonment/drop-off                         |
| Fairness metric        | L1 divergence                         | Transparent but not statistically formal             |
| Quota horizon          | Single month                          | No multi-wave carry-over or smoothing                |

---

## Limitations & Future Extensions

The prototype solves the core routing problem cleanly, but a production-grade system would extend across several dimensions. The goal is not complexity for its own sake, but robustness under real-world variability.

### A. Real Panel-Supply Modelling

Replace uniform sampling with:

* empirical demographic distributions
* source-level behavioural differences
* time-of-day/day-of-week patterns
* panel fatigue and response-propensity modelling

### B. Bayesian Incidence Updating

* Maintain posterior incidence distributions
* Update in real time as qualification data arrives
* Propagate uncertainty into routing decisions
* Adjust urgency dynamically

### C. Stochastic LOI & Dropout

* Category-specific completion-time distributions
* Respondent-level speed variation
* Probability of abandonment
* LOI-risk-aware routing

### D. Multi-Wave Quota Smoothing

* Track quota debt/credit across waves
* Avoid hard resets
* Deliver consistent customer sample even with incidence volatility

### E. ILP Benchmarking

* Build small ILPs to benchmark greedy behaviour
* Understand failure modes and efficiency gaps

### F. Engineering Hardening

* Modular Python packages
* Unit tests
* Pre-computed eligibility/feasibility checks
* Profiling and vectorisation
* Feature-flagged policy components

### G. Monitoring & Drift Detection

* Continuous incidence drift tracking
* Demographic deviation alerts
* LOI-risk monitoring
* Automatic recalibration of weights or constraints

---

## How to Run

1. Open `allocator.ipynb`
2. Run all cells in order

The notebook will:

* load data
* simulate respondents
* run the allocation algorithm
* validate constraints
* run robustness analyses
* compare performance to the naïve baseline

All results in this README are reproducible.

---

## Summary

I framed this as an **online routing problem**: a stream of respondents, each with gender, age, and region, and a goal to route each one so that by the end of the month:

* every category gets its ~200 completes
* the average respondent stays under 8 minutes
* the screened population resembles the national population
* and the total respondent cost is minimised

The allocator scores categories based on quota urgency, incidence, time feasibility, and demographic balance. It selects the best option, updates quotas and LOI, and proceeds until no useful actions remain.

Validation is done through simulation and Monte-Carlo estimation of success probability under incidence uncertainty. The allocator consistently meets constraints and uses **20–25% fewer respondents** than a naïve policy.

The trade-off is realism vs complexity: I chose a simple, interpretable online policy that performs strongly and is easy to reason about. With more time, I’d extend respondent modelling, make incidence adaptive, and add more sophisticated fairness and cost metrics. But the core logic—**how do we decide who sees what, under these constraints?**—is implemented, validated, and robust.

---

## Final Notes

This solution focuses on decision quality, not code theatrics. It provides:

* A realistic allocator
* Transparent logic
* Clear validation
* Quantified robustness
* Measurable improvement over a naïve baseline
* Honest articulation of assumptions and limitations

It reflects how I think about measurement, resource allocation, uncertainty, and system constraints—the foundation of Tracksuit’s survey engine.

---

