# Survey Allocation Results – LP vs Greedy

**Author:** Nastaran Saffaryazdi  
**Date:** 8/12/2025  

---

## 1. Overview

This report summarizes the allocation of survey categories to respondents using two approaches:

1. **Linear Programming (LP)** – a planning-focused approach using expected values.  
2. **Greedy heuristic** – a practical, production-ready approach that sequentially assigns respondents to categories.  

**Goals:**

- Ensure ≥ 200 qualified respondents per category.  
- Keep the mean survey time per respondent ≤ 480 seconds.  
- Maintain approximate demographic representativeness.  

---

## 2. Total Respondents & Survey Time

| Method | Total Respondents | Total Expected Time (s) | Mean Expected Time (s) | Constraints Satisfied? |
|--------|-------------------|-------------------------|------------------------|------------------------|
| LP     | 40,879            | 1,675,588               | 40.99                  | ✅ (demographics relaxed)|
| Greedy | 9,683             | –                       | 479.99                 | ✅                      |

**Interpretation:**

- LP overestimates total respondents because it works at the category level and does not mix categories per respondent.  
- Greedy efficiently mixes categories, using fewer respondents while meeting all constraints.  

---

## 3. Key Observations

1. **LP Insights for Planning**  
   - Identifies high-risk or “expensive” categories that require more respondents.  
   - Helps plan for resource allocation and potential bottlenecks.  
   - Provides a baseline for expected costs if categories are not mixed efficiently.  

2. **Greedy Insights for Production**  
   - Significantly reduces the number of respondents needed (9,683 vs 40,879).  
   - Mean survey time per respondent is within the 480s limit.  
   - Works well for daily allocation in a live production environment.  

---

## 4. Monte Carlo Validation & Risk Assessment

While the LP solution satisfies all constraints **in expectation**, real-world survey qualification is stochastic. To evaluate operational risk, **Monte Carlo simulations** were performed by repeatedly simulating respondent qualification as Bernoulli trials using each category’s incidence rate.

Monte Carlo results in average:

- Mean qualified completes: **≈200.4**  
- 5th percentile: **≈180**  
- Median: **≈200**  
- 95th percentile: **≈220**  
- Probability of missing target (< 200): **≈49%**

### Interpretation

- Although the expected number of qualified respondents is ≈200, almost **half of simulated months fail** to meet the contractual target.  
- This reveals a **high-variance, high-risk categories** where planning to exactly 200 is statistically fragile.  
- The LP solution alone does not expose this risk because it optimizes **expected values**, not outcome distributions.

### Planning Implication

- To reduce the probability of failure to an acceptable level (~5%), a **buffered planning target** is required.  
- Based on variance, planning for approximately **220 expected qualified completes** (~10% buffer) would reduce failure probability from ~49% to ~5%.  


## 5. Recommendations

- **Use Greedy allocation for production deployments:** efficient, meets constraints, fewer respondents.  
- **Use LP + Monte Carlo for planning and risk assessment:** helps understand fragile categories, category-level cost forecasts, and potential bottlenecks.  
- **Introduce category-specific buffers** for high-variance categories.  
- **Monitor survey times and category incidence:** adjust heuristics dynamically to maintain efficiency.  
- **Future improvement:** consider individual-level LP to further optimize respondent allocation in complex survey scenarios.  

---

## 6. Summary

| Aspect                 | LP                     | Greedy |
|------------------------|-----------------------|--------|
| Respondent Count       | High (overestimate)    | Low (efficient) |
| Mean Survey Time       | 41s (per category)     | 480s (per respondent) |
| Constraints            | ✅ (demographics relaxed) | ✅ |
| Production Suitability | Planning / Stress-test | Daily allocation / Production |

**Conclusion:** LP provides strategic insights and risk assessment, while Greedy is optimized for operational efficiency. Monte Carlo validation shows the risk in low-incidence categories and helps define safe planning buffers. Both methods complement each other to ensure cost-effective and constraint-compliant survey allocation.
