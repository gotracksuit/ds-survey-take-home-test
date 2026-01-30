"""
Bundle-based allocation engine for static survey version planning.

This module implements a fundamentally different approach than the dynamic
allocation in policy_engine.py. Instead of adapting what categories to show
based on current quotas, we pre-compute fixed "survey bundles" (versions)
where each bundle contains a static set of category qualifiers.

Key concepts:
- Bundle: A fixed set of categories shown to all respondents assigned to that bundle
- BundlePlan: A complete allocation plan with multiple bundles and respondent counts
- Planning is done BEFORE any data collection (no quota adaptation)
- Validation uses Monte Carlo to prove P(success) >= TARGET_CONFIDENCE

The main advantage is operational simplicity: each survey version is fixed,
making it easier to implement, audit, and ensure demographic representativeness.

Bonferroni Confidence Adjustment:
---------------------------------
To ensure that P(ALL categories meet target) >= TARGET_CONFIDENCE, we use a
Bonferroni-style correction. If we have K categories total, we set each
per-category confidence to:
    conf_cat = 1 - (1 - TARGET_CONFIDENCE) / K

This is conservative (overshoots slightly) but guarantees the joint probability.
For K=78 categories and TARGET_CONFIDENCE=0.95:
    conf_cat ≈ 1 - 0.05/78 ≈ 0.99936

This means we size each category to have ~99.94% individual success probability.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter

import config


@dataclass
class Bundle:
    """
    A survey bundle (version) - a fixed set of categories shown to all assigned respondents.
    
    Attributes:
        bundle_id: Unique identifier for this bundle
        categories: List of category IDs included in this bundle
        planned_respondents: Pre-computed number of respondents to assign to this bundle
        expected_time: Expected interview time in seconds (sum of IR*length for each category)
    """
    bundle_id: int
    categories: List[int]
    planned_respondents: int = 0
    expected_time: float = 0.0

    @property
    def num_categories(self) -> int:
        return len(self.categories)


@dataclass
class BundlePlan:
    """
    A complete bundle-based allocation plan.
    
    Contains all survey bundles with their category assignments and planned
    respondent counts. This is the output of the planning phase, computed
    BEFORE any data collection begins.
    
    Attributes:
        name: Human-readable plan name (e.g., "Strategy A: Bottleneck First")
        description: Explanation of the strategy used to create bundles
        bundles: List of all survey bundles
        category_data: Reference to category data for lookups
    """
    name: str
    description: str
    bundles: List[Bundle]
    category_data: pd.DataFrame = field(repr=False)

    @property
    def total_planned_respondents(self) -> int:
        return sum(b.planned_respondents for b in self.bundles)

    @property
    def total_categories(self) -> int:
        return sum(b.num_categories for b in self.bundles)

    @property
    def num_bundles(self) -> int:
        return len(self.bundles)

    def get_bundle_for_category(self, category_id: int) -> Optional[Bundle]:
        """Find which bundle contains a given category."""
        for bundle in self.bundles:
            if category_id in bundle.categories:
                return bundle
        return None


@dataclass
class BundleSimulationResult:
    """
    Results from simulating a BundlePlan with fixed N.
    
    Unlike SimulationResult which runs until quotas are met, this uses
    the fixed planned_respondents counts from the BundlePlan.
    
    Includes per-category exposure tracking for demographic validation.
    """
    plan_name: str
    total_respondents: int
    category_completions: Dict[int, int]
    category_exposures: Dict[int, int]  # How many saw each category's qualifier
    category_exposure_demographics: Dict[int, Dict[str, Counter]]  # Per-category demo counts
    interview_times: List[float]
    all_quotas_met: bool
    mean_time_ok: bool
    demographics_ok: bool

    @property
    def mean_interview_time(self) -> float:
        return float(np.mean(self.interview_times)) if self.interview_times else 0.0

    @property
    def q95_interview_time(self) -> float:
        return float(np.percentile(self.interview_times, 95)) if self.interview_times else 0.0

    @property
    def min_category_completions(self) -> int:
        return min(self.category_completions.values()) if self.category_completions else 0

    @property 
    def all_constraints_met(self) -> bool:
        return self.all_quotas_met and self.mean_time_ok and self.demographics_ok


# =============================================================================
# SAMPLE SIZE CALCULATION WITH CONFIDENCE
# =============================================================================

def required_n_for_confidence(
    p: float,
    target: int,
    conf: float,
    rng_seed: int = 42,
    num_mc_samples: int = 5000
) -> int:
    """
    Find minimum n such that P(Binomial(n, p) >= target) >= conf.
    
    Uses Monte Carlo approximation with efficient search:
    1. Start at theoretical minimum ceil(target/p)
    2. Step by 50 until success probability exceeds conf
    3. Binary search to find minimum n
    
    Args:
        p: Probability of success per trial (category incidence rate)
        target: Required number of successes (completions)
        conf: Required confidence level (e.g., 0.9994 after Bonferroni)
        rng_seed: Random seed for reproducibility
        num_mc_samples: Number of Monte Carlo samples for probability estimation
    
    Returns:
        Minimum n satisfying the confidence requirement
    
    Note:
        For very high confidence requirements, this may return large values.
        The Monte Carlo approach avoids scipy dependencies while remaining accurate.
    """
    rng = np.random.default_rng(rng_seed)
    
    def success_prob(n: int) -> float:
        """Estimate P(Binomial(n,p) >= target) via Monte Carlo."""
        samples = rng.binomial(n, p, size=num_mc_samples)
        return np.mean(samples >= target)
    
    # Start at theoretical minimum
    n_min = max(int(np.ceil(target / p)), target)
    step = 50
    
    # Phase 1: Find upper bound by stepping
    n = n_min
    while success_prob(n) < conf:
        n += step
        if n > 100_000:  # Safety limit
            break
    
    # Phase 2: Binary search to minimize
    lo, hi = max(n_min, n - step), n
    while lo < hi:
        mid = (lo + hi) // 2
        if success_prob(mid) >= conf:
            hi = mid
        else:
            lo = mid + 1
    
    return lo


def compute_bonferroni_confidence(
    target_confidence: float,
    num_categories: int
) -> float:
    """
    Compute per-category confidence using Bonferroni correction.
    
    To ensure P(ALL K categories succeed) >= target_confidence, we need
    each category to succeed with probability:
        conf_cat = 1 - (1 - target_confidence) / K
    
    This is conservative (the actual joint probability may be higher due to
    independence), but it guarantees the target is met.
    
    Args:
        target_confidence: Desired probability that ALL categories meet target
        num_categories: Total number of categories (K)
    
    Returns:
        Per-category confidence level
    
    Example:
        >>> compute_bonferroni_confidence(0.95, 78)
        0.9993589743589743  # Each category needs ~99.94% success
    """
    alpha = 1 - target_confidence  # Total allowable failure probability
    per_category_alpha = alpha / num_categories
    return 1 - per_category_alpha


# =============================================================================
# BUNDLE CONSTRUCTION STRATEGIES
# =============================================================================

def _calculate_expected_time(category_ids: List[int], category_lookup: pd.DataFrame) -> float:
    """Calculate expected interview time for a set of categories."""
    total = 0.0
    for cat_id in category_ids:
        cat = category_lookup.loc[cat_id]
        total += cat["effective_incidence_rate"] * cat["survey_length_seconds"]
    return total


def _greedy_pack_categories(
    ordered_category_ids: List[int],
    category_data: pd.DataFrame,
    max_expected_time: float = config.MAX_MEAN_INTERVIEW_TIME_SECONDS,
    max_categories: int = config.MAX_CATEGORY_QUALIFIERS_PER_RESPONDENT
) -> List[List[int]]:
    """
    Greedily pack categories into bundles respecting time and count constraints.
    
    Each category is assigned to exactly one bundle (partition).
    Categories are processed in the given order and added to the current bundle
    if constraints allow; otherwise a new bundle is started.
    
    Args:
        ordered_category_ids: Categories in priority order for packing
        category_data: DataFrame with category info
        max_expected_time: Maximum expected interview time per bundle
        max_categories: Maximum categories per bundle
    
    Returns:
        List of bundles, where each bundle is a list of category IDs
    """
    lookup = category_data.set_index("category_id")
    bundles: List[List[int]] = []
    current_bundle: List[int] = []
    current_time = 0.0

    for cat_id in ordered_category_ids:
        cat = lookup.loc[cat_id]
        time_contribution = cat["effective_incidence_rate"] * cat["survey_length_seconds"]

        # Check if category fits in current bundle
        would_exceed_time = (current_time + time_contribution) > max_expected_time
        would_exceed_count = len(current_bundle) >= max_categories

        if would_exceed_time or would_exceed_count:
            # Start new bundle
            if current_bundle:
                bundles.append(current_bundle)
            current_bundle = [cat_id]
            current_time = time_contribution
        else:
            current_bundle.append(cat_id)
            current_time += time_contribution

    # Don't forget the last bundle
    if current_bundle:
        bundles.append(current_bundle)

    return bundles


def create_bundles_strategy_a_bottleneck_first(df: pd.DataFrame) -> List[Bundle]:
    """
    Strategy A: Bottleneck First Partition
    
    Sort categories by required_exposed (highest first = lowest IR = hardest to fill).
    Then greedily pack into bundles. This ensures low-IR categories get priority
    placement and their bundles get sized appropriately.
    """
    ordered = df.sort_values("required_exposed", ascending=False)["category_id"].tolist()
    category_lists = _greedy_pack_categories(ordered, df)
    lookup = df.set_index("category_id")
    
    bundles = []
    for i, cats in enumerate(category_lists):
        expected_time = _calculate_expected_time(cats, lookup)
        bundles.append(Bundle(
            bundle_id=i + 1,
            categories=cats,
            expected_time=expected_time
        ))
    return bundles


def create_bundles_strategy_b_efficiency_first(df: pd.DataFrame) -> List[Bundle]:
    """
    Strategy B: Efficiency First Partition
    
    Sort categories by value_density (highest first = best demand/cost ratio).
    Packing in this order tends to create bundles with high expected completions
    per respondent.
    """
    ordered = df.sort_values("value_density", ascending=False)["category_id"].tolist()
    category_lists = _greedy_pack_categories(ordered, df)
    lookup = df.set_index("category_id")
    
    bundles = []
    for i, cats in enumerate(category_lists):
        expected_time = _calculate_expected_time(cats, lookup)
        bundles.append(Bundle(
            bundle_id=i + 1,
            categories=cats,
            expected_time=expected_time
        ))
    return bundles


def create_bundles_strategy_c_balanced_interleaving(df: pd.DataFrame) -> List[Bundle]:
    """
    Strategy C: Balanced Interleaving Partition
    
    Alternate between bottleneck (low IR) and stable (high IR) categories
    in the ordering, then pack. This creates bundles with mixed difficulty,
    potentially reducing variance in total sample size.
    """
    bottleneck_queue = df.sort_values("required_exposed", ascending=False)["category_id"].tolist()
    stability_queue = df.sort_values("stability_score", ascending=False)["category_id"].tolist()

    # Interleave
    ordered: List[int] = []
    used: set = set()
    b_idx, s_idx = 0, 0
    pick_bottleneck = True

    while len(ordered) < len(df):
        queue = bottleneck_queue if pick_bottleneck else stability_queue
        idx = b_idx if pick_bottleneck else s_idx

        while idx < len(queue):
            candidate = queue[idx]
            idx += 1
            if candidate not in used:
                ordered.append(candidate)
                used.add(candidate)
                break

        if pick_bottleneck:
            b_idx = idx
        else:
            s_idx = idx
        pick_bottleneck = not pick_bottleneck

    category_lists = _greedy_pack_categories(ordered, df)
    lookup = df.set_index("category_id")
    
    bundles = []
    for i, cats in enumerate(category_lists):
        expected_time = _calculate_expected_time(cats, lookup)
        bundles.append(Bundle(
            bundle_id=i + 1,
            categories=cats,
            expected_time=expected_time
        ))
    return bundles


# =============================================================================
# PLANNED RESPONDENT SIZING
# =============================================================================

def size_bundles_for_confidence(
    bundles: List[Bundle],
    category_data: pd.DataFrame,
    target_confidence: float = config.TARGET_CONFIDENCE,
    target_completions: int = config.TARGET_QUALIFIED_COMPLETIONS,
    rng_seed: int = config.RANDOM_SEED,
    valid_completion_rate: float = 1.0
) -> None:
    """
    Compute planned_respondents for each bundle to achieve target confidence.
    
    For each bundle, we find the minimum N such that P(all categories in bundle
    reach target) >= adjusted confidence. The confidence is adjusted using
    Bonferroni correction across ALL categories (not just bundle categories).
    
    The planned_respondents for a bundle = max(required_n) over all categories
    in that bundle. This ensures the hardest category in the bundle drives sizing.
    
    Modifies bundles in-place.
    
    Args:
        bundles: List of Bundle objects to size
        category_data: DataFrame with category incidence rates
        target_confidence: Joint confidence for all categories meeting target
        target_completions: Required completions per category
        rng_seed: Random seed for reproducibility
        valid_completion_rate: Fraction of completions passing quality checks (0-1).
                               Effective probability becomes IR * valid_completion_rate.
    """
    lookup = category_data.set_index("category_id")
    total_categories = sum(len(b.categories) for b in bundles)
    
    # Bonferroni adjustment: ensure joint probability
    conf_per_category = compute_bonferroni_confidence(target_confidence, total_categories)
    
    for bundle in bundles:
        max_n = 0
        for cat_id in bundle.categories:
            # Effective probability = IR * valid_completion_rate
            # This accounts for some completions failing quality checks
            p = lookup.loc[cat_id, "effective_incidence_rate"] * valid_completion_rate
            n = required_n_for_confidence(p, target_completions, conf_per_category, rng_seed)
            max_n = max(max_n, n)
        bundle.planned_respondents = max_n


# =============================================================================
# BUNDLE PLAN CREATION
# =============================================================================

def create_bundle_plan_a(
    df: pd.DataFrame, 
    rng_seed: int = config.RANDOM_SEED,
    valid_completion_rate: float = 1.0
) -> BundlePlan:
    """Create bundle plan using Strategy A: Bottleneck First."""
    bundles = create_bundles_strategy_a_bottleneck_first(df)
    size_bundles_for_confidence(bundles, df, rng_seed=rng_seed, valid_completion_rate=valid_completion_rate)
    return BundlePlan(
        name="Bundle Plan A: Bottleneck First",
        description="Partitions categories by required_exposed (hardest first), sizes for confidence",
        bundles=bundles,
        category_data=df
    )


def create_bundle_plan_b(
    df: pd.DataFrame, 
    rng_seed: int = config.RANDOM_SEED,
    valid_completion_rate: float = 1.0
) -> BundlePlan:
    """Create bundle plan using Strategy B: Efficiency First."""
    bundles = create_bundles_strategy_b_efficiency_first(df)
    size_bundles_for_confidence(bundles, df, rng_seed=rng_seed, valid_completion_rate=valid_completion_rate)
    return BundlePlan(
        name="Bundle Plan B: Efficiency First",
        description="Partitions categories by value_density (most efficient first), sizes for confidence",
        bundles=bundles,
        category_data=df
    )


def create_bundle_plan_c(
    df: pd.DataFrame, 
    rng_seed: int = config.RANDOM_SEED,
    valid_completion_rate: float = 1.0
) -> BundlePlan:
    """Create bundle plan using Strategy C: Balanced Interleaving."""
    bundles = create_bundles_strategy_c_balanced_interleaving(df)
    size_bundles_for_confidence(bundles, df, rng_seed=rng_seed, valid_completion_rate=valid_completion_rate)
    return BundlePlan(
        name="Bundle Plan C: Balanced Interleaving",
        description="Alternates bottleneck/stable categories then packs, sizes for confidence",
        bundles=bundles,
        category_data=df
    )


def get_all_bundle_plans(
    df: pd.DataFrame, 
    rng_seed: int = config.RANDOM_SEED,
    valid_completion_rate: float = 1.0
) -> List[BundlePlan]:
    """Create all bundle plan strategies for comparison."""
    return [
        create_bundle_plan_a(df, rng_seed, valid_completion_rate),
        create_bundle_plan_b(df, rng_seed, valid_completion_rate),
        create_bundle_plan_c(df, rng_seed, valid_completion_rate),
    ]


# =============================================================================
# FIXED-N SIMULATION FOR BUNDLE PLANS
# =============================================================================

class BundleRespondentGenerator:
    """
    Generates virtual respondents for bundle-based simulation.
    
    Similar to RespondentGenerator but designed for fixed-N simulation
    where we track per-category exposure demographics.
    
    Supports two modes:
    - Random sampling (default): Demographics sampled randomly from national distribution
    - Quota sampling: Demographics are forced to exactly match national distribution
                      by pre-computing the quota cells and cycling through them
    """

    def __init__(
        self, 
        category_data: pd.DataFrame, 
        random_seed: int = config.RANDOM_SEED,
        use_quota_sampling: bool = False
    ):
        self.category_data = category_data.set_index("category_id")
        self.category_ids = list(self.category_data.index)
        self.rng = np.random.default_rng(random_seed)
        self._next_id = 1
        self.use_quota_sampling = use_quota_sampling
        self._quota_sequence: List[Dict[str, str]] = []
        self._quota_idx = 0

    def reset_for_bundle(self, n_respondents: int) -> None:
        """
        Reset generator for a new bundle with n_respondents.
        
        If using quota sampling, pre-compute the demographic sequence
        to exactly match the national distribution.
        """
        self._quota_idx = 0
        if self.use_quota_sampling:
            self._quota_sequence = self._build_quota_sequence(n_respondents)
            # Shuffle to randomize order while maintaining exact counts
            self.rng.shuffle(self._quota_sequence)

    def _build_quota_sequence(self, n: int) -> List[Dict[str, str]]:
        """
        Build a sequence of n demographic profiles that exactly matches
        the national distribution (within rounding).
        
        Uses a cross-product approach: for each demographic dimension,
        compute the target counts, then combine them.
        """
        # Compute target counts for each demographic dimension
        targets: Dict[str, Dict[str, int]] = {}
        for demo_type, distribution in config.DEMOGRAPHICS_DISTRIBUTION.items():
            targets[demo_type] = {}
            remaining = n
            sorted_cats = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
            for i, (cat, pct) in enumerate(sorted_cats):
                if i == len(sorted_cats) - 1:
                    # Last category gets remainder to ensure sum = n
                    targets[demo_type][cat] = remaining
                else:
                    count = int(round(pct * n))
                    count = min(count, remaining)  # Don't exceed remaining
                    targets[demo_type][cat] = count
                    remaining -= count
        
        # Build sequences for each dimension
        sequences: Dict[str, List[str]] = {}
        for demo_type, counts in targets.items():
            seq = []
            for cat, count in counts.items():
                seq.extend([cat] * count)
            # Shuffle this dimension
            self.rng.shuffle(seq)
            sequences[demo_type] = seq
        
        # Combine into respondent profiles
        profiles = []
        for i in range(n):
            profile = {
                demo_type: sequences[demo_type][i]
                for demo_type in sequences
            }
            profiles.append(profile)
        
        return profiles

    def generate(self) -> Tuple[int, Dict[int, bool], Dict[str, str]]:
        """Generate a respondent: (id, qualifications, demographics)."""
        qualifications = {
            cat_id: self.rng.random() < self.category_data.loc[cat_id, "effective_incidence_rate"]
            for cat_id in self.category_ids
        }
        
        if self.use_quota_sampling and self._quota_sequence:
            # Use pre-computed demographic profile
            demographics = self._quota_sequence[self._quota_idx]
            self._quota_idx = (self._quota_idx + 1) % len(self._quota_sequence)
        else:
            demographics = self._generate_demographics()
        
        resp_id = self._next_id
        self._next_id += 1
        return resp_id, qualifications, demographics

    def _generate_demographics(self) -> Dict[str, str]:
        """Sample demographic attributes based on config distributions."""
        demographics = {}
        for demo_type, distribution in config.DEMOGRAPHICS_DISTRIBUTION.items():
            options = list(distribution.keys())
            probabilities = list(distribution.values())
            demographics[demo_type] = self.rng.choice(options, p=probabilities)
        return demographics


def check_demographic_representativeness(
    exposure_demographics: Dict[int, Dict[str, Counter]],
    tolerance: float = 0.05
) -> Tuple[bool, Dict[int, Dict[str, float]]]:
    """
    Check if per-category exposure demographics match national distribution.
    
    For each category, compute the maximum absolute deviation from expected
    distribution for each demographic dimension.
    
    Args:
        exposure_demographics: Per-category demographics counters
        tolerance: Maximum allowed absolute percentage-point deviation (default 5%).
                   Note: With small sample sizes (n<500), a 2% tolerance is often
                   unrealistic due to sampling variance. For n=300, the standard
                   error for a proportion of 0.5 is ~2.9%, so deviations of 3-5%
                   are expected just from random chance.
    
    Returns:
        Tuple of (all_ok, deviations_dict)
        - all_ok: True if all categories within tolerance
        - deviations_dict: {cat_id: {demo_type: max_deviation}}
    """
    deviations: Dict[int, Dict[str, float]] = {}
    all_ok = True
    
    for cat_id, demo_counters in exposure_demographics.items():
        deviations[cat_id] = {}
        for demo_type, counter in demo_counters.items():
            total = sum(counter.values())
            if total == 0:
                continue
            
            expected_dist = config.DEMOGRAPHICS_DISTRIBUTION.get(demo_type, {})
            max_dev = 0.0
            for category, expected_pct in expected_dist.items():
                actual_count = counter.get(category, 0)
                actual_pct = actual_count / total
                dev = abs(actual_pct - expected_pct)
                max_dev = max(max_dev, dev)
            
            deviations[cat_id][demo_type] = max_dev
            if max_dev > tolerance:
                all_ok = False
    
    return all_ok, deviations


def run_simulation_for_bundle_plan(
    plan: BundlePlan,
    random_seed: int = config.RANDOM_SEED,
    verbose: bool = False,
    demographic_tolerance: float = 0.05,
    use_quota_sampling: bool = False,
    valid_completion_rate: float = 1.0
) -> BundleSimulationResult:
    """
    Simulate a BundlePlan with FIXED planned_respondents per bundle.
    
    Key differences from dynamic simulation:
    - Each bundle has a fixed N (no "run until quota" behavior)
    - All respondents in a bundle see the same set of category qualifiers
    - We track per-category exposure demographics for representativeness validation
    
    Args:
        plan: The BundlePlan to simulate
        random_seed: Random seed for reproducibility
        verbose: Print progress information
        demographic_tolerance: Max allowed demographic deviation (default 5%).
                               With random sampling, expect ~3% deviation for n=300.
        use_quota_sampling: If True, force exact demographic match within each bundle
                            by pre-computing quota cells. This guarantees demographics_ok=True
                            but is less realistic as real survey panels may not achieve this.
        valid_completion_rate: Fraction of qualified completions that pass quality checks (0-1).
                               Simulates data quality issues like speeders, straightliners, etc.
                               E.g., 0.9 means 10% of completions are randomly invalidated.
    
    Returns:
        BundleSimulationResult with completions, exposures, and validation flags
    """
    generator = BundleRespondentGenerator(
        plan.category_data, random_seed, use_quota_sampling=use_quota_sampling
    )
    lookup = plan.category_data.set_index("category_id")
    
    # Create a separate RNG for valid completion checks (to not affect other randomness)
    quality_rng = np.random.default_rng(random_seed + 999)
    
    # Tracking structures
    category_completions: Dict[int, int] = defaultdict(int)
    category_exposures: Dict[int, int] = defaultdict(int)
    category_exposure_demographics: Dict[int, Dict[str, Counter]] = defaultdict(
        lambda: {demo_type: Counter() for demo_type in config.DEMOGRAPHICS_DISTRIBUTION}
    )
    interview_times: List[float] = []
    total_respondents = 0
    
    for bundle in plan.bundles:
        if verbose:
            print(f"  Simulating bundle {bundle.bundle_id}: {bundle.planned_respondents} respondents, "
                  f"{len(bundle.categories)} categories")
        
        # Reset quota sequence for this bundle if using quota sampling
        generator.reset_for_bundle(bundle.planned_respondents)
        
        for _ in range(bundle.planned_respondents):
            _, qualifications, demographics = generator.generate()
            total_respondents += 1
            interview_time = 0.0
            
            # Each respondent in bundle sees ALL categories in that bundle (exposure)
            for cat_id in bundle.categories:
                category_exposures[cat_id] += 1
                
                # Track exposure demographics
                for demo_type, demo_value in demographics.items():
                    category_exposure_demographics[cat_id][demo_type][demo_value] += 1
                
                # Check qualification (Bernoulli)
                if qualifications.get(cat_id, False):
                    # Apply valid completion rate - some completions fail quality checks
                    if quality_rng.random() < valid_completion_rate:
                        category_completions[cat_id] += 1
                    interview_time += lookup.loc[cat_id, "survey_length_seconds"]
            
            interview_times.append(interview_time)
    
    # Validation checks
    target = config.TARGET_QUALIFIED_COMPLETIONS
    all_cats = list(lookup.index)
    all_quotas_met = all(category_completions[c] >= target for c in all_cats)
    mean_time = float(np.mean(interview_times)) if interview_times else 0.0
    mean_time_ok = mean_time <= config.MAX_MEAN_INTERVIEW_TIME_SECONDS
    demographics_ok, _ = check_demographic_representativeness(
        dict(category_exposure_demographics), demographic_tolerance
    )
    
    return BundleSimulationResult(
        plan_name=plan.name,
        total_respondents=total_respondents,
        category_completions=dict(category_completions),
        category_exposures=dict(category_exposures),
        category_exposure_demographics={
            cat_id: {k: dict(v) for k, v in demo_counters.items()}
            for cat_id, demo_counters in category_exposure_demographics.items()
        },
        interview_times=interview_times,
        all_quotas_met=all_quotas_met,
        mean_time_ok=mean_time_ok,
        demographics_ok=demographics_ok
    )


# =============================================================================
# MONTE CARLO VALIDATION FOR BUNDLE PLANS
# =============================================================================

@dataclass
class MonteCarloResult:
    """Aggregated results from Monte Carlo validation of a BundlePlan."""
    plan_name: str
    num_runs: int
    success_rate_all_quotas: float
    success_rate_mean_time: float
    success_rate_demographics: float
    success_rate_all_constraints: float
    total_respondents_mean: float
    total_respondents_std: float
    mean_time_avg: float
    mean_time_std: float
    q95_time_avg: float
    min_completions_avg: float
    min_completions_std: float
    # Store individual run results for detailed analysis
    run_results: List[BundleSimulationResult] = field(default_factory=list, repr=False)


def run_monte_carlo_for_plan(
    plan: BundlePlan,
    num_runs: int = config.MONTE_CARLO_RUNS,
    base_seed: int = config.RANDOM_SEED,
    verbose: bool = False,
    demographic_tolerance: float = 0.05,
    use_quota_sampling: bool = False,
    valid_completion_rate: float = 1.0
) -> MonteCarloResult:
    """
    Run Monte Carlo validation to measure P(success) under fixed N plan.
    
    This is the key validation step: we prove that the pre-computed plan
    achieves P(all constraints met) >= TARGET_CONFIDENCE empirically.
    
    Args:
        plan: The BundlePlan to validate
        num_runs: Number of Monte Carlo simulations
        base_seed: Starting random seed (incremented for each run)
        verbose: Print per-run progress
        demographic_tolerance: Max demographic deviation allowed (default 5%)
        use_quota_sampling: Force exact demographic match within bundles
        valid_completion_rate: Fraction of completions passing quality checks (0-1)
    
    Returns:
        MonteCarloResult with success rates and statistics
    """
    results: List[BundleSimulationResult] = []
    
    for i in range(num_runs):
        seed = base_seed + i
        result = run_simulation_for_bundle_plan(
            plan, random_seed=seed, verbose=False, 
            demographic_tolerance=demographic_tolerance,
            use_quota_sampling=use_quota_sampling,
            valid_completion_rate=valid_completion_rate
        )
        results.append(result)
        
        if verbose:
            status = "✓" if result.all_constraints_met else "✗"
            print(f"    Run {i+1}/{num_runs}: N={result.total_respondents:,}, "
                  f"Min={result.min_category_completions}, Mean={result.mean_interview_time:.1f}s {status}")
    
    # Compute statistics
    success_quotas = sum(1 for r in results if r.all_quotas_met)
    success_time = sum(1 for r in results if r.mean_time_ok)
    success_demo = sum(1 for r in results if r.demographics_ok)
    success_all = sum(1 for r in results if r.all_constraints_met)
    
    total_ns = [r.total_respondents for r in results]
    mean_times = [r.mean_interview_time for r in results]
    q95_times = [r.q95_interview_time for r in results]
    min_comps = [r.min_category_completions for r in results]
    
    return MonteCarloResult(
        plan_name=plan.name,
        num_runs=num_runs,
        success_rate_all_quotas=success_quotas / num_runs,
        success_rate_mean_time=success_time / num_runs,
        success_rate_demographics=success_demo / num_runs,
        success_rate_all_constraints=success_all / num_runs,
        total_respondents_mean=float(np.mean(total_ns)),
        total_respondents_std=float(np.std(total_ns)),
        mean_time_avg=float(np.mean(mean_times)),
        mean_time_std=float(np.std(mean_times)),
        q95_time_avg=float(np.mean(q95_times)),
        min_completions_avg=float(np.mean(min_comps)),
        min_completions_std=float(np.std(min_comps)),
        run_results=results
    )


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_bundle_plan_to_csv(
    plan: BundlePlan,
    output_path: str = "bundle_plan.csv"
) -> None:
    """
    Export bundle plan to CSV for implementation.
    
    Output format:
        bundle_id, planned_respondents, expected_time, num_categories, category_id, category_name
    
    Each row represents a (bundle, category) pair.
    """
    lookup = plan.category_data.set_index("category_id")
    rows = []
    
    for bundle in plan.bundles:
        for cat_id in bundle.categories:
            rows.append({
                "bundle_id": bundle.bundle_id,
                "planned_respondents": bundle.planned_respondents,
                "expected_time": round(bundle.expected_time, 2),
                "num_categories": bundle.num_categories,
                "category_id": cat_id,
                "category_name": lookup.loc[cat_id, "category_name"],
                "incidence_rate": round(lookup.loc[cat_id, "effective_incidence_rate"], 4),
                "survey_length_seconds": round(lookup.loc[cat_id, "survey_length_seconds"], 2)
            })
    
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Bundle plan exported to: {output_path}")


def export_bundle_summary_to_csv(
    plan: BundlePlan,
    output_path: str = "bundle_summary.csv"
) -> None:
    """Export high-level bundle summary (one row per bundle)."""
    lookup = plan.category_data.set_index("category_id")
    rows = []
    
    for bundle in plan.bundles:
        cat_names = [lookup.loc[c, "category_name"] for c in bundle.categories]
        rows.append({
            "bundle_id": bundle.bundle_id,
            "planned_respondents": bundle.planned_respondents,
            "expected_time_seconds": round(bundle.expected_time, 2),
            "num_categories": bundle.num_categories,
            "categories": "; ".join(cat_names)
        })
    
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Bundle summary exported to: {output_path}")
