"""
Configuration for the Survey Allocation Optimiser.

All configurable parameters are centralised here for easy "what-if" analysis.
"""

from typing import Dict

# Reproducibility - fixed seed ensures identical results across runs
RANDOM_SEED: int = 42

# Contract targets
TARGET_QUALIFIED_COMPLETIONS: int = 200  # Required completions per category
TARGET_CONFIDENCE: float = 0.95  # P(all categories meet target) >= this

# Time constraints (seconds)
MAX_MEAN_INTERVIEW_TIME_SECONDS: float = 480.0  # 8 minutes - hard constraint
MAX_Q95_INTERVIEW_TIME_SECONDS: float = 720.0   # 12 minutes - soft constraint for outliers

# Respondent burden - prevents survey fatigue even if time budget allows more
MAX_CATEGORY_QUALIFIERS_PER_RESPONDENT: int = 10

# Data quality - set < 1.0 to simulate losing responses to quality checks
VALID_COMPLETION_RATE: float = 1.0

# Seasonality adjustments: {category_id: multiplier}
# Values < 1.0 reduce IR (e.g., Suncare in winter)
# Values > 1.0 increase IR (e.g., Hot drinks in winter)
# Default is empty (no adjustments)
SEASONALITY_MULTIPLIERS: Dict[int, float] = {}

# Predefined seasonality scenarios for "next month" IR changes
# Each scenario maps category_id -> multiplier
SEASONALITY_SCENARIOS: Dict[str, Dict[int, float]] = {
    # Winter scenario (June-August in Southern Hemisphere)
    # - Outdoor/summer products decline
    # - Hot beverages and indoor activities increase
    "winter": {
        60: 0.50,   # Suncare - 50% drop (less sun exposure)
        41: 0.70,   # Camping Equipments - 30% drop
        52: 0.80,   # Outdoor Apparel - 20% drop
        46: 0.85,   # Outdoor Pursuits Apparel - 15% drop
        4: 0.60,    # Self Tan (Female Only) - 40% drop
        16: 1.25,   # Fresh Coffee - 25% increase
        67: 1.20,   # Coffee - 20% increase
        66: 1.15,   # Chocolate - 15% increase (comfort food)
        71: 1.15,   # Chocolate confectionery - 15% increase
        37: 1.10,   # Beddings - 10% increase
    },
    
    # Summer scenario (December-February in Southern Hemisphere)
    # - Outdoor/summer products increase
    # - Hot beverages decline
    "summer": {
        60: 1.30,   # Suncare - 30% increase
        41: 1.25,   # Camping Equipments - 25% increase
        52: 1.20,   # Outdoor Apparel - 20% increase
        46: 1.15,   # Outdoor Pursuits Apparel - 15% increase
        4: 1.25,    # Self Tan (Female Only) - 25% increase
        28: 1.20,   # Alcohol Free Drinks - 20% increase
        29: 1.15,   # Pre-mixed alcoholic drinks/RTDs - 15% increase
        16: 0.85,   # Fresh Coffee - 15% drop
        67: 0.90,   # Coffee - 10% drop
        66: 0.90,   # Chocolate - 10% drop
    },
    
    # Back-to-school scenario (January-February)
    # - Education and children's products increase
    "back_to_school": {
        20: 1.30,   # Vocational Training - 30% increase
        23: 1.25,   # Baby and Child Products - 25% increase
        5: 1.20,    # Baby Feeding - 20% increase
        6: 1.15,    # Baby products - 15% increase
        55: 1.20,   # Children's Television - 20% increase
        11: 1.15,   # Nappies & wipes - 15% increase
    },
    
    # Holiday season scenario (November-December)
    # - Gift items and indulgent products increase
    "holiday": {
        66: 1.35,   # Chocolate - 35% increase
        71: 1.30,   # Chocolate confectionery - 30% increase
        74: 1.25,   # Chocolate bars and blocks - 25% increase
        72: 1.20,   # Sugar confectionery - 20% increase
        24: 1.25,   # Luxury Leather Bags/Accessories - 25% increase
        10: 1.20,   # Designer Women's Clothing - 20% increase
        51: 1.15,   # Female Fashion - 15% increase
        15: 1.20,   # Dark Spirits - 20% increase
        39: 1.15,   # White Spirits - 15% increase
        29: 1.20,   # Pre-mixed alcoholic drinks/RTDs - 20% increase
    },
    
    # New Year / Health kick scenario (January)
    # - Health and fitness products increase
    "new_year": {
        47: 1.40,   # Nutrition & Fitness Services - 40% increase
        14: 1.35,   # Weight-loss - 35% increase
        38: 1.30,   # Natural Health Supplements - 30% increase
        56: 1.25,   # Mental Performance Supplements - 25% increase
        54: 1.25,   # Healthy Snacks - 25% increase
        69: 1.20,   # Healthy Beverages - 20% increase
        77: 1.20,   # Breakfast or Health Foods - 20% increase
        35: 1.25,   # Women's Activewear (Female Only) - 25% increase
        28: 1.15,   # Alcohol Free Drinks - 15% increase
        34: 1.20,   # Alcohol Free Beer - 20% increase
        66: 0.80,   # Chocolate - 20% drop (post-holiday guilt)
        75: 0.85,   # Fast Food - 15% drop
    },
}

# Demographics distribution (for validation - checks per-category exposure representativeness)
DEMOGRAPHICS_DISTRIBUTION: Dict[str, Dict[str, float]] = {
    "gender": {
        "Female": 0.51,
        "Male": 0.49,
    },
    "age": {
        "18-34": 0.32,
        "35-54": 0.35,
        "55+": 0.33,
    },
    "region": {
        "Auckland": 0.33,
        "Rest of North Island": 0.43,
        "South Island": 0.24,
    },
}

# Monte Carlo settings for robustness testing
MONTE_CARLO_RUNS: int = 10

# File paths
CATEGORY_DATA_FILE: str = "fake_category_data.csv"
