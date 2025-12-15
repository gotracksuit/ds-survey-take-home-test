import pandas as pd
import numpy as np

TARGET_QUALIFIED = 200
TIME_BUDGET = 480 #

GENDER_DIST = {"Male": 0.495, "Female": 0.505}
AGE_DIST = {"18-64": 0.771, "65+": 0.229}
REGION_DIST = {"Urban": 0.80, "Rural": 0.20}

def load_categories(enriched=False):
    if enriched:
        return pd.read_csv('data/fake_category_data_enriched.csv')
    return pd.read_csv('data/fake_category_data.csv')

def generate_respondents(n, seed=42):
    np.random.seed(seed)
    return pd.DataFrame({
        'respondent_id': range(n),
        'gender': np.random.choice(list(GENDER_DIST.keys()), n, p=list(GENDER_DIST.values())),
        'age_group': np.random.choice(list(AGE_DIST.keys()), n, p=list(AGE_DIST.values())),
        'region': np.random.choice(list(REGION_DIST.keys()), n, p=list(REGION_DIST.values())),
    })
