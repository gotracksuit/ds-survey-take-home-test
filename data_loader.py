import pandas as pd

def load_categories(path):
    df = pd.read_csv(path)

    categories = df["category_id"].tolist()
    incidence = dict(zip(df["category_id"], df["incidence_rate"]))
    lengths = dict(zip(df["category_id"], df["category_length_seconds"]))

    return categories, incidence, lengths
