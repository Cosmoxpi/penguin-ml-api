from pathlib import Path
import pandas as pd

def load_data():
    """ Load the processed dataset if available
    If not, load the raw dataset"""

    processed_file=Path("data/processed/penguins_clean.csv")
    raw_file=Path("data/raw/penguins.csv")

    if processed_file.exists():
        print("Loading processed dataset...")
        return pd.read_csv(processed_file)
    elif raw_file.exists():
        print("Processed dataset not found, Loading raw dataset...")
        return pd.read_csv(raw_file)
    else:
        raise FileNotFoundError("Neither raw nor processed dataset was found")
