import pandas as pd

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the penguins dataset for modeling.

    Steps:
    - Drop the rows with missing essential values
    - Encode target column 'species' to numeric labels
    - Return X(features) and y(target)
    """

    # Remove rows with missing important columns
    clean_df=df.dropna(subset=[
        "species",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]).copy()

    # Encode species as integer labels
    clean_df["species_id"]=clean_df["species"].astype("category").cat.codes

    # Select feature columns
    X=clean_df[[
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]]

    # Target column
    y=clean_df['species_id']
    return X,y
