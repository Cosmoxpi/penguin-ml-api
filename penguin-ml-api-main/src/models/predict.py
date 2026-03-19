import pickle
import pandas as pd
from pathlib import Path

# Mapping numeric labels back to specific names
SPECIES_MAP={
    0: "Adelie",
    1: "Chinstrap",
    2: "Gentoo"
}

# Expected input feature columns
FEATURE_COLUMNS = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

MODEL_PATH=Path("artifacts/model.pkl")

def predict(input_data:dict)->str:
    """

    Predict penguin species from input feature dictionary.
    Example input_data:
    {
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": 3750
    }
    """

    # Load model
    with open(MODEL_PATH,"rb") as f:
        model=pickle.load(f)

    # Convert input dict to DataFrame
    df=pd.DataFrame([input_data])

    # Ensure only required columns are used
    X=df[FEATURE_COLUMNS]

    # Get prediction (return numeric code)
    pred_code=model.predict(X)[0]

    # Covert numeric code to species name
    return SPECIES_MAP.get(pred_code,"Unknown")
