import pandas as pd
from src.models.preprocess import preprocess

def test_preprocess():
    df = pd.DataFrame({
        "species": ["A", None],
        "bill_length_mm": [10.0, 20.0],
        "bill_depth_mm": [5.0, None],
        "flipper_length_mm": [100.0, 200.0],
        "body_mass_g": [3000.0, 4000.0],
    })

    X, y = preprocess(df)

    # Only first row is valid after dropna
    assert len(X) == 1
    assert len(y) == 1

    # Encoded species label must be an integer
    assert isinstance(y.iloc[0], (int, float))
