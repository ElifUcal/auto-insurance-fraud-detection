import os
import pandas as pd
from typing import Optional


def load_raw_claims_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load the raw insurance claims dataset from CSV.

    Args:
        filepath (Optional[str]): Path to the raw CSV file. If None, resolves to
          the default 'data/raw/insurance_claims.csv'.

    Returns:
        pd.DataFrame: Raw claims dataset loaded into a pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if filepath is None:
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        filepath = os.path.join(base_dir, "data", "raw", "insurance_claims.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset file not found at: {filepath}. Please place it under"
            " 'data/raw/'."
        )

    # Missing values in this Kaggle dataset are commonly represented as '?'
    df = pd.read_csv(filepath, na_values=["?", "MISSING", "none", ""])
    return df


if __name__ == "__main__":
    df = load_raw_claims_data()
    print(
        f"Data loaded successfully! Shape: {df.shape[0]} rows, {df.shape[1]}"
        " columns."
    )
    print("\nTarget variable ('fraud_reported') distribution (%):")
    print(df["fraud_reported"].value_counts(normalize=True) * 100)