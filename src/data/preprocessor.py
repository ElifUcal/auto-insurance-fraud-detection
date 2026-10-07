import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate insurance domain-specific features from raw columns.

    Args:
        df (pd.DataFrame): Input claims dataframe.

    Returns:
        pd.DataFrame: Dataframe augmented with engineered features.
    """
    df = df.copy()

    # 1. Date and temporal features
    if "policy_bind_date" in df.columns and "incident_date" in df.columns:
        df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
        df["incident_date"] = pd.to_datetime(df["incident_date"])

        # Days elapsed between policy bind and incident
        df["days_to_incident"] = (
            df["incident_date"] - df["policy_bind_date"]
        ).dt.days

        # Early claim flag: Claims occurring within the first 30 days carry higher risk
        df["is_immediate_claim"] = (df["days_to_incident"] <= 30).astype(int)

        # Day of week (0: Monday, 6: Sunday)
        df["incident_day_of_week"] = df["incident_date"].dt.dayofweek
        df["is_weekend"] = (df["incident_day_of_week"] >= 5).astype(int)

    # 2. Risky incident hour flag (Late night / early morning: 23:00 - 05:00)
    if "incident_hour_of_the_day" in df.columns:
        df["is_night_incident"] = df["incident_hour_of_the_day"].apply(
            lambda h: 1 if (h >= 23 or h <= 5) else 0
        )

    # 3. Financial and claim severity ratios
    if (
        "total_claim_amount" in df.columns
        and "policy_annual_premium" in df.columns
    ):
        df["claim_to_premium_ratio"] = df["total_claim_amount"] / (
            df["policy_annual_premium"] + 1e-5
        )

    if "vehicle_claim" in df.columns and "total_claim_amount" in df.columns:
        df["vehicle_to_total_ratio"] = df["vehicle_claim"] / (
            df["total_claim_amount"] + 1e-5
        )

    if "injury_claim" in df.columns and "total_claim_amount" in df.columns:
        df["injury_to_total_ratio"] = df["injury_claim"] / (
            df["total_claim_amount"] + 1e-5
        )

    # 4. Incident severity ordinal mapping
    severity_map = {
        "Trivial Damage": 0,
        "Minor Damage": 1,
        "Major Damage": 2,
        "Total Loss": 3,
    }
    if "incident_severity" in df.columns:
        df["severity_score"] = (
            df["incident_severity"].map(severity_map).fillna(1)
        )

    return df


def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw claims data, engineer features, impute missing values,

    and apply one-hot encoding.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Fully processed, encoded dataframe ready for modeling.
    """
    df = df.copy()

    # Step 1: Feature engineering
    df = engineer_features(df)

    # Step 2: Drop non-predictive identifiers, high-cardinality leakage columns
    drop_cols = [
        "_c39",
        "policy_number",
        "policy_bind_date",
        "incident_date",
        "insured_zip",
        "incident_location",
    ]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # Step 3: Target variable binarization (Y -> 1, N -> 0)
    if "fraud_reported" in df.columns:
        df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

    # Step 4: Missing value imputation
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Step 5: One-Hot Encoding for categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df_encoded


def run_anova_feature_selection(
    X: pd.DataFrame, y: pd.Series, top_k: int = 15
):
    """Run ANOVA F-test (f_classif) to score and select features with highest

    statistical correlation to fraud target.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
        top_k (int): Number of top features to select.

    Returns:
        tuple[list[str], pd.DataFrame]: Selected feature names and complete
        ANOVA score table.
    """
    selector = SelectKBest(score_func=f_classif, k=min(top_k, X.shape[1]))
    selector.fit(X, y)

    scores_df = (
        pd.DataFrame({
            "Feature": X.columns,
            "ANOVA_F_Score": selector.scores_,
            "p_value": selector.pvalues_,
        })
        .dropna()
        .sort_values(by="ANOVA_F_Score", ascending=False)
    )

    selected_features = scores_df.head(top_k)["Feature"].tolist()
    return selected_features, scores_df


if __name__ == "__main__":
    from src.data.loader import load_raw_claims_data

    print("1. Loading raw insurance claims data...")
    raw_df = load_raw_claims_data()

    print("2. Performing feature engineering and preprocessing...")
    processed_df = clean_and_preprocess(raw_df)

    # Perform ANOVA F-Test evaluation
    if "fraud_reported" in processed_df.columns:
        X = processed_df.drop(columns=["fraud_reported"])
        y = processed_df["fraud_reported"]

        top_features, report = run_anova_feature_selection(X, y, top_k=15)
        print("\n--- ANOVA F-Test: Top 15 Most Significant Features ---")
        print(report.head(15).to_string(index=False))

    # Save processed dataset for modeling (Person 2) and dashboard (Person 3)
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    output_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "claims_cleaned_v2.csv")

    processed_df.to_csv(out_path, index=False)
    print(f"\nEngineered dataset successfully saved to: {out_path}")