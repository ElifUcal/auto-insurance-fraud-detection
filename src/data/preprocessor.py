import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, SelectKBest

def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham veriyi temizler, gereksiz kolonları eler, kategorik değişkenleri kodlar.
    """
    df = df.copy()

    # 1. Gereksiz veya ID kolonlarını düşürme
    drop_cols = ["_c39", "policy_number", "policy_bind_date", "incident_date", 
                 "insured_zip", "incident_location"]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # 2. Hedef değişkeni ikili (binary) formata çevirme (Y=1, N=0)
    if "fraud_reported" in df.columns:
        df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

    # 3. Basit Eksik Veri Tamamlama
    # Kategorik sütunlardaki eksik değerleri 'Unknown' ile doldur
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # Sayısal sütunlardaki eksikleri medyan ile doldur
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # 4. One-Hot Encoding (Kategorikleri sayısallaştırma)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df_encoded


def run_anova_feature_selection(X: pd.DataFrame, y: pd.Series, top_k: int = 15):
    """
    ANOVA F-Testi (f_classif) uygulayarak hedef değişkenle en yüksek 
    ilişkisi olan en iyi top_k özelliği seçer ve raporlar.
    """
    # Sadece sayısal/one-hot sütunları kullan
    selector = SelectKBest(score_func=f_classif, k=min(top_k, X.shape[1]))
    selector.fit(X, y)

    scores_df = pd.DataFrame({
        "Feature": X.columns,
        "ANOVA_F_Score": selector.scores_,
        "p_value": selector.pvalues_
    }).dropna().sort_values(by="ANOVA_F_Score", ascending=False)

    selected_features = scores_df.head(top_k)["Feature"].tolist()
    return selected_features, scores_df


if __name__ == "__main__":
    import os
    from src.data.loader import load_raw_claims_data

    print("1. Ham veri yükleniyor...")
    raw_df = load_raw_claims_data()

    print("2. Veri temizleniyor ve encode ediliyor...")
    processed_df = clean_and_preprocess(raw_df)

    # ANOVA Testi Raporu
    if "fraud_reported" in processed_df.columns:
        X = processed_df.drop(columns=["fraud_reported"])
        y = processed_df["fraud_reported"]
        
        top_features, report = run_anova_feature_selection(X, y, top_k=10)
        print("\n--- ANOVA F-Testine Göre En Anlamlı 10 Özellik ---")
        print(report.head(10).to_string(index=False))

    # Temiz veriyi kaydet (Kişi 2 ve Kişi 3 için)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "claims_cleaned_v1.csv")
    
    processed_df.to_csv(out_path, index=False)
    print(f"\nİşlenmiş veri başarıyla kaydedildi: {out_path}")