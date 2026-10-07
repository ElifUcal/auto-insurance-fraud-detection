import os
import pandas as pd
from typing import Optional

def load_raw_claims_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Kaggle'dan indirilen ham sigorta hasar verisini okur.
    Eksik veya hatalı biçimlendirilmiş verileri denetler.
    """
    if filepath is None:
        # Varsayılan dosya konumu
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filepath = os.path.join(base_dir, "data", "raw", "insurance_claims.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {filepath}. Lütfen data/raw içine yerleştirin.")
    
    # Kaggle verisinde eksik değerler genelde '?' olarak yer alır
    df = pd.read_csv(filepath, na_values=["?", "MISSING", "none", ""])
    return df

if __name__ == "__main__":
    df = load_raw_claims_data()
    print(f"Veri başarıyla yüklendi! Satır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")
    print("\nHedef Değişken Dağılımı:")
    print(df["fraud_reported"].value_counts(normalize=True) * 100)