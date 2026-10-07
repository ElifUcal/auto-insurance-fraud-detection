import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from src.models.evaluate import evaluate_model

DATA_PATH = "data/processed/claims_cleaned_v1.csv"
MODEL_DIR = "models"

def train_and_compare():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"İşlenmiş veri bulunamadı: {DATA_PATH}. Önce veri pipeline'ını çalıştırın.")

    # 1. Veri Yükleme
    df = pd.read_csv(DATA_PATH)
    target_col = "fraud_reported"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Train (%80) ve Test (%20) Ayrımı (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Doğrusal ve olasılıksal modeller için ölçekleme
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. İstenen 3 Modelin Tanımlanması
    candidate_models = {
        "Gaussian Naive Bayes (Referans)": {
            "model": GaussianNB(),
            "X_tr": X_train_scaled,
            "X_te": X_test_scaled,
        },
        "Logistic Regression (Doğrusal Baseline)": {
            "model": LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42),
            "X_tr": X_train_scaled,
            "X_te": X_test_scaled,
        },
        "Random Forest (Ağaç Tabanlı)": {
            "model": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42),
            "X_tr": X_train,
            "X_te": X_test,
        },
    }

    best_model_name = None
    best_f1_score = -1.0
    best_estimator = None

    # 4. Modelleri Eğit, Değerlendir ve Yarıştır
    for name, config in candidate_models.items():
        clf = config["model"]
        clf.fit(config["X_tr"], y_train)

        preds = clf.predict(config["X_te"])
        probs = clf.predict_proba(config["X_te"])[:, 1] if hasattr(clf, "predict_proba") else None

        metrics = evaluate_model(y_test, preds, probs, model_name=name)

        if metrics["f1"] > best_f1_score:
            best_f1_score = metrics["f1"]
            best_model_name = name
            best_estimator = clf

    # 5. En Yüksek F1 Veren Modeli models/best_model.joblib Olarak Kaydet
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best_estimator, save_path)

    print("\n" + "="*50)
    print(f"🏆 YARIŞMA BİRİNCİSİ: {best_model_name}")
    print(f"🎯 En Yüksek F1-Score: {best_f1_score:.4f}")
    print(f"💾 Kaydedilen Model: {save_path}")
    print("="*50)

if __name__ == "__main__":
    train_and_compare()