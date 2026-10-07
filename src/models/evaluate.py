import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Sınıf dengesizliği için kapsamlı metrikler ve Confusion Matrix hesaplar.
    """
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*20} {model_name} {'='*20}")
    print(f"F1-Score:   {f1:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"ROC-AUC:    {roc:.4f}")
    print("\nConfusion Matrix (Hata Matrisi):")
    print(f"TN (Doğru Negatif): {cm[0,0]:<4} | FP (Yanlış Pozitif): {cm[0,1]:<4}")
    print(f"FN (Yanlış Negatif): {cm[1,0]:<4} | TP (Doğru Pozitif): {cm[1,1]:<4}")
    print("\nSınıflandırma Detayları:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Dolandırıcılık"], zero_division=0))

    return {
        "model_name": model_name,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc,
        "confusion_matrix": cm,
    }