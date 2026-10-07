# 🛡️ Auto Insurance Fraud Detection & Decision Support System

Uçtan uca makine öğrenmesi destekli sigorta sahtekarlık tespiti, açıklanabilir yapay zeka (XAI) ve eksper karar destek paneli.

---

## 📌 Proje Özeti
Trafik kazası sonrası sigorta şirketine bildirilen hasar taleplerinin şüpheli olup olmadığını (hasar tutarı uyumsuzlukları, zamanlama anormallikleri vb.) tespit eden ve sigorta eksperlerine **SHAP** tabanlı gerekçelendirilmiş risk skorları sunan kurumsal bir analitik platform.

## 👥 Ekip ve Rol Dağılımı
* **Kişi 1 (Elif Naz Uçal) (Data Engineer / Scientist):** Veri temizleme, sınıf dengesizliği (imbalance) yönetimi, feature engineering pipeline.
* **Kişi 2 (Beren ) (ML Engineer / XAI Specialist):** Model eğitimi (LightGBM/XGBoost/CatBoost), hiperparametre optimizasyonu, SHAP entegrasyonu.
* **Kişi 3 (Ennur) (Full-Stack / Dashboard Engineer):** Streamlit tabanlı eksper karar destek paneli, KPI izleme ekranları, vaka analiz arayüzü.

## 📁 Dizin Yapısı
```text
├── data/              # Ham ve işlenmiş veri dosyaları
├── notebooks/         # EDA ve modelleme deneyleri
├── models/            # Eğitilmiş model binary'leri (.joblib)
├── src/               # Ortak veri ve model kaynak kodları
└── app/               # Streamlit karar destek arayüzü