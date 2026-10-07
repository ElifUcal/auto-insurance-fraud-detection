import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sigorta Sahtekarlığı Tespit Paneli",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = "models/best_model.joblib"
DATA_PATH = "data/processed/claims_cleaned_v1.csv"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

model = load_model()
df = load_data()

st.title("🛡️ Sigorta Hasar Sahtekarlığı Tespit & Karar Destek Sistemi")
st.markdown("Eksperler için makine öğrenmesi destekli dosya inceleme ve risk önceliklendirme arayüzü.")

tab1, tab2 = st.tabs(["📊 Veri Tabanı İnceleme", "🔍 Yeni Hasar Dosyası Değerlendir"])

# --- TAB 1: VERİ İNCELEME ---
with tab1:
    st.subheader("İşlenmiş Hasar Verileri")
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam Dosya", len(df))
        col2.metric("Sahtekarlık Olay Sayısı", int(df["fraud_reported"].sum()))
        col3.metric("Dolandırıcılık Oranı", f"%{(df['fraud_reported'].mean() * 100):.1f}")
        
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.warning("Veri seti bulunamadı. Lütfen veri pipeline'ını çalıştırın.")

# --- TAB 2: YENİ DOSYA RİSK ANALİZİ ---
with tab2:
    st.subheader("Eksper Karar Destek Ekranı")
    if model is None:
        st.error("Kayıtlı model (`models/best_model.joblib`) bulunamadı.")
    elif df is None:
        st.error("Veri seti şablonu bulunamadı.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            total_claim = st.number_input("Toplam Talep Tutarı ($)", min_value=100, max_value=200000, value=50000, step=1000)
            vehicle_claim = st.number_input("Araç Hasar Tutarı ($)", min_value=0, max_value=100000, value=35000, step=1000)
            property_claim = st.number_input("Mülk Hasar Tutarı ($)", min_value=0, max_value=50000, value=5000, step=500)
        with col_b:
            injury_claim = st.number_input("Yaralanma Talep Tutarı ($)", min_value=0, max_value=50000, value=10000, step=500)
            incident_severity = st.selectbox("Hasar Boyutu", ["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"])
            insured_hobby_chess = st.checkbox("Sigortalının Hobisi Satranç mı?", value=False)
            insured_hobby_crossfit = st.checkbox("Sigortalının Hobisi Cross-Fit mi?", value=False)

        if st.button("🚨 Dosya Riskini Hesapla", type="primary"):
            feature_cols = [c for c in df.columns if c != "fraud_reported"]
            sample_input = pd.DataFrame(0, index=[0], columns=feature_cols)

            if "total_claim_amount" in sample_input.columns: sample_input["total_claim_amount"] = total_claim
            if "vehicle_claim" in sample_input.columns: sample_input["vehicle_claim"] = vehicle_claim
            if "property_claim" in sample_input.columns: sample_input["property_claim"] = property_claim
            if "injury_claim" in sample_input.columns: sample_input["injury_claim"] = injury_claim

            sev_col = f"incident_severity_{incident_severity}"
            if sev_col in sample_input.columns: sample_input[sev_col] = 1
            if insured_hobby_chess and "insured_hobbies_chess" in sample_input.columns:
                sample_input["insured_hobbies_chess"] = 1
            if insured_hobby_crossfit and "insured_hobbies_cross-fit" in sample_input.columns:
                sample_input["insured_hobbies_cross-fit"] = 1

            try:
                prob = model.predict_proba(sample_input)[0][1]
                pred = 1 if prob >= 0.5 else 0

                st.markdown("---")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Model Sahtekarlık İhtimali", f"%{prob * 100:.1f}")
                    st.progress(float(prob))
                with col_res2:
                    if pred == 1:
                        st.error("⚠️ **YÜKSEK RİSK:** Dosya sahtekarlık şüphesi taşıyor. Denetime sevk edilmeli!")
                    else:
                        st.success("✅ **DÜŞÜK RİSK:** Standart inceleme süreci uygulanabilir.")
            except Exception as e:
                st.error(f"Tahmin hesaplanırken hata oluştu: {e}")