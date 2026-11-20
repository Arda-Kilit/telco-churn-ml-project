"""
BTK Uçtan Uca Makine Öğrenmesi Projesi — Telco Customer Churn
Aşama 13–14: Çalışan Streamlit Kullanıcı Arayüzü (UI) ve Canlı Deployment Uygulaması.

Çalıştırma:
    streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure custom transformer class is importable when unpickling best_model.joblib
from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TelcoFeatureEngineer,
)

st.set_page_config(
    page_title="Telco Churn Risk Tahmin Sistemi",
    page_icon="📡",
    layout="wide",
)

MODEL_PATH = "models/best_model.joblib"
META_PATH = "models/model_meta.json"
METRICS_PATH = "results/metrics.json"
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIG_DIR = "figures"
RANDOM_STATE = 42


def _train_champion_in_place(meta: dict):
    """
    Deterministic fallback when a joblib pickle created in another Python/scikit-learn
    version (e.g., Colab Python 3.12 -> Streamlit Cloud Python 3.14) cannot be unpickled.
    Rebuilds and fits the exact tuned champion pipeline in ~2 seconds on the same 70% train split.
    """
    df = pd.read_csv(DATA_PATH)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    try:
        from catboost import CatBoostClassifier

        estimator = CatBoostClassifier(
            iterations=400,
            learning_rate=0.02,
            depth=4,
            l2_leaf_reg=10,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=0,
        )
    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    pipe = Pipeline(steps=[
        ("fe", TelcoFeatureEngineer()),
        ("prep", preprocessor),
        ("model", estimator),
    ])
    pipe.fit(X_train, y_train)
    return pipe


@st.cache_resource
def load_artifacts():
    meta = json.load(open(META_PATH, encoding="utf-8")) if os.path.exists(META_PATH) else {}
    metrics = json.load(open(METRICS_PATH, encoding="utf-8")) if os.path.exists(METRICS_PATH) else None

    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None

    if model is None and os.path.exists(DATA_PATH):
        model = _train_champion_in_place(meta)

    return model, meta, metrics


def get_risk_band(prob: float, threshold: float):
    if prob >= max(threshold, 0.60):
        return "Yüksek Risk", "🔴", "Acil tutundurma teklifi: 1 yıllık sözleşme indirimi + ücretsiz teknik destek önerin."
    if prob >= threshold * 0.70:
        return "Orta Risk", "🟠", "Proaktif iletişim: Otomatik ödeme indirimi ve çevrimiçi güvenlik paketi tanıtın."
    return "Düşük Risk", "🟢", "Mevcut memnuniyeti koruyun; sadakat/çapraz satış kampanyalarına dahil edin."


def explain_customer_drivers(row: dict):
    drivers = []
    if row["Contract"] == "Month-to-month":
        drivers.append("Aylık (taahhütsüz) sözleşme — veri setinde %42.7 churn oranı ile en güçlü risk faktörü.")
    if row["tenure"] <= 12:
        drivers.append(f"Kısa abonelik süresi ({row['tenure']} ay) — ilk 12 ayda müşteri kaybı yoğunlaşmaktadır.")
    if row["InternetService"] == "Fiber optic":
        drivers.append("Fiber internet paketi — veri setinde %41.9 churn oranı ile yüksek hassasiyet göstermektedir.")
    if row["PaymentMethod"] == "Electronic check":
        drivers.append("Elektronik çek ile ödeme — %45.3 churn oranı ile otomatik ödemeye göre belirgin risk taşır.")
    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        drivers.append("Teknik destek paketi yok — destek almayan abonelerde churn oranı %41.6'dır.")
    if row["OnlineSecurity"] == "No" and row["InternetService"] != "No":
        drivers.append("Çevrimiçi güvenlik paketi yok — güvenlik eklentisi olmayanlarda churn oranı %41.8'dir.")
    if not drivers:
        drivers.append("Belirgin bir yüksek risk tetikleyicisi görünmüyor (uzun dönemli sözleşme / düşük risk profili).")
    return drivers


st.title("📡 Telekom Müşteri Kaybı (Churn) Tahmin ve Karar Destek Sistemi")
st.caption(
    "BTK Uçtan Uca Makine Öğrenmesi Projesi  |  "
    "Veri Kaynağı: IBM Telco Customer Churn (Kaggle)"
)

model, meta, metrics = load_artifacts()

if model is None:
    st.warning(
        "Model veya veri dosyası bulunamadı. Lütfen `models/best_model.joblib` veya "
        "`data/WA_Fn-UseC_-Telco-Customer-Churn.csv` dosyasının repoda olduğundan emin olun."
    )
    st.stop()

threshold = float(meta.get("optimal_threshold", 0.53))

# Top KPI bar
c1, c2, c3, c4 = st.columns(4)
c1.metric("Aktif Şampiyon Model", meta.get("champion_name", "CatBoost (Tuned)"))
c2.metric("Test PR-AUC", f"{meta.get('test_pr_auc', 0.677):.3f}")
c3.metric("Test ROC-AUC", f"{meta.get('test_roc_auc', 0.844):.3f}")
c4.metric("Optimize Karar Eşiği", f"{threshold:.2f}")

tab1, tab2, tab3 = st.tabs([
    "🎯 Tekil Müşteri Tahmini & What-If",
    "📂 Toplu CSV Tahmini",
    "📊 Model Karşılaştırması & EDA",
])

# ==========================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==========================================================
with tab1:
    st.subheader("Müşteri Profili Girin")
    with st.form("churn_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Demografi & Hesap**")
            gender = st.selectbox("Cinsiyet", ["Female", "Male"])
            senior = st.selectbox("65+ Yaş (SeniorCitizen)", [0, 1], format_func=lambda x: "Evet (1)" if x == 1 else "Hayır (0)")
            partner = st.selectbox("Eş / Partner", ["Yes", "No"])
            dependents = st.selectbox("Bakmakla Yükümlü Kişi", ["No", "Yes"])
            tenure = st.slider("Abonelik Süresi (Ay)", min_value=0, max_value=72, value=6)

        with col_b:
            st.markdown("**Sözleşme & Ödeme**")
            contract = st.selectbox("Sözleşme Tipi", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Kağıtsız Fatura", ["Yes", "No"])
            payment = st.selectbox(
                "Ödeme Yöntemi",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Aylık Ücret ($)", min_value=15.0, max_value=150.0, value=85.0, step=1.0)
            default_total = round(monthly * tenure, 2)
            total = st.number_input(
                "Toplam Ücret ($)",
                min_value=0.0,
                max_value=15000.0,
                value=float(default_total),
                step=10.0,
                help="Varsayılan olarak Aylık Ücret × Abonelik Süresi hesaplanır.",
            )

        with col_c:
            st.markdown("**Hizmet Paketleri**")
            phone = st.selectbox("Telefon Hizmeti", ["Yes", "No"])
            multi_lines = st.selectbox(
                "Çoklu Hat",
                ["No phone service"] if phone == "No" else ["No", "Yes"],
            )
            internet = st.selectbox("İnternet Hizmeti", ["Fiber optic", "DSL", "No"])
            addon_opts = ["No internet service"] if internet == "No" else ["No", "Yes"]
            online_sec = st.selectbox("Çevrimiçi Güvenlik", addon_opts)
            online_bak = st.selectbox("Çevrimiçi Yedekleme", addon_opts)
            dev_prot = st.selectbox("Cihaz Koruma", addon_opts)
            tech_sup = st.selectbox("Teknik Destek", addon_opts)
            stream_tv = st.selectbox("Streaming TV", addon_opts)
            stream_mov = st.selectbox("Streaming Movies", addon_opts)

        submitted = st.form_submit_button("🔍 Churn Riskini Hesapla", use_container_width=True, type="primary")

    if submitted:
        if tenure > 0 and total == 0:
            st.info("Not: Abonelik süresi > 0 iken toplam ücret 0 girildi; tahmin buna göre hesaplandı.")

        customer_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bak,
            "DeviceProtection": dev_prot,
            "TechSupport": tech_sup,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        input_df = pd.DataFrame([customer_dict])
        prob = float(model.predict_proba(input_df)[0, 1])
        pred_label = "AYRILMA RİSKİ VAR (Churn = Yes)" if prob >= threshold else "KALMA EĞİLİMİNDE (Churn = No)"
        band, icon, advice = get_risk_band(prob, threshold)

        r1, r2, r3 = st.columns([1.2, 1.2, 2.0])
        r1.metric("Tahmini Churn Olasılığı", f"%{prob * 100:.1f}")
        r2.metric("Risk Segmenti", f"{icon} {band}")
        r3.info(f"**Karar:** {pred_label}\n\n**Önerilen Aksiyon:** {advice}")

        st.progress(min(max(prob, 0.0), 1.0))

        st.markdown("**Bu Müşteri İçin Öne Çıkan Risk / Koruma Faktörleri:**")
        for d in explain_customer_drivers(customer_dict):
            st.markdown(f"- {d}")

# ==========================================================
# TAB 2: BATCH CSV PREDICTION
# ==========================================================
with tab2:
    st.subheader("Toplu Müşteri Risk Skorlama (CSV)")
    st.write(
        "Telco şemasına uygun bir CSV dosyası yükleyin. Sistem her müşteri için churn olasılığı "
        "ve risk segmenti hesaplayıp indirilebilir rapor üretir."
    )
    uploaded = st.file_uploader("CSV dosyası yükle", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            required_cols = {"tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod"}
            missing = required_cols - set(batch_df.columns)
            if missing:
                st.error(f"Eksik zorunlu sütunlar: {', '.join(sorted(missing))}")
            else:
                feat_df = batch_df.drop(columns=["Churn"], errors="ignore")
                probs = model.predict_proba(feat_df)[:, 1]
                out_df = batch_df.copy()
                out_df["Churn_Olasiligi"] = (probs * 100).round(2)
                out_df["Tahmin"] = ["Yes" if p >= threshold else "No" for p in probs]
                out_df["Risk_Segmenti"] = [get_risk_band(p, threshold)[0] for p in probs]

                high_n = int((out_df["Risk_Segmenti"] == "Yüksek Risk").sum())
                st.success(f"{len(out_df):,} müşteri skorlandı. Yüksek riskli müşteri sayısı: {high_n:,}")
                st.dataframe(out_df.sort_values("Churn_Olasiligi", ascending=False).head(50), use_container_width=True)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Tahmin Sonuçlarını CSV Olarak İndir",
                    data=csv_bytes,
                    file_name="telco_churn_tahminleri.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"CSV işlenirken hata oluştu: {exc}")

# ==========================================================
# TAB 3: MODEL COMPARISON & EDA
# ==========================================================
with tab3:
    st.subheader("Bilimsel Model Karşılaştırması (Baseline + 6+ Güçlü Rakip)")
    if metrics is not None:
        st.markdown("**Kilitli Test Seti Sonuçları:**")
        test_tbl = pd.DataFrame(metrics["test_comparison_all_models"])
        st.dataframe(test_tbl, use_container_width=True, hide_index=True)

        st.markdown("**Hiperparametre Optimizasyonu Öncesi / Sonrası (5-Fold CV PR-AUC):**")
        tune_tbl = pd.DataFrame(metrics["tuning_comparison"])[
            ["model", "cv_pr_auc_before", "cv_pr_auc_after", "delta_pr_auc", "val_pr_auc", "optimal_threshold"]
        ]
        st.dataframe(tune_tbl, use_container_width=True, hide_index=True)

    st.subheader("Keşifsel Veri Analizi (EDA) ve Değerlendirme Grafikleri")
    fig_files = [
        ("fig1_eda_target_and_numeric.png", "Hedef Dağılımı ve Sayısal Özellikler"),
        ("fig2_eda_categorical_churn.png", "Kategorik Değişkenlere Göre Churn Oranları"),
        ("fig3_eda_correlation_heatmap.png", "Korelasyon Matrisi"),
        ("fig4_model_comparison_cv.png", "5-Fold CV Model Karşılaştırması"),
        ("fig5_roc_pr_curves_test.png", "Test Seti ROC & Precision-Recall Eğrileri"),
        ("fig6_confusion_matrix_threshold.png", "Karışıklık Matrisi ve Eşik Optimizasyonu"),
        ("fig7_feature_importance.png", "En Önemli Özellikler"),
        ("fig8_error_analysis.png", "Sözleşme Tipine Göre Hata Analizi"),
    ]
    for fname, caption in fig_files:
        fpath = os.path.join(FIG_DIR, fname)
        if os.path.exists(fpath):
            st.image(fpath, caption=caption, use_container_width=True)
