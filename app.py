"""
BTK Uçtan Uca Makine Öğrenmesi Projesi — Telco Customer Churn
Aşama 13–14: Modern & Profesyonel Streamlit Karar Destek Arayüzü

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

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TelcoFeatureEngineer,
)

st.set_page_config(
    page_title="Telco Churn AI | Karar Destek Sistemi",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/best_model.joblib"
META_PATH = "models/model_meta.json"
METRICS_PATH = "results/metrics.json"
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIG_DIR = "figures"
RANDOM_STATE = 42

# ==========================================================
# CUSTOM EXECUTIVE UI STYLING (CSS)
# ==========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
        max-width: 1220px;
    }

    /* Executive Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0B1F3A 0%, #163A5F 58%, #1E293B 100%);
        border-radius: 16px;
        padding: 26px 30px;
        margin-bottom: 1.35rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 12px 30px rgba(11, 31, 58, 0.22);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 6px;
        height: 100%;
        background: linear-gradient(180deg, #E11D48 0%, #38BDF8 100%);
    }
    .hero-kicker {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #7DD3FC;
        margin-bottom: 6px;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 0.96rem;
        color: #CBD5E1;
        margin: 0;
        line-height: 1.45;
    }

    /* Executive KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 1.4rem;
    }
    @media (max-width: 900px) {
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .kpi-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-top: 3px solid #38BDF8;
        border-radius: 12px;
        padding: 15px 18px;
    }
    .kpi-card.accent {
        border-top-color: #E11D48;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.78;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 0.78rem;
        opacity: 0.72;
    }

    /* Prediction Result Card */
    .result-card {
        border-radius: 14px;
        padding: 22px 24px;
        margin: 14px 0 18px 0;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(15, 23, 42, 0.55);
    }
    .result-high { border-left: 6px solid #E11D48; }
    .result-mid  { border-left: 6px solid #F59E0B; }
    .result-low  { border-left: 6px solid #10B981; }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 8px;
    }
    .pill-high { background: rgba(225, 29, 72, 0.18); color: #FB7185; border: 1px solid rgba(225, 29, 72, 0.4); }
    .pill-mid  { background: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .pill-low  { background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }

    .driver-item {
        background: rgba(148, 163, 184, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 9px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.90rem;
        line-height: 1.45;
    }

    /* Responsible Use Footer */
    .disclaimer-box {
        margin-top: 1.6rem;
        padding: 14px 18px;
        border-radius: 10px;
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.22);
        font-size: 0.83rem;
        line-height: 1.5;
        opacity: 0.88;
    }

    /* Tabs polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# MODEL LOADING (VERSION-PROOF WITH DETERMINISTIC FALLBACK)
# ==========================================================
def _train_champion_in_place():
    """
    Deterministic fallback when a joblib pickle created in another Python/scikit-learn
    version (e.g., Colab Python 3.12 -> Streamlit Cloud Python 3.14) cannot be unpickled.
    Rebuilds and fits the exact tuned champion pipeline on the same 70% train split.
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

    pipe = Pipeline(
        steps=[
            ("fe", TelcoFeatureEngineer()),
            ("prep", preprocessor),
            ("model", estimator),
        ]
    )
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
        model = _train_champion_in_place()

    return model, meta, metrics


def get_risk_profile(prob: float, threshold: float):
    if prob >= max(threshold, 0.60):
        return (
            "Yüksek Risk",
            "🔴",
            "result-high",
            "pill-high",
            "Acil tutundurma teklifi: 1 yıllık sözleşme indirimi + 3 ay ücretsiz Teknik Destek önerin.",
        )
    if prob >= threshold * 0.70:
        return (
            "Orta Risk",
            "🟠",
            "result-mid",
            "pill-mid",
            "Proaktif iletişim: Otomatik ödeme indirimi ve Çevrimiçi Güvenlik paketi tanıtın.",
        )
    return (
        "Düşük Risk",
        "🟢",
        "result-low",
        "pill-low",
        "Mevcut memnuniyeti koruyun; sadakat programı veya çapraz satış tekliflerine dahil edin.",
    )


def explain_customer_drivers(row: dict):
    drivers = []
    if row["Contract"] == "Month-to-month":
        drivers.append(("⚠️ Risk", "Aylık (taahhütsüz) sözleşme — veri setinde %42,7 churn oranı ile en güçlü risk faktörüdür."))
    elif row["Contract"] == "Two year":
        drivers.append(("✅ Koruyucu", "İki yıllık uzun dönemli sözleşme — veri setinde churn oranı yalnızca %2,8'dir."))

    if row["tenure"] <= 12:
        drivers.append(("⚠️ Risk", f"Kısa abonelik süresi ({row['tenure']} ay) — müşteri kaybı ilk 12 ayda yoğunlaşır."))
    elif row["tenure"] >= 48:
        drivers.append(("✅ Koruyucu", f"Uzun abonelik sadakati ({row['tenure']} ay) — kayıp olasılığını belirgin biçimde azaltır."))

    if row["InternetService"] == "Fiber optic":
        drivers.append(("⚠️ Risk", "Fiber internet paketi — veri setinde %41,9 churn oranı ile yüksek fiyat/hizmet hassasiyeti taşır."))
    if row["PaymentMethod"] == "Electronic check":
        drivers.append(("⚠️ Risk", "Elektronik çek ile ödeme — %45,3 churn oranı ile otomatik ödemeye (%15,2) göre 3 kat risklidir."))
    elif "automatic" in row["PaymentMethod"]:
        drivers.append(("✅ Koruyucu", "Otomatik ödeme talimatı — abonelik devamlılığını destekleyen güçlü bir koruyucu sinyaldir."))

    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        drivers.append(("⚠️ Risk", "Teknik destek paketi yok — destek almayan abonelerde churn oranı %41,6'dır."))
    if row["OnlineSecurity"] == "No" and row["InternetService"] != "No":
        drivers.append(("⚠️ Risk", "Çevrimiçi güvenlik paketi yok — güvenlik eklentisi olmayanlarda churn oranı %41,8'dir."))

    if not drivers:
        drivers.append(("ℹ️ Özet", "Müşteri profili dengeli risk sinyalleri göstermektedir."))
    return drivers


# ==========================================================
# DEMO PRESETS FOR 1-CLICK LIVE PRESENTATION
# ==========================================================
PRESETS = {
    "high": {
        "gender": "Female", "senior": 0, "partner": "No", "dependents": "No", "tenure": 4,
        "contract": "Month-to-month", "paperless": "Yes", "payment": "Electronic check",
        "monthly": 89.0, "total": 356.0, "phone": "Yes", "multi_lines": "No",
        "internet": "Fiber optic", "online_sec": "No", "online_bak": "No",
        "dev_prot": "No", "tech_sup": "No", "stream_tv": "Yes", "stream_mov": "No",
    },
    "mid": {
        "gender": "Male", "senior": 0, "partner": "Yes", "dependents": "No", "tenure": 18,
        "contract": "Month-to-month", "paperless": "Yes", "payment": "Mailed check",
        "monthly": 65.0, "total": 1170.0, "phone": "Yes", "multi_lines": "No",
        "internet": "DSL", "online_sec": "Yes", "online_bak": "No",
        "dev_prot": "No", "tech_sup": "No", "stream_tv": "No", "stream_mov": "No",
    },
    "low": {
        "gender": "Female", "senior": 0, "partner": "Yes", "dependents": "Yes", "tenure": 60,
        "contract": "Two year", "paperless": "No", "payment": "Credit card (automatic)",
        "monthly": 85.0, "total": 5100.0, "phone": "Yes", "multi_lines": "Yes",
        "internet": "DSL", "online_sec": "Yes", "online_bak": "Yes",
        "dev_prot": "Yes", "tech_sup": "Yes", "stream_tv": "Yes", "stream_mov": "Yes",
    },
}

if "preset_data" not in st.session_state:
    st.session_state["preset_data"] = PRESETS["high"].copy()

# ==========================================================
# SIDEBAR — QUICK DEMO PRESETS & PROJECT INFO
# ==========================================================
with st.sidebar:
    st.markdown("### ⚡ Hızlı Demo Senaryoları")
    st.caption("Sunum sırasında tek tıkla farklı müşteri profillerini yükleyebilirsiniz:")
    if st.button("🔴 Yüksek Riskli Müşteri", use_container_width=True):
        st.session_state["preset_data"] = PRESETS["high"].copy()
    if st.button("🟠 Orta Riskli Müşteri", use_container_width=True):
        st.session_state["preset_data"] = PRESETS["mid"].copy()
    if st.button("🟢 Düşük Riskli Müşteri", use_container_width=True):
        st.session_state["preset_data"] = PRESETS["low"].copy()

    st.divider()
    st.markdown("### 📌 Proje Künyesi")
    st.markdown(
        "- **Geliştirici:** Arda KİLİT\n"
        "- **Program:** BTK Uçtan Uca ML\n"
        "- **Veri Seti:** IBM Telco (7.043 kayıt)\n"
        "- **Model:** CatBoost (Tuned)\n"
        "- **Kod:** [GitHub Deposu](https://github.com/Arda-Kilit/telco-churn-ml-project)"
    )

# ==========================================================
# HERO HEADER & KPI STRIP
# ==========================================================
model, meta, metrics = load_artifacts()

if model is None:
    st.error(
        "Model veya veri dosyası bulunamadı. Lütfen `models/best_model.joblib` veya "
        "`data/WA_Fn-UseC_-Telco-Customer-Churn.csv` dosyasının repoda olduğundan emin olun."
    )
    st.stop()

threshold = float(meta.get("optimal_threshold", 0.53))
champ_name = meta.get("champion_name", "CatBoost")
test_pr = float(meta.get("test_pr_auc", 0.6772))
test_roc = float(meta.get("test_roc_auc", 0.8445))
test_rec = float(meta.get("test_recall", 0.7616))

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-kicker">BTK Uçtan Uca Makine Öğrenmesi Projesi · Arda KİLİT</div>
        <div class="hero-title">📡 Telekom Müşteri Kaybı (Churn) Karar Destek Sistemi</div>
        <div class="hero-sub">
            Sızıntısız Pipeline, 8 alan özniteliği ve 7 güçlü algoritmanın bilimsel karşılaştırmasıyla
            müşteri kaybını erken tespit eden ve tutundurma aksiyonu öneren etkileşimli analitik platform.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="kpi-grid">
        <div class="kpi-card accent">
            <div class="kpi-label">Şampiyon Model</div>
            <div class="kpi-value">{champ_name}</div>
            <div class="kpi-sub">7 güçlü algoritma arasından seçildi</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Kilitli Test PR-AUC</div>
            <div class="kpi-value">{test_pr:.4f}</div>
            <div class="kpi-sub">Dummy Baseline'a göre 2,55× artış</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Kilitli Test ROC-AUC</div>
            <div class="kpi-value">{test_roc:.4f}</div>
            <div class="kpi-sub">n = 1.057 kilitli test müşterisi</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Churn Yakalama (Recall)</div>
            <div class="kpi-value">%{test_rec * 100:.1f}</div>
            <div class="kpi-sub">Optimize karar eşiği: {threshold:.2f}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Tekil Risk Analizi & What-If",
        "📂 Toplu CSV Skorlama",
        "📊 Model Karşılaştırması & EDA",
    ]
)

# ==========================================================
# TAB 1: SINGLE CUSTOMER PREDICTION & WHAT-IF
# ==========================================================
with tab1:
    p = st.session_state["preset_data"]
    st.markdown("#### Müşteri Profili ve What-If Simülatörü")
    st.caption("Sol menüden hazır senaryo seçebilir veya aşağıdan özellikleri değiştirerek anlık churn riskini görebilirsiniz.")

    with st.form("churn_form"):
        col_a, col_b, col_c = st.columns(3, gap="medium")

        with col_a:
            st.markdown("##### 👤 Demografi & Hesap")
            gender = st.selectbox("Cinsiyet", ["Female", "Male"], index=["Female", "Male"].index(p["gender"]))
            senior = st.selectbox(
                "65+ Yaş (SeniorCitizen)",
                [0, 1],
                index=[0, 1].index(p["senior"]),
                format_func=lambda x: "Evet (1)" if x == 1 else "Hayır (0)",
            )
            partner = st.selectbox("Eş / Partner", ["Yes", "No"], index=["Yes", "No"].index(p["partner"]))
            dependents = st.selectbox("Bakmakla Yükümlü Kişi", ["No", "Yes"], index=["No", "Yes"].index(p["dependents"]))
            tenure = st.slider("Abonelik Süresi (Ay)", min_value=0, max_value=72, value=int(p["tenure"]))

        with col_b:
            st.markdown("##### 💳 Sözleşme & Fatura")
            contract_opts = ["Month-to-month", "One year", "Two year"]
            contract = st.selectbox("Sözleşme Tipi", contract_opts, index=contract_opts.index(p["contract"]))
            paperless = st.selectbox("Kağıtsız Fatura", ["Yes", "No"], index=["Yes", "No"].index(p["paperless"]))
            pay_opts = [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
            payment = st.selectbox("Ödeme Yöntemi", pay_opts, index=pay_opts.index(p["payment"]))
            monthly = st.number_input("Aylık Ücret ($)", min_value=15.0, max_value=150.0, value=float(p["monthly"]), step=1.0)
            total = st.number_input(
                "Toplam Ücret ($)",
                min_value=0.0,
                max_value=15000.0,
                value=float(p["total"]),
                step=10.0,
                help="Genellikle Aylık Ücret × Abonelik Süresi civarındadır.",
            )

        with col_c:
            st.markdown("##### 🌐 Hizmet Paketleri")
            phone = st.selectbox("Telefon Hizmeti", ["Yes", "No"], index=["Yes", "No"].index(p["phone"]))
            multi_opts = ["No phone service"] if phone == "No" else ["No", "Yes"]
            multi_idx = multi_opts.index(p["multi_lines"]) if p["multi_lines"] in multi_opts else 0
            multi_lines = st.selectbox("Çoklu Hat", multi_opts, index=multi_idx)

            net_opts = ["Fiber optic", "DSL", "No"]
            internet = st.selectbox("İnternet Hizmeti", net_opts, index=net_opts.index(p["internet"]))
            addon_opts = ["No internet service"] if internet == "No" else ["No", "Yes"]

            def _idx(val):
                return addon_opts.index(val) if val in addon_opts else 0

            online_sec = st.selectbox("Çevrimiçi Güvenlik", addon_opts, index=_idx(p["online_sec"]))
            online_bak = st.selectbox("Çevrimiçi Yedekleme", addon_opts, index=_idx(p["online_bak"]))
            dev_prot = st.selectbox("Cihaz Koruma", addon_opts, index=_idx(p["dev_prot"]))
            tech_sup = st.selectbox("Teknik Destek", addon_opts, index=_idx(p["tech_sup"]))
            stream_tv = st.selectbox("Streaming TV", addon_opts, index=_idx(p["stream_tv"]))
            stream_mov = st.selectbox("Streaming Movies", addon_opts, index=_idx(p["stream_mov"]))

        submitted = st.form_submit_button("🔍 Churn Riskini Hesapla", use_container_width=True, type="primary")

    if submitted:
        if tenure > 0 and total == 0:
            st.info("Not: Abonelik süresi 0'dan büyükken toplam ücret 0 girildi; tahmin bu girdiye göre hesaplandı.")

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
        band, icon, card_cls, pill_cls, advice = get_risk_profile(prob, threshold)

        st.markdown(
            f"""
            <div class="result-card {card_cls}">
                <span class="badge-pill {pill_cls}">{icon} {band.upper()}</span>
                <div style="display:flex; flex-wrap:wrap; align-items:baseline; gap:24px; margin-bottom:10px;">
                    <div style="font-size:2.4rem; font-weight:800; line-height:1;">%{prob * 100:.1f}</div>
                    <div style="font-size:1.05rem; font-weight:700;">{pred_label}</div>
                </div>
                <div style="font-size:0.93rem; opacity:0.92;">
                    <strong>Önerilen CRM Aksiyonu:</strong> {advice}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(prob, 0.0), 1.0))

        st.markdown("##### Bu Müşteri İçin Öne Çıkan Risk ve Koruma Sinyalleri")
        for tag, text in explain_customer_drivers(customer_dict):
            st.markdown(
                f'<div class="driver-item"><strong>{tag}:</strong> {text}</div>',
                unsafe_allow_html=True,
            )

# ==========================================================
# TAB 2: BATCH CSV PREDICTION
# ==========================================================
with tab2:
    st.markdown("#### Toplu Müşteri Risk Skorlama (CSV)")
    st.caption(
        "Telco şemasına uygun bir CSV dosyası yükleyin; sistem her müşteri için olasılık, "
        "risk segmenti ve tutundurma önceliği hesaplayıp indirilebilir rapor üretir."
    )
    uploaded = st.file_uploader("Müşteri listesi (.csv) yükleyin", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            required_cols = {"tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod"}
            missing = required_cols - set(batch_df.columns)
            if missing:
                st.error(f"Eksik zorunlu sütunlar tespit edildi: {', '.join(sorted(missing))}")
            else:
                feat_df = batch_df.drop(columns=["Churn"], errors="ignore")
                probs = model.predict_proba(feat_df)[:, 1]
                out_df = batch_df.copy()
                out_df["Churn_Olasiligi_%"] = (probs * 100).round(2)
                out_df["Tahmin"] = ["Yes" if pr >= threshold else "No" for pr in probs]
                out_df["Risk_Segmenti"] = [get_risk_profile(pr, threshold)[0] for pr in probs]

                high_n = int((out_df["Risk_Segmenti"] == "Yüksek Risk").sum())
                mid_n = int((out_df["Risk_Segmenti"] == "Orta Risk").sum())
                low_n = int((out_df["Risk_Segmenti"] == "Düşük Risk").sum())

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Toplam Skorlanan", f"{len(out_df):,}")
                m2.metric("🔴 Yüksek Risk", f"{high_n:,}")
                m3.metric("🟠 Orta Risk", f"{mid_n:,}")
                m4.metric("🟢 Düşük Risk", f"{low_n:,}")

                st.dataframe(
                    out_df.sort_values("Churn_Olasiligi_%", ascending=False).head(50),
                    use_container_width=True,
                )
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Risk Skorlama Raporunu CSV Olarak İndir",
                    data=csv_bytes,
                    file_name="telco_churn_tahminleri.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as exc:
            st.error(f"CSV işlenirken beklenmeyen bir hata oluştu: {exc}")

# ==========================================================
# TAB 3: MODEL COMPARISON & EDA GALLERY
# ==========================================================
with tab3:
    st.markdown("#### Bilimsel Model Karşılaştırması (Baseline + 6 Güçlü Rakip)")
    if metrics is not None:
        test_tbl = pd.DataFrame(metrics["test_comparison_all_models"])[
            [
                "model",
                "test_pr_auc",
                "test_roc_auc",
                "test_f1_at_opt",
                "test_recall_at_opt",
                "test_precision_at_opt",
                "test_accuracy_at_opt",
                "threshold_used",
            ]
        ].rename(
            columns={
                "model": "Model",
                "test_pr_auc": "Test PR-AUC",
                "test_roc_auc": "Test ROC-AUC",
                "test_f1_at_opt": "Test F1",
                "test_recall_at_opt": "Recall",
                "test_precision_at_opt": "Precision",
                "test_accuracy_at_opt": "Accuracy",
                "threshold_used": "Eşik",
            }
        )
        st.dataframe(
            test_tbl.style.format(
                {
                    "Test PR-AUC": "{:.4f}",
                    "Test ROC-AUC": "{:.4f}",
                    "Test F1": "{:.4f}",
                    "Recall": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Accuracy": "{:.4f}",
                    "Eşik": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("##### Hiperparametre Optimizasyonu Öncesi / Sonrası (5-Fold CV PR-AUC)")
        tune_tbl = pd.DataFrame(metrics["tuning_comparison"])[
            ["model", "cv_pr_auc_before", "cv_pr_auc_after", "delta_pr_auc", "val_pr_auc", "optimal_threshold"]
        ].rename(
            columns={
                "model": "Aday Model",
                "cv_pr_auc_before": "Önce CV PR-AUC",
                "cv_pr_auc_after": "Sonra CV PR-AUC",
                "delta_pr_auc": "Artış (Δ)",
                "val_pr_auc": "Validation PR-AUC",
                "optimal_threshold": "Optimal Eşik",
            }
        )
        st.dataframe(
            tune_tbl.style.format(
                {
                    "Önce CV PR-AUC": "{:.4f}",
                    "Sonra CV PR-AUC": "{:.4f}",
                    "Artış (Δ)": "+{:.4f}",
                    "Validation PR-AUC": "{:.4f}",
                    "Optimal Eşik": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Keşifsel Veri Analizi (EDA) ve Değerlendirme Grafikleri")
    fig_files = [
        ("fig1_eda_target_and_numeric.png", "Şekil 1: Hedef Dağılımı ve Sayısal Özellikler"),
        ("fig2_eda_categorical_churn.png", "Şekil 2: Kategorik Değişkenlere Göre Churn Oranları"),
        ("fig3_eda_correlation_heatmap.png", "Şekil 3: Korelasyon Matrisi"),
        ("fig4_model_comparison_cv.png", "Şekil 4: 5-Fold CV Model Karşılaştırması"),
        ("fig5_roc_pr_curves_test.png", "Şekil 5: Test Seti ROC & Precision-Recall Eğrileri"),
        ("fig6_confusion_matrix_threshold.png", "Şekil 6: Karışıklık Matrisi ve Eşik Optimizasyonu"),
        ("fig7_feature_importance.png", "Şekil 7: En Önemli 15 Özellik (CatBoost)"),
        ("fig8_error_analysis.png", "Şekil 8: Sözleşme Tipine Göre Hata Analizi"),
    ]
    existing_figs = [(os.path.join(FIG_DIR, fn), cap) for fn, cap in fig_files if os.path.exists(os.path.join(FIG_DIR, fn))]
    for i in range(0, len(existing_figs), 2):
        cols = st.columns(2, gap="medium")
        for col, (fpath, caption) in zip(cols, existing_figs[i : i + 2]):
            with col:
                st.image(fpath, caption=caption, use_container_width=True)

# ==========================================================
# RESPONSIBLE USE & MODEL LIMITATIONS FOOTER (YÖNERGE BÖLÜM 11)
# ==========================================================
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Model Sınırlılığı ve Sorumlu Kullanım Notu:</strong>
        Bu uygulama, IBM Telco Customer Churn kesit veri seti üzerinde eğitilmiş olasılıksal bir tahmin modeli sunar;
        kesin sonuç garanti etmez ve yanlış pozitif (FP) veya yanlış negatif (FN) tahminler içerebilir.
        Model çıktıları, iş kararlarında tek başına değil, müşteri ilişkileri uzman değerlendirmesiyle birlikte kullanılmalıdır.
    </div>
    """,
    unsafe_allow_html=True,
)
