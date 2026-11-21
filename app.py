"""
BTK Uçtan Uca Makine Öğrenmesi Projesi — Telco Customer Churn
Aşama 13–14: Üst Düzey SaaS Kalitesinde Streamlit Karar Destek Platformu

Öne Çıkan Özellikler:
- Dinamik SVG Risk Göstergesi (Eşik İşaretli) & Önce/Sonra Görsel Karşılaştırma
- Gerçek CatBoost Modeliyle Canlı "What-If Tutundurma & Yıllık Gelir Koruma Simülatörü"
- Sol Menüde Tek Tıkla Anında Çalışan Demo Senaryoları + Dinamik Karar Eşiği (Bonus)
- Toplu CSV Skorlama + Tek Tıkla Hazır Örnek Veri Testi + Risk Segment Filtresi
- İnteraktif Model Karşılaştırması, Model Kartı (Bonus) ve 8 Gerçek Analiz Grafiği

Çalıştırma:
    streamlit run app.py
"""

import json
import math
import os
import re

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
    page_title="Telco Churn AI | Karar Destek Platformu",
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


def _html(raw: str) -> str:
    """Strips per-line leading whitespace so Streamlit Markdown never renders HTML as a code block."""
    return "\n".join(line.strip() for line in raw.strip().splitlines() if line.strip())


# ==========================================================
# EXECUTIVE SAAS CSS THEME (OBSIDIAN + ELECTRIC TEAL + CORAL)
# ==========================================================
st.markdown(
    _html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.5rem;
            max-width: 1260px;
        }

        /* Hero Banner */
        .hero-wrap {
            background: radial-gradient(circle at 86% 18%, rgba(13, 148, 136, 0.36), transparent 44%),
                        radial-gradient(circle at 10% 88%, rgba(244, 63, 94, 0.24), transparent 42%),
                        linear-gradient(135deg, #070B14 0%, #0F172A 55%, #111827 100%);
            border-radius: 18px;
            padding: 26px 32px;
            margin-bottom: 1.25rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 18px 40px rgba(2, 6, 23, 0.35);
            position: relative;
            overflow: hidden;
        }
        .hero-wrap::before {
            content: "";
            position: absolute;
            top: 0; left: 0; width: 100%; height: 4px;
            background: linear-gradient(90deg, #0D9488 0%, #38BDF8 50%, #F43F5E 100%);
        }
        .hero-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #5EEAD4;
            background: rgba(13, 148, 136, 0.20);
            border: 1px solid rgba(94, 234, 212, 0.35);
            margin-bottom: 10px;
        }
        .hero-title {
            font-size: 1.92rem;
            font-weight: 800;
            color: #FFFFFF;
            margin: 0 0 8px 0;
            line-height: 1.18;
            letter-spacing: -0.02em;
        }
        .hero-desc {
            font-size: 0.94rem;
            color: #CBD5E1;
            margin: 0;
            max-width: 840px;
            line-height: 1.5;
        }

        /* KPI Strip */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 1.35rem;
        }
        @media (max-width: 900px) {
            .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        }
        .kpi-box {
            background: rgba(15, 23, 42, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 14px;
            padding: 15px 18px;
            position: relative;
            overflow: hidden;
        }
        .kpi-box::after {
            content: "";
            position: absolute;
            top: 0; left: 0; width: 100%; height: 3px;
            background: #0D9488;
        }
        .kpi-box.coral::after { background: #F43F5E; }
        .kpi-box.sky::after { background: #38BDF8; }
        .kpi-lbl {
            font-size: 0.73rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            opacity: 0.72;
            margin-bottom: 5px;
        }
        .kpi-val {
            font-size: 1.62rem;
            font-weight: 800;
            line-height: 1.12;
            margin-bottom: 4px;
        }
        .kpi-sub { font-size: 0.77rem; opacity: 0.74; }

        /* Result & Simulation Panels */
        .panel-card {
            border-radius: 16px;
            padding: 20px 22px;
            border: 1px solid rgba(148, 163, 184, 0.24);
            background: rgba(15, 23, 42, 0.56);
            margin-bottom: 12px;
        }
        .badge-status {
            display: inline-block;
            padding: 5px 14px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .st-high { background: rgba(244, 63, 94, 0.18); color: #FDA4AF; border: 1px solid rgba(244, 63, 94, 0.45); }
        .st-mid  { background: rgba(245, 158, 11, 0.18); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.45); }
        .st-low  { background: rgba(16, 185, 129, 0.18); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.45); }

        .sim-card {
            border-radius: 15px;
            padding: 18px 22px;
            background: linear-gradient(135deg, rgba(13, 148, 136, 0.16), rgba(56, 189, 248, 0.08));
            border: 1px solid rgba(94, 234, 212, 0.38);
            margin-bottom: 12px;
        }
        .bar-track {
            width: 100%;
            height: 10px;
            background: rgba(148, 163, 184, 0.20);
            border-radius: 999px;
            overflow: hidden;
            margin: 5px 0 10px 0;
        }
        .bar-fill-before { height: 100%; border-radius: 999px; background: #F43F5E; }
        .bar-fill-after  { height: 100%; border-radius: 999px; background: #10B981; }

        .driver-chip {
            background: rgba(148, 163, 184, 0.10);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 10px;
            padding: 11px 14px;
            margin-bottom: 9px;
            font-size: 0.89rem;
            line-height: 1.45;
        }
        .driver-chip.risk { border-left: 4px solid #F43F5E; }
        .driver-chip.safe { border-left: 4px solid #10B981; }

        .model-card-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin: 12px 0 18px 0;
        }
        @media (max-width: 900px) {
            .model-card-grid { grid-template-columns: 1fr; }
        }
        .mc-item {
            background: rgba(15, 23, 42, 0.52);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            padding: 14px 16px;
            font-size: 0.87rem;
            line-height: 1.5;
        }
        .mc-title {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #5EEAD4;
            margin-bottom: 6px;
        }

        .disclaimer-card {
            margin-top: 1.8rem;
            padding: 14px 18px;
            border-radius: 12px;
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.22);
            font-size: 0.82rem;
            line-height: 1.5;
            opacity: 0.88;
        }

        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 10px 18px;
            font-weight: 600;
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


# ==========================================================
# VERSION-PROOF MODEL LOADER & DATA CACHE
# ==========================================================
def _train_champion_in_place():
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


@st.cache_data
def load_sample_batch():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    return df.sample(n=min(60, len(df)), random_state=RANDOM_STATE).drop(columns=["Churn"], errors="ignore")


def get_risk_profile(prob: float, threshold: float):
    if prob >= max(threshold, 0.60):
        return (
            "Yüksek Risk",
            "🔴",
            "st-high",
            "#F43F5E",
            "Acil tutundurma teklifi: 1 yıllık sözleşme indirimi + 3 ay ücretsiz Teknik Destek önerin.",
        )
    if prob >= threshold * 0.70:
        return (
            "Orta Risk",
            "🟠",
            "st-mid",
            "#F59E0B",
            "Proaktif iletişim: Otomatik ödeme indirimi ve Çevrimiçi Güvenlik paketi tanıtın.",
        )
    return (
        "Düşük Risk",
        "🟢",
        "st-low",
        "#10B981",
        "Mevcut memnuniyeti koruyun; sadakat programı veya çapraz satış tekliflerine dahil edin.",
    )


def render_svg_gauge(prob: float, threshold: float, color: str) -> str:
    pct = max(0.0, min(1.0, prob))
    radius = 84
    circumference = math.pi * radius
    dash_filled = circumference * pct
    dash_empty = circumference - dash_filled

    thr_angle = math.pi * (1.0 - max(0.0, min(1.0, threshold)))
    tx1 = 110 + (radius - 10) * math.cos(thr_angle)
    ty1 = 105 - (radius - 10) * math.sin(thr_angle)
    tx2 = 110 + (radius + 10) * math.cos(thr_angle)
    ty2 = 105 - (radius + 10) * math.sin(thr_angle)

    return (
        '<div style="display:flex; justify-content:center; align-items:center; padding:6px 0;">'
        '<svg viewBox="0 0 220 132" width="246" height="148" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M 26 105 A 84 84 0 0 1 194 105" fill="none" stroke="rgba(148,163,184,0.22)" stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M 26 105 A 84 84 0 0 1 194 105" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" stroke-dasharray="{dash_filled:.1f} {dash_empty:.1f}"/>'
        f'<line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" stroke="#F8FAFC" stroke-width="2.6" stroke-linecap="round"/>'
        f'<text x="110" y="88" text-anchor="middle" fill="currentColor" font-size="31" font-weight="800" font-family="Inter, sans-serif">%{pct * 100:.1f}</text>'
        f'<text x="110" y="114" text-anchor="middle" fill="#94A3B8" font-size="10.5" font-weight="600" font-family="Inter, sans-serif">CHURN OLASILIĞI · EŞİK {threshold:.2f}</text>'
        "</svg></div>"
    )


def explain_customer_drivers(row: dict):
    drivers = []
    if row["Contract"] == "Month-to-month":
        drivers.append(("risk", "Aylık (Taahhütsüz) Sözleşme", "Veri setinde %42,7 churn oranı ile en güçlü kayıp tetikleyicisi."))
    elif row["Contract"] == "Two year":
        drivers.append(("safe", "İki Yıllık Sözleşme", "Veri setinde churn oranı yalnızca %2,8 düzeyinde."))
    elif row["Contract"] == "One year":
        drivers.append(("safe", "Bir Yıllık Sözleşme", "Aylık sözleşmeye (%42,7) kıyasla belirgin biçimde düşük risk (%11,3)."))

    if row["tenure"] <= 12:
        drivers.append(("risk", f"Kısa Abonelik Süresi ({row['tenure']} ay)", "Müşteri kaybı ilk 12 aylık erken dönemde yoğunlaşır."))
    elif row["tenure"] >= 48:
        drivers.append(("safe", f"Yüksek Sadakat ({row['tenure']} ay)", "Uzun abonelik geçmişi ayrılma olasılığını belirgin biçimde düşürür."))

    if row["InternetService"] == "Fiber optic":
        drivers.append(("risk", "Fiber İnternet Paketi", "Veri setinde %41,9 churn oranı ile yüksek fiyat/memnuniyet hassasiyeti."))
    if row["PaymentMethod"] == "Electronic check":
        drivers.append(("risk", "Elektronik Çek Ödemesi", "%45,3 churn oranı ile otomatik ödemeye (%15,2) göre 3 kat riskli."))
    elif "automatic" in row["PaymentMethod"]:
        drivers.append(("safe", "Otomatik Ödeme Talimatı", "Abonelik devamlılığını koruyan güçlü bir ödeme sinyali."))

    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        drivers.append(("risk", "Teknik Destek Yok", "Destek paketi olmayan abonelerde churn oranı %41,6'dır."))
    if row["OnlineSecurity"] == "No" and row["InternetService"] != "No":
        drivers.append(("risk", "Çevrimiçi Güvenlik Yok", "Güvenlik eklentisi olmayanlarda churn oranı %41,8'dir."))

    if not drivers:
        drivers.append(("safe", "Dengeli Müşteri Profili", "Belirgin bir yüksek risk sinyali tespit edilmedi."))
    return drivers


# ==========================================================
# 1-CLICK DEMO PRESETS (AUTO-RUNS PREDICTION IMMEDIATELY)
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

model, meta, metrics = load_artifacts()
if model is None:
    st.error("Model veya veri dosyası bulunamadı. Lütfen repo dosyalarını kontrol edin.")
    st.stop()

default_thr = float(meta.get("optimal_threshold", 0.53))
champ_name = meta.get("champion_name", "CatBoost")
test_pr = float(meta.get("test_pr_auc", 0.6772))
test_roc = float(meta.get("test_roc_auc", 0.8445))
test_rec = float(meta.get("test_recall", 0.7616))

if "preset_data" not in st.session_state:
    st.session_state["preset_data"] = PRESETS["high"].copy()
    st.session_state["auto_predict"] = True
if "active_threshold" not in st.session_state:
    st.session_state["active_threshold"] = round(default_thr, 2)

# ==========================================================
# SIDEBAR: 1-CLICK DEMO PRESETS + LIVE THRESHOLD + PROJECT INFO
# ==========================================================
with st.sidebar:
    st.markdown("### ⚡ Canlı Demo Senaryoları")
    st.caption("Tıkladığınız anda profil yüklenir ve tahmin otomatik güncellenir:")
    if st.button("🔴 Yüksek Riskli Müşteri", use_container_width=True):
        st.session_state["preset_data"] = PRESETS["high"].copy()
        st.session_state["auto_predict"] = True
    if st.button("🟠 Orta Riskli Müşteri", use_container_width=True):
        st.session_state["preset_data"] = PRESETS["mid"].copy()
        st.session_state["auto_predict"] = True
    if st.button("🟢 Düşük Riskli Müşteri", use_container_width=True):
        st.session_state["preset_data"] = PRESETS["low"].copy()
        st.session_state["auto_predict"] = True

    st.divider()
    st.markdown("### 🎚️ Karar Eşiği (Threshold)")
    threshold = st.slider(
        "Churn Alarm Eşiği",
        min_value=0.15,
        max_value=0.85,
        value=float(st.session_state["active_threshold"]),
        step=0.01,
        help="Validation setinde F1-skorunu maksimize eden optimal eşik 0,53'tür.",
    )
    st.session_state["active_threshold"] = threshold
    if abs(threshold - default_thr) > 0.005:
        if st.button(f"↩️ Optimal Eşiğe Dön ({default_thr:.2f})", use_container_width=True):
            st.session_state["active_threshold"] = round(default_thr, 2)
            st.rerun()

    st.divider()
    st.markdown("### 📌 Proje Künyesi")
    st.markdown(
        "- **Geliştirici:** Arda KİLİT\n"
        "- **Atölye:** BTK Uçtan Uca ML\n"
        "- **Veri Seti:** IBM Telco (7.043 kayıt)\n"
        "- **Şampiyon:** CatBoost (Tuned)\n"
        "- **GitHub:** [telco-churn-ml-project](https://github.com/Arda-Kilit/telco-churn-ml-project)"
    )

# ==========================================================
# HERO HEADER & KPI STRIP
# ==========================================================
st.markdown(
    _html(
        """
        <div class="hero-wrap">
            <div class="hero-pill">BTK Uçtan Uca Makine Öğrenmesi · Arda KİLİT</div>
            <div class="hero-title">📡 Telekom Müşteri Kaybı (Churn) Karar Destek Platformu</div>
            <div class="hero-desc">
                Sızıntısız Pipeline, 8 alan özniteliği ve 7 güçlü algoritmanın bilimsel karşılaştırmasıyla
                müşteri kaybını erken yakalayan ve gerçek zamanlı tutundurma simülasyonu sunan yapay zekâ platformu.
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

st.markdown(
    _html(
        f"""
        <div class="kpi-grid">
            <div class="kpi-box coral">
                <div class="kpi-lbl">Şampiyon Model</div>
                <div class="kpi-val">{champ_name}</div>
                <div class="kpi-sub">7 güçlü algoritma arasından lider</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-lbl">Kilitli Test PR-AUC</div>
                <div class="kpi-val">{test_pr:.4f}</div>
                <div class="kpi-sub">Dummy Baseline'a göre 2,55× artış</div>
            </div>
            <div class="kpi-box sky">
                <div class="kpi-lbl">Kilitli Test ROC-AUC</div>
                <div class="kpi-val">{test_roc:.4f}</div>
                <div class="kpi-sub">n = 1.057 kilitli test müşterisi</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-lbl">Churn Yakalama (Recall)</div>
                <div class="kpi-val">%{test_rec * 100:.1f}</div>
                <div class="kpi-sub">Aktif karar eşiği: {threshold:.2f}</div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Risk Analizi & What-If",
        "📂 Toplu CSV Skorlama",
        "🏆 Model Karşılaştırması & Kartı",
        "📈 EDA & Hata Analizi",
    ]
)

# ==========================================================
# TAB 1: SINGLE RISK ANALYSIS + LIVE WHAT-IF RETENTION SIMULATOR
# ==========================================================
with tab1:
    p = st.session_state["preset_data"]
    st.markdown("#### Müşteri Profili ve Canlı Tutundurma Simülatörü")
    st.caption("Sol menüden hazır senaryo seçin (otomatik hesaplanır) veya alanları değiştirip aşağıdaki butona basın.")

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

        submitted = st.form_submit_button("🔍 Churn Riskini ve Tutundurma Etkisini Hesapla", use_container_width=True, type="primary")

    if submitted or st.session_state.get("auto_predict", False):
        if submitted:
            st.session_state["auto_predict"] = True

        customer_dict = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multi_lines,
            "InternetService": internet, "OnlineSecurity": online_sec, "OnlineBackup": online_bak,
            "DeviceProtection": dev_prot, "TechSupport": tech_sup, "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }
        input_df = pd.DataFrame([customer_dict])
        prob = float(model.predict_proba(input_df)[0, 1])
        pred_label = "AYRILMA RİSKİ VAR (Churn = Yes)" if prob >= threshold else "KALMA EĞİLİMİNDE (Churn = No)"
        band, icon, pill_cls, gauge_col, advice = get_risk_profile(prob, threshold)
        annual_rev = monthly * 12.0

        res_left, res_right = st.columns([1.02, 1.48], gap="large")
        with res_left:
            gauge_html = render_svg_gauge(prob, threshold, gauge_col)
            st.markdown(
                f'<div class="panel-card" style="text-align:center;">'
                f'<span class="badge-status {pill_cls}">{icon} {band}</span>'
                f"{gauge_html}"
                f'<div style="font-size:1.03rem; font-weight:800; margin-top:4px;">{pred_label}</div>'
                f'<div style="font-size:0.83rem; opacity:0.75; margin-top:6px;">Yıllık Abonelik Değeri: <strong>${annual_rev:,.0f}</strong></div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        with res_right:
            st.markdown(
                _html(
                    f"""
                    <div class="panel-card">
                        <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#5EEAD4; margin-bottom:6px;">
                            Önerilen CRM Aksiyonu
                        </div>
                        <div style="font-size:0.98rem; line-height:1.5;">{advice}</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

            # Real CatBoost Live What-If Retention Simulation
            sim_dict = customer_dict.copy()
            sim_actions = []
            if sim_dict["Contract"] == "Month-to-month":
                sim_dict["Contract"] = "One year"
                sim_actions.append("1 Yıllık Sözleşme")
            if sim_dict["PaymentMethod"] == "Electronic check":
                sim_dict["PaymentMethod"] = "Credit card (automatic)"
                sim_actions.append("Otomatik Ödeme")
            if sim_dict["InternetService"] != "No" and sim_dict["TechSupport"] == "No":
                sim_dict["TechSupport"] = "Yes"
                sim_actions.append("Teknik Destek")
            if sim_dict["InternetService"] != "No" and sim_dict["OnlineSecurity"] == "No":
                sim_dict["OnlineSecurity"] = "Yes"
                sim_actions.append("Çevrimiçi Güvenlik")

            if sim_actions:
                sim_prob = float(model.predict_proba(pd.DataFrame([sim_dict]))[0, 1])
                delta_pp = (prob - sim_prob) * 100
                st.markdown(
                    _html(
                        f"""
                        <div class="sim-card">
                            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#5EEAD4; margin-bottom:6px;">
                                💡 Canlı What-If Tutundurma Simülasyonu (Gerçek CatBoost Çıktısı)
                            </div>
                            <div style="font-size:0.89rem; margin-bottom:10px;">
                                <strong>Önerilen Paket:</strong> {" + ".join(sim_actions)}
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.84rem;">
                                <span>Mevcut Risk</span><strong>%{prob*100:.1f}</strong>
                            </div>
                            <div class="bar-track"><div class="bar-fill-before" style="width:{min(100, prob*100):.1f}%;"></div></div>
                            <div style="display:flex; justify-content:space-between; font-size:0.84rem;">
                                <span>Teklif Sonrası Risk</span><strong style="color:#5EEAD4;">%{sim_prob*100:.1f}</strong>
                            </div>
                            <div class="bar-track"><div class="bar-fill-after" style="width:{min(100, sim_prob*100):.1f}%;"></div></div>
                            <div style="font-size:0.88rem; margin-top:4px;">
                                Net Risk Azalışı: <strong style="color:#34D399;">-{delta_pp:.1f} puan</strong>
                                &nbsp;·&nbsp; Korunan Yıllık Gelir Potansiyeli: <strong>${annual_rev:,.0f}</strong>
                            </div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    _html(
                        """
                        <div class="sim-card">
                            <div style="font-size:0.82rem; font-weight:700; color:#5EEAD4; margin-bottom:4px;">
                                ✅ Optimal Koruma Profili
                            </div>
                            <div style="font-size:0.89rem;">
                                Bu müşteri zaten uzun dönemli sözleşme, otomatik ödeme ve destek/güvenlik paketleriyle düşük risk bandında yer almaktadır.
                            </div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown("##### Müşteri Bazlı Risk ve Koruma Sinyalleri")
        d_cols = st.columns(2, gap="medium")
        for idx, (kind, title, desc) in enumerate(explain_customer_drivers(customer_dict)):
            icon_d = "⚠️" if kind == "risk" else "✅"
            with d_cols[idx % 2]:
                st.markdown(
                    f'<div class="driver-chip {kind}"><strong>{icon_d} {title}:</strong> {desc}</div>',
                    unsafe_allow_html=True,
                )

# ==========================================================
# TAB 2: BATCH CSV PREDICTION + 1-CLICK DEMO DATASET
# ==========================================================
with tab2:
    st.markdown("#### Toplu Müşteri Risk Skorlama (CSV)")
    st.caption("Kendi CSV dosyanızı yükleyin veya canlı demo için tek tıkla 60 müşterilik örnek listeyi skorlayın.")

    b_col1, b_col2 = st.columns([2.2, 1.0], gap="medium")
    with b_col1:
        uploaded = st.file_uploader("Müşteri listesi (.csv) yükleyin", type=["csv"])
    with b_col2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        use_sample = st.button("⚡ 60 Örnek Müşteriyle Hemen Dene", use_container_width=True)

    batch_df = None
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"CSV okunamadı: {exc}")
    elif use_sample:
        batch_df = load_sample_batch()

    if batch_df is not None:
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
            at_risk_rev = float(out_df.loc[out_df["Risk_Segmenti"] == "Yüksek Risk", "MonthlyCharges"].sum() * 12)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Toplam Skorlanan", f"{len(out_df):,}")
            m2.metric("🔴 Yüksek Risk", f"{high_n:,}")
            m3.metric("🟠 Orta / 🟢 Düşük", f"{mid_n:,} / {low_n:,}")
            m4.metric("Riskteki Yıllık Gelir", f"${at_risk_rev:,.0f}")

            seg_filter = st.multiselect(
                "Risk Segmentine Göre Filtrele",
                ["Yüksek Risk", "Orta Risk", "Düşük Risk"],
                default=["Yüksek Risk", "Orta Risk", "Düşük Risk"],
            )
            filtered_df = out_df[out_df["Risk_Segmenti"].isin(seg_filter)].sort_values("Churn_Olasiligi_%", ascending=False)
            st.dataframe(filtered_df.head(60), use_container_width=True)

            csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Risk Skorlama Raporunu CSV Olarak İndir",
                data=csv_bytes,
                file_name="telco_churn_tahminleri.csv",
                mime="text/csv",
                type="primary",
            )

# ==========================================================
# TAB 3: MODEL COMPARISON & MODEL CARD (YÖNERGE BONUS)
# ==========================================================
with tab3:
    st.markdown("#### Şampiyon Model Kartı (Model Card)")
    st.markdown(
        _html(
            """
            <div class="model-card-grid">
                <div class="mc-item">
                    <div class="mc-title">Mimari & Hiperparametreler</div>
                    <strong>CatBoostClassifier</strong> (Tuned)<br>
                    iterations=400 · depth=4 · lr=0,02 · L2=10<br>
                    auto_class_weights="Balanced" · seed=42
                </div>
                <div class="mc-item">
                    <div class="mc-title">Deney & Sızıntı Güvenliği</div>
                    Stratified %70 / %15 / %15 (4.930 / 1.056 / 1.057)<br>
                    5-Fold CV · Eşit bütçeli RandomizedSearchCV (n=20)<br>
                    Sıfır veri sızıntılı scikit-learn Pipeline
                </div>
                <div class="mc-item">
                    <div class="mc-title">Kilitli Test & Hata Profili</div>
                    PR-AUC: <strong>0,6772</strong> · ROC-AUC: <strong>0,8445</strong><br>
                    TP=214 · TN=593 · FP=183 · FN=67 (%76,2 Recall)<br>
                    2 yıllık sözleşmede %98,1 doğruluk
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown("#### Kilitli Test Seti Model Karşılaştırması (Baseline + 6 Güçlü Rakip)")
    if metrics is not None:
        test_tbl = pd.DataFrame(metrics["test_comparison_all_models"])[
            [
                "model", "test_pr_auc", "test_roc_auc", "test_f1_at_opt",
                "test_recall_at_opt", "test_precision_at_opt", "test_accuracy_at_opt", "threshold_used",
            ]
        ].rename(
            columns={
                "model": "Model", "test_pr_auc": "Test PR-AUC", "test_roc_auc": "Test ROC-AUC",
                "test_f1_at_opt": "Test F1", "test_recall_at_opt": "Recall",
                "test_precision_at_opt": "Precision", "test_accuracy_at_opt": "Accuracy",
                "threshold_used": "Eşik",
            }
        )
        c_chart, c_tbl = st.columns([1.0, 1.35], gap="large")
        with c_chart:
            st.markdown("##### Test PR-AUC & ROC-AUC Grafiği")
            st.bar_chart(test_tbl.set_index("Model")[["Test PR-AUC", "Test ROC-AUC"]], height=345)
        with c_tbl:
            st.markdown("##### Detaylı Performans Tablosu (n = 1.057)")
            st.dataframe(
                test_tbl.style.format(
                    {
                        "Test PR-AUC": "{:.4f}", "Test ROC-AUC": "{:.4f}", "Test F1": "{:.4f}",
                        "Recall": "{:.4f}", "Precision": "{:.4f}", "Accuracy": "{:.4f}", "Eşik": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("##### Eşit Bütçeli Hiperparametre Optimizasyonu (Önce vs. Sonra)")
        tune_tbl = pd.DataFrame(metrics["tuning_comparison"])[
            ["model", "cv_pr_auc_before", "cv_pr_auc_after", "delta_pr_auc", "val_pr_auc", "optimal_threshold"]
        ].rename(
            columns={
                "model": "Aday Model", "cv_pr_auc_before": "Önce CV PR-AUC",
                "cv_pr_auc_after": "Sonra CV PR-AUC", "delta_pr_auc": "Artış (Δ)",
                "val_pr_auc": "Validation PR-AUC", "optimal_threshold": "Optimal Eşik",
            }
        )
        st.dataframe(
            tune_tbl.style.format(
                {
                    "Önce CV PR-AUC": "{:.4f}", "Sonra CV PR-AUC": "{:.4f}",
                    "Artış (Δ)": "+{:.4f}", "Validation PR-AUC": "{:.4f}", "Optimal Eşik": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

# ==========================================================
# TAB 4: EDA & ERROR ANALYSIS GALLERY
# ==========================================================
with tab4:
    st.markdown("#### Keşifsel Veri Analizi (EDA), Değerlendirme ve Hata Analizi Grafikleri")
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
# RESPONSIBLE USE FOOTER (YÖNERGE BÖLÜM 11)
# ==========================================================
st.markdown(
    _html(
        """
        <div class="disclaimer-card">
            <strong>⚠️ Model Sınırlılığı ve Sorumlu Kullanım Notu:</strong>
            Bu platform, IBM Telco Customer Churn kesit veri setiyle eğitilmiş olasılıksal bir tahmin modeli sunar;
            kesin sonuç garanti etmez ve yanlış pozitif (FP) veya yanlış negatif (FN) tahminler içerebilir.
            Çıktılar, iş kararlarında tek başına değil, müşteri ilişkileri uzman değerlendirmesiyle birlikte kullanılmalıdır.
        </div>
        """
    ),
    unsafe_allow_html=True,
)
