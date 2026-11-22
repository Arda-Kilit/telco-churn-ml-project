"""
BTK Uçtan Uca Makine Öğrenmesi Projesi — Telco Customer Churn
Aşama 13–14: Özgün Editorial-Graphite Tasarımlı Streamlit Karar Destek Platformu

Çalıştırma:
    streamlit run app.py
"""

import json
import math
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
    page_title="Telco Churn Intelligence | Arda KİLİT",
    page_icon="◈",
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
    """Strips per-line indentation so Streamlit Markdown never renders HTML as a code block."""
    return "\n".join(line.strip() for line in raw.strip().splitlines() if line.strip())


# ==========================================================
# BESPOKE EDITORIAL-GRAPHITE & CHAMPAGNE DESIGN SYSTEM (CSS)
# Calm matte surfaces, warm ivory typography, hairline geometry
# ==========================================================
st.markdown(
    _html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        :root {
            --bg-canvas: #0F1115;
            --bg-card: #161920;
            --bg-elevated: #1D212B;
            --border-subtle: rgba(244, 241, 234, 0.09);
            --border-strong: rgba(217, 167, 96, 0.34);
            --ink-primary: #F4F1EA;
            --ink-muted: #9CA3AF;
            --gold: #D9A760;
            --sage: #34D399;
            --crimson: #FB7185;
            --amber: #FBBF24;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--ink-primary);
        }
        .stApp {
            background-color: var(--bg-canvas);
        }
        [data-testid="stHeader"] {
            background: rgba(15, 17, 21, 0.85);
            backdrop-filter: blur(10px);
        }
        [data-testid="stSidebar"] {
            background-color: #13161C;
            border-right: 1px solid var(--border-subtle);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.8rem;
            max-width: 1240px;
        }

        /* Editorial Masthead */
        .masthead {
            background: linear-gradient(145deg, #161920 0%, #12151B 100%);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 26px 32px;
            margin-bottom: 1.25rem;
            position: relative;
            overflow: hidden;
        }
        .masthead::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent 0%, #D9A760 28%, #34D399 72%, transparent 100%);
        }
        .mh-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 12px;
        }
        .mh-kicker {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--gold);
        }
        .live-badge {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            font-size: 0.74rem;
            font-weight: 600;
            color: #A7F3D0;
            background: rgba(52, 211, 153, 0.10);
            border: 1px solid rgba(52, 211, 153, 0.28);
            padding: 4px 12px;
            border-radius: 999px;
        }
        .live-dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: #34D399;
            box-shadow: 0 0 8px #34D399;
        }
        .mh-title {
            font-family: 'Newsreader', Georgia, serif;
            font-size: 2.15rem;
            font-weight: 700;
            color: #F8F6F1;
            margin: 0 0 8px 0;
            line-height: 1.16;
            letter-spacing: -0.015em;
        }
        .mh-sub {
            font-size: 0.93rem;
            color: var(--ink-muted);
            margin: 0;
            max-width: 780px;
            line-height: 1.55;
        }

        /* Quiet Luxury KPI Strip */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 1.45rem;
        }
        @media (max-width: 900px) {
            .kpi-row { grid-template-columns: repeat(2, 1fr); }
        }
        .kpi-tile {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 16px 20px;
            transition: border-color 0.2s ease;
        }
        .kpi-tile.featured {
            border-color: var(--border-strong);
            background: linear-gradient(180deg, rgba(217, 167, 96, 0.08) 0%, #161920 100%);
        }
        .kpi-kicker {
            font-size: 0.70rem;
            font-weight: 700;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: var(--ink-muted);
            margin-bottom: 6px;
        }
        .kpi-num {
            font-family: 'Newsreader', Georgia, serif;
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--ink-primary);
            line-height: 1.1;
            font-variant-numeric: tabular-nums;
            margin-bottom: 4px;
        }
        .kpi-tile.featured .kpi-num { color: var(--gold); }
        .kpi-meta {
            font-size: 0.78rem;
            color: var(--ink-muted);
        }

        /* Editorial Section Headers */
        .sec-head {
            font-family: 'Newsreader', Georgia, serif;
            font-size: 1.38rem;
            font-weight: 700;
            color: var(--ink-primary);
            margin: 0.3rem 0 0.2rem 0;
        }
        .sec-sub {
            font-size: 0.87rem;
            color: var(--ink-muted);
            margin-bottom: 1rem;
        }

        /* Dossier / Analysis Cards */
        .dossier-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 22px 24px;
            margin-bottom: 14px;
        }
        .status-tag {
            display: inline-block;
            padding: 4px 13px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .tag-high { background: rgba(251, 113, 133, 0.14); color: #FDA4AF; border: 1px solid rgba(251, 113, 133, 0.36); }
        .tag-mid  { background: rgba(251, 191, 36, 0.14); color: #FDE68A; border: 1px solid rgba(251, 191, 36, 0.36); }
        .tag-low  { background: rgba(52, 211, 153, 0.14); color: #A7F3D0; border: 1px solid rgba(52, 211, 153, 0.36); }

        .whatif-box {
            background: linear-gradient(135deg, rgba(52, 211, 153, 0.08) 0%, rgba(217, 167, 96, 0.06) 100%);
            border: 1px solid rgba(52, 211, 153, 0.28);
            border-radius: 15px;
            padding: 20px 22px;
            margin-bottom: 14px;
        }
        .wi-bar-bg {
            width: 100%;
            height: 8px;
            background: rgba(244, 241, 234, 0.08);
            border-radius: 999px;
            overflow: hidden;
            margin: 6px 0 12px 0;
        }
        .wi-fill-now { height: 100%; border-radius: 999px; background: #FB7185; }
        .wi-fill-opt { height: 100%; border-radius: 999px; background: #34D399; }

        .signal-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 13px 16px;
            margin-bottom: 10px;
            font-size: 0.87rem;
            line-height: 1.5;
        }
        .signal-card.risk { border-left: 3px solid #FB7185; }
        .signal-card.safe { border-left: 3px solid #34D399; }
        .sig-badge {
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-right: 6px;
        }
        .sig-badge.risk { color: #FDA4AF; }
        .sig-badge.safe { color: #6EE7B7; }

        /* Model Card Grid */
        .mc-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin: 10px 0 20px 0;
        }
        @media (max-width: 900px) {
            .mc-grid { grid-template-columns: 1fr; }
        }
        .mc-box {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-top: 2px solid var(--gold);
            border-radius: 12px;
            padding: 16px 18px;
            font-size: 0.86rem;
            line-height: 1.55;
            color: #D1D5DB;
        }
        .mc-lbl {
            font-size: 0.70rem;
            font-weight: 700;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: var(--gold);
            margin-bottom: 7px;
        }

        /* Refined Tabs & Buttons */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px 8px 0 0;
            padding: 10px 18px;
            font-weight: 600;
            font-size: 0.88rem;
            color: var(--ink-muted);
        }
        .stTabs [aria-selected="true"] {
            color: var(--ink-primary) !important;
            background: var(--bg-card) !important;
            border-bottom: 2px solid var(--gold) !important;
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #D9A760 0%, #B8863B 100%);
            color: #0F1115;
            border: none;
            font-weight: 700;
            border-radius: 10px;
            padding: 0.62rem 1.2rem;
        }
        div.stButton > button[kind="primary"]:hover {
            filter: brightness(1.06);
        }

        /* Responsible Use Footer */
        .footer-note {
            margin-top: 2rem;
            padding: 14px 18px;
            border-radius: 12px;
            background: rgba(244, 241, 234, 0.03);
            border: 1px solid var(--border-subtle);
            font-size: 0.80rem;
            line-height: 1.55;
            color: var(--ink-muted);
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


# ==========================================================
# VERSION-PROOF MODEL LOADER & CACHED SAMPLE DATA
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
            "tag-high",
            "#FB7185",
            "Acil tutundurma müdahalesi: 1 yıllık taahhüt indirimi ve 3 ay ücretsiz Teknik Destek paketi sunun.",
        )
    if prob >= threshold * 0.70:
        return (
            "Orta Risk",
            "tag-mid",
            "#FBBF24",
            "Proaktif iletişim: Otomatik ödeme indirimi ve Çevrimiçi Güvenlik eklentisiyle bağlılığı güçlendirin.",
        )
    return (
        "Düşük Risk",
        "tag-low",
        "#34D399",
        "Mevcut memnuniyeti koruyun; sadakat programı veya ek paket çapraz satış kampanyalarına dahil edin.",
    )


def render_editorial_dial(prob: float, threshold: float, color: str) -> str:
    """Precision SVG semi-circular dial with hairline scale ticks and threshold marker."""
    pct = max(0.0, min(1.0, prob))
    r = 86
    circ = math.pi * r
    filled = circ * pct
    empty = circ - filled

    thr_ang = math.pi * (1.0 - max(0.0, min(1.0, threshold)))
    tx1 = 115 + (r - 11) * math.cos(thr_ang)
    ty1 = 108 - (r - 11) * math.sin(thr_ang)
    tx2 = 115 + (r + 11) * math.cos(thr_ang)
    ty2 = 108 - (r + 11) * math.sin(thr_ang)

    return (
        '<div style="display:flex; justify-content:center; align-items:center; padding:8px 0 4px 0;">'
        '<svg viewBox="0 0 230 134" width="256" height="150" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M 29 108 A 86 86 0 0 1 201 108" fill="none" stroke="rgba(244,241,234,0.10)" stroke-width="14" stroke-linecap="round"/>'
        f'<path d="M 29 108 A 86 86 0 0 1 201 108" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round" stroke-dasharray="{filled:.1f} {empty:.1f}"/>'
        f'<line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" stroke="#D9A760" stroke-width="2.6" stroke-linecap="round"/>'
        f'<text x="115" y="90" text-anchor="middle" fill="#F8F6F1" font-size="33" font-weight="700" font-family="Newsreader, Georgia, serif">%{pct * 100:.1f}</text>'
        f'<text x="115" y="116" text-anchor="middle" fill="#9CA3AF" font-size="10" font-weight="600" letter-spacing="0.08em" font-family="Plus Jakarta Sans, sans-serif">CHURN OLASILIĞI · EŞİK {threshold:.2f}</text>'
        "</svg></div>"
    )


def explain_customer_drivers(row: dict):
    drivers = []
    if row["Contract"] == "Month-to-month":
        drivers.append(("risk", "Aylık Sözleşme", "Veri setinde %42,7 churn oranı ile en güçlü ayrılma tetikleyicisidir."))
    elif row["Contract"] == "Two year":
        drivers.append(("safe", "İki Yıllık Sözleşme", "Veri setinde churn oranı yalnızca %2,8 düzeyindedir."))
    elif row["Contract"] == "One year":
        drivers.append(("safe", "Bir Yıllık Sözleşme", "Aylık sözleşmeye (%42,7) kıyasla belirgin biçimde düşük risk (%11,3)."))

    if row["tenure"] <= 12:
        drivers.append(("risk", f"Erken Dönem ({row['tenure']} Ay)", "Müşteri kaybı ilk 12 aylık abonelik döneminde yoğunlaşmaktadır."))
    elif row["tenure"] >= 48:
        drivers.append(("safe", f"Yüksek Sadakat ({row['tenure']} Ay)", "Uzun abonelik geçmişi kayıp olasılığını güçlü biçimde azaltır."))

    if row["InternetService"] == "Fiber optic":
        drivers.append(("risk", "Fiber İnternet", "Veri setinde %41,9 churn oranı ile yüksek fiyat/memnuniyet hassasiyeti gösterir."))
    if row["PaymentMethod"] == "Electronic check":
        drivers.append(("risk", "Elektronik Çek", "%45,3 churn oranı ile otomatik ödemeye (%15,2) göre yaklaşık 3 kat risklidir."))
    elif "automatic" in row["PaymentMethod"]:
        drivers.append(("safe", "Otomatik Ödeme", "Abonelik devamlılığını destekleyen güçlü bir koruma sinyalidir."))

    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        drivers.append(("risk", "Teknik Destek Eksik", "Destek paketi olmayan abonelerde churn oranı %41,6 seviyesindedir."))
    if row["OnlineSecurity"] == "No" and row["InternetService"] != "No":
        drivers.append(("risk", "Çevrimiçi Güvenlik Eksik", "Güvenlik eklentisi bulunmayanlarda churn oranı %41,8 seviyesindedir."))

    if not drivers:
        drivers.append(("safe", "Dengeli Profil", "Müşteri profili belirgin bir yüksek risk tetikleyicisi taşımamaktadır."))
    return drivers


# ==========================================================
# 1-CLICK DEMO PRESETS
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
# SIDEBAR — CALM EXECUTIVE CONTROLS
# ==========================================================
with st.sidebar:
    st.markdown(
        _html(
            """
            <div style="padding:4px 0 12px 0; border-bottom:1px solid rgba(244,241,234,0.09); margin-bottom:14px;">
                <div style="font-size:0.70rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#D9A760;">
                    KONTROL PANELİ
                </div>
                <div style="font-family:'Newsreader',Georgia,serif; font-size:1.25rem; font-weight:700; color:#F4F1EA; margin-top:2px;">
                    Canlı Demo Senaryoları
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    st.caption("Seçtiğiniz profil anında yüklenir ve risk göstergesi otomatik güncellenir.")
    if st.button("Yüksek Risk Profili (Örnek 1)", use_container_width=True):
        st.session_state["preset_data"] = PRESETS["high"].copy()
        st.session_state["auto_predict"] = True
    if st.button("Orta Risk Profili (Örnek 2)", use_container_width=True):
        st.session_state["preset_data"] = PRESETS["mid"].copy()
        st.session_state["auto_predict"] = True
    if st.button("Düşük Risk Profili (Örnek 3)", use_container_width=True):
        st.session_state["preset_data"] = PRESETS["low"].copy()
        st.session_state["auto_predict"] = True

    st.divider()
    st.markdown(
        '<div style="font-size:0.72rem; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#D9A760; margin-bottom:6px;">KARAR EŞİĞİ (THRESHOLD)</div>',
        unsafe_allow_html=True,
    )
    threshold = st.slider(
        "Churn Alarm Eşiği",
        min_value=0.15,
        max_value=0.85,
        value=float(st.session_state["active_threshold"]),
        step=0.01,
        help="Doğrulama setinde F1-skorunu maksimize eden optimal eşik 0,53'tür.",
    )
    st.session_state["active_threshold"] = threshold
    if abs(threshold - default_thr) > 0.005:
        if st.button(f"Optimal Eşiğe Sıfırla ({default_thr:.2f})", use_container_width=True):
            st.session_state["active_threshold"] = round(default_thr, 2)
            st.rerun()

    st.divider()
    st.markdown(
        _html(
            """
            <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#D9A760; margin-bottom:8px;">
                PROJE KÜNYESİ
            </div>
            <div style="font-size:0.84rem; line-height:1.7; color:#D1D5DB;">
                <strong>Geliştirici:</strong> Arda KİLİT<br>
                <strong>Program:</strong> BTK Uçtan Uca ML<br>
                <strong>Veri Seti:</strong> IBM Telco (7.043)<br>
                <strong>Model:</strong> CatBoost (Tuned)<br>
                <strong>Kod:</strong> <a href="https://github.com/Arda-Kilit/telco-churn-ml-project" style="color:#D9A760;">GitHub Deposu</a>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

# ==========================================================
# EDITORIAL MASTHEAD & KPI STRIP
# ==========================================================
st.markdown(
    _html(
        """
        <div class="masthead">
            <div class="mh-top">
                <span class="mh-kicker">BTK Uçtan Uca Makine Öğrenmesi · Arda KİLİT</span>
                <span class="live-badge"><span class="live-dot"></span>Canlı Model · CatBoost Tuned</span>
            </div>
            <div class="mh-title">Telekom Müşteri Kaybı Karar Destek Sistemi</div>
            <div class="mh-sub">
                Sızıntısız Pipeline mimarisi, 8 alan özniteliği ve 7 güçlü algoritmanın bilimsel karşılaştırmasıyla
                müşteri kaybını erken yakalayan ve canlı What-If tutundurma etkisi hesaplayan analitik platform.
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

st.markdown(
    _html(
        f"""
        <div class="kpi-row">
            <div class="kpi-tile featured">
                <div class="kpi-kicker">Kilitli Test PR-AUC</div>
                <div class="kpi-num">{test_pr:.4f}</div>
                <div class="kpi-meta">Baseline'a göre 2,55× artış</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-kicker">Kilitli Test ROC-AUC</div>
                <div class="kpi-num">{test_roc:.4f}</div>
                <div class="kpi-meta">n = 1.057 kilitli test kaydı</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-kicker">Churn Yakalama (Recall)</div>
                <div class="kpi-num">%{test_rec * 100:.1f}</div>
                <div class="kpi-meta">Aktif karar eşiği: {threshold:.2f}</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-kicker">Şampiyon Algoritma</div>
                <div class="kpi-num">{champ_name}</div>
                <div class="kpi-meta">7 güçlü rakip arasından seçildi</div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Tekil Risk & What-If",
        "Toplu CSV Skorlama",
        "Model Kartı & Kıyaslama",
        "EDA & Hata Analizi",
    ]
)

# ==========================================================
# TAB 1: SINGLE RISK ANALYSIS & LIVE WHAT-IF SIMULATOR
# ==========================================================
with tab1:
    p = st.session_state["preset_data"]
    st.markdown('<div class="sec-head">Müşteri Profili ve Tutundurma Simülatörü</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Sol menüden hazır senaryo seçebilir veya müşteri özelliklerini değiştirip anlık risk etkisini ölçebilirsiniz.</div>',
        unsafe_allow_html=True,
    )

    with st.form("churn_form"):
        col_a, col_b, col_c = st.columns(3, gap="medium")

        with col_a:
            st.markdown("**01 · Demografi ve Hesap**")
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
            st.markdown("**02 · Sözleşme ve Fatura**")
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
            st.markdown("**03 · Hizmet Paketleri**")
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

        submitted = st.form_submit_button("Churn Riskini ve Tutundurma Etkisini Hesapla", use_container_width=True, type="primary")

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
        pred_label = "Ayrılma Riski Mevcut (Churn = Yes)" if prob >= threshold else "Kalma Eğiliminde (Churn = No)"
        band, tag_cls, dial_col, advice = get_risk_profile(prob, threshold)
        annual_rev = monthly * 12.0

        res_left, res_right = st.columns([1.0, 1.48], gap="large")
        with res_left:
            dial_svg = render_editorial_dial(prob, threshold, dial_col)
            st.markdown(
                f'<div class="dossier-card" style="text-align:center;">'
                f'<span class="status-tag {tag_cls}">{band}</span>'
                f"{dial_svg}"
                f'<div style="font-size:1.02rem; font-weight:700; color:#F8F6F1; margin-top:4px;">{pred_label}</div>'
                f'<div style="font-size:0.82rem; color:#9CA3AF; margin-top:6px;">Yıllık Abonelik Değeri: <strong style="color:#D9A760;">${annual_rev:,.0f}</strong></div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        with res_right:
            st.markdown(
                _html(
                    f"""
                    <div class="dossier-card">
                        <div style="font-size:0.70rem; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#D9A760; margin-bottom:6px;">
                            ÖNERİLEN CRM AKSİYONU
                        </div>
                        <div style="font-size:0.96rem; line-height:1.55; color:#F4F1EA;">{advice}</div>
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
                        <div class="whatif-box">
                            <div style="font-size:0.70rem; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#34D399; margin-bottom:6px;">
                                CANLI WHAT-IF TUTUNDURMA SİMÜLASYONU · GERÇEK CATBOOST ÇIKTISI
                            </div>
                            <div style="font-size:0.88rem; color:#E5E7EB; margin-bottom:12px;">
                                <strong>Simüle Edilen Paket:</strong> {" + ".join(sim_actions)}
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.83rem; color:#D1D5DB;">
                                <span>Mevcut Churn Riski</span><strong>%{prob * 100:.1f}</strong>
                            </div>
                            <div class="wi-bar-bg"><div class="wi-fill-now" style="width:{min(100, prob * 100):.1f}%;"></div></div>
                            <div style="display:flex; justify-content:space-between; font-size:0.83rem; color:#D1D5DB;">
                                <span>Teklif Sonrası Risk</span><strong style="color:#34D399;">%{sim_prob * 100:.1f}</strong>
                            </div>
                            <div class="wi-bar-bg"><div class="wi-fill-opt" style="width:{min(100, sim_prob * 100):.1f}%;"></div></div>
                            <div style="font-size:0.86rem; color:#F4F1EA; margin-top:4px;">
                                Net Risk Azalışı: <strong style="color:#34D399;">-{delta_pp:.1f} puan</strong>
                                &nbsp;·&nbsp; Korunan Yıllık Gelir: <strong style="color:#D9A760;">${annual_rev:,.0f}</strong>
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
                        <div class="whatif-box">
                            <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#34D399; margin-bottom:5px;">
                                OPTİMAL KORUMA PROFİLİ
                            </div>
                            <div style="font-size:0.88rem; color:#E5E7EB;">
                                Bu müşteri halihazırda uzun dönemli sözleşme, otomatik ödeme ve destek/güvenlik paketleriyle düşük risk bandında yer almaktadır.
                            </div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="sec-head" style="font-size:1.15rem; margin-top:8px;">Müşteri Bazlı Risk ve Koruma Sinyalleri</div>', unsafe_allow_html=True)
        d_cols = st.columns(2, gap="medium")
        for idx, (kind, title, desc) in enumerate(explain_customer_drivers(customer_dict)):
            lbl_txt = "RİSK" if kind == "risk" else "KORUMA"
            with d_cols[idx % 2]:
                st.markdown(
                    f'<div class="signal-card {kind}">'
                    f'<span class="sig-badge {kind}">{lbl_txt}</span>'
                    f'<strong style="color:#F8F6F1;">{title}:</strong> '
                    f'<span style="color:#D1D5DB;">{desc}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ==========================================================
# TAB 2: BATCH CSV SCORING + 1-CLICK DEMO DATASET
# ==========================================================
with tab2:
    st.markdown('<div class="sec-head">Toplu Müşteri Risk Skorlama</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Kendi CSV listenizi yükleyin veya canlı demo için tek tıkla 60 müşterilik örnek veriyi skorlayın.</div>',
        unsafe_allow_html=True,
    )

    b_col1, b_col2 = st.columns([2.2, 1.0], gap="medium")
    with b_col1:
        uploaded = st.file_uploader("Müşteri listesi (.csv) yükleyin", type=["csv"])
    with b_col2:
        st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
        use_sample = st.button("60 Örnek Müşteriyle Hemen Dene", use_container_width=True, type="primary")

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
            m2.metric("Yüksek Risk", f"{high_n:,}")
            m3.metric("Orta / Düşük Risk", f"{mid_n:,} / {low_n:,}")
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
                "Risk Skorlama Raporunu CSV Olarak İndir",
                data=csv_bytes,
                file_name="telco_churn_tahminleri.csv",
                mime="text/csv",
                type="primary",
            )

# ==========================================================
# TAB 3: MODEL CARD & BENCHMARK COMPARISON
# ==========================================================
with tab3:
    st.markdown('<div class="sec-head">Şampiyon Model Kartı (Model Card)</div>', unsafe_allow_html=True)
    st.markdown(
        _html(
            """
            <div class="mc-grid">
                <div class="mc-box">
                    <div class="mc-lbl">Mimari ve Hiperparametreler</div>
                    <strong style="color:#F8F6F1;">CatBoostClassifier (Tuned)</strong><br>
                    iterations=400 · depth=4 · lr=0,02 · L2=10<br>
                    auto_class_weights="Balanced" · seed=42
                </div>
                <div class="mc-box">
                    <div class="mc-lbl">Deney ve Sızıntı Güvenliği</div>
                    Stratified %70 / %15 / %15 (4.930 / 1.056 / 1.057)<br>
                    5-Fold CV · Eşit bütçeli RandomizedSearchCV (n=20)<br>
                    Tam sızıntısız scikit-learn Pipeline
                </div>
                <div class="mc-box">
                    <div class="mc-lbl">Kilitli Test ve Hata Profili</div>
                    PR-AUC: <strong style="color:#D9A760;">0,6772</strong> · ROC-AUC: <strong style="color:#F8F6F1;">0,8445</strong><br>
                    TP=214 · TN=593 · FP=183 · FN=67 (%76,2 Recall)<br>
                    İki yıllık sözleşmede %98,1 doğruluk
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-head">Kilitli Test Seti Model Karşılaştırması (Baseline + 6 Güçlü Rakip)</div>', unsafe_allow_html=True)
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
            st.markdown("**Test PR-AUC & ROC-AUC Karşılaştırması**")
            st.bar_chart(test_tbl.set_index("Model")[["Test PR-AUC", "Test ROC-AUC"]], height=345)
        with c_tbl:
            st.markdown("**Detaylı Performans Tablosu (n = 1.057)**")
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

        st.markdown("**Eşit Bütçeli Hiperparametre Optimizasyonu (Önce vs. Sonra)**")
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
    st.markdown('<div class="sec-head">Keşifsel Veri Analizi ve Değerlendirme Galerisi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Eğitim ve kilitli test sürecinden üretilen 8 gerçek bilimsel analiz grafiği.</div>',
        unsafe_allow_html=True,
    )
    fig_files = [
        ("fig1_eda_target_and_numeric.png", "Şekil 1 · Hedef Dağılımı ve Sayısal Özellikler"),
        ("fig2_eda_categorical_churn.png", "Şekil 2 · Kategorik Değişkenlere Göre Churn Oranları"),
        ("fig3_eda_correlation_heatmap.png", "Şekil 3 · Korelasyon Matrisi"),
        ("fig4_model_comparison_cv.png", "Şekil 4 · 5-Fold CV Model Karşılaştırması"),
        ("fig5_roc_pr_curves_test.png", "Şekil 5 · Test Seti ROC & Precision-Recall Eğrileri"),
        ("fig6_confusion_matrix_threshold.png", "Şekil 6 · Karışıklık Matrisi ve Eşik Optimizasyonu"),
        ("fig7_feature_importance.png", "Şekil 7 · En Önemli 15 Özellik (CatBoost)"),
        ("fig8_error_analysis.png", "Şekil 8 · Sözleşme Tipine Göre Hata Analizi"),
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
        <div class="footer-note">
            <strong style="color:#F4F1EA;">Model Sınırlılığı ve Sorumlu Kullanım Notu:</strong>
            Bu platform, IBM Telco Customer Churn kesit veri setiyle eğitilmiş olasılıksal bir tahmin modeli sunar;
            kesin sonuç garanti etmez ve yanlış pozitif (FP) veya yanlış negatif (FN) tahminler içerebilir.
            Çıktılar, iş kararlarında tek başına değil, müşteri ilişkileri uzman değerlendirmesiyle birlikte kullanılmalıdır.
        </div>
        """
    ),
    unsafe_allow_html=True,
)
