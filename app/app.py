"""
Air Quality Classification Dashboard
--------------------------------------
UAS Pembelajaran Mesin - Genap 2025/2026
Universitas Dian Nuswantoro

A Streamlit application presenting an end-to-end Machine Learning solution
for classifying air quality using meteorological & contextual features only
(no direct pollutant sensor data).
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Resolve all data/model file paths relative to THIS script's location,
# so the app works regardless of the working directory it's launched from
# (Streamlit Community Cloud runs scripts with the repo root as cwd, not
# the folder the script lives in).
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def app_path(filename):
    return os.path.join(APP_DIR, filename)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Air Quality Classification",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME / CUSTOM CSS
# ---------------------------------------------------------------------------
PALETTE = {
    "Good": "#8FD9A8",
    "Moderate": "#FFE08A",
    "Poor": "#FFB37B",
    "Hazardous": "#FF8A8A",
}
CLASS_ORDER = ["Good", "Moderate", "Poor", "Hazardous"]

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;600;700&family=Nunito:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}
h1, h2, h3, h4, .app-title {
    font-family: 'Quicksand', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}

/* ========== LIGHT THEME ========== */
[data-theme="light"] .stApp,
.stApp {
    background: linear-gradient(160deg, #D6EFFA 0%, #E8E4F8 40%, #FAE9F0 75%, #FDF6EC 100%);
    min-height: 100vh;
}
[data-theme="light"] section[data-testid="stSidebar"],
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #B8E0F5 0%, #C8D8F5 50%, #D8CCF0 100%) !important;
    border-right: 2px solid rgba(255,255,255,0.55);
    box-shadow: 3px 0 20px rgba(130,150,220,0.12);
}
[data-theme="light"] .nav-btn {
    background: rgba(255,255,255,0.45);
    color: #1E2A5E;
    border: 1px solid rgba(100,130,210,0.18);
}
[data-theme="light"] .nav-btn:hover {
    background: rgba(255,255,255,0.72);
    color: #1E2A5E;
    border-color: rgba(100,130,210,0.38);
}
[data-theme="light"] .nav-btn.active {
    background: linear-gradient(135deg, #7BBFE8 0%, #9AAEF5 100%);
    color: #1E1E38;
    border-color: transparent;
    box-shadow: 0 4px 14px rgba(100,130,220,0.30);
}
[data-theme="light"] .info-card {
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(255,255,255,0.7);
    color: inherit;
}
[data-theme="light"] div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(255,255,255,0.65);
}
[data-theme="light"] .hero-card {
    background: linear-gradient(135deg, #8ECFEE 0%, #AABEF5 50%, #C9AEF0 100%);
    color: #1E1E38;
}
[data-theme="light"] .hero-card h1 { color: #1E1E38; }

/* ========== DARK THEME ========== */
[data-theme="dark"] .stApp {
    background: linear-gradient(160deg, #0D1B2A 0%, #121828 40%, #1A1230 75%, #1C1522 100%);
    min-height: 100vh;
}
[data-theme="dark"] section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F2035 0%, #131A30 50%, #1A1230 100%) !important;
    border-right: 2px solid rgba(100,140,220,0.18);
    box-shadow: 3px 0 24px rgba(30,50,120,0.25);
}
[data-theme="dark"] .nav-btn {
    background: rgba(255,255,255,0.06);
    color: #A8C8E8;
    border: 1px solid rgba(100,140,220,0.18);
}
[data-theme="dark"] .nav-btn:hover {
    background: rgba(100,140,220,0.18);
    color: #D8EEFF;
    border-color: rgba(100,140,220,0.38);
}
[data-theme="dark"] .nav-btn.active {
    background: linear-gradient(135deg, #2A6FA8 0%, #3A5AB8 100%);
    color: #E8F4FF;
    border-color: transparent;
    box-shadow: 0 4px 16px rgba(50,100,220,0.40);
}
[data-theme="dark"] .info-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(100,140,220,0.18);
    color: #C8D8EE;
}
[data-theme="dark"] div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(100,140,220,0.18);
}
[data-theme="dark"] .hero-card {
    background: linear-gradient(135deg, #1A4060 0%, #1E2A5E 50%, #2A1A50 100%);
    color: #E8F4FF;
    box-shadow: 0 12px 35px rgba(20,60,160,0.35), 0 2px 8px rgba(20,60,160,0.18);
}
[data-theme="dark"] .hero-card h1 { color: #E8F4FF; }
[data-theme="dark"] .hero-card p  { color: #A8C8E8; }
[data-theme="dark"] .footer-note  { border-top-color: rgba(100,140,220,0.18); color: #7090B0; }
[data-theme="dark"] .stButton > button {
    background: linear-gradient(135deg, #2A6FA8 0%, #3A5AB8 100%);
    color: #E8F4FF;
}

/* ========== SHARED STYLES ========== */
.hero-card {
    padding: 2.6rem 2.4rem;
    border-radius: 26px;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    transition: background 0.3s ease;
}
.hero-card::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.10);
    pointer-events: none;
}
.hero-card::after {
    content: '';
    position: absolute;
    bottom: -30px; left: 30px;
    width: 140px; height: 140px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
    pointer-events: none;
}
.hero-card h1 {
    margin: 0 0 0.5rem 0;
    font-size: 2.3rem;
    position: relative; z-index:1;
}
.hero-card p {
    margin: 0;
    font-size: 1.05rem;
    opacity: 0.85;
    line-height: 1.6;
    position: relative; z-index:1;
}

.info-card {
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 6px 20px rgba(100,120,210,0.10);
    transition: box-shadow 0.25s ease, transform 0.2s ease;
}
.info-card:hover {
    box-shadow: 0 10px 30px rgba(100,120,210,0.18);
    transform: translateY(-2px);
}

/* Custom nav buttons (replaces radio) */
.nav-btn {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    width: 100%;
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.3rem;
    border-radius: 12px;
    font-family: 'Quicksand', sans-serif;
    font-weight: 600;
    font-size: 0.97rem;
    cursor: pointer;
    text-align: left;
    transition: background 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease, color 0.2s ease;
    text-decoration: none !important;
}
.nav-btn:hover {
    transform: translateX(3px);
}
.nav-btn.active {
    font-weight: 700;
}
/* Hide the stButton wrapper margin for nav buttons */
.nav-btn-wrapper > div > button {
    background: none !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
    width: 100%;
    text-align: left;
}

.metric-pill {
    display: inline-block;
    padding: 0.32rem 0.95rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.88rem;
    color: #2B2B45;
    margin-right: 0.45rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    transition: transform 0.15s ease;
}
.metric-pill:hover { transform: scale(1.05); }

.pred-card {
    padding: 2rem;
    border-radius: 22px;
    text-align: center;
    margin-top: 1.2rem;
    box-shadow: 0 8px 28px rgba(0,0,0,0.13);
    animation: popIn 0.4s cubic-bezier(0.175,0.885,0.32,1.275);
}
@keyframes popIn {
    from { transform: scale(0.88); opacity: 0; }
    to   { transform: scale(1);    opacity: 1; }
}
.pred-label { font-size: 0.95rem; opacity: 0.72; margin-bottom: 0.4rem; font-family:'Nunito',sans-serif; }
.pred-value { font-size: 2.6rem; font-weight: 800; font-family:'Quicksand',sans-serif; line-height:1.1; }
.pred-icon  { font-size: 3rem; margin-bottom: 0.4rem; display:block; }

.footer-note {
    text-align: center;
    opacity: 0.55;
    font-size: 0.82rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(150,160,220,0.18);
    font-family: 'Nunito', sans-serif;
}

div[data-testid="stMetric"] {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(100,120,210,0.10);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(100,120,210,0.18);
}

.stButton > button {
    background: linear-gradient(135deg, #7BBFE8 0%, #9AAEF5 100%);
    color: #1E1E38;
    font-family: 'Quicksand', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 14px;
    padding: 0.65rem 1.5rem;
    box-shadow: 0 4px 14px rgba(100,130,220,0.25);
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    cursor: pointer;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(100,130,220,0.32);
    filter: brightness(1.08);
}
.stButton > button:active { transform: translateY(0); }

.stTabs [data-baseweb="tab"] {
    font-family: 'Quicksand', sans-serif;
    font-weight: 600;
    border-radius: 10px 10px 0 0;
}
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DATA & MODEL LOADING (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_raw_data():
    df = pd.read_csv(app_path("updated_pollution_dataset.csv"))
    return df


@st.cache_data
def load_features(df):
    pollutant_columns = ["PM2.5", "PM10", "NO2", "SO2", "CO"]
    df_features = df.drop(columns=pollutant_columns).copy()
    # Same consistency fix as in the notebook: cap Humidity at 100%
    df_features["Humidity"] = df_features["Humidity"].clip(upper=100)
    return df_features


@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(app_path("rf_air_quality_model.pkl"))
    scaler = joblib.load(app_path("scaler.pkl"))
    return model, scaler


@st.cache_data
def get_test_split(_df_features):
    """Reproduce the exact same train/val/test split used in the notebook
    (random_state=42) so evaluation here matches the reported notebook results."""
    X = _df_features.drop(columns=["Air Quality"])
    y = _df_features["Air Quality"]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1765, stratify=y_trainval, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, X_trainval, y_trainval


raw_df = load_raw_data()
df_features = load_features(raw_df)
model, scaler = load_model_and_scaler()
FEATURE_COLS = ["Temperature", "Humidity", "Proximity_to_Industrial_Areas", "Population_Density"]


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
NAV_PAGES = [
    ("🏠", "Beranda"),
    ("📊", "Dashboard EDA"),
    ("🔮", "Model Demo"),
    ("📈", "Evaluasi Model"),
    ("💡", "Interpretasi & Insight"),
    ("📖", "Dokumentasi"),
]

if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Beranda"

st.sidebar.markdown(
    "<div style='text-align:center; padding:0.8rem 0 0.4rem;'>"
    "<span style='font-size:2.4rem;'>🌤️</span><br>"
    "<span style='font-family:Quicksand,sans-serif; font-weight:700; font-size:1.15rem;'>Air Quality ML</span>"
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.caption("UAS Pembelajaran Mesin · Genap 2025/2026")
st.sidebar.markdown("<div style='margin:0.4rem 0 0.8rem; border-bottom:1px solid rgba(130,150,220,0.25);'></div>", unsafe_allow_html=True)

# Render custom styled navigation buttons
for icon, label in NAV_PAGES:
    page_key = f"{icon} {label}"
    is_active = st.session_state["page"] == page_key
    active_cls = "active" if is_active else ""
    clicked = st.sidebar.button(
        f"{icon}  {label}",
        key=f"nav_{label}",
        use_container_width=True,
    )
    if clicked:
        st.session_state["page"] = page_key
        st.rerun()

# Inject active styling via JS-based class targeting is not feasible;
# instead we use the button label approach with CSS overriding below.
# Highlight active button by injecting a targeted CSS snippet:
active_label = st.session_state["page"]
st.sidebar.markdown(
    f"""
    <style>
    /* Highlight active nav button */
    section[data-testid="stSidebar"] button[kind="secondary"] {{
        background: rgba(255,255,255,0.30);
        border: 1px solid rgba(100,140,220,0.25);
        border-radius: 12px;
        font-family: 'Quicksand', sans-serif;
        font-weight: 600;
        font-size: 0.97rem;
        text-align: left;
        justify-content: flex-start;
        padding: 0.55rem 0.85rem;
        margin-bottom: 0.15rem;
        transition: background 0.2s ease, transform 0.15s ease;
        color: inherit;
    }}
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {{
        background: rgba(255,255,255,0.50);
        transform: translateX(3px);
    }}
    [data-theme="dark"] section[data-testid="stSidebar"] button[kind="secondary"] {{
        background: rgba(255,255,255,0.07);
        border-color: rgba(100,140,220,0.20);
        color: #A8C8E8;
    }}
    [data-theme="dark"] section[data-testid="stSidebar"] button[kind="secondary"]:hover {{
        background: rgba(100,140,220,0.18);
        color: #D8EEFF;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div style='margin:0.8rem 0 0.4rem; border-bottom:1px solid rgba(130,150,220,0.25);'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div style='font-size:0.78rem; opacity:0.72; line-height:1.7; padding:0 0.3rem;'>"
    "🤖 Model: <b>Random Forest</b> (tuned)<br>"
    "🌡️ Temperature &nbsp;💧 Humidity<br>"
    "🏭 Proximity Industrial<br>"
    "🏙️ Population Density"
    "</div>",
    unsafe_allow_html=True,
)

page = st.session_state["page"]


def class_badge(label):
    color = PALETTE.get(label, "#ccc")
    return f"<span class='metric-pill' style='background:{color};'>{label}</span>"


# ---------------------------------------------------------------------------
# PAGE 1 — HOME
# ---------------------------------------------------------------------------
if page == "🏠 Beranda":
    st.markdown(
        """
        <div class="hero-card">
            <div style="font-size:3.2rem; margin-bottom:0.5rem;">🌤️ ☁️ 🍃</div>
            <h1>Air Quality Classification</h1>
            <p>Memprediksi kategori kualitas udara dari data meteorologi &amp; konteks lingkungan —
            tanpa bergantung pada sensor pollutant langsung. 💨</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", f"{len(raw_df):,}")
    col2.metric("Fitur Digunakan", "4")
    col3.metric("Kelas Target", "4")
    col4.metric("Model Terbaik", "Random Forest")

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("""
#### Tentang Project Ini
Polusi udara biasanya dipantau lewat sensor pollutant langsung (PM2.5, PM10, NO2, SO2, CO) yang mahal untuk
dipasang secara luas. Project ini menguji pendekatan alternatif: **memprediksi kategori kualitas udara hanya dari
data cuaca dan konteks lingkungan** — Temperature, Humidity, Proximity to Industrial Areas, dan Population Density —
tanpa data pollutant sama sekali.

Pendekatan ini relevan karena stasiun cuaca jauh lebih umum dan murah dibanding sensor pollutant khusus, sehingga
model semacam ini bisa berfungsi sebagai early-warning proxy berbiaya rendah di daerah tanpa infrastruktur
pemantauan pollutant yang memadai.

Gunakan menu di sidebar kiri untuk menjelajahi eksplorasi data, mencoba prediksi langsung, melihat evaluasi model,
serta interpretasi hasilnya.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### 📊 Distribusi Kelas Air Quality")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    counts = df_features["Air Quality"].value_counts().reindex(CLASS_ORDER)
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in CLASS_ORDER],
                  edgecolor='white', linewidth=1.5, width=0.55)
    ax.set_ylabel("Jumlah Data", fontsize=11)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontweight="bold", fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis='y', color='#ccc', linestyle='--', alpha=0.4)
    st.pyplot(fig, transparent=True)


# ---------------------------------------------------------------------------
# PAGE 2 — EDA DASHBOARD
# ---------------------------------------------------------------------------
elif page == "📊 Dashboard EDA":
    st.markdown("## 📊 Dashboard Exploratory Data Analysis")
    st.caption("Eksplorasi interaktif dataset setelah pembersihan data (Humidity di-cap ke 100%, kolom pollutant dihapus).")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Ringkasan Data", "Distribusi Fitur", "Korelasi", "Fitur vs Air Quality"]
    )

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Statistik Deskriptif**")
            st.dataframe(df_features[FEATURE_COLS].describe().T.style.format("{:.2f}"), use_container_width=True)
        with c2:
            st.markdown("**Data Quality**")
            quality_df = pd.DataFrame({
                "Cek": ["Missing values", "Duplikat", "Humidity > 100% (sebelum fix)", "Total baris"],
                "Hasil": [df_features.isnull().sum().sum(), df_features.duplicated().sum(), 195, len(df_features)],
            })
            st.table(quality_df)
        st.markdown("**Preview Data**")
        st.dataframe(df_features.head(10), use_container_width=True)

    with tab2:
        selected_feature = st.selectbox("Pilih fitur:", FEATURE_COLS)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df_features[selected_feature], kde=True, ax=axes[0], color="#7FB3D5")
        axes[0].set_title(f"Distribusi {selected_feature}")
        sns.boxplot(y=df_features[selected_feature], ax=axes[1], color="#F5B7B1")
        axes[1].set_title(f"Boxplot {selected_feature}")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(6, 5))
        corr = df_features[FEATURE_COLS].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
        st.info("Korelasi antar fitur rendah–sedang → tidak ada masalah multikolinearitas serius.")

    with tab4:
        selected_feature2 = st.selectbox("Pilih fitur untuk dibandingkan per kelas:", FEATURE_COLS, key="biv")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df_features, x="Air Quality", y=selected_feature2, order=CLASS_ORDER,
                    palette=PALETTE, ax=ax)
        ax.set_title(f"{selected_feature2} berdasarkan Air Quality")
        st.pyplot(fig)


# ---------------------------------------------------------------------------
# PAGE 3 — MODEL DEMO
# ---------------------------------------------------------------------------
elif page == "🔮 Model Demo":
    st.markdown("## 🔮 Coba Prediksi Air Quality")
    st.caption("Masukkan kondisi meteorologi & lingkungan, lalu lihat prediksi model Random Forest secara langsung.")

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        temperature = st.slider("🌡️ Temperature (°C)", 10.0, 60.0, 30.0, 0.1)
        humidity = st.slider("💧 Humidity (%)", 0.0, 100.0, 70.0, 0.1)
    with c2:
        proximity = st.slider("🏭 Proximity to Industrial Areas (km)", 0.0, 30.0, 8.0, 0.1)
        population = st.slider("🏙️ Population Density (people/km²)", 0, 1000, 500, 10)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Prediksi Air Quality", use_container_width=True):
        input_df = pd.DataFrame([[temperature, humidity, proximity, population]], columns=FEATURE_COLS)
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        proba_df = pd.DataFrame({"Kelas": model.classes_, "Probabilitas": proba}).set_index("Kelas").reindex(CLASS_ORDER)

        color = PALETTE.get(prediction, "#ccc")
        PRED_ICONS = {"Good": "🌿", "Moderate": "🌤️", "Poor": "🌫️", "Hazardous": "☠️"}
        icon = PRED_ICONS.get(prediction, "🌀")
        st.markdown(
            f"""
            <div class="pred-card" style="background:linear-gradient(135deg,{color},{color}cc);">
                <span class="pred-icon">{icon}</span>
                <div class="pred-label">Hasil Prediksi Kualitas Udara</div>
                <div class="pred-value">{prediction}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📊 Probabilitas Tiap Kelas")
        fig, ax = plt.subplots(figsize=(8, 3.2))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        bars = ax.barh(proba_df.index, proba_df["Probabilitas"],
                       color=[PALETTE[c] for c in proba_df.index],
                       edgecolor='white', linewidth=1.2, height=0.55)
        ax.set_xlim(0, 1)
        for i, v in enumerate(proba_df["Probabilitas"]):
            ax.text(v + 0.015, i, f"{v*100:.1f}%", va="center", fontweight="bold", fontsize=11)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.tick_params(bottom=False)
        ax.grid(axis='x', color='#ccc', linestyle='--', alpha=0.35)
        st.pyplot(fig, transparent=True)


# ---------------------------------------------------------------------------
# PAGE 4 — MODEL EVALUATION
# ---------------------------------------------------------------------------
elif page == "📈 Evaluasi Model":
    st.markdown("## 📈 Evaluasi Model")
    st.caption("Dihitung langsung dari model tersimpan pada test set (data yang tidak pernah dilihat saat training).")

    X_train, X_val, X_test, y_train, y_val, y_test, X_trainval, y_trainval = get_test_split(df_features)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    macro_f1 = report_dict["macro avg"]["f1-score"]
    roc_auc_val = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc*100:.2f}%")
    c2.metric("Macro F1-Score", f"{macro_f1:.3f}")
    c3.metric("ROC-AUC (macro)", f"{roc_auc_val:.3f}")
    c4.metric("Jumlah Data Test", f"{len(X_test)}")

    tab1, tab2, tab3 = st.tabs(["Classification Report", "Confusion Matrix", "ROC Curve"])

    with tab1:
        report_df = pd.DataFrame(report_dict).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    with tab2:
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

    with tab3:
        y_test_bin = label_binarize(y_test, classes=model.classes_)
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, cls in enumerate(model.classes_):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc_i = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc_i:.3f})", color=PALETTE.get(cls, None))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)


# ---------------------------------------------------------------------------
# PAGE 5 — INTERPRETATION & INSIGHTS
# ---------------------------------------------------------------------------
elif page == "💡 Interpretasi & Insight":
    st.markdown("## 💡 Interpretasi Model & Insight Bisnis")

    st.markdown("#### 🌬️ Feature Importance (Random Forest)")
    importance_df = pd.DataFrame({
        "Fitur": FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    colors_imp = ["#A8DDF0", "#B5C8F5", "#C5B8F0", "#F0C5D8"]
    ax.barh(importance_df["Fitur"], importance_df["Importance"],
            color=colors_imp, edgecolor='white', linewidth=1.2, height=0.55)
    for i, v in enumerate(importance_df["Importance"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontweight="bold", fontsize=10)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(bottom=False)
    ax.grid(axis='x', color='#ccc', linestyle='--', alpha=0.35)
    st.pyplot(fig, transparent=True)

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("""
#### 📌 Insight Bisnis
- **Proximity to Industrial Areas** dan **Temperature** adalah dua faktor paling dominan dalam menentukan kategori
  kualitas udara — semakin dekat ke kawasan industri dan semakin tinggi suhu, semakin besar kemungkinan kategori
  udara masuk **Poor** atau **Hazardous**.
- **Humidity** dan **Population Density** berkontribusi lebih kecil namun tetap membantu model menangkap pola
  non-linear antar kelas.
- **Implikasi praktis**: pemerintah daerah atau instansi lingkungan dapat memprioritaskan pemantauan tambahan di
  wilayah dengan kombinasi suhu tinggi dan jarak dekat ke kawasan industri, bahkan sebelum data sensor pollutant
  langsung tersedia.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Analisis SHAP (Opsional)")
    st.caption("Klik tombol di bawah untuk menghitung SHAP value secara live (butuh beberapa detik).")
    if st.button("⚙️ Jalankan Analisis SHAP"):
        with st.spinner("Menghitung SHAP values pada sample data test..."):
            import shap
            X_train, X_val, X_test, y_train, y_val, y_test, X_trainval, y_trainval = get_test_split(df_features)
            explainer = shap.TreeExplainer(model)
            X_sample = X_test.sample(min(150, len(X_test)), random_state=42)
            shap_values = explainer.shap_values(X_sample)
            fig = plt.figure(figsize=(8, 4))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", class_names=model.classes_, show=False)
            st.pyplot(fig)


# ---------------------------------------------------------------------------
# PAGE 6 — DOCUMENTATION
# ---------------------------------------------------------------------------
elif page == "📖 Dokumentasi":
    st.markdown("## 📖 Dokumentasi Project")

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("""
### Dataset
- **Sumber**: Kaggle — [Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment)
- **Ukuran**: 5.000 baris x 10 kolom (data mentah)
- **Fitur yang digunakan**: Temperature, Humidity, Proximity_to_Industrial_Areas, Population_Density
- **Target**: Air Quality (Good, Moderate, Poor, Hazardous)
- Kolom pollutant (PM2.5, PM10, NO2, SO2, CO) sengaja **dihapus** untuk menghindari data leakage, karena label
  Air Quality diturunkan langsung dari nilai pollutant tersebut.

### Metodologi
1. **Data Cleaning** — cek missing value, duplikat, dan inconsistency (Humidity di-cap ke maksimum 100%).
2. **EDA** — analisis univariat, multivariat, korelasi, dan distribusi kelas.
3. **Split Data** — train (70%) / validation (15%) / test (15%), stratified.
4. **Modeling** — Logistic Regression & Random Forest, dituning dengan GridSearchCV (5-fold CV).
5. **Model Selection** — 10-Fold Stratified Cross-Validation + Paired T-Test untuk memilih model terbaik secara
   statistik.
6. **Evaluasi Final** — model terbaik dilatih ulang pada data train+validation, dievaluasi sekali di test set
   (Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix).
7. **Interpretasi** — SHAP & feature importance untuk memahami kontribusi tiap fitur.
8. **Deployment** — model & scaler diekspor ke `.pkl`, disajikan lewat dashboard Streamlit ini.

### Cara Menggunakan Dashboard
- **Dashboard EDA** — eksplorasi data secara interaktif.
- **Model Demo** — masukkan nilai fitur secara manual untuk melihat prediksi model.
- **Evaluasi Model** — lihat performa model pada test set.
- **Interpretasi & Insight** — pahami fitur mana yang paling berpengaruh terhadap prediksi.

### Repository
Source code lengkap (notebook, model training, dan aplikasi ini) tersedia di GitHub:
[`sabrinaangel/air-quality-prediction-ML`](https://github.com/sabrinaangel/air-quality-prediction-ML)
    """)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    "<div class='footer-note'>☁️ Air Quality Classification &nbsp;·&nbsp; UAS Pembelajaran Mesin &nbsp;·&nbsp; Universitas Dian Nuswantoro &nbsp;🍃</div>",
    unsafe_allow_html=True,
)
