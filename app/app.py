"""
Air Quality Classification Dashboard
--------------------------------------
UAS Pembelajaran Mesin - Genap 2025/2026
Universitas Dian Nuswantoro

A Streamlit application presenting an end-to-end Machine Learning solution
for classifying air quality using meteorological & contextual features only
(no direct pollutant sensor data).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, .app-title {
    font-family: 'Quicksand', sans-serif !important;
    font-weight: 700 !important;
}

.stApp {
    background: linear-gradient(180deg, #EAF6FB 0%, #F3F0FA 55%, #FBF3F7 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #CDEBF7 0%, #DCE3F7 100%);
    border-right: 1px solid rgba(255,255,255,0.4);
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Quicksand', sans-serif;
    font-weight: 600;
    font-size: 1rem;
}

.hero-card {
    background: linear-gradient(135deg, #A8DDF0 0%, #C9C9F2 100%);
    padding: 2.2rem 2rem;
    border-radius: 22px;
    color: #2B2B45;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(120,140,220,0.18);
}
.hero-card h1 {
    margin: 0 0 0.4rem 0;
    font-size: 2.1rem;
    color: #2B2B45;
}
.hero-card p {
    margin: 0;
    font-size: 1.02rem;
    opacity: 0.85;
}

.info-card {
    background: rgba(255,255,255,0.72);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 14px rgba(140,150,200,0.12);
    border: 1px solid rgba(255,255,255,0.6);
}

.metric-pill {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    color: #2B2B45;
    margin-right: 0.4rem;
}

.footer-note {
    text-align: center;
    opacity: 0.55;
    font-size: 0.8rem;
    margin-top: 2rem;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.75);
    padding: 0.9rem 1rem;
    border-radius: 16px;
    box-shadow: 0 3px 10px rgba(140,150,200,0.10);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DATA & MODEL LOADING (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_raw_data():
    df = pd.read_csv("updated_pollution_dataset.csv")
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
    model = joblib.load("rf_air_quality_model.pkl")
    scaler = joblib.load("scaler.pkl")
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
st.sidebar.markdown("### 🌤️ Air Quality ML")
st.sidebar.caption("UAS Pembelajaran Mesin · Genap 2025/2026")
page = st.sidebar.radio(
    "Navigasi",
    [
        "🏠 Beranda",
        "📊 Dashboard EDA",
        "🔮 Model Demo",
        "📈 Evaluasi Model",
        "💡 Interpretasi & Insight",
        "📖 Dokumentasi",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.8rem; opacity:0.75;'>"
    "Model: <b>Random Forest</b> (tuned)<br>"
    "Fitur: Temperature, Humidity,<br>Proximity to Industrial Areas,<br>Population Density"
    "</div>",
    unsafe_allow_html=True,
)


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
            <h1>🌤️ Air Quality Classification</h1>
            <p>Predicting air quality categories from meteorological & contextual data —
            without relying on direct pollutant sensors.</p>
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

    st.markdown("#### Distribusi Kelas Air Quality")
    fig, ax = plt.subplots(figsize=(8, 3))
    counts = df_features["Air Quality"].value_counts().reindex(CLASS_ORDER)
    ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in CLASS_ORDER])
    ax.set_ylabel("Jumlah Data")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)


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
        st.markdown(
            f"""
            <div style="background:{color}; padding:1.6rem; border-radius:18px; text-align:center; margin-top:1rem;">
                <div style="font-size:1rem; opacity:0.75;">Hasil Prediksi</div>
                <div style="font-size:2.2rem; font-weight:700; font-family:'Quicksand',sans-serif;">{prediction}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Probabilitas Tiap Kelas")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(proba_df.index, proba_df["Probabilitas"], color=[PALETTE[c] for c in proba_df.index])
        ax.set_xlim(0, 1)
        for i, v in enumerate(proba_df["Probabilitas"]):
            ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)


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

    st.markdown("#### Feature Importance (Random Forest)")
    importance_df = pd.DataFrame({
        "Fitur": FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(importance_df["Fitur"], importance_df["Importance"], color="#A9C7E0")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

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
    "<div class='footer-note'>Air Quality Classification · UAS Pembelajaran Mesin · Universitas Dian Nuswantoro</div>",
    unsafe_allow_html=True,
)
