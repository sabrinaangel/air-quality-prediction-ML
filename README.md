# 🌤️ Air Quality Classification Using Meteorological Parameters

**UAS Pembelajaran Mesin (Machine Learning) — Genap 2025/2026**
**Universitas Dian Nuswantoro — Fakultas Ilmu Komputer**

An end-to-end Machine Learning capstone project that classifies air quality into four categories (`Good`, `Moderate`, `Poor`, `Hazardous`) using **only meteorological and environmental context features** — without relying on direct pollutant sensor readings.

🔗 **Live Demo:** [air-quality-prediction-ml-15596.streamlit.app](https://air-quality-prediction-ml-15596.streamlit.app/)

---

## 📌 Problem Statement

Air pollution monitoring typically relies on direct pollutant sensors (PM2.5, PM10, NO2, SO2, CO), which are expensive to install and maintain at scale. This project explores an alternative approach: **can air quality be classified using only meteorological and contextual features** — Temperature, Humidity, Proximity to Industrial Areas, and Population Density — **without any pollutant readings**?

This is a practically useful problem, since meteorological stations are far more common and cheaper to deploy than dedicated pollutant sensors. A model that can reasonably estimate air quality from weather and demographic context alone could serve as a low-cost, early-warning proxy in areas without dense pollutant-monitoring infrastructure.

**Success Criteria:** Macro-averaged F1-score above 0.80 and ROC-AUC above 0.90 on a held-out test set, with particular attention to recall on the minority `Hazardous` class.

---

## 📊 Dataset

- **Source:** Kaggle — [Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment) by Mujtaba Matin
- **Size:** 5,000 rows × 10 columns (raw)
- **Features used:** `Temperature`, `Humidity`, `Proximity_to_Industrial_Areas`, `Population_Density`
- **Target:** `Air Quality` — 4 classes (`Good`, `Moderate`, `Poor`, `Hazardous`)
- **Excluded features:** `PM2.5`, `PM10`, `NO2`, `SO2`, `CO` — deliberately dropped to avoid data leakage, since these pollutant values are used to derive the target label directly.

---

## 🧪 Methodology

1. **Feature Selection** — Pollutant columns removed to prevent data leakage; only meteorological/contextual features retained.
2. **Exploratory Data Analysis (EDA)** — Data quality checks (missing values, duplicates), a consistency check that identified and fixed invalid `Humidity` values above 100%, descriptive statistics, univariate & multivariate analysis, correlation heatmap, and outlier detection.
3. **Data Splitting** — Stratified **70% train / 15% validation / 15% test** split, preserving class proportions across all three sets.
4. **Baseline Modeling** — Logistic Regression (with `StandardScaler`) and Random Forest trained with default hyperparameters.
5. **Hyperparameter Tuning** — `GridSearchCV` (5-fold CV) applied to both models within the training set only.
6. **Model Selection** — 10-Fold Stratified Cross-Validation combined with a **Paired T-Test** (α = 0.05) to statistically compare the tuned models on train+validation data.
7. **Final Evaluation** — The selected model was retrained on train+validation and evaluated **once** on the untouched test set, using Accuracy, Precision, Recall, F1-Score, ROC-AUC (macro, one-vs-rest), and a Confusion Matrix.
8. **Model Interpretation** — SHAP (SHapley Additive exPlanations) and native feature importance used to explain which features drive predictions.
9. **Deployment** — Final model and scaler exported (`.pkl`) and served through an interactive Streamlit dashboard.

---

## 📈 Results

| Model | Baseline Val. Accuracy | Tuned Val. Accuracy | 10-Fold CV Mean Accuracy | Final Test Accuracy | Final Test Macro F1 | Final Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 81.89% | 81.89% | 81.65% | 80.27% | 0.751 | 0.945 |
| **Random Forest** | 87.08% | 87.22% | **87.11%** | **89.20%** | **0.855** | **0.973** |

**Random Forest** outperforms Logistic Regression at every stage of evaluation, and the difference is statistically significant (Paired T-Test, *p* < 0.05). It was selected as the final deployed model.

### Key Insights
- **`Proximity_to_Industrial_Areas`** and **`Temperature`** are the two most influential predictors of air quality, according to both SHAP analysis and exploratory data patterns.
- Locations closer to industrial areas and with higher temperatures are more likely to fall into the `Poor` or `Hazardous` categories.
- `Humidity` and `Population_Density` contribute secondary, non-linear signal that helps the model separate classes further.

---

## 📁 Repository Structure

```
air-quality-prediction-ML/
├── data/
│   └── updated_pollution_dataset.csv       # Raw dataset
├── notebooks/
│   └── air_quality_prediction.ipynb        # Full EDA, modeling, tuning, evaluation & SHAP
├── models/
│   ├── rf_air_quality_model.pkl            # Final tuned Random Forest model
│   └── scaler.pkl                          # StandardScaler used for Logistic Regression
├── app/
│   ├── app.py                              # Streamlit dashboard application
│   ├── requirements.txt                    # App-specific dependencies
│   ├── rf_air_quality_model.pkl            # Model copy used by the deployed app
│   ├── scaler.pkl                          # Scaler copy used by the deployed app
│   └── updated_pollution_dataset.csv       # Dataset copy used by the deployed app
├── reports/
│   └── final_report.pdf                    # Technical report (background, methodology, results, conclusion)
├── requirements.txt                        # Project-level dependencies
└── README.md
└── .gitignore
```

> **Note:** The dataset and model files are duplicated inside `app/` because Streamlit Community Cloud runs the app with the repository root as its working directory. Keeping self-contained copies alongside `app.py` ensures the deployed app can always locate them regardless of execution context.

---

## 🖥️ Streamlit Application

The deployed dashboard (`app/app.py`) provides:

1. **🏠 Home** — Project overview and dataset summary.
2. **📊 EDA Dashboard** — Interactive exploration of feature distributions, correlations, and class patterns.
3. **🔮 Model Demo** — Manually input meteorological values and get a real-time air quality prediction with class probabilities.
4. **📈 Model Evaluation** — Live-computed Accuracy, Macro F1, ROC-AUC, classification report, confusion matrix, and ROC curves on the test set.
5. **💡 Interpretation & Insights** — Feature importance chart and on-demand SHAP analysis, with business-oriented insights.
6. **📖 Documentation** — Dataset description, methodology summary, and usage guide.

**Live app:** [air-quality-prediction-ml-15596.streamlit.app](https://air-quality-prediction-ml-15596.streamlit.app/)

---

## ⚙️ Running Locally

**Notebook:**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/air_quality_prediction.ipynb
```

**Streamlit App:**
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · SHAP · Streamlit · Joblib

---

## 📚 References

- Matin, M. *Air Quality and Pollution Assessment* [Dataset]. Kaggle. https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS).

---

## 👤 Author

**Sabrina Angel**
Teknik Informatika, Universitas Dian Nuswantoro
NIM: A11.2024.15596 · Kelas A11.4402

Submitted for **UAS Pembelajaran Mesin — Genap 2025/2026**
