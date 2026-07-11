# 🌤️ Air Quality Classification Using Meteorology Parameters

An end-to-end Machine Learning research project to classify air quality categories (`Good`, `Moderate`, `Poor`, `Hazardous`) based **strictly on meteorological and environmental features** (e.g., Temperature, Humidity, Industrial Proximity) without relying on direct pollutant readings (`PM2.5`, `PM10`, `NO2`, `SO2`, `CO`).

> 💡 **Research Purpose:** By removing chemical pollutant features, this model avoids **data leakage** and evaluates whether low-cost, widely available weather station data alone can accurately predict air quality levels.

---

## 📊 Dataset Overview
* **Source:** Kaggle - Air Quality and Pollution Assessment Dataset
* **Observations:** 5,000 instances
* **Target Feature:** `Air_Quality` (4 Classes: `Good`, `Moderate`, `Poor`, `Hazardous`)
* **Input Features:** `Temperature`, `Humidity`, `Wind_Speed`, `Pressure`, `Proximity_to_Industrial`, `Population_Density`

---

## 🚀 Key Methodology & Results

1. **Feature Selection (Preventing Data Leakage):** Chemical pollutant columns were intentionally dropped to test pure meteorological predictability.
2. **Scaling & Cross-Validation:** Utilized a 10-Fold Stratified Cross-Validation inside a Scikit-Learn `Pipeline` with `StandardScaler` to prevent information leakage across validation folds.
3. **Statistical Hypothesis Testing:** Conducted a **Paired T-Test** to verify performance differences between models.

### 📈 Model Performance Comparison

| Model | Scaling | Mean CV Accuracy | Paired T-Test ($p$-value) | Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | `StandardScaler` | ~80% | — | Baseline Model |
| **Random Forest** | Not Required | **~86%** | **$p < 0.05$** | **Significantly Superior** |

> **Key Finding:** Random Forest significantly outperforms Logistic Regression. Feature importance analysis reveals that **Proximity to Industrial Areas** and **Temperature** are the most dominant non-pollutant factors influencing air quality classification.

---

## 📁 Repository Structure
```text
├── air_quality_prediction.ipynb   # Complete Jupyter Notebook (EDA, Pipeline, Models, T-Test)
├── updated_pollution_dataset.csv  # Raw Dataset
├── rf_air_quality_model.pkl       # Saved Random Forest Model
├── scaler.pkl                     # Saved StandardScaler Object
└── README.md                      # Project Documentation
