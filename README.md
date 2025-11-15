
# Project Canary

### Short-Term Forecasting of Ground-Level O₃ and NO₂ Using Multi-Source Data Fusion

### Smart India Hackathon 2025 – Problem Statement SIH25178

---

## 🌍 Overview

**Project Canary** is a hybrid deep-learning model for forecasting **NO₂** and **O₃** concentrations by fusing **satellite data**, **meteorological reanalysis**, and **ground-station observations**.
The system uses a CNN + BiLSTM + Attention pipeline to capture spatial pollutant behavior, temporal evolution, and weather-driven influences.

---

## 🚀 Key Highlights

* **Multi-source fusion** of Sentinel-5P, ERA5, and CPCB data.
* **Hybrid deep learning architecture** combining spatial and temporal modeling.
* **Simultaneous forecasting** of NO₂ and O₃.
* **Spatial raster encoding** via CNN.
* **Temporal dynamics** captured via BiLSTM.
* **Interpretability** through attention weights.

---

## 📡 Data Sources

### Satellite (Sentinel-5P TROPOMI)

* NO₂ and O₃ tropospheric column densities
* Processed from GeoTIFF
* Resampled to **10 × 11** spatial grid aligned with Delhi AOI

### Meteorological (ERA5)

* Hourly variables:
  Temp, RH, wind (u/v), precipitation, solar radiation, boundary-layer height, dewpoint, etc.
* Interpolated to station coordinates

### Ground-Station (CPCB)

* Hourly NO₂ & O₃ measurements
* Used as supervised labels

---

## 🛠 Data Pipeline

1. **Ingestion**

   * Load Sentinel-5P rasters
   * Load ERA5 weather series
   * Load CPCB pollutant readings

2. **Preprocessing**

   * Temporal alignment
   * Imputation (previous-day fill, KNN)
   * Z-score normalization
   * Derived features (dewpoint, cyclical time features)

3. **Spatial Grid Preparation**

   * Crop satellite rasters to region
   * Resample to fixed **10×11 grid**
   * Stack channels (NO₂, O₃, meteo grids)

4. **Final Input Shapes**

   * Satellite tensors: `(T, 9, 10, 11)`
   * Tabular station features: `(T, 18)`
   * Labels: `(T, 2)` for NO₂ & O₃

---

## 🧠 Model Architecture

```
Satellite Raster (T, C, 10, 11)
        ↓
     CNN Encoder
   (Conv → ReLU → MaxPool × 3)
        ↓
   64-Dim Spatial Embedding
```

```
Station Features (T, 18)
        ↓
        MLP
        ↓
   64-Dim Tabular Embedding
```

**Fusion → BiLSTM → Attention → Dual Regression Heads**

### Components (compact summary)

* **CNN Encoder:** Learns spatial pollutant patterns from raster grids.
* **MLP Block:** Encodes meteorological + derived features.
* **Fusion Layer:** Concatenates spatial and tabular embeddings.
* **BiLSTM:** Models temporal evolution across past 24 hours.
* **Attention:** Highlights key timesteps (e.g., rush hours, wind changes).
* **Output Heads:**

  * NO₂ concentration
  * O₃ concentration

---

## 📈 Model Performance

### **NO₂**

* **MAE:** 12.35
* **RMSE:** 17.89
* **R²:** 0.755

### **O₃**

* **MAE:** 14.65
* **RMSE:** 20.91
* **R²:** 0.666

---

## 🔧 Tech Stack

* **PyTorch** – modeling
* **scikit-learn** – preprocessing & feature engineering
* **NumPy / Pandas** – time-series & data handling
* **Rasterio** – GeoTIFF reading & spatial ops

---

## 📚 References
Attention mechanism based CNN-LSTM hybrid deep learning model for atmospheric ozone concentration prediction](https://www.nature.com/articles/s41598-025-05877-2)

