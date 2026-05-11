# 🏡 Texas Real Estate Intelligence System  
**ML System for Price Prediction, Market Analysis & Property Recommendation (Houston, TX)**  
🚀 Deployed ML system on AWS EC2 with real-time inference  

🔗 **Live App**: https://meet-real-estate.streamlit.app/  
---
## ⚡ Technical Snapshot
- Built a **full ML pipeline (sklearn Pipeline)** with:
  - OneHot Encoding + Target Encoding  
  - PCA for dimensionality reduction  
  - Log-transformed target for variance stabilization  

- Engineered **high-signal features from noisy real-world data**:
  - Parsed unstructured furnishing text → structured numerical features  
  - Derived **area ratios (super-built, carpet, built-up)**  
  - Designed **Luxury Score (weighted amenities)**  
  - Created **sector/society-level aggregations**  

- Solved real-world data challenges:
  - Outlier handling using **IQR + domain filtering**  
  - Missing value imputation using **feature relationships & grouped statistics**  

- Benchmarked **10+ ML models**:
  - Linear, Ridge, Lasso  
  - Decision Tree, Random Forest, Extra Trees  
  - Gradient Boosting, AdaBoost, XGBoost  
  - SVR, MLP  
  - Evaluated using **K-Fold Cross Validation (k=10)** with **R² & MAE**  

- Built a **content-based recommendation system**:
  - Feature similarity + K-Means clustering  
  - Context-aware property suggestions  

- Deployed on **AWS EC2**:
  - Streamlit UI  
  - Pickle-serialized model + pipeline for real-time inference  

---

<img width="1841" height="813" alt="image" src="https://github.com/user-attachments/assets/5b598e4b-a3cf-4eda-82e9-91ff930fdce8" />
*Unified interface integrating prediction, analysis, and recommendation*

---

## 🧩 Core Components

---

### 📈 1. Price Prediction Engine

- Built using **sklearn Pipeline** for consistent preprocessing and modeling  
- Applied **log transformation** to handle skewed price distribution  
- Compared 10+ regression models to select best-performing approach  

**Evaluation:**
- K-Fold Cross Validation (k=10)  
- Metrics: **R² Score, Mean Absolute Error (MAE)**  

---

### 📊 2. Data Analysis & Visualization

- Performed in-depth analysis to understand pricing behavior  
- Identified:
  - Skewed distributions  
  - Extreme outliers in price per sq. ft.  
  - Differences between house vs flat pricing  

**Techniques:**
- Histogram, Box Plot, ECDF  
- Skewness & Kurtosis  
- Quantile analysis & IQR-based outlier detection  

---

<img width="1833" height="614" alt="image" src="https://github.com/user-attachments/assets/0ac025e5-838f-477f-aa00-3285ac84a2b4" />
*Price distribution and outlier analysis*

---

### 🤖 3. Recommendation System

- Content-based filtering using **feature similarity**  
- Recommends properties based on:
  - Location  
  - Area & pricing patterns  
  - Amenities & furnishing  

- Enhanced using:
  - Feature engineering  
  - **K-Means clustering** for grouping similar listings  

---

<img width="1826" height="803" alt="image" src="https://github.com/user-attachments/assets/4be141f2-6deb-4fc0-9336-7fb1cc16741a" />
*Context-aware property recommendations*

---

## 🧠 Data Engineering & Feature Design

- Cleaned and standardized raw real estate data  
- Extracted structured values using **regex parsing**  
- Converted furnishing details into numerical feature columns  
- Engineered features:
  - Area ratios (super-built, carpet, built-up)  
  - Luxury score (weighted amenities)  
  - Sector & society-level aggregations  
  - Price bins & density features  

---

## 🏗️ ML Pipeline & System Design

Raw Listings → Cleaning → Feature Engineering → Encoding → PCA → Model → Recommendation → Deployment  

- OneHot Encoding + Target Encoding  
- PCA for dimensionality reduction  
- Designed pipeline to **prevent data leakage**  

---

## ☁️ Deployment

- Hosted on **AWS EC2**  
- Interactive UI built with **Streamlit**  
- Model + preprocessing pipeline serialized using **Pickle**  
- Supports real-time predictions and recommendations  

---
