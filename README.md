# 🏡 Texas Real Estate Intelligence System  
**Prediction • Analysis • Recommendation (Houston, TX)**

🔗 **Live App**: *https://meet-real-estate.streamlit.app/*  
📂 **Repository**: https://github.com/Meetpaghadar/real_estate  

---

## ⚡ Overview

An end-to-end machine learning system built on real-world Houston housing data that enables:

- 📈 **Price Prediction** using advanced regression models  
- 📊 **Market Analysis & Visualization** for data-driven insights  
- 🤖 **Property Recommendation** based on similarity  

Designed to handle **messy, unstructured real estate data** and deployed on **AWS EC2 for real-time interaction**.

---
<img width="1841" height="813" alt="image" src="https://github.com/user-attachments/assets/5b598e4b-a3cf-4eda-82e9-91ff930fdce8" />



*Unified Streamlit interface integrating prediction, analysis, and recommendations*

---

## 🧩 Core Components

---

### 📈 1. Price Prediction Engine

- Built a robust ML pipeline using **sklearn Pipeline**
- Applied **log transformation** to handle skewed pricing data  
- Evaluated 10+ models:
  - Linear, Ridge, Lasso  
  - Decision Tree, Random Forest, Extra Trees  
  - Gradient Boosting, AdaBoost, XGBoost  
  - SVR, MLP  

**Evaluation Strategy:**
- K-Fold Cross Validation (k=10)  
- Metrics: **R² Score, Mean Absolute Error (MAE)**  

---

### 📊 2. Data Analysis & Visualization

- Performed deep exploratory analysis on pricing trends  
- Identified:
  - Skewed price distribution  
  - Outliers in price per sq. ft.  
  - Behavioral differences between houses vs flats  

**Techniques Used:**
- Histogram, Box Plot, ECDF  
- Skewness & Kurtosis analysis  
- Quantile-based binning  
- IQR-based outlier detection  


---

<img width="1833" height="614" alt="image" src="https://github.com/user-attachments/assets/0ac025e5-838f-477f-aa00-3285ac84a2b4" />
*Price distribution, outlier detection, and transformation insights*
<img width="1842" height="828" alt="image" src="https://github.com/user-attachments/assets/494a7fa7-39d9-41c6-b5e1-80e071129505" />


---

### 🤖 3. Recommendation System

- Content-based filtering using **feature similarity**
- Suggests properties based on:
  - Location  
  - Area & price characteristics  
  - Amenities & furnishing  

Enhanced using:
- Feature engineering  
- Clustering (K-Means) for grouping similar listings  

---

<img width="1826" height="803" alt="image" src="https://github.com/user-attachments/assets/4be141f2-6deb-4fc0-9336-7fb1cc16741a" />

*Context-aware property recommendations based on selected listing*

---

## 🧠 Data Engineering & Feature Design

- Handled missing values using **feature relationships & grouped imputation**  
- Extracted structured data from text using **regex parsing**  
- Converted furnishing details into numerical features  
- Engineered high-impact features:
  - Area ratios (super-built, carpet, built-up)  
  - Luxury score (weighted amenities)  
  - Sector & society aggregations  
  - Price bins & density features  

---

## 🏗️ ML Pipeline & System Design

Raw Listings → Cleaning → Feature Engineering → Encoding → PCA → Model → Recommendation Layer → Deployment

- OneHot Encoding + Target Encoding  
- PCA for dimensionality reduction  
- Modular pipeline to prevent data leakage  

---

## ☁️ Deployment

- 🚀 Deployed on **AWS EC2**  
- 🌐 Interactive UI using **Streamlit**  
- 📦 Model serialized using Pickle  
- ⚙️ End-to-end pipeline ensures consistent inference  

---

## 🛠️ Tech Stack

**Languages:** Python  
**Libraries:** Pandas, NumPy, Scikit-learn, XGBoost  
**Visualization:** Matplotlib, Seaborn  
**ML Concepts:** Regression, Clustering, Feature Engineering, Pipelines  
**Deployment:** AWS EC2, Streamlit  

---

## 🚧 Challenges Solved

- Handling extreme outliers in real estate pricing  
- Converting unstructured furnishing text into usable features  
- Managing inconsistent area measurements  
- Preventing data leakage in ML pipeline  

---

## 📈 Impact

- Improved prediction reliability on noisy housing data  
- Enabled intuitive property discovery through recommendations  
- Demonstrates ability to build **end-to-end production-ready ML systems**

---

## 🔮 Future Improvements

- Hyperparameter tuning  
- Hybrid recommendation system  
- Scalable deployment (Docker + cloud services)  
- User personalization  

---

## ⭐

If you found this project interesting, consider giving it a star!

# 🏡 Texas Real Estate Recommender System

**An intelligent real estate recommendation system leveraging machine learning to personalize property discovery in Houston, Texas.**

---

## 🖼️ Project Preview

![Project Preview](assets/demo.png)

*Example recommendation output / UI preview*

---

## 💡 Problem Statement

- Real estate search is overwhelming due to high listing volume and noisy data.
- Generic filters often fail to personalize options based on buyer intent.
- Buyers and agents need smarter recommendation workflows for faster decision-making.

---

## 🧠 Solution Overview

This project delivers a machine learning-powered recommendation engine tailored for Houston listings.  
It combines feature engineering, similarity scoring, and ranking logic to surface properties that best match user preferences (budget, location, amenities, and home characteristics).

- ML-based recommendation workflow for personalized listing discovery
- Feature-based similarity and ranking for relevant property matches
- User-centric output design focused on practical decision support

---

## ⚙️ Tech Stack

### Machine Learning
- Scikit-learn
- XGBoost
- Cosine similarity / feature-based ranking

### Backend / Deployment
- Python
- REST-style service integration (model-serving ready architecture)
- **AWS EC2 deployment for scalable cloud hosting**

### Data Processing
- Pandas
- NumPy

### Other Tools
- Jupyter Notebook
- Git / GitHub
- Matplotlib / Seaborn (EDA and model insights)

---

## 🧩 System Architecture

### High-Level Pipeline
1. Data ingestion from real estate sources
2. Data cleaning and preprocessing
3. Feature engineering and transformation
4. Model training and evaluation
5. Recommendation engine with similarity scoring
6. Cloud deployment on AWS EC2

*Optional architecture diagram placeholder:*  
`assets/architecture.png`

---

## 📊 Key Features

- Personalized property recommendations based on user preferences
- Houston-focused, location-aware filtering and matching
- Feature-based similarity scoring for relevant results
- Scalable cloud-ready serving strategy via AWS EC2
- Clean, modular, and structured end-to-end ML pipeline

---

## 📈 Model Highlights

- **Model type:** Supervised ML for structured tabular real estate data + recommendation similarity layer
- **Why this approach:** Tree-based and ensemble methods handle non-linear feature interactions effectively
- **Why it works well:** Strong feature engineering and ranking logic improve recommendation relevance

---

## 🌍 Real-World Impact

- **Real estate platforms:** Improve user engagement through personalized discovery
- **Buyers:** Reduce search friction and decision fatigue
- **Agents:** Prioritize high-fit listings for clients and accelerate close cycles

---

## 🔥 What Makes This Project Stand Out

- End-to-end ML workflow from raw data to deployable recommendations
- Real-world housing data and practical constraints, not a toy dataset
- Cloud deployment readiness with AWS EC2
- Business-relevant use case with clear product value
- Scalable architecture that can evolve into a production-grade platform

---

## 📌 Future Improvements

- Integrate user behavior signals (clicks, saves, inquiries) for adaptive recommendations
- Add map intelligence and richer geospatial proximity features
- Launch a full web application with real-time recommendation APIs
