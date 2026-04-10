# Texas/Houston Real Estate Intelligence Platform

Production-ready end-to-end Data Science + ML + Analytics project for Texas real estate, with a Houston-first focus.

## Highlights

- **Real raw ingestion:** Drop Zillow / scraper / Thunderbit CSVs into `data/raw/`. The loader understands:
  - Zillow “property blob” exports (`houston_zillow_raw.csv` style)
  - Wide Zillow scraper CSVs (`dataset_zillow-scraper_*.csv`)
  - Semicolon-separated Zillow dumps (`899b569d-*.csv`)
  - Thunderbit-style listing sheets
  - Skips Zillow **Metro** / **ZHVI** time-series files (not listing-level)
- **Bootstrap augmentation:** After deduplication, rows are resampled with small noise to reach a large training set (default **18,000** raw rows so cleaning still leaves **10k+**). Augmented rows are labeled `data_tier=synthetic_augmented` and `is_synthetic_augment=True`.
- Optional fallback: `usa_real_estate.csv` or Kaggle if `data/raw/` has no listing CSVs.
- Stronger feature engineering:
  - `price_per_sqft`, `property_age`, `bed_bath_ratio`
  - location target encoding
  - geo clustering
  - amenity extraction from descriptions
  - custom `luxury_score`
- Model comparison and selection:
  - Linear Regression
  - Random Forest + hyperparameter tuning
  - XGBoost (or GradientBoosting fallback)
  - 5-fold cross-validation
- Content-based recommender using cosine similarity.
- Streamlit dashboard with:
  - price prediction
  - market analytics
  - geospatial maps
  - recommendation explorer

## Project Structure

```text
real_estate_texas/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing_missing_outliers.ipynb
│   ├── feature_engineering.ipynb
│   ├── feature_selection.ipynb        # Filter + MI + RF importance (like base feature_selection.ipynb)
│   ├── model_selection.ipynb          # CV across model families (like base model_selection.ipynb)
│   ├── modeling.ipynb
│   ├── recommender_system.ipynb       # Multi-view cosine blend (like base recommender logic)
│   ├── _compose_capstone_eda.py
│   ├── _compose_ml_notebooks.py
│
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_selection.py           # Writes selected_features.json + optional reduced CSV
│   ├── model_selection.py             # Broad CV comparison → model_selection_cv.json
│   ├── train.py                       # GridSearch finalists; uses selected_features.json if present
│   ├── evaluate.py
│   ├── recommender.py                 # Weighted numeric + cat + geo (+ optional text)
│
├── app/
│   ├── streamlit_app.py
│
├── models/
│   ├── pipeline.pkl
│
├── requirements.txt
├── Procfile
├── README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run End-to-End Pipeline

From `real_estate_texas/`:

```bash
python src/data_collection.py
python src/preprocessing.py
python src/feature_engineering.py
python src/feature_selection.py          # optional: models/selected_features.json
python src/model_selection.py          # optional: compare families (CV)
python src/train.py                    # uses selection JSON if present; add --no_feature_selection to ignore
python src/evaluate.py
python src/recommender.py --build      # optional: only saves pickle if n_rows ≤ 8000 (else on-the-fly)
```

Artifacts created:

- Raw data: `data/raw/texas_houston_raw.csv`
- Cleaned data: `data/processed/texas_houston_clean.csv`
- Features data: `data/processed/texas_houston_features.csv`
- **Feature selection:** `models/selected_features.json`, `data/processed/texas_houston_features_selected.csv`
- **Model comparison:** `models/model_selection_cv.json`
- Best model pipeline: `models/pipeline.pkl`
- Metrics: `models/metrics.json`

## Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Dashboard sections:

- **Prediction**: estimate property prices from listing features.
- **Analysis**: distributions, box plots, property-type share, map visualizations.
- **Recommendations**: similar listings via **blended** numeric + categorical + geo cosine similarity (optional TF-IDF); large datasets compute similarity on demand (no multi-GB pickle).

## Data Sources

- **Primary:** Your files under `data/raw/` (scraped / exported listings).
- **Augmentation:** Statistical bootstrap from real rows (not independent random “fake Houston”); preserves correlations while expanding N.
- **Fallback:** Kaggle USA dataset if no raw CSVs are present.

### Notebooks (recruiter / capstone)

| Notebook | Contents |
|----------|----------|
| **`EDA.ipynb`** | Full pipeline: overview, duplicates, **missing values**, **univariate** (numeric + categorical), **bivariate**, **multivariate** (heatmap + scatter matrix), **outlier impact** (raw vs clean), **geospatial**, data lineage (real vs augmented), **QQ / correlation-with-target**, ZIP-level view, **modeling takeaways**. |
| **`preprocessing_missing_outliers.ipynb`** | Documents **missing-value strategy** and **IQR outlier treatment** aligned with `src/preprocessing.py` (like your base project’s separate missing / outlier notebooks). |
| **`feature_engineering.ipynb`** | Rationale + validation plots + multicollinearity check + interview notes. |
| **`feature_selection.ipynb`** | Correlation vs target, **mutual information**, redundant pairs, **RF importances**, run `select_features`. |
| **`model_selection.ipynb`** | **Ridge / Lasso / ElasticNet / RF / HGBT / XGBoost** CV + hold-out table (parallels base `model_selection.ipynb`). |
| **`modeling.ipynb`** | Final `train_models()` / artifact inspection. |
| **`recommender_system.ipynb`** | Multi-view similarity matrix + top-k recommendations. |

Regenerate notebooks: `python notebooks/_compose_capstone_eda.py` and `python notebooks/_compose_ml_notebooks.py`

## Deployment

### Option 1: Streamlit Cloud

1. Push this folder to GitHub.
2. In Streamlit Cloud, choose repo and set app path: `app/streamlit_app.py`.
3. Ensure `requirements.txt` is in root.

### Option 2: Heroku

1. Ensure Heroku CLI is installed.
2. Create app and deploy:

```bash
heroku create your-app-name
git push heroku main
```

`Procfile` is pre-configured.

### Option 3: Vercel

Use Vercel only if you split frontend/backend. For the current Streamlit app, Streamlit Cloud or Heroku is recommended.

## Engineering Notes

- Logging and exceptions are integrated across modules.
- Modular design separates ingestion, cleaning, features, training, evaluation, and serving.
- The model is stored as a packaged object with selected feature columns and pipeline.

## Next Improvements

- Add unit tests (`pytest`) and CI workflow.
- Integrate live geocoding and richer amenities extraction.
- Add drift monitoring and scheduled retraining job.
