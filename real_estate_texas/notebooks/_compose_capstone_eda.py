"""One-off composer for capstone EDA / preprocessing notebooks. Run from repo root: python notebooks/_compose_capstone_eda.py"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(lines: str) -> dict:
    parts = [ln + "\n" for ln in lines.strip("\n").split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": parts}


def code(src: str) -> dict:
    parts = [ln + "\n" for ln in src.strip("\n").split("\n")]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": parts}


META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
}

EDA_CELLS = [
    md(
        r"""
# Capstone-level exploratory data analysis
## Texas / Houston residential listings (Zillow-style exports + augmentation)

**Purpose:** Demonstrate a recruiter-ready analytics workflow: data understanding → quality → univariate → bivariate → multivariate → outliers → geo → modeling implications.

**Artifacts (pipeline stages):**
| Stage | Path |
|-------|------|
| Raw merged | `data/raw/texas_houston_raw.csv` |
| Cleaned (outliers + rules) | `data/processed/texas_houston_clean.csv` |
| Engineered features | `data/processed/texas_houston_features.csv` |

**How to run:** Open this notebook from `notebooks/` (paths assume `..` = project root). Re-run the pipeline first if data changed: `python src/data_collection.py` → `preprocessing` → `feature_engineering`.
"""
    ),
    md(
        r"""
## 0. Data quality report artifacts (production-style profiling)

Before visual EDA, generate formal quality artifacts from `src/data_quality.py`:
- `reports/data_quality_summary.json`
- `reports/data_quality_columns.csv`
- `reports/data_quality_duplicate_samples.csv`
"""
    ),
    code(
        r"""
import sys
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_quality import build_quality_report

dq = build_quality_report(
    input_csv="data/processed/texas_houston_features.csv",
    output_dir="reports",
)
dq
"""
    ),
    code(
        r"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 180)
sns.set_theme(style="whitegrid", context="notebook")

PROJECT_ROOT = Path("..").resolve()
RAW_PATH = PROJECT_ROOT / "data/raw/texas_houston_raw.csv"
CLEAN_PATH = PROJECT_ROOT / "data/processed/texas_houston_clean.csv"
FEAT_PATH = PROJECT_ROOT / "data/processed/texas_houston_features.csv"

raw = pd.read_csv(RAW_PATH, low_memory=False)
clean = pd.read_csv(CLEAN_PATH, low_memory=False)
feat = pd.read_csv(FEAT_PATH, low_memory=False)

# Primary working copy for EDA: engineered features (richest schema)
df = feat.copy()
print("Shapes  raw:", raw.shape, "  clean:", clean.shape, "  features:", feat.shape)
"""
    ),
    md(
        r"""
## 1. Dataset overview & schema

High-level inventory: row counts across pipeline stages, dtypes, memory, and duplicate risk on business keys.
"""
    ),
    code(
        r"""
summary = pd.DataFrame(
    {
        "stage": ["raw", "clean", "features"],
        "rows": [len(raw), len(clean), len(feat)],
        "cols": [raw.shape[1], clean.shape[1], feat.shape[1]],
    }
)
display(summary)

print("--- dtypes (features) ---")
display(feat.dtypes.astype(str).to_frame("dtype"))

print("--- memory (MB) ---")
print(round(feat.memory_usage(deep=True).sum() / 1e6, 2))
"""
    ),
    code(
        r"""
# Duplicate checks (listing URL preferred; fallback address + price)
def dup_report(frame: pd.DataFrame, name: str):
    n = len(frame)
    if "listing_url" in frame.columns:
        u = frame["listing_url"].astype(str).str.strip()
        valid = u.notna() & (u != "") & (u.lower() != "nan")
        d = frame.loc[valid].duplicated(subset=["listing_url"]).sum()
        print(f"{name}: rows={n} | dup listing_url (non-empty) = {int(d)}")
    if "address" in frame.columns and "price" in frame.columns:
        d2 = frame.duplicated(subset=["address", "price"]).sum()
        print(f"{name}: dup address+price = {int(d2)}")

dup_report(raw, "raw")
dup_report(clean, "clean")
"""
    ),
    md(
        r"""
## 2. Missing values (MCAR / MAR awareness)

We quantify missingness **before** modeling, visualize patterns, and tie decisions to `src/preprocessing.py` (median imputation for numerics, empty string for text).
"""
    ),
    code(
        r"""
miss = df.isna().mean().sort_values(ascending=False)
miss = miss[miss > 0].to_frame("missing_rate")
display(miss.head(25))

fig = px.bar(
    miss.head(20).reset_index(),
    x="missing_rate",
    y="index",
    orientation="h",
    title="Top 20 columns by missing rate (features dataset)",
)
fig.update_layout(template="plotly_white", yaxis_title=None, height=520)
fig.show()
"""
    ),
    code(
        r"""
# Missingness heatmap on a sample of key columns (readability)
key_cols = [c for c in ["price", "sqft", "lot_size", "bedrooms", "bathrooms", "year_built", "latitude", "longitude", "property_type", "location", "listing_url", "description"] if c in df.columns]
sample = df[key_cols].sample(min(400, len(df)), random_state=42)
plt.figure(figsize=(12, 7))
sns.heatmap(sample.isna(), cbar=False, yticklabels=False)
plt.title("Missing pattern (sample of rows × key columns)")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
## 3. Univariate analysis — numeric

Distributions, skewness, kurtosis, and log-transform exploration for heavy-tailed `price`.
"""
    ),
    code(
        r"""
num_cols = [
    c
    for c in ["price", "sqft", "lot_size", "bedrooms", "bathrooms", "year_built", "price_per_sqft", "property_age", "luxury_score"]
    if c in df.columns
]
desc = df[num_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
desc["skew"] = df[num_cols].skew()
desc["kurtosis"] = df[num_cols].kurtosis()
display(desc.round(3))
"""
    ),
    code(
        r"""
fig = make_subplots(rows=2, cols=2, subplot_titles=("price", "log1p(price)", "sqft", "price_per_sqft"))
fig.add_trace(go.Histogram(x=df["price"], nbinsx=70, name="price"), row=1, col=1)
fig.add_trace(go.Histogram(x=np.log1p(df["price"]), nbinsx=70, name="log1p price"), row=1, col=2)
if "sqft" in df.columns:
    fig.add_trace(go.Histogram(x=df["sqft"], nbinsx=60, name="sqft"), row=2, col=1)
if "price_per_sqft" in df.columns:
    fig.add_trace(go.Histogram(x=df["price_per_sqft"], nbinsx=60, name="$/sqft"), row=2, col=2)
fig.update_layout(template="plotly_white", height=700, showlegend=False, title_text="Univariate — core numerics")
fig.show()
"""
    ),
    md(
        r"""
## 4. Univariate analysis — categorical

Property type mix, **BHK (bedrooms)** distribution, and concentration of listings by `location` / `zipcode`.
"""
    ),
    code(
        r"""
if "property_type" in df.columns:
    vc = df["property_type"].fillna("unknown").value_counts().head(15).reset_index()
    vc.columns = ["property_type", "count"]
    fig = px.bar(vc, x="property_type", y="count", title="Top property types")
    fig.update_layout(template="plotly_white")
    fig.show()

bhk = df["bedrooms"].fillna(-1).clip(0, 8).astype(int).astype(str) + " BR"
vc2 = bhk.value_counts().reset_index()
vc2.columns = ["bhk", "count"]
fig = px.pie(vc2, names="bhk", values="count", title="Bedroom (BHK) mix")
fig.update_layout(template="plotly_white")
fig.show()
"""
    ),
    code(
        r"""
top_n = 18
if "location" in df.columns:
    locs = df["location"].value_counts().head(top_n).reset_index()
    locs.columns = ["location", "count"]
    fig = px.bar(locs, x="count", y="location", orientation="h", title=f"Top {top_n} locations by listing count")
    fig.update_layout(template="plotly_white", height=520)
    fig.show()
"""
    ),
    md(
        r"""
## 5. Bivariate analysis

Relationships with **price**: built area, bathrooms, property type, and location (median / spread).
"""
    ),
    code(
        r"""
sub = df.sample(min(3500, len(df)), random_state=7)
fig = px.scatter(
    sub,
    x="sqft",
    y="price",
    color="bedrooms",
    opacity=0.45,
    trendline="ols",
    title="Price vs sqft (OLS trend; colored by bedrooms)",
)
fig.update_layout(template="plotly_white")
fig.show()
"""
    ),
    code(
        r"""
if "property_type" in df.columns:
    sub2 = df[df["property_type"].isin(df["property_type"].value_counts().head(8).index)]
    fig = px.violin(sub2, x="property_type", y="price", box=True, points=False, title="Price distribution by property type (top 8)")
    fig.update_layout(template="plotly_white", xaxis_tickangle=-30, height=520)
    fig.show()

med_bed = df.groupby(df["bedrooms"].fillna(0).clip(0, 8).astype(int))["price"].median().reset_index()
med_bed.columns = ["bedrooms", "median_price"]
fig = px.bar(med_bed, x="bedrooms", y="median_price", title="Median price by bedroom count")
fig.update_layout(template="plotly_white")
fig.show()
"""
    ),
    md(
        r"""
## 6. Multivariate analysis

Correlation structure among numerics (multicollinearity check for linear models). Pairwise relationships for a compact feature subset.
"""
    ),
    code(
        r"""
corr_cols = [c for c in num_cols if c in df.columns]
C = df[corr_cols].corr()
fig = go.Figure(data=go.Heatmap(z=C.values, x=list(C.columns), y=list(C.index), colorscale="RdBu", zmid=0))
fig.update_layout(title="Correlation heatmap (numeric features)", template="plotly_white", height=640)
fig.show()
"""
    ),
    code(
        r"""
pair_cols = [c for c in ["price", "sqft", "bedrooms", "bathrooms", "price_per_sqft", "luxury_score"] if c in df.columns]
extra = ["property_type"] if "property_type" in df.columns else []
base = df[pair_cols + extra].dropna()
base = base.sample(min(2500, len(base)), random_state=3)
fig = px.scatter_matrix(
    base,
    dimensions=pair_cols,
    color="property_type" if "property_type" in base.columns else None,
    opacity=0.35,
    title="Scatter matrix (sample; multivariate pairwise views)",
)
fig.update_layout(template="plotly_white", height=900)
fig.show()
"""
    ),
    md(
        r"""
## 7. Outlier treatment (linked to `src/preprocessing.py`)

We quantify how many rows **raw → clean** drops using IQR on `price`, `sqft`, `bedrooms`, `bathrooms` (same spirit as the production script). Extreme values are often data errors or ultra-luxury tails—document the tradeoff for interviews.
"""
    ),
    code(
        r"""
def iqr_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return pd.Series(True, index=s.index)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (s >= lo) & (s <= hi)


r = raw.copy()
for c in ["price", "sqft", "bedrooms", "bathrooms"]:
    if c in r.columns:
        r[c] = pd.to_numeric(r[c], errors="coerce")

r2 = r.dropna(subset=["price"])
r2 = r2[(r2["price"] > 30_000) & (r2["price"] < 20_000_000)]
mask = pd.Series(True, index=r2.index)
for c in ["price", "sqft", "bedrooms", "bathrooms"]:
    if c in r2.columns:
        mask &= iqr_mask(r2[c].fillna(r2[c].median()))
approx_kept = int(mask.sum())
print(f"Raw rows: {len(raw)} | After price filter + IQR (replicated logic): ~{approx_kept}")
print(f"Actual clean rows: {len(clean)}")
"""
    ),
    code(
        r"""
fig = go.Figure()
fig.add_trace(go.Box(y=raw["price"], name="raw price"))
fig.add_trace(go.Box(y=clean["price"], name="clean price"))
fig.update_layout(title="Price: before vs after cleaning", template="plotly_white", yaxis_title="USD")
fig.show()
"""
    ),
    md(
        r"""
## 8. Geospatial EDA

Listings mapped by `latitude` / `longitude` (required for Houston-area spatial patterns). Color encodes price; size can reflect sqft.
"""
    ),
    code(
        r"""
geo = df.dropna(subset=["latitude", "longitude"]).sample(min(3000, len(df)), random_state=11)
fig = px.scatter_mapbox(
    geo,
    lat="latitude",
    lon="longitude",
    color="price",
    size="sqft" if "sqft" in geo.columns else None,
    hover_name="location" if "location" in geo.columns else None,
    zoom=8,
    mapbox_style="carto-positron",
    title="Spatial distribution of listings (sample)",
)
fig.update_layout(height=650)
fig.show()
"""
    ),
    md(
        r"""
## 9. Data lineage: scraped / exported vs augmented rows

`data_tier` and `is_synthetic_augment` explain train-set size. Interview talking point: **bootstrap augmentation** expands N while preserving marginal distributions; test leakage controls belong in CV design for target encoding.
"""
    ),
    code(
        r"""
if "data_tier" in df.columns:
    display(df.groupby("data_tier", dropna=False).agg(rows=("price", "size"), median_price=("price", "median"), median_sqft=("sqft", "median")))
    fig = px.box(df, x="data_tier", y="price", title="Price by data tier")
    fig.update_layout(template="plotly_white")
    fig.show()
else:
    print("No data_tier column — re-run preprocessing with latest schema.")
"""
    ),
    md(
        r"""
## 9b. Distributional diagnostics & correlation with target

**Normality:** `price` is rarely normal; `log1p(price)` is closer to symmetric. **Spearman** complements Pearson for monotonic relationships under skew.
"""
    ),
    code(
        r"""
from scipy.stats import probplot

sample_log = np.log1p(df["price"].dropna().sample(min(5000, len(df)), random_state=0))
fig_p = plt.figure(figsize=(6, 5))
probplot(sample_log, dist="norm", plot=plt)
plt.title("QQ plot — log1p(price) vs normal (sample)")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
rows = []
for c in [x for x in ["sqft", "bedrooms", "bathrooms", "lot_size", "year_built", "price_per_sqft", "luxury_score", "property_age"] if x in df.columns]:
    sub = df[["price", c]].dropna()
    if len(sub) < 50:
        continue
    pear = sub["price"].corr(sub[c], method="pearson")
    spear = sub["price"].corr(sub[c], method="spearman")
    rows.append({"feature": c, "pearson": pear, "spearman": spear})
corr_tgt = pd.DataFrame(rows)
corr_tgt["_abs_s"] = corr_tgt["spearman"].abs()
corr_tgt = corr_tgt.sort_values("_abs_s", ascending=False).drop(columns=["_abs_s"])
display(corr_tgt.round(4))
"""
    ),
    md(
        r"""
## 9c. Zip code / neighborhood concentration (if available)

High cardinality `zipcode` is often **target-encoded** or **clustered** (see `location_cluster` in features). Here we inspect listing counts and median price by ZIP (top 25 by count).
"""
    ),
    code(
        r"""
if "zipcode" in df.columns:
    z = df.copy()
    z["zipcode"] = z["zipcode"].astype(str).str.replace(r"\.0$", "", regex=True)
    g = z.groupby("zipcode").agg(listings=("price", "size"), median_price=("price", "median")).reset_index()
    g = g.sort_values("listings", ascending=False).head(25)
    fig = px.bar(g, x="zipcode", y="listings", title="Top 25 ZIPs by listing count", text="median_price")
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
    fig.show()
else:
    print("No zipcode column in features file.")
"""
    ),
    md(
        r"""
## 10. Takeaways for modeling & deployment

**Summary (fill in after you run cells):**
- **Target:** heavy-tailed `price` → training uses `log1p(price)` in `src/train.py`.
- **Leakage note:** `location_target_enc` uses global median by location; for production-grade CV, use nested CV or smoothing.
- **Outliers:** IQR removes extreme numeric tails; revisit if luxury segment is analytically important.
- **Next steps:** drift monitoring on new scrapes, SHAP for explainability, separate holdout ZIPs or time-based split if `listing_date` becomes available.

This notebook is intentionally long to mirror a **capstone report** and separate notebooks for *preprocessing / missing / outliers* in the same repo.
"""
    ),
]


PRE_CELLS = [
    md(
        r"""
# Data quality, missing values & outlier treatment

Companion to **`EDA.ipynb`**. This notebook narrates **what** `src/preprocessing.py` implements and **why**, with visual before/after checks on `data/raw/texas_houston_raw.csv` → `data/processed/texas_houston_clean.csv`.

**Base-project parallel:** Gurgaon workflow split *missing_value_handling*, *outlier_handling*, *preprocessing* — here we document the same decisions for the Texas pipeline.
"""
    ),
    code(
        r"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

pd.set_option("display.max_columns", 60)
PROJECT_ROOT = Path("..").resolve()
raw = pd.read_csv(PROJECT_ROOT / "data/raw/texas_houston_raw.csv", low_memory=False)
clean = pd.read_csv(PROJECT_ROOT / "data/processed/texas_houston_clean.csv", low_memory=False)
print(raw.shape, "->", clean.shape)
"""
    ),
    md(
        r"""
## 1. Missing values — strategy

| Column group | Strategy in `preprocessing.py` |
|--------------|----------------------------------|
| Numeric (`bedrooms`, `bathrooms`, `sqft`, …) | `median` imputation after coercion |
| `property_type`, `location` | fill unknown / strip strings |
| `description` | empty string |
| Metadata (`address`, `listing_url`, …) | preserved when present |

Below: missing rates on **raw** input.
"""
    ),
    code(
        r"""
mr = raw.isna().mean().sort_values(ascending=False)
display(mr[mr > 0].head(25).to_frame("missing_rate"))
"""
    ),
    md(
        r"""
## 2. Outlier removal — IQR on core numerics

The cleaner applies IQR with factor 1.5 on `price`, `sqft`, `bedrooms`, `bathrooms` **after** basic price bounds. We visualize impact on price.
"""
    ),
    code(
        r"""
fig = px.histogram(raw, x="price", nbins=80, title="RAW: price histogram")
fig.update_layout(template="plotly_white")
fig.show()
fig2 = px.histogram(clean, x="price", nbins=80, title="CLEAN: price histogram")
fig2.update_layout(template="plotly_white")
fig2.show()
"""
    ),
    code(
        r"""
plt.figure(figsize=(10, 5))
sns.kdeplot(raw["price"].dropna(), label="raw", fill=True, alpha=0.35)
sns.kdeplot(clean["price"].dropna(), label="clean", fill=True, alpha=0.35)
plt.title("Price density: raw vs clean")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
## 3. Texas / Houston filter

Rows kept if `state` is TX (or similar) **or** `location` contains Houston (case-insensitive), matching `preprocessing.py` intent for regional focus.
"""
    ),
    code(
        r"""
# Quick sanity: state distribution in clean
if "state" in clean.columns:
    display(clean["state"].value_counts().head(10))
"""
    ),
    md(
        r"""
## 4. Missingness after cleaning

Core numerics should be filled post–median impute; sparse URL fields may remain null by design.
"""
    ),
    code(
        r"""
core = [c for c in ["price", "sqft", "bedrooms", "bathrooms", "year_built", "latitude", "longitude"] if c in clean.columns]
display(clean[core].isna().mean().to_frame("missing_rate_after_clean"))
"""
    ),
    md(
        r"""
## 5. Reproduce from CLI

`python src/preprocessing.py`

Tune IQR columns, multiplier, or price bounds in `src/preprocessing.py` if the business requires retaining luxury tails.
"""
    ),
]


FEATURE_CELLS = [
    md(
        r"""
# Feature engineering — capstone narrative

Mirrors **`src/feature_engineering.py`** with **rationale**, **validation plots**, and **interview notes**.

**Derived fields:** `price_per_sqft`, `property_age`, `bed_bath_ratio`, `location_target_enc`, `location_cluster` (KMeans on lat/lon), regex-based amenity flags, `amenity_count`, `luxury_score`.
"""
    ),
    code(
        r"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

clean = pd.read_csv(PROJECT_ROOT / "data/processed/texas_houston_clean.csv", low_memory=False)
print("Clean shape:", clean.shape)
clean.head(3)
"""
    ),
    md(
        r"""
## 1. Manual preview — price per sqft

Stabilizes comparability across different home sizes before tree/linear models.
"""
    ),
    code(
        r"""
tmp = clean.copy()
tmp["price_per_sqft_preview"] = tmp["price"] / tmp["sqft"].replace(0, np.nan)
fig = px.histogram(tmp.dropna(subset=["price_per_sqft_preview"]), x="price_per_sqft_preview", nbins=80, title="price / sqft (preview)")
fig.update_layout(template="plotly_white")
fig.show()
"""
    ),
    md(
        r"""
## 2. Run production pipeline

`engineer_features()` writes `data/processed/texas_houston_features.csv`.
"""
    ),
    code(
        r"""
from src.feature_engineering import engineer_features

out = engineer_features()
print("Wrote:", out)
feat = pd.read_csv(PROJECT_ROOT / "data/processed/texas_houston_features.csv")
new_cols = sorted(set(feat.columns) - set(clean.columns))
print("New columns:", new_cols)
"""
    ),
    md(
        r"""
## 3. Validate engineered distributions
"""
    ),
    code(
        r"""
for col in ["price_per_sqft", "property_age", "luxury_score", "location_cluster"]:
    if col in feat.columns:
        display(feat[col].describe(percentiles=[0.05, 0.5, 0.95]).to_frame(col))
"""
    ),
    code(
        r"""
sub = feat.sample(min(4000, len(feat)), random_state=2)
fig = px.scatter(sub, x="price_per_sqft", y="price", color="location_cluster", opacity=0.4, title="Price vs price_per_sqft (by geo cluster)")
fig.update_layout(template="plotly_white")
fig.show()

if "luxury_score" in feat.columns:
    fig2 = px.scatter(sub, x="luxury_score", y="price", trendline="ols", title="Price vs luxury_score")
    fig2.update_layout(template="plotly_white")
    fig2.show()
"""
    ),
    md(
        r"""
## 4. Multicollinearity spot-check

`sqft`, `price_per_sqft`, and `price` are mechanically related — tree ensembles handle this well; linear models may need regularization.
"""
    ),
    code(
        r"""
num = [c for c in ["price", "sqft", "price_per_sqft", "property_age", "bed_bath_ratio", "luxury_score", "amenity_count"] if c in feat.columns]
cm = feat[num].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Correlation: engineered + core numerics")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
## 5. Interview notes

- **Target encoding:** use fold-wise or smoothed encoding in production to avoid leakage.
- **Clusters:** interpret `location_cluster` as a non-linear location proxy.
- **Text:** regex amenities are a baseline; NLP embeddings are a natural upgrade.
"""
    ),
]


def dump(name: str, cells: list) -> None:
    path = Path(__file__).parent / name
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": META, "cells": cells}
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Wrote", path)


if __name__ == "__main__":
    dump("EDA.ipynb", EDA_CELLS)
    dump("preprocessing_missing_outliers.ipynb", PRE_CELLS)
    dump("feature_engineering.ipynb", FEATURE_CELLS)
