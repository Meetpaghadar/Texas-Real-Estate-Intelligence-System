"""Generate feature_selection, model_selection, recommender_system notebooks. Run: python notebooks/_compose_ml_notebooks.py"""
from __future__ import annotations

import json
from pathlib import Path

META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
}


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in s.strip("\n").split("\n")]}


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [ln + "\n" for ln in s.strip("\n").split("\n")]}


def dump(name: str, cells: list) -> None:
    p = Path(__file__).parent / name
    p.write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": META, "cells": cells}, indent=1), encoding="utf-8")
    print("Wrote", p)


FS_CELLS = [
    md(
        r"""
# Feature selection (capstone track)

Mirrors a **dedicated feature_selection notebook** workflow (like the Gurgaon project): combine **filter** methods (correlation, redundancy) with **embedded** signals (Random Forest importances) and **mutual information** vs `log1p(price)`.

**Outputs (from `src/feature_selection.py`):**
- `models/selected_features.json` — ranked scores + final column list  
- `data/processed/texas_houston_features_selected.csv` — reduced training table  

Downstream: `train.py` auto-loads `selected_features.json` when present.
"""
    ),
    code(
        r"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection import (
    correlation_with_target,
    drop_high_correlation_pairs,
    mutual_information_scores,
    random_forest_importance,
    select_features,
    DEFAULT_CANDIDATES,
)

feat_path = PROJECT_ROOT / "data/processed/texas_houston_features.csv"
df = pd.read_csv(feat_path, low_memory=False)
y_log = pd.Series(np.log1p(df["price"].astype(float)), index=df.index)
candidates = [c for c in DEFAULT_CANDIDATES if c in df.columns]
print("Candidates:", len(candidates))
"""
    ),
    md("## 1. Correlation with target (log price)"),
    code(
        r"""
corr_tgt = correlation_with_target(df, y_log, candidates)
fig = px.bar(corr_tgt.reset_index(), x="index", y=0, title="|Correlation| with log1p(price) (numeric features)")
fig.update_layout(template="plotly_white", xaxis_title="feature", yaxis_title="correlation")
fig.show()
display(corr_tgt.to_frame("corr").round(4))
"""
    ),
    md("## 2. Mutual information (non-linear strength)"),
    code(
        r"""
mi = mutual_information_scores(df, y_log, candidates)
fig = px.bar(mi.reset_index(), x="index", y=0, title="Mutual information vs log1p(price)")
fig.update_layout(template="plotly_white", xaxis_title="feature", yaxis_title="MI")
fig.show()
"""
    ),
    md("## 3. Redundant pairs (multicollinearity) — drop weaker vs target"),
    code(
        r"""
kept, pairs = drop_high_correlation_pairs(df, candidates, corr_tgt, threshold=0.92)
print("Dropped from high pairwise corr:", len(pairs), "pairs")
pairs[:15]
"""
    ),
    md("## 4. Random Forest importances (mixed-type quick encoding)"),
    code(
        r"""
imp = random_forest_importance(df, y_log, candidates)
fig = px.bar(imp.reset_index(), x="index", y=0, title="RF feature importance")
fig.update_layout(template="plotly_white", xaxis_title="feature")
fig.show()
"""
    ),
    md("## 5. Run full selector + inspect JSON"),
    code(
        r"""
report = select_features(
    input_csv="data/processed/texas_houston_features.csv",
    output_json="models/selected_features.json",
    output_csv="data/processed/texas_houston_features_selected.csv",
)
print("Selected count:", len(report["selected_features"]))
print(report["selected_features"])
"""
    ),
]

MS_CELLS = [
    md(
        r"""
# Model selection (cross-validation + hold-out)

Analogous to a **model_selection** notebook: compare **linear, regularized linear, forest, histogram boosting, XGBoost/GBM** on the same preprocessing.

**Script:** `src/model_selection.py` → `models/model_selection_cv.json`  

Use CV **R²** and **RMSE on log target** for stability; **test RMSE in dollars** for business interpretation.
Also export **segment diagnostics** (`models/model_diagnostics.json`) to show where errors are high/low by property type, cluster, and BHK.
"""
    ),
    code(
        r"""
import sys
from pathlib import Path

import pandas as pd
import json

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_selection import compare_models

result = compare_models(use_selected_features=True)
display(result)

diag_path = PROJECT_ROOT / "models/model_diagnostics.json"
if diag_path.exists():
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    print("Best model:", diag.get("best_model"))
    print("Overall diagnostics:", diag.get("overall"))
    print("Available segment reports:", list(diag.get("segments", {}).keys()))
"""
    ),
    md("## Compare using all engineered features (ignore selection file)"),
    code(
        r"""
result_all = compare_models(use_selected_features=False)
display(result_all)
"""
    ),
    md("## Notes for interviews"),
    md(
        r"""
- **Why log target?** Stabilizes variance of home prices.  
- **Why both CV and hold-out?** CV estimates generalization; hold-out mimics deployment slice.  
- **Leakage:** `location_target_enc` is fitted on full data in `feature_engineering.py` — mention fold-wise encoding as an improvement.  
- **Next:** learning curves, `RandomizedSearchCV`, or early stopping for boosting.
"""
    ),
]

REC_CELLS = [
    md(
        r"""
# Recommender system — multi-view similarity

The Gurgaon base project blended **multiple cosine matrices** with weights `(0.5, 0.8, 1.0)`. Here we implement the same idea:

| View | Content |
|------|---------|
| Numeric | bedrooms, baths, sqft, $/sqft, luxury, amenities |
| Categorical | location, property_type (one-hot cosine) |
| Geo | latitude / longitude |
| Text (optional) | TF-IDF on `description` |

**CLI:** `python src/recommender.py --build` (only materializes a pickle if **n_rows ≤ 8000**; else similarity is computed on demand to avoid GB-sized matrices).
"""
    ),
    code(
        r"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender import build_similarity_matrices, get_similar_properties

df = pd.read_csv(PROJECT_ROOT / "data/processed/texas_houston_features.csv", low_memory=False)
S, meta = build_similarity_matrices(df)
print("Combined similarity shape:", S.shape)
print("Meta:", meta)
"""
    ),
    code(
        r"""
# Top-5 similar listings to row 0
print(get_similar_properties(0, top_k=5).to_string(index=False))
"""
    ),
    md("## Weight sensitivity (optional experiment)"),
    code(
        r"""
# Re-import with different weights requires editing DEFAULT_WEIGHTS in src/recommender.py
# or wrapping build_similarity_matrices — documented for capstone extension.
print("Default blend weights (numeric, cat, geo):", (0.5, 0.8, 1.0))
"""
    ),
]


if __name__ == "__main__":
    dump("feature_selection.ipynb", FS_CELLS)
    dump("model_selection.ipynb", MS_CELLS)
    dump("recommender_system.ipynb", REC_CELLS)
