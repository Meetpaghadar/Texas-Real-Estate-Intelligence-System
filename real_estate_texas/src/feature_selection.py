"""
Feature selection pipeline (filter + embedded methods), aligned with a classic DS workflow:
- Mutual information vs target
- Pearson correlation with target (numeric)
- Drop redundant features (high pairwise correlation)
- Random Forest feature importances (mixed-type quick encoding)

Writes models/selected_features.json and optional reduced CSV for training.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_regression

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("feature_selection")

DEFAULT_CANDIDATES = [
    "bedrooms",
    "bathrooms",
    "sqft",
    "lot_size",
    "year_built",
    "location",
    "property_type",
    "latitude",
    "longitude",
    "price_per_sqft",
    "property_age",
    "bed_bath_ratio",
    "location_target_enc",
    "location_cluster",
    "amenity_count",
    "luxury_score",
    "has_pool",
    "has_garage",
    "has_gym",
    "has_garden",
    "has_security",
    "has_fireplace",
    "has_office",
]


def _numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    num = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num].copy()
    return X, num


def correlation_with_target(df: pd.DataFrame, y: pd.Series, cols: list[str]) -> pd.Series:
    num, names = _numeric_frame(df, cols)
    if not names:
        return pd.Series(dtype=float)
    aligned = pd.concat([num, y], axis=1).dropna()
    if len(aligned) < 50:
        return pd.Series(dtype=float)
    yv = aligned.iloc[:, -1]
    Xn = aligned.iloc[:, :-1]
    return Xn.corrwith(yv).reindex(names).sort_values(key=abs, ascending=False)


def mutual_information_scores(df: pd.DataFrame, y: pd.Series, cols: list[str], random_state: int = 42) -> pd.Series:
    num, names = _numeric_frame(df, cols)
    if not names:
        return pd.Series(dtype=float)
    X = num.fillna(num.median())
    yv = y.values
    mask = ~np.isnan(yv)
    mi = mutual_info_regression(X.loc[mask], yv[mask], random_state=random_state)
    return pd.Series(mi, index=names).sort_values(ascending=False)


def drop_high_correlation_pairs(
    df: pd.DataFrame,
    cols: list[str],
    corr_with_y: pd.Series,
    threshold: float = 0.92,
) -> tuple[list[str], list[tuple[str, str, float, str]]]:
    """Among highly correlated pairs, drop the feature with weaker |corr(target)|."""
    num, names = _numeric_frame(df, cols)
    if len(names) < 2:
        return names, []
    c = num.corr().abs()
    names_list = list(c.columns)
    drop: set[str] = set()
    pairs: list[tuple[str, str, float, str]] = []
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            ri, rj = names_list[i], names_list[j]
            v = c.loc[ri, rj]
            if pd.notna(v) and v >= threshold:
                cr = corr_with_y.reindex([ri, rj]).abs()
                lose = cr.idxmin() if cr.notna().all() else rj
                pairs.append((ri, rj, float(v), lose))
                drop.add(lose)
    kept = [x for x in names if x not in drop]
    return kept, pairs


def random_forest_importance(df: pd.DataFrame, y: pd.Series, cols: list[str], random_state: int = 42) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(dtype=float)
    X = df[present].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in present if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols))
    if cat_cols:
        transformers.append(
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols)
        )
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    X_t = pre.fit_transform(X)
    names_out = num_cols + cat_cols
    rf = RandomForestRegressor(n_estimators=120, max_depth=16, random_state=random_state, n_jobs=-1)
    rf.fit(X_t, y)
    return pd.Series(rf.feature_importances_, index=names_out).sort_values(ascending=False)


def select_features(
    input_csv: str = "data/processed/texas_houston_features.csv",
    output_json: str = "models/selected_features.json",
    output_csv: str | None = "data/processed/texas_houston_features_selected.csv",
    mi_top_k: int = 12,
    min_importance_pct: float = 0.01,
) -> dict:
    root = Path(__file__).resolve().parents[1]
    path = root / input_csv
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)
    target_col = "price"
    if target_col not in df.columns:
        raise ValueError("price column required")

    candidates = [c for c in DEFAULT_CANDIDATES if c in df.columns]
    y_log = pd.Series(np.log1p(df[target_col].astype(float)), index=df.index)

    corr_tgt = correlation_with_target(df, pd.Series(y_log), candidates)
    mi = mutual_information_scores(df, pd.Series(y_log), candidates)
    kept_num, high_corr_pairs = drop_high_correlation_pairs(df, candidates, corr_tgt, threshold=0.92)
    imp = random_forest_importance(df, y_log, candidates)

    cat_keep = [c for c in ["location", "property_type"] if c in candidates]
    imp_floor = imp.max() * min_importance_pct if len(imp) else 0
    imp_keep = list(imp[imp >= imp_floor].index)

    mi_ranked = [c for c in mi.index if c in kept_num]
    mi_keep = mi_ranked[:mi_top_k]

    selected = set(cat_keep)
    selected.update(mi_keep)
    selected.update(imp_keep)
    selected.update(c for c in kept_num if c in selected)

    selected_ordered = [c for c in candidates if c in selected]
    if len(selected_ordered) < 4:
        selected_ordered = candidates

    report = {
        "selected_features": selected_ordered,
        "correlation_with_log_price": {k: float(v) for k, v in corr_tgt.dropna().round(4).items()},
        "mutual_information": {k: float(v) for k, v in mi.items()},
        "random_forest_importance": {k: float(v) for k, v in imp.items()},
        "high_correlation_pairs": [{"a": a, "b": b, "corr": r, "dropped": d} for a, b, r, d in high_corr_pairs[:40]],
        "notes": "location/property_type always included. Numerics: MI top-k + RF importance floor, after dropping redundant high-corr pairs.",
    }

    out_json = root / output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote selection report to %s (%s features)", out_json, len(selected_ordered))

    if output_csv:
        meta = [
            c
            for c in [
                target_col,
                "address",
                "listing_url",
                "description",
                "latitude",
                "longitude",
                "data_tier",
                "is_synthetic_augment",
            ]
            if c in df.columns
        ]
        out_cols = list(dict.fromkeys(meta + selected_ordered))
        csv_path = root / output_csv
        df[out_cols].to_csv(csv_path, index=False)
        logger.info("Wrote reduced feature CSV: %s", csv_path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature selection and write JSON + optional CSV.")
    parser.add_argument("--input_csv", default="data/processed/texas_houston_features.csv")
    parser.add_argument("--output_json", default="models/selected_features.json")
    parser.add_argument("--output_csv", default="data/processed/texas_houston_features_selected.csv")
    parser.add_argument("--mi_top_k", type=int, default=12)
    parser.add_argument("--no_csv", action="store_true")
    args = parser.parse_args()
    select_features(
        input_csv=args.input_csv,
        output_json=args.output_json,
        output_csv=None if args.no_csv else args.output_csv,
        mi_top_k=args.mi_top_k,
    )
