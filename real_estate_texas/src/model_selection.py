"""
Broad model comparison (cross-validation + hold-out), similar in spirit to a dedicated model_selection notebook:
Linear / regularized linear / tree / boosting families, then optional fine-tuning via train.py.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("model_selection")


def _get_xgb():
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
        )
    except Exception:
        return GradientBoostingRegressor(random_state=42, n_estimators=200, max_depth=5)


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = X.select_dtypes(exclude=["number"]).columns.tolist()
    try:
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        oh = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", oh)]), cat),
        ]
    )


def load_feature_list(root: Path, candidates: list[str]) -> list[str]:
    p = root / "models/selected_features.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        sel = data.get("selected_features", [])
        use = [c for c in sel if c in candidates]
        if len(use) >= 4:
            logger.info("Using %s features from selected_features.json", len(use))
            return use
    return candidates


def compare_models(
    input_csv: str = "data/processed/texas_houston_features.csv",
    output_json: str = "models/model_selection_cv.json",
    cv_splits: int = 5,
    use_selected_features: bool = True,
) -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    path = root / input_csv
    df = pd.read_csv(path, low_memory=False)

    candidate_features = [
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
        "sqft_per_bedroom",
        "bath_per_bedroom",
        "dist_to_houston_mi",
        "dist_to_location_center_mi",
        "luxury_x_sqft",
        "is_luxury_segment",
        "property_type_rarity",
    ]
    for c in ["has_pool", "has_garage", "has_gym", "has_garden", "has_security", "has_fireplace", "has_office"]:
        if c in df.columns:
            candidate_features.append(c)

    candidates = [c for c in candidate_features if c in df.columns]
    if use_selected_features:
        feature_cols = load_feature_list(root, candidates)
    else:
        feature_cols = candidates

    X = df[feature_cols].copy()
    y = np.log1p(df["price"].astype(float))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    model_defs: list[tuple[str, object]] = [
        ("linear_regression", LinearRegression()),
        ("ridge", Ridge(alpha=2.0, random_state=42)),
        ("lasso", Lasso(alpha=0.01, random_state=42, max_iter=5000)),
        ("elastic_net", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000)),
        ("random_forest", RandomForestRegressor(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1)),
        ("hist_gradient_boosting", HistGradientBoostingRegressor(max_depth=8, learning_rate=0.06, random_state=42)),
        ("xgboost_or_gradient_boosting", _get_xgb()),
    ]

    rows = []
    fitted_models: dict[str, Pipeline] = {}
    for name, model in model_defs:
        pipe = Pipeline([("preprocessor", _preprocessor(X_train)), ("model", model)])
        try:
            pipe.fit(X_train, y_train)
            cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
            cv_nmse = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
            pred = pipe.predict(X_test)
            te_r2 = r2_score(np.expm1(y_test), np.expm1(pred))
            te_rmse = float(np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred))))
            te_mae = float(mean_absolute_error(np.expm1(y_test), np.expm1(pred)))
        except Exception as exc:
            logger.warning("Skip %s: %s", name, exc)
            continue
        rows.append(
            {
                "model": name,
                "cv_r2_mean": float(np.mean(cv_r2)),
                "cv_r2_std": float(np.std(cv_r2)),
                "cv_rmse_log_target": float(np.sqrt(-np.mean(cv_nmse))),
                "test_r2": te_r2,
                "test_rmse": te_rmse,
                "test_mae": te_mae,
            }
        )
        fitted_models[name] = pipe
        logger.info(
            "%s | CV R2: %.4f (+/- %.4f) | CV RMSE (log1p $): %.4f | test R2: %.4f | test RMSE ($): %.0f",
            name,
            np.mean(cv_r2),
            np.std(cv_r2),
            np.sqrt(-np.mean(cv_nmse)),
            te_r2,
            te_rmse,
        )

    result = pd.DataFrame(rows).sort_values("test_r2", ascending=False)

    # Segment-level diagnostics for the top model (recruiter-facing error analysis).
    diagnostics = {}
    if len(result) > 0:
        best = str(result.iloc[0]["model"])
        best_pipe = fitted_models.get(best)
        if best_pipe is not None:
            pred_dollars = np.expm1(best_pipe.predict(X_test))
            true_dollars = np.expm1(y_test.values)
            diag_df = X_test.copy()
            diag_df["_y_true"] = true_dollars
            diag_df["_y_pred"] = pred_dollars
            diag_df["_abs_err"] = np.abs(diag_df["_y_true"] - diag_df["_y_pred"])
            diag_df["_ape"] = diag_df["_abs_err"] / np.maximum(diag_df["_y_true"], 1.0)

            diagnostics["best_model"] = best
            diagnostics["overall"] = {
                "mae": float(diag_df["_abs_err"].mean()),
                "mape": float(diag_df["_ape"].mean()),
                "p90_abs_err": float(diag_df["_abs_err"].quantile(0.9)),
            }

            seg_cols = [c for c in ["property_type", "location_cluster"] if c in diag_df.columns]
            seg_reports = {}
            for c in seg_cols:
                g = (
                    diag_df.groupby(c)
                    .agg(
                        n=("_y_true", "size"),
                        mae=("_abs_err", "mean"),
                        mape=("_ape", "mean"),
                        median_price=("_y_true", "median"),
                    )
                    .sort_values("n", ascending=False)
                    .head(12)
                )
                seg_reports[c] = g.reset_index().to_dict(orient="records")
            diagnostics["segments"] = seg_reports

            if "bedrooms" in diag_df.columns:
                b = diag_df["bedrooms"].fillna(-1)
                b = np.where(b >= 5, "5+", b.astype(int).astype(str))
                gb = (
                    pd.DataFrame({"bhk": b, "abs_err": diag_df["_abs_err"], "ape": diag_df["_ape"]})
                    .groupby("bhk")
                    .agg(n=("abs_err", "size"), mae=("abs_err", "mean"), mape=("ape", "mean"))
                    .sort_values("bhk")
                )
                diagnostics["segments"]["bhk"] = gb.reset_index().to_dict(orient="records")
    out_path = root / output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.to_json(orient="records", indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    diag_path = root / "models/model_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    logger.info("Wrote %s", diag_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validate multiple regressors and save comparison JSON.")
    parser.add_argument("--input_csv", default="data/processed/texas_houston_features.csv")
    parser.add_argument("--output_json", default="models/model_selection_cv.json")
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--all_features", action="store_true", help="Ignore selected_features.json")
    args = parser.parse_args()
    df = compare_models(
        input_csv=args.input_csv,
        output_json=args.output_json,
        cv_splits=args.cv_splits,
        use_selected_features=not args.all_features,
    )
    print(df.to_string(index=False))
