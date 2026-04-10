import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("train")


def _get_xgb_model():
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=350,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor

        logger.warning("xgboost unavailable, falling back to GradientBoostingRegressor.")
        return GradientBoostingRegressor(random_state=42)


def _build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def _load_selected_features(root: Path) -> Optional[list[str]]:
    p = root / "models/selected_features.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        sel = data.get("selected_features", [])
        return sel if isinstance(sel, list) and len(sel) >= 4 else None
    except Exception:
        return None


def train_models(
    input_csv: str = "data/processed/texas_houston_features.csv",
    model_output: str = "models/pipeline.pkl",
    metrics_output: str = "models/metrics.json",
    use_feature_selection: bool = True,
) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    input_path = root / input_csv
    model_path = root / model_output
    metrics_path = root / metrics_output
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Features dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    target_col = "price"
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
    ]
    for c in ["has_pool", "has_garage", "has_gym", "has_garden", "has_security", "has_fireplace", "has_office"]:
        if c in df.columns and c not in candidate_features:
            candidate_features.append(c)
    feature_cols = [c for c in candidate_features if c in df.columns]
    if use_feature_selection:
        picked = _load_selected_features(root)
        if picked:
            feature_cols = [c for c in picked if c in df.columns]
            logger.info("Training with feature-selection list (%s features).", len(feature_cols))
    X = df[feature_cols].copy()
    y = np.log1p(df[target_col].copy())

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "linear_regression": (LinearRegression(), {}),
        "random_forest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [12, 20, None],
                "model__min_samples_split": [2, 5],
            },
        ),
        "xgboost_or_gbm": (
            _get_xgb_model(),
            {
                "model__n_estimators": [200, 350],
            },
        ),
    }

    best_name = ""
    best_pipeline = None
    best_r2 = -np.inf
    all_metrics: dict[str, dict[str, float]] = {}

    for model_name, (model, grid_params) in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        search = GridSearchCV(
            pipe,
            grid_params if grid_params else {"model": [model]},
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        preds_log = search.predict(X_test)
        preds = np.expm1(preds_log)
        y_true = np.expm1(y_test)

        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2 = r2_score(y_true, preds)
        all_metrics[model_name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }
        logger.info("%s | R2: %.4f | RMSE: %.2f | MAE: %.2f", model_name, r2, rmse, mae)

        if r2 > best_r2:
            best_r2 = r2
            best_name = model_name
            best_pipeline = search.best_estimator_

    assert best_pipeline is not None
    package = {
        "model_name": best_name,
        "feature_columns": feature_cols,
        "pipeline": best_pipeline,
    }
    joblib.dump(package, model_path)
    all_metrics["best_model"] = {"name": best_name, "r2": float(best_r2)}
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    logger.info("Saved best model '%s' to %s", best_name, model_path)
    return model_path, metrics_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare real-estate price models.")
    parser.add_argument("--input_csv", type=str, default="data/processed/texas_houston_features.csv")
    parser.add_argument("--model_output", type=str, default="models/pipeline.pkl")
    parser.add_argument("--metrics_output", type=str, default="models/metrics.json")
    parser.add_argument("--no_feature_selection", action="store_true", help="Ignore models/selected_features.json")
    args = parser.parse_args()
    train_models(
        input_csv=args.input_csv,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
        use_feature_selection=not args.no_feature_selection,
    )
