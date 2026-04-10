import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("feature_engineering")
HOUSTON_LAT, HOUSTON_LON = 29.7604, -95.3698


def _amenity_flags(text: str) -> dict[str, int]:
    text = (text or "").lower()
    patterns = {
        "has_pool": r"\bpool\b",
        "has_garage": r"\bgarage\b",
        "has_gym": r"\bgym\b",
        "has_garden": r"\bgarden\b",
        "has_security": r"\bsecurity|gated\b",
        "has_fireplace": r"\bfireplace\b",
        "has_office": r"\boffice\b",
    }
    return {key: int(bool(re.search(pattern, text))) for key, pattern in patterns.items()}


def _luxury_score(df: pd.DataFrame) -> pd.Series:
    amenity_cols = [
        "has_pool",
        "has_garage",
        "has_gym",
        "has_garden",
        "has_security",
        "has_fireplace",
        "has_office",
    ]
    amenity_sum = df[amenity_cols].sum(axis=1)
    score = (
        (df["sqft"] / df["sqft"].median())
        + (df["price_per_sqft"] / df["price_per_sqft"].median())
        + (amenity_sum / len(amenity_cols))
    ) / 3
    return np.round(score * 10, 2)


def _haversine_miles(lat: pd.Series, lon: pd.Series, lat0, lon0) -> np.ndarray:
    r = 3958.7613
    a1 = np.radians(lat.astype(float).values)
    b1 = np.radians(lon.astype(float).values)
    a2 = np.radians(np.asarray(lat0, dtype=float))
    b2 = np.radians(np.asarray(lon0, dtype=float))
    da = a2 - a1
    db = b2 - b1
    h = np.sin(da / 2.0) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(db / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def engineer_features(
    input_csv: str = "data/processed/texas_houston_clean.csv",
    output_csv: str = "data/processed/texas_houston_features.csv",
) -> Path:
    root = Path(__file__).resolve().parents[1]
    input_path = root / input_csv
    output_path = root / output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input cleaned dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    current_year = datetime.now().year

    df["price_per_sqft"] = df["price"] / df["sqft"].replace(0, np.nan)
    df["price_per_sqft"] = df["price_per_sqft"].fillna(df["price_per_sqft"].median())

    df["property_age"] = (current_year - df["year_built"]).clip(lower=0)
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"].replace(0, np.nan)
    df["bed_bath_ratio"] = df["bed_bath_ratio"].fillna(df["bed_bath_ratio"].median())
    df["sqft_per_bedroom"] = df["sqft"] / df["bedrooms"].replace(0, np.nan)
    df["sqft_per_bedroom"] = df["sqft_per_bedroom"].fillna(df["sqft_per_bedroom"].median())
    df["bath_per_bedroom"] = df["bathrooms"] / df["bedrooms"].replace(0, np.nan)
    df["bath_per_bedroom"] = df["bath_per_bedroom"].fillna(df["bath_per_bedroom"].median())

    # Location target encoding using neighborhood median price.
    location_price = df.groupby("location")["price"].median()
    df["location_target_enc"] = df["location"].map(location_price)
    df["location_target_enc"] = df["location_target_enc"].fillna(df["price"].median())

    if {"latitude", "longitude"}.issubset(df.columns):
        geo = df[["latitude", "longitude"]].fillna(df[["latitude", "longitude"]].median())
        n_clusters = 8 if len(df) >= 500 else 4
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["location_cluster"] = km.fit_predict(geo)
        df["dist_to_houston_mi"] = _haversine_miles(geo["latitude"], geo["longitude"], HOUSTON_LAT, HOUSTON_LON)
        # Local centrality signal by location: distance to each location centroid.
        cent = df.groupby("location")[["latitude", "longitude"]].median()
        c_lat = df["location"].map(cent["latitude"])
        c_lon = df["location"].map(cent["longitude"])
        df["dist_to_location_center_mi"] = _haversine_miles(geo["latitude"], geo["longitude"], c_lat, c_lon)
    else:
        df["location_cluster"] = 0
        df["dist_to_houston_mi"] = 0.0
        df["dist_to_location_center_mi"] = 0.0

    amenity_df = df["description"].fillna("").apply(_amenity_flags).apply(pd.Series)
    df = pd.concat([df, amenity_df], axis=1)
    df["amenity_count"] = amenity_df.sum(axis=1)
    df["luxury_score"] = _luxury_score(df)
    df["luxury_x_sqft"] = df["luxury_score"] * np.log1p(df["sqft"].clip(lower=0))
    df["is_luxury_segment"] = (df["luxury_score"] >= df["luxury_score"].quantile(0.8)).astype(int)

    # Rarity signal: uncommon property types can carry a premium in specific pockets.
    type_freq = df["property_type"].astype(str).value_counts(normalize=True)
    df["property_type_rarity"] = 1.0 - df["property_type"].astype(str).map(type_freq).fillna(0.0)

    df.to_csv(output_path, index=False)
    logger.info("Saved engineered features to %s with %s rows.", output_path, len(df))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature engineering for real-estate dataset.")
    parser.add_argument("--input_csv", type=str, default="data/processed/texas_houston_clean.csv")
    parser.add_argument("--output_csv", type=str, default="data/processed/texas_houston_features.csv")
    args = parser.parse_args()
    engineer_features(input_csv=args.input_csv, output_csv=args.output_csv)
