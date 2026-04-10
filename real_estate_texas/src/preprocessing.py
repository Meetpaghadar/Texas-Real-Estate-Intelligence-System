import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.property_type_utils import resolve_property_type_series

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("preprocessing")


def remove_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    filtered = df.copy()
    for col in numeric_cols:
        q1 = filtered[col].quantile(0.25)
        q3 = filtered[col].quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = filtered[(filtered[col] >= lower) & (filtered[col] <= upper)]
    return filtered


def preprocess_data(
    input_csv: str = "data/raw/texas_houston_raw.csv",
    output_csv: str = "data/processed/texas_houston_clean.csv",
) -> Path:
    root = Path(__file__).resolve().parents[1]
    input_path = root / input_csv
    output_path = root / output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    logger.info("Reading raw data from %s", input_path)
    df = pd.read_csv(input_path)

    expected_cols = [
        "price",
        "bedrooms",
        "bathrooms",
        "sqft",
        "lot_size",
        "property_type",
        "year_built",
        "location",
        "latitude",
        "longitude",
        "description",
        "state",
        "zipcode",
        "address",
        "listing_url",
        "source_file",
        "data_tier",
        "is_synthetic_augment",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
    if "is_synthetic_augment" in df.columns:
        df["is_synthetic_augment"] = df["is_synthetic_augment"].fillna(False).astype(bool)

    numeric_cols = ["price", "bedrooms", "bathrooms", "sqft", "lot_size", "year_built", "latitude", "longitude"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["location"] = df["location"].fillna("unknown").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str)
    df["state"] = df["state"].fillna("TX").astype(str).str.upper()

    core_required = ["price", "sqft", "bedrooms"]
    before = len(df)
    df = df.dropna(subset=core_required).copy()
    logger.info("Dropped %s rows missing any of %s (kept %s).", before - len(df), core_required, len(df))

    df["property_type"] = resolve_property_type_series(df).astype(str)

    # Impute remaining numerics (rows without lat/lon or baths are still usable for modeling/maps).
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())
    df["latitude"] = df["latitude"].fillna(df["latitude"].median())
    df["longitude"] = df["longitude"].fillna(df["longitude"].median())
    df["lot_size"] = df["lot_size"].fillna(df["lot_size"].median())
    df["year_built"] = df["year_built"].fillna(df["year_built"].median())

    df = df[(df["price"] > 30_000) & (df["price"] < 20_000_000)].copy()
    df = remove_outliers_iqr(df, ["price", "sqft", "bedrooms", "bathrooms"])

    # Keep Houston-focused rows while preserving broader Texas entries.
    tx_mask = df["state"].isin(["TX", "TEXAS"])
    hou_mask = df["location"].str.contains("houston", case=False, na=False)
    df = df[tx_mask | hou_mask].copy()

    sig = df["listing_url"].astype(str).str.strip()
    sig = sig.mask(sig.eq("") | sig.eq("nan"), df["address"].astype(str) + "|" + df["price"].astype(str) + "|" + df["sqft"].astype(str))
    df = df.assign(_dedupe_sig=sig).drop_duplicates(subset=["_dedupe_sig"], keep="first").drop(columns=["_dedupe_sig"])
    df = df.reset_index(drop=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved cleaned data to %s with %s rows.", output_path, len(df))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw real-estate data.")
    parser.add_argument("--input_csv", type=str, default="data/raw/texas_houston_raw.csv")
    parser.add_argument("--output_csv", type=str, default="data/processed/texas_houston_clean.csv")
    args = parser.parse_args()
    preprocess_data(input_csv=args.input_csv, output_csv=args.output_csv)
