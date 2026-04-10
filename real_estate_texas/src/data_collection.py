import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.property_type_utils import resolve_property_type_series
from src.raw_ingest import bootstrap_augment, dedupe_listings, ingest_raw_folder

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("data_collection")

DEFAULT_INPUT_CSV = "data/raw/usa_real_estate.csv"
DEFAULT_OUTPUT_CSV = "data/raw/texas_houston_raw.csv"
# Raw file row count before cleaning/outliers; set high enough that processed data stays 10k+.
TARGET_ROWS = 18_000


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "bed": "bedrooms",
        "beds": "bedrooms",
        "bath": "bathrooms",
        "baths": "bathrooms",
        "sqft_living": "sqft",
        "house_size": "sqft",
        "acre_lot": "lot_size",
        "lot": "lot_size",
        "type": "property_type",
        "city": "location",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    if "zip_code" in df.columns and "zipcode" not in df.columns:
        df = df.rename(columns={"zip_code": "zipcode"})
    return df


def _download_kaggle_dataset() -> Optional[pd.DataFrame]:
    try:
        import kagglehub
    except Exception:
        logger.warning("kagglehub is not installed; skipping Kaggle download.")
        return None
    try:
        logger.info("Downloading USA real estate dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("ahmedshahriarsakib/usa-real-estate-dataset")
        csv_files = list(Path(dataset_path).glob("*.csv"))
        if not csv_files:
            return None
        df = pd.read_csv(csv_files[0], low_memory=False)
        logger.info("Downloaded Kaggle dataset with %s rows.", len(df))
        return df
    except Exception as exc:
        logger.exception("Kaggle download failed: %s", exc)
        return None


def _prepare_texas_houston_subset(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(df).copy()
    if "state" not in df.columns:
        df["state"] = np.nan
    if "location" not in df.columns:
        df["location"] = np.nan
    state_series = df["state"].astype(str).str.upper()
    loc_series = df["location"].astype(str).str.upper()
    tx_mask = state_series.isin(["TX", "TEXAS"]) | loc_series.str.contains("TEXAS", na=False)
    houston_mask = loc_series.str.contains("HOUSTON", na=False)
    subset = df[tx_mask | houston_mask].copy()
    required_cols = [
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
    ]
    for col in required_cols:
        if col not in subset.columns:
            subset[col] = np.nan
    subset = subset[subset["price"].notna()].copy()
    subset = subset[(subset["price"] > 20_000) & (subset["price"] < 20_000_000)]
    subset["state"] = "TX"
    return subset[required_cols]


def _finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    extra = ["address", "listing_url", "source_file", "data_tier", "is_synthetic_augment"]
    for col in extra:
        if col not in df.columns:
            df[col] = np.nan if col != "is_synthetic_augment" else False
    df["is_synthetic_augment"] = df["is_synthetic_augment"].fillna(False).astype(bool)
    return df


def collect_data(input_csv: str = DEFAULT_INPUT_CSV, output_csv: str = DEFAULT_OUTPUT_CSV) -> Path:
    root = Path(__file__).resolve().parents[1]
    output_path = root / output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = root / "data/raw"

    merged = ingest_raw_folder(raw_dir)
    merged = dedupe_listings(merged)

    if merged.empty:
        logger.warning("No rows from data/raw CSVs; falling back to usa_real_estate.csv or Kaggle.")
        input_path = root / input_csv
        if input_path.exists():
            source_df = pd.read_csv(input_path, low_memory=False)
        else:
            source_df = _download_kaggle_dataset()
        if source_df is None or source_df.empty:
            raise RuntimeError("No data: add CSVs under data/raw/ or place usa_real_estate.csv / configure Kaggle.")
        merged = _prepare_texas_houston_subset(source_df)
        merged["source_file"] = input_path.name if input_path.exists() else "kaggle"
        merged["address"] = merged.get("location", "")
        merged["listing_url"] = np.nan
        merged["data_tier"] = "supplemental_kaggle"
        merged["is_synthetic_augment"] = False
    else:
        merged = _finalize_schema(merged)

    merged["property_type"] = resolve_property_type_series(merged).astype(str)

    merged = bootstrap_augment(merged, TARGET_ROWS)
    merged = _finalize_schema(merged)

    out_cols = [
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
    for c in out_cols:
        if c not in merged.columns:
            merged[c] = np.nan if c != "is_synthetic_augment" else False
    merged = merged[out_cols].drop_duplicates().reset_index(drop=True)
    merged.to_csv(output_path, index=False)
    real_n = (~merged["is_synthetic_augment"]).sum()
    logger.info(
        "Saved %s rows to %s (real/raw: %s, augmented: %s).",
        len(merged),
        output_path,
        real_n,
        len(merged) - real_n,
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Houston/Texas data from data/raw + optional augmentation.")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()
    collect_data(input_csv=args.input_csv, output_csv=args.output_csv)
