"""
Load listing-level CSVs from data/raw/ (Zillow exports, scrapers, Thunderbit, etc.)
into a unified schema. Metro/ZHVI time-series files are skipped.
"""
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.property_type_utils import normalize_property_type

logger = logging.getLogger("raw_ingest")

OUTPUT_NAMES = frozenset({"texas_houston_raw.csv", "texas_houston_scraped.csv", "usa_real_estate.csv"})
HOUSTON_LAT, HOUSTON_LON = 29.7604, -95.3698


def _parse_money(x) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)
    s = re.sub(r"[^\d.]", "", str(x))
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_sqft_text(x) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).lower().replace(",", "")
    m = re.search(r"([\d.]+)\s*sq", s)
    if m:
        return float(m.group(1))
    m = re.search(r"([\d.]+)", s)
    return float(m.group(1)) if m else None


def _row_from_zillow_dict(d: dict) -> dict | None:
    if not isinstance(d, dict):
        return None
    price_obj = d.get("price") or {}
    val = price_obj.get("value") if isinstance(price_obj, dict) else price_obj
    price = _parse_money(val)
    if price is None or price < 1_000:
        return None
    addr = d.get("address") or {}
    loc = d.get("location") or {}
    lot = d.get("lotSizeWithUnit") or {}
    hdp = d.get("hdpView") or {}
    url = hdp.get("hdpUrl")
    if url and isinstance(url, str) and url.startswith("/"):
        url = "https://www.zillow.com" + url
    street = addr.get("streetAddress") or ""
    city = addr.get("city") or "Houston"
    zc = addr.get("zipcode")
    return {
        "price": price,
        "bedrooms": pd.to_numeric(d.get("bedrooms"), errors="coerce"),
        "bathrooms": pd.to_numeric(d.get("bathrooms"), errors="coerce"),
        "sqft": pd.to_numeric(d.get("livingArea"), errors="coerce"),
        "lot_size": pd.to_numeric(lot.get("lotSize"), errors="coerce"),
        "property_type": normalize_property_type(d.get("propertyType") or "unknown"),
        "year_built": pd.to_numeric(d.get("yearBuilt"), errors="coerce"),
        "location": str(city).strip(),
        "latitude": pd.to_numeric(loc.get("latitude"), errors="coerce"),
        "longitude": pd.to_numeric(loc.get("longitude"), errors="coerce"),
        "description": "",
        "state": str(addr.get("state") or "TX").upper(),
        "zipcode": zc,
        "address": f"{street}, {city}, TX {zc or ''}".strip(),
        "listing_url": url,
        "source_file": "zillow_property_blob",
    }


def load_zillow_property_blob_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    col = "property" if "property" in df.columns else None
    if col is None:
        return pd.DataFrame()
    rows = []
    for raw in df[col].dropna():
        s = str(raw).strip()
        if not s.startswith("{"):
            continue
        try:
            d = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            continue
        r = _row_from_zillow_dict(d)
        if r:
            rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty:
        logger.info("Zillow blob %s: %s rows", path.name, len(out))
    return out


def load_zillow_wide_scraper_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "detailUrl" not in df.columns and "url" not in df.columns:
        return pd.DataFrame()
    price_col = "price" if "price" in df.columns else None
    if price_col is None:
        return pd.DataFrame()
    lat_c = "latLong/latitude" if "latLong/latitude" in df.columns else "hdpData/homeInfo/latitude"
    lon_c = "latLong/longitude" if "latLong/longitude" in df.columns else "hdpData/homeInfo/longitude"
    bed_c = "hdpData/homeInfo/bedrooms" if "hdpData/homeInfo/bedrooms" in df.columns else "beds"
    bath_c = "hdpData/homeInfo/bathrooms" if "hdpData/homeInfo/bathrooms" in df.columns else "baths"
    sq_c = "hdpData/homeInfo/livingArea" if "hdpData/homeInfo/livingArea" in df.columns else "area"
    lot_c = "hdpData/homeInfo/lotAreaValue" if "hdpData/homeInfo/lotAreaValue" in df.columns else None
    url_c = "detailUrl" if "detailUrl" in df.columns else "url"
    city_c = "addressCity" if "addressCity" in df.columns else "city"
    zip_c = "addressZipcode" if "addressZipcode" in df.columns else "zipcode"
    type_c = "hdpData/homeInfo/homeType" if "hdpData/homeInfo/homeType" in df.columns else None
    out = pd.DataFrame(
        {
            "price": df[price_col].map(_parse_money),
            "bedrooms": pd.to_numeric(df[bed_c], errors="coerce") if bed_c in df.columns else np.nan,
            "bathrooms": pd.to_numeric(df[bath_c], errors="coerce") if bath_c in df.columns else np.nan,
            "sqft": pd.to_numeric(df[sq_c], errors="coerce") if sq_c in df.columns else np.nan,
            "lot_size": pd.to_numeric(df[lot_c], errors="coerce") if lot_c and lot_c in df.columns else np.nan,
            "property_type": (
                df[type_c].map(normalize_property_type)
                if type_c and type_c in df.columns
                else "unknown"
            ),
            "year_built": (
                pd.to_numeric(df["hdpData/homeInfo/yearBuilt"], errors="coerce")
                if "hdpData/homeInfo/yearBuilt" in df.columns
                else np.nan
            ),
            "location": df[city_c].astype(str) if city_c in df.columns else "Houston",
            "latitude": pd.to_numeric(df[lat_c], errors="coerce") if lat_c in df.columns else np.nan,
            "longitude": pd.to_numeric(df[lon_c], errors="coerce") if lon_c in df.columns else np.nan,
            "description": "",
            "state": df["addressState"].astype(str).str.upper() if "addressState" in df.columns else "TX",
            "zipcode": df[zip_c] if zip_c in df.columns else np.nan,
            "address": df["address"].astype(str) if "address" in df.columns else "",
            "listing_url": df[url_c].astype(str) if url_c in df.columns else "",
            "source_file": path.name,
        }
    )
    out = out[out["price"].notna() & (out["price"] >= 10_000)].copy()
    if not out.empty:
        logger.info("Zillow wide %s: %s rows", path.name, len(out))
    return out


def load_semicolon_zillow_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";", low_memory=False, on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(path, sep=";", low_memory=False, error_bad_lines=False, warn_bad_lines=False)
    if "price" not in df.columns or "city" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "price": df["price"].map(_parse_money),
            "bedrooms": pd.to_numeric(df.get("beds"), errors="coerce"),
            "bathrooms": pd.to_numeric(df.get("baths"), errors="coerce"),
            "sqft": pd.to_numeric(df.get("area"), errors="coerce"),
            "lot_size": np.nan,
            "property_type": (
                df["homeType"].map(normalize_property_type)
                if "homeType" in df.columns
                else pd.Series("unknown", index=df.index, dtype=object)
            ),
            "year_built": pd.to_numeric(df.get("yearBuilt"), errors="coerce"),
            "location": df["city"].astype(str),
            "latitude": pd.to_numeric(df.get("latitude"), errors="coerce"),
            "longitude": pd.to_numeric(df.get("longitude"), errors="coerce"),
            "description": df.get("description", "").astype(str).fillna(""),
            "state": df.get("state", "TX").astype(str).str.upper(),
            "zipcode": df.get("zipcode"),
            "address": df.get("fullAddress", df.get("street", "")).astype(str),
            "listing_url": df.get("url", "").astype(str),
            "source_file": path.name,
        }
    )
    out = out[out["price"].notna()].copy()
    if not out.empty:
        logger.info("Zillow semicolon %s: %s rows", path.name, len(out))
    return out


def load_thunderbit_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "Listing URL" not in df.columns and "listing url" not in [c.lower() for c in df.columns]:
        return pd.DataFrame()
    col_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    def pick(*names):
        for n in names:
            for k, v in col_map.items():
                if n in k:
                    return df[v]
        return None

    pricing = pick("pricing")
    if pricing is None:
        return pd.DataFrame()
    living = pick("living_area", "living area")
    beds = pick("number_of_bedrooms", "bedroom")
    baths = pick("number_of_bathrooms", "bathroom")
    city = pick("city")
    state = pick("state")
    zc = pick("zipcode", "zip")
    street = pick("street_address", "street")
    yb = pick("built_year", "built")
    desc = pick("description")
    url = pick("listing_url", "listing")
    home_type = pick("property_type", "home_type", "hometype", "type")

    lat = pd.Series(np.nan, index=df.index)
    lon = pd.Series(np.nan, index=df.index)
    sq = living.map(_parse_sqft_text) if living is not None else np.nan
    out = pd.DataFrame(
        {
            "price": pricing.map(_parse_money),
            "bedrooms": pd.to_numeric(beds, errors="coerce") if beds is not None else np.nan,
            "bathrooms": pd.to_numeric(baths, errors="coerce") if baths is not None else np.nan,
            "sqft": pd.to_numeric(sq, errors="coerce"),
            "lot_size": np.nan,
            "property_type": (
                home_type.map(normalize_property_type)
                if home_type is not None
                else pd.Series("unknown", index=df.index, dtype=object)
            ),
            "year_built": pd.to_numeric(yb, errors="coerce") if yb is not None else np.nan,
            "location": city.astype(str) if city is not None else "Houston",
            "latitude": lat,
            "longitude": lon,
            "description": desc.astype(str).fillna("") if desc is not None else "",
            "state": state.astype(str).str.upper() if state is not None else "TX",
            "zipcode": zc if zc is not None else np.nan,
            "address": street.astype(str) if street is not None else "",
            "listing_url": url.astype(str) if url is not None else "",
            "source_file": path.name,
        }
    )
    out = out[out["price"].notna() & (out["price"] >= 10_000)].copy()
    mask_tx = out["state"].str.contains("TX", na=False) | out["location"].str.contains("Houston", case=False, na=False)
    out = out[mask_tx].copy()
    miss_geo = out["latitude"].isna() | out["longitude"].isna()
    if miss_geo.any():
        rng = np.random.default_rng(42)
        out.loc[miss_geo, "latitude"] = HOUSTON_LAT + rng.normal(0, 0.08, size=miss_geo.sum())
        out.loc[miss_geo, "longitude"] = HOUSTON_LON + rng.normal(0, 0.1, size=miss_geo.sum())
    if not out.empty:
        logger.info("Thunderbit %s: %s rows", path.name, len(out))
    return out


def _should_skip_file(name: str) -> bool:
    ln = name.lower()
    if name in OUTPUT_NAMES:
        return True
    if ln.startswith("metro_") or "zhvi" in ln or "metro_zhvi" in ln:
        return True
    # Windows "file (1).csv" duplicate copies — same scrape ingested multiple times
    stem = Path(name).stem
    if re.search(r" \(\d+\)$", stem):
        return True
    return False


def _first_line(path: Path) -> str:
    with path.open(encoding="utf-8", errors="ignore") as fh:
        return fh.readline()


def ingest_raw_folder(raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return pd.DataFrame()
    chunks: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("*.csv")):
        if _should_skip_file(path.name):
            continue
        try:
            peek = pd.read_csv(path, nrows=2, low_memory=False)
            cols = set(peek.columns.astype(str))
            line1 = _first_line(path)
            semicolon_rich = line1.count(";") > line1.count(",") and line1.count(";") > 8

            if "property" in cols and len(cols) <= 6:
                part = load_zillow_property_blob_csv(path)
            elif semicolon_rich:
                part = load_semicolon_zillow_csv(path)
                if part.empty:
                    part = load_zillow_wide_scraper_csv(path)
            elif "detailUrl" in cols or any(c.startswith("hdpData/") for c in cols):
                part = load_zillow_wide_scraper_csv(path)
            elif any("listing" in c.lower() and "url" in c.lower() for c in cols):
                part = load_thunderbit_csv(path)
            else:
                part = load_zillow_wide_scraper_csv(path)
                if part.empty:
                    part = load_zillow_property_blob_csv(path)
                if part.empty:
                    part = load_semicolon_zillow_csv(path)
                if part.empty:
                    part = load_thunderbit_csv(path)
            if not part.empty:
                chunks.append(part)
        except Exception as exc:
            logger.warning("Skip %s: %s", path.name, exc)
    if not chunks:
        return pd.DataFrame()
    merged = pd.concat(chunks, ignore_index=True)
    merged["data_tier"] = "real_raw"
    return merged


def dedupe_listings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    key = df["listing_url"].astype(str).fillna("")
    key = key.mask(key == "", df["address"].astype(str) + "|" + df["price"].astype(str))
    df["_dedupe_key"] = key
    df = df.drop_duplicates(subset=["_dedupe_key"], keep="first").drop(columns=["_dedupe_key"])
    return df.reset_index(drop=True)


def bootstrap_augment(
    df: pd.DataFrame,
    target_rows: int,
    seed: int = 42,
    price_jitter: float = 0.04,
    sqft_jitter: float = 0.05,
    geo_jitter: float = 0.0004,
) -> pd.DataFrame:
    """Resample real rows with small noise to reach target_rows (simulated variety, same market structure)."""
    n = len(df)
    if n == 0:
        raise ValueError("Cannot augment empty dataframe")
    if n >= target_rows:
        return df.assign(is_synthetic_augment=False)
    need = target_rows - n
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=need)
    synth = df.iloc[idx].copy().reset_index(drop=True)
    p_noise = rng.normal(1.0, price_jitter, size=need)
    synth["price"] = (synth["price"].values * p_noise).clip(25_000, 18_000_000)
    synth["sqft"] = (synth["sqft"].fillna(df["sqft"].median()).values * rng.normal(1.0, sqft_jitter, size=need)).clip(
        400, 12000
    )
    synth["lot_size"] = (
        synth["lot_size"].fillna(df["lot_size"].median()).values * rng.normal(1.0, sqft_jitter, size=need)
    ).clip(500, 50000)
    synth["year_built"] = (
        synth["year_built"].fillna(df["year_built"].median()).values + rng.integers(-3, 4, size=need)
    ).clip(1920, 2026)
    synth["latitude"] = synth["latitude"].values + rng.normal(0, geo_jitter, size=need)
    synth["longitude"] = synth["longitude"].values + rng.normal(0, geo_jitter, size=need)
    synth["bathrooms"] = np.round(
        (synth["bathrooms"].fillna(df["bathrooms"].median()).values + rng.normal(0, 0.15, size=need)).clip(0.5, 8),
        1,
    )
    synth["listing_url"] = np.nan
    synth["address"] = synth["address"].astype(str) + " [aug]"
    synth["data_tier"] = "synthetic_augmented"
    synth["is_synthetic_augment"] = True
    real_part = df.assign(is_synthetic_augment=False)
    out = pd.concat([real_part, synth], ignore_index=True)
    logger.info("Augmented %s real rows with %s synthetic-like rows -> %s total", n, need, len(out))
    return out
