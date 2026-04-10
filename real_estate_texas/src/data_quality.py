"""
Data quality profiler for recruiter-grade EDA/cleaning narrative.

Outputs:
- reports/data_quality_summary.json
- reports/data_quality_columns.csv
- reports/data_quality_duplicate_samples.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        miss = float(s.isna().mean())
        uniq = int(s.nunique(dropna=True))
        uniq_rate = float(uniq / max(n, 1))
        row = {
            "column": c,
            "dtype": dtype,
            "missing_rate": miss,
            "n_unique": uniq,
            "unique_rate": uniq_rate,
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            s2 = pd.to_numeric(s, errors="coerce")
            row.update(
                {
                    "p01": float(s2.quantile(0.01)) if s2.notna().any() else np.nan,
                    "p50": float(s2.quantile(0.5)) if s2.notna().any() else np.nan,
                    "p99": float(s2.quantile(0.99)) if s2.notna().any() else np.nan,
                    "skew": float(s2.skew()) if s2.notna().sum() > 2 else np.nan,
                }
            )
        elif pd.api.types.is_bool_dtype(s):
            sb = s.astype(float)
            row.update(
                {
                    "p01": float(sb.quantile(0.01)) if sb.notna().any() else np.nan,
                    "p50": float(sb.quantile(0.5)) if sb.notna().any() else np.nan,
                    "p99": float(sb.quantile(0.99)) if sb.notna().any() else np.nan,
                    "skew": float(sb.skew()) if sb.notna().sum() > 2 else np.nan,
                }
            )
        else:
            top = s.astype(str).value_counts(dropna=False).head(3)
            row["top_values"] = "; ".join([f"{k}:{v}" for k, v in top.items()])
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["missing_rate", "unique_rate"], ascending=[False, False])
    return out


def _duplicate_report(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dup_meta = {}
    if "listing_url" in df.columns:
        u = df["listing_url"].astype(str).str.strip()
        valid = ~u.isin(["", "nan", "None"])
        d_url = df.loc[valid].duplicated(subset=["listing_url"], keep=False)
        dup_meta["duplicate_listing_url_rows"] = int(d_url.sum())
    if {"address", "price"}.issubset(df.columns):
        d_addr = df.duplicated(subset=["address", "price"], keep=False)
        dup_meta["duplicate_address_price_rows"] = int(d_addr.sum())
    sig_cols = [c for c in ["address", "price", "sqft", "bedrooms", "bathrooms", "zipcode"] if c in df.columns]
    if sig_cols:
        d_sig = df.duplicated(subset=sig_cols, keep=False)
        dup_meta["duplicate_signature_rows"] = int(d_sig.sum())
        sample = df.loc[d_sig, sig_cols].head(200).copy()
    else:
        sample = pd.DataFrame()
    return sample, dup_meta


def build_quality_report(
    input_csv: str = "data/processed/texas_houston_features.csv",
    output_dir: str = "reports",
) -> dict:
    root = Path(__file__).resolve().parents[1]
    in_path = root / input_csv
    out_dir = root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, low_memory=False)
    col_prof = _profile_columns(df)
    dup_sample, dup_meta = _duplicate_report(df)

    obj = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_columns_over_20pct": col_prof.loc[col_prof["missing_rate"] > 0.2, "column"].tolist(),
        "high_cardinality_columns_over_80pct_unique": col_prof.loc[col_prof["unique_rate"] > 0.8, "column"].tolist(),
        "duplicates": dup_meta,
    }

    (out_dir / "data_quality_summary.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
    col_prof.to_csv(out_dir / "data_quality_columns.csv", index=False)
    dup_sample.to_csv(out_dir / "data_quality_duplicate_samples.csv", index=False)
    return obj


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate data quality profiling artifacts.")
    ap.add_argument("--input_csv", default="data/processed/texas_houston_features.csv")
    ap.add_argument("--output_dir", default="reports")
    args = ap.parse_args()
    rep = build_quality_report(input_csv=args.input_csv, output_dir=args.output_dir)
    print(json.dumps(rep, indent=2))
