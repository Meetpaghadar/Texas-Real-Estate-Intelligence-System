"""Normalize and infer property_type so analytics are not dominated by 'unknown'."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_UNKNOWN_LIKE = frozenset(
    {"unknown", "nan", "none", "", "other", "n/a", "na", "—", "-"}
)


def _clean_token(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_property_type(value) -> str:
    """Map vendor strings to a small set of labels."""
    raw = _clean_token(value) if pd.notna(value) else ""
    if raw in _UNKNOWN_LIKE:
        return "unknown"
    # Zillow / common MLS
    aliases = {
        "single_family": "single_family",
        "singlefamily": "single_family",
        "single family": "single_family",
        "single_family_home": "single_family",
        "sfr": "single_family",
        "house": "single_family",
        "detached": "single_family",
        "condo": "condo",
        "condominium": "condo",
        "co-op": "condo",
        "coop": "condo",
        "townhouse": "townhouse",
        "townhome": "townhouse",
        "town home": "townhouse",
        "multi_family": "multi_family",
        "multifamily": "multi_family",
        "multi-family": "multi_family",
        "duplex": "multi_family",
        "triplex": "multi_family",
        "fourplex": "multi_family",
        "apartment": "apartment",
        "apartments": "apartment",
        "lot": "land",
        "land": "land",
        "mobile": "mobile",
        "manufactured": "mobile",
        "mfd/mobile home": "mobile",
        "modular": "mobile",
    }
    if raw in aliases:
        return aliases[raw]
    for key, lab in aliases.items():
        if key in raw:
            return lab
    return raw if raw else "unknown"


def infer_property_type_from_text(description: str) -> str | None:
    t = (description or "").lower()
    if not t.strip():
        return None
    rules = [
        (r"\bcondominium\b|\bcondo\b", "condo"),
        (r"\btown[- ]?house\b|\btownhome\b|\btown home\b", "townhouse"),
        (r"\bduplex\b|\btriplex\b|\bfourplex\b|\bmulti[- ]family\b", "multi_family"),
        (r"\bapartment\b|\bapt\.?\b", "apartment"),
        (r"\bmobile home\b|\bmanufactured\b|\brv park\b", "mobile"),
        (r"\bvacant land\b|\bbuildable lot\b|\bunimproved\b", "land"),
        (r"\bsingle[- ]family\b|\bdetached home\b|\bfamily home\b", "single_family"),
    ]
    for pat, lab in rules:
        if re.search(pat, t):
            return lab
    return None


def infer_property_type_heuristic(
    description: str,
    sqft: float | None,
    bedrooms: float | None,
) -> str:
    """When metadata is missing, use description then simple size/bed heuristics."""
    from_desc = infer_property_type_from_text(description)
    if from_desc:
        return from_desc
    sf = float(sqft) if sqft is not None and not pd.isna(sqft) else np.nan
    br = float(bedrooms) if bedrooms is not None and not pd.isna(bedrooms) else np.nan
    if not np.isnan(sf) and sf < 1100 and not np.isnan(br) and br <= 2:
        return "condo"
    if not np.isnan(sf) and sf >= 1800:
        return "single_family"
    if not np.isnan(br) and br >= 4:
        return "single_family"
    return "single_family"


def resolve_property_type_series(df: pd.DataFrame) -> pd.Series:
    """Return a Series aligned to df with normalized + inferred types."""
    if "property_type" in df.columns:
        base = df["property_type"].map(normalize_property_type)
    else:
        base = pd.Series("unknown", index=df.index, dtype=object)
    desc = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
    sq = df.get("sqft", pd.Series(np.nan, index=df.index))
    br = df.get("bedrooms", pd.Series(np.nan, index=df.index))
    out = base.astype(str).copy()
    mask = out.isin(["unknown", ""]) | out.isna()
    if mask.any():
        for idx in df.index[mask]:
            out.loc[idx] = infer_property_type_heuristic(
                str(desc.loc[idx]),
                float(sq.loc[idx]) if pd.notna(sq.loc[idx]) else None,
                float(br.loc[idx]) if pd.notna(br.loc[idx]) else None,
            )
    out = out.map(normalize_property_type)
    return out.replace({"unknown": "single_family"})
