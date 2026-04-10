"""
Content-based recommender with multiple similarity views (like blending cosine_sim1/2/3 in the base project):
  - numeric lifestyle / size signals
  - categorical (location, property type)
  - geospatial (lat/lon)
Optional: TF-IDF on descriptions when non-empty.

Weights default (0.5, 0.8, 1.0) mirror the Gurgaon-style blend; combined matrix is row-normalized for stable scores.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("recommender")

DEFAULT_WEIGHTS = (0.5, 0.8, 1.0)  # numeric, categorical, geo (base-project style)
TEXT_WEIGHT = 0.35
ARTIFACT_PATH = "models/recommender_artifact.pkl"
# Full n×n cosine matrices explode in memory/disk; only pickle for modest n.
MAX_ROWS_FOR_ARTIFACT = 8000


def _numeric_block(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in ["bedrooms", "bathrooms", "sqft", "price_per_sqft", "luxury_score", "amenity_count"] if c in df.columns]
    if not cols:
        return np.zeros((len(df), 1)), []
    X = df[cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(numeric_only=True))
    mat = StandardScaler().fit_transform(X)
    return mat, cols


def _categorical_block(df: pd.DataFrame):
    cols = [c for c in ["location", "property_type"] if c in df.columns]
    if not cols:
        return None
    try:
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        oh = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", oh),
        ]
    )
    return pipe.fit_transform(df[cols].astype(str))


def _geo_block(df: pd.DataFrame) -> np.ndarray | None:
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return None
    g = df[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
    g = g.fillna(g.median())
    return StandardScaler().fit_transform(g)


def _text_block(df: pd.DataFrame):
    if "description" not in df.columns:
        return None
    texts = df["description"].fillna("").astype(str)
    if texts.str.strip().str.len().sum() < 10:
        return None
    v = TfidfVectorizer(max_features=600, min_df=2, stop_words="english")
    return v.fit_transform(texts)


def encode_similarity_blocks(df: pd.DataFrame) -> dict:
    """Fit transforms on the full frame once; used for row-vs-all similarity (avoids O(n²) RAM)."""
    Xn, num_cols = _numeric_block(df)
    Xc = _categorical_block(df)
    Xg = _geo_block(df)
    Xt = _text_block(df)
    parts_meta = [
        ("numeric", DEFAULT_WEIGHTS[0], Xn is not None),
        ("categorical", DEFAULT_WEIGHTS[1], Xc is not None),
        ("geo", DEFAULT_WEIGHTS[2], Xg is not None),
    ]
    if Xt is not None:
        parts_meta.append(("text", TEXT_WEIGHT, True))
    return {
        "Xn": Xn,
        "Xc": Xc,
        "Xg": Xg,
        "Xt": Xt,
        "numeric_cols": num_cols,
        "parts_meta": parts_meta,
    }


def similarity_scores_for_row(blocks: dict, row_index: int) -> np.ndarray:
    """Cosine similarity from one row to all rows per block; same weight blend as full matrix."""
    Xn, Xc, Xg, Xt = blocks["Xn"], blocks["Xc"], blocks["Xg"], blocks["Xt"]
    n = Xn.shape[0]
    parts: list[tuple[float, np.ndarray]] = []

    s_num = cosine_similarity(Xn[row_index : row_index + 1], Xn, dense_output=True).ravel()
    parts.append((DEFAULT_WEIGHTS[0], s_num))

    if Xc is not None:
        if issparse(Xc):
            s_cat = cosine_similarity(Xc.getrow(row_index), Xc, dense_output=True).ravel()
        else:
            s_cat = cosine_similarity(Xc[row_index : row_index + 1], Xc, dense_output=True).ravel()
        parts.append((DEFAULT_WEIGHTS[1], s_cat))
    else:
        parts.append((DEFAULT_WEIGHTS[1], np.ones(n)))

    if Xg is not None:
        s_geo = cosine_similarity(Xg[row_index : row_index + 1], Xg, dense_output=True).ravel()
        parts.append((DEFAULT_WEIGHTS[2], s_geo))
    else:
        parts.append((DEFAULT_WEIGHTS[2], np.ones(n)))

    if Xt is not None:
        if issparse(Xt):
            s_txt = cosine_similarity(Xt.getrow(row_index), Xt, dense_output=True).ravel()
        else:
            s_txt = cosine_similarity(Xt[row_index : row_index + 1], Xt, dense_output=True).ravel()
        parts.append((TEXT_WEIGHT, s_txt))

    w_sum = sum(w for w, _ in parts)
    out = sum(w * s for w, s in parts) / w_sum
    out[row_index] = -1.0
    return out


def build_similarity_matrices(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    blocks = encode_similarity_blocks(df)
    Xn, Xc, Xg, Xt = blocks["Xn"], blocks["Xc"], blocks["Xg"], blocks["Xt"]
    S_num = cosine_similarity(Xn)

    if Xc is not None:
        S_cat = cosine_similarity(Xc)
    else:
        S_cat = np.ones_like(S_num)

    S_geo = cosine_similarity(Xg) if Xg is not None else np.ones_like(S_num)

    parts = [(DEFAULT_WEIGHTS[0], S_num), (DEFAULT_WEIGHTS[1], S_cat), (DEFAULT_WEIGHTS[2], S_geo)]
    if Xt is not None:
        S_txt = cosine_similarity(Xt)
        parts.append((TEXT_WEIGHT, S_txt))

    w_sum = sum(w for w, _ in parts)
    S_combined = sum(w * S for w, S in parts) / w_sum
    np.fill_diagonal(S_combined, 1.0)
    meta = {"numeric_cols": blocks["numeric_cols"], "weights": [p[0] for p in parts], "has_text": Xt is not None}
    return S_combined, meta


def save_recommender_artifact(
    data_csv: str = "data/processed/texas_houston_features.csv",
    output_pkl: str = ARTIFACT_PATH,
    max_rows: int = MAX_ROWS_FOR_ARTIFACT,
) -> Path | None:
    root = Path(__file__).resolve().parents[1]
    path = root / data_csv
    df = pd.read_csv(path, low_memory=False)
    if len(df) > max_rows:
        logger.warning(
            "Skipping recommender pickle: %s rows > max_rows=%s (would be ~%.1f GB dense float32). Similarity is computed on the fly in the app.",
            len(df),
            max_rows,
            (len(df) ** 2 * 4) / 1e9,
        )
        return None
    S, meta = build_similarity_matrices(df)
    S = S.astype(np.float32)
    out = root / output_pkl
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"similarity": S, "meta": meta, "n_rows": len(df)}, out)
    logger.info("Saved recommender artifact: %s (shape %s, float32)", out, S.shape)
    return out


def load_similarity_matrix(root: Path) -> tuple[np.ndarray | None, int]:
    p = root / ARTIFACT_PATH
    if not p.exists():
        return None, 0
    blob = joblib.load(p)
    return blob["similarity"], int(blob.get("n_rows", 0))


def get_similar_properties(
    listing_index: int,
    data_csv: str = "data/processed/texas_houston_features.csv",
    top_k: int = 5,
    use_artifact: bool = True,
) -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / data_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Features dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    n_rows = len(df)
    if listing_index < 0 or listing_index >= n_rows:
        raise IndexError(f"listing_index out of range [0, {n_rows - 1}]")

    scores: np.ndarray | None = None
    if use_artifact:
        S, n_stored = load_similarity_matrix(root)
        if S is not None and n_stored == n_rows:
            scores = S[listing_index].copy()
            scores[listing_index] = -1.0

    if scores is None:
        blocks = encode_similarity_blocks(df)
        scores = similarity_scores_for_row(blocks, listing_index)
    similar_idx = scores.argsort()[::-1][:top_k]

    columns = [c for c in ["price", "bedrooms", "bathrooms", "sqft", "location", "property_type", "listing_url"] if c in df.columns]
    result = df.iloc[similar_idx][columns].copy()
    result["similarity"] = scores[similar_idx]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build recommender artifact or print recommendations.")
    parser.add_argument("--build", action="store_true", help="Precompute similarity matrix and save to models/")
    parser.add_argument("--listing_index", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--data_csv", type=str, default="data/processed/texas_houston_features.csv")
    args = parser.parse_args()
    if args.build:
        save_recommender_artifact(data_csv=args.data_csv)
    else:
        print(get_similar_properties(args.listing_index, data_csv=args.data_csv, top_k=args.top_k).to_string(index=False))
