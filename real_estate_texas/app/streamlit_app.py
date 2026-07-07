from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommender import encode_similarity_blocks, similarity_scores_for_row

_R_MI = 3958.7613
HOUSTON_LAT, HOUSTON_LON = 29.7604, -95.3698

st.set_page_config(page_title="Texas Real Estate Intelligence", layout="wide")
st.title("Texas & Houston Real Estate Intelligence Dashboard")

DATA_PATH = ROOT / "data/processed/texas_houston_features.csv"
MODEL_PATH = ROOT / "models/pipeline.pkl"


def _haversine_miles(lat: pd.Series, lon: pd.Series, lat0: float, lon0: float) -> np.ndarray:
    lat1 = np.radians(lat.astype(float).values)
    lon1 = np.radians(lon.astype(float).values)
    lat2 = np.radians(lat0)
    lon2 = np.radians(lon0)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * _R_MI * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _default_for_column(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return 0
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return float(s.median())
    mode = s.mode(dropna=True)
    return mode.iloc[0] if len(mode) else ""


def build_prediction_row(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    bedrooms: float,
    bathrooms: float,
    sqft: float,
    lot_size: float,
    year_built: int,
    location: str,
    property_type: str,
    latitude: float,
    longitude: float,
    price_per_sqft: float,
    luxury_score: float,
    amenity_count: int,
) -> pd.DataFrame:
    """Align inference row with training schema (all columns the fitted pipeline expects)."""
    loc_mask = df["location"] == location
    row: dict = {
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "sqft": float(sqft),
        "lot_size": float(lot_size),
        "year_built": float(year_built),
        "location": str(location),
        "property_type": str(property_type),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "price_per_sqft": float(price_per_sqft),
        "property_age": float(2026 - year_built),
        "bed_bath_ratio": float(bedrooms / max(bathrooms, 1)),
        "sqft_per_bedroom": float(sqft / max(bedrooms, 1)),
        "bath_per_bedroom": float(bathrooms / max(bedrooms, 1)),
        "location_target_enc": float(df.loc[loc_mask, "price"].median() if loc_mask.any() else df["price"].median()),
        "amenity_count": int(amenity_count),
        "luxury_score": float(luxury_score),
        "luxury_x_sqft": float(luxury_score * np.log1p(max(sqft, 0))),
        "is_luxury_segment": int(luxury_score >= df["luxury_score"].quantile(0.8)),
        "dist_to_houston_mi": float(_haversine_miles(pd.Series([latitude]), pd.Series([longitude]), HOUSTON_LAT, HOUSTON_LON)[0]),
    }
    cent_lat = float(df.loc[loc_mask, "latitude"].median()) if loc_mask.any() else float(latitude)
    cent_lon = float(df.loc[loc_mask, "longitude"].median()) if loc_mask.any() else float(longitude)
    row["dist_to_location_center_mi"] = float(
        _haversine_miles(pd.Series([latitude]), pd.Series([longitude]), cent_lat, cent_lon)[0]
    )
    type_freq = df["property_type"].astype(str).value_counts(normalize=True)
    row["property_type_rarity"] = float(1.0 - type_freq.get(str(property_type), 0.0))

    for flag in ["has_pool", "has_garage", "has_gym", "has_garden", "has_security", "has_fireplace", "has_office"]:
        row[flag] = 0

    if "location_cluster" in feature_columns and "location_cluster" in df.columns and loc_mask.any():
        row["location_cluster"] = int(df.loc[loc_mask, "location_cluster"].mode().iloc[0])

    out = pd.DataFrame([row])
    for col in feature_columns:
        if col not in out.columns:
            out[col] = _default_for_column(df, col)
    return out[feature_columns].copy()


def _coerce_prediction_dtypes(pred_df: pd.DataFrame, ref_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Match training dtypes so SimpleImputer/encoders do not fail at transform time."""
    out = pred_df.copy()
    for col in feature_columns:
        if col not in ref_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(ref_df[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        else:
            out[col] = out[col].astype(str)
    return out[feature_columns]


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("Feature dataset not found. Run the training pipeline first.")
        st.stop()
    return pd.read_csv(DATA_PATH, low_memory=False).reset_index(drop=True)


@st.cache_data
def load_enriched_data() -> pd.DataFrame:
    """Load once and add UI helper columns (avoids recomputing on every rerun)."""
    out = load_data().copy()
    if "address" in out.columns:
        out["street"] = out["address"].fillna("").astype(str).str.split(",").str[0].str.strip()
        out.loc[out["street"].eq(""), "street"] = "(no address)"
    else:
        out["street"] = "(no address)"
    out["bhk_label"] = out["bedrooms"].apply(
        lambda x: "5+ BHK" if pd.notna(x) and float(x) >= 5 else (f"{int(x)} BHK" if pd.notna(x) else "Unknown")
    )
    return out


@st.cache_resource
def load_model():
    import joblib

    if not MODEL_PATH.exists():
        st.error("Model artifact not found. Run training first.")
        st.stop()
    pkg = joblib.load(MODEL_PATH)
    trained_ver = pkg.get("sklearn_version")
    if trained_ver and trained_ver != sklearn.__version__:
        st.warning(
            f"Model was trained with scikit-learn **{trained_ver}** but this app runs **{sklearn.__version__}**. "
            f"Pin `scikit-learn=={trained_ver}` in requirements.txt and use Python 3.10."
        )
    return pkg


@st.cache_data(show_spinner="Preparing recommender encodings (one-time per data refresh)…")
def _recommender_blocks_cached(data_mtime: float, n_rows: int) -> dict:
    del n_rows  # bust cache when row count changes
    return encode_similarity_blocks(load_data())


with st.spinner("Loading dataset and model…"):
    df = load_enriched_data()
    package = load_model()
pipeline = package["pipeline"]
feature_columns = list(package["feature_columns"])

tabs = st.tabs(["Prediction", "Analysis", "Recommendations"])

with tabs[0]:
    st.subheader("Price Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        bedrooms = st.slider("Bedrooms", 1, 8, 3)
        bathrooms = st.slider("Bathrooms", 1.0, 8.0, 2.0, 0.5)
        sqft = st.slider("Built Area (sqft)", 500, 8000, 2000, 50)
        lot_size = st.slider("Lot Size", 600, 15000, 3200, 100)
    with col2:
        year_built = st.slider("Year Built", 1940, 2026, 2005, 1)
        location = st.selectbox("Location", sorted(df["location"].dropna().unique().tolist()))
        property_type = st.selectbox("Property Type", sorted(df["property_type"].dropna().unique().tolist()))
        latitude = st.number_input("Latitude", value=float(df["latitude"].median()))
    with col3:
        longitude = st.number_input("Longitude", value=float(df["longitude"].median()))
        price_per_sqft = st.number_input("Price per sqft", value=float(df["price_per_sqft"].median()))
        luxury_score = st.number_input("Luxury score", value=float(df["luxury_score"].median()))
        amenity_count = st.slider("Amenity count", 0, 10, int(df["amenity_count"].median()))

    if st.button("Predict Price", type="primary"):
        pred_df = None
        try:
            pred_df = _coerce_prediction_dtypes(
                build_prediction_row(
                    df,
                    feature_columns,
                    bedrooms=bedrooms,
                    bathrooms=bathrooms,
                    sqft=sqft,
                    lot_size=lot_size,
                    year_built=year_built,
                    location=location,
                    property_type=property_type,
                    latitude=latitude,
                    longitude=longitude,
                    price_per_sqft=price_per_sqft,
                    luxury_score=luxury_score,
                    amenity_count=amenity_count,
                ),
                df,
                feature_columns,
            )
            pred_log = pipeline.predict(pred_df)[0]
            prediction = float(np.expm1(pred_log))
            st.success(f"Estimated property price: ${prediction:,.0f}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            with st.expander("Debug (features sent to model)"):
                st.write("Expected columns:", feature_columns)
                if pred_df is not None:
                    st.dataframe(pred_df)

with tabs[1]:
    st.subheader("Market Analytics")
    st.caption("Use the filters below, then adjust each chart’s dropdown to change how it is grouped or colored.")

    street_counts = df["street"].value_counts()
    top_streets = ["All"] + street_counts.head(80).index.tolist()
    loc_options = ["All"] + sorted(df["location"].dropna().unique().tolist())

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        filt_location = st.selectbox("Location", loc_options, key="flt_loc")
    with f2:
        filt_street = st.selectbox("Street", top_streets, key="flt_street")
    with f3:
        filt_bhk = st.selectbox("BHK (bedrooms)", ["All", "1", "2", "3", "4", "5+"], key="flt_bhk")
    with f4:
        pt_vals = sorted(df["property_type"].dropna().unique().tolist())
        filt_pt = st.selectbox("Property type", ["All"] + pt_vals, key="flt_pt")

    pmin, pmax = int(df["price"].min()), int(df["price"].max())
    fp1, fp2 = st.columns(2)
    with fp1:
        min_price = st.slider("Min price", pmin, pmax, pmin, key="flt_pmin")
    with fp2:
        max_price = st.slider("Max price", pmin, pmax, pmax, key="flt_pmax")

    d = df.copy()
    if filt_location != "All":
        d = d[d["location"] == filt_location]
    if filt_street != "All":
        d = d[d["street"] == filt_street]
    if filt_bhk != "All":
        if filt_bhk == "5+":
            d = d[d["bedrooms"] >= 5]
        else:
            d = d[d["bedrooms"] == int(filt_bhk)]
    if filt_pt != "All":
        d = d[d["property_type"] == filt_pt]
    d = d[(d["price"] >= min_price) & (d["price"] <= max_price)]

    st.caption(f"Rows after filters: **{len(d):,}**")

    r1, r2 = st.columns(2)
    with r1:
        box_group = st.selectbox(
            "Price box plot — group by",
            ["location", "property_type", "bedrooms", "street"],
            key="chart_box_g",
        )
        plot_box = d.copy()
        if box_group == "location":
            vc = plot_box["location"].value_counts().head(18).index
            plot_box = plot_box[plot_box["location"].isin(vc)]
            xcol = "location"
        elif box_group == "property_type":
            xcol = "property_type"
        elif box_group == "bedrooms":
            plot_box["bedrooms"] = plot_box["bedrooms"].apply(
                lambda x: f"{int(x)} BHK" if pd.notna(x) else "Unknown"
            )
            xcol = "bedrooms"
        else:
            svc = plot_box["street"].value_counts().head(15).index
            plot_box = plot_box[plot_box["street"].isin(svc)]
            xcol = "street"
        if len(plot_box) > 0:
            st.plotly_chart(
                px.box(plot_box, x=xcol, y="price", title=f"Price by {box_group.replace('_', ' ')}"),
                use_container_width=True,
            )
        else:
            st.info("No rows for this chart with current filters.")

    with r2:
        hist_color = st.selectbox(
            "Price histogram — color by",
            ["(none)", "property_type", "bedrooms", "location"],
            key="chart_hist_c",
        )
        hc = None if hist_color == "(none)" else hist_color
        plot_h = d[d["location"].isin(d["location"].value_counts().head(8).index)] if hc == "location" else d
        if len(plot_h) > 0:
            fig_h = px.histogram(plot_h, x="price", nbins=45, color=hc, title="Price distribution") if hc else px.histogram(plot_h, x="price", nbins=45, title="Price distribution")
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("No rows for histogram.")

    r3, r4 = st.columns(2)
    with r3:
        map_color = st.selectbox("Map — point color", ["price", "property_type", "bedrooms", "location"], key="chart_map_c")
        map_df = d.dropna(subset=["latitude", "longitude"])
        if len(map_df) > 0:
            sample = map_df.sample(min(1500, len(map_df)))
            if map_color == "bedrooms":
                sample = sample.copy()
                sample["bedrooms"] = sample["bedrooms"].astype(str)
            mfig = px.scatter_mapbox(
                sample,
                lat="latitude",
                lon="longitude",
                color=map_color if map_color != "price" else "price",
                size="sqft",
                hover_name="location",
                hover_data=["street", "price", "bedrooms"] if "street" in sample.columns else ["price", "bedrooms"],
                zoom=8,
                mapbox_style="carto-positron",
                title="Listings map",
            )
            st.plotly_chart(mfig, use_container_width=True)
        else:
            st.info("No geo data for map.")

    with r4:
        share_mode = st.selectbox("Share chart", ["Property type", "BHK (bedrooms)"], key="chart_share")
        if len(d) > 0:
            if share_mode == "Property type":
                tc = d["property_type"].value_counts().reset_index()
                tc.columns = ["property_type", "count"]
                st.plotly_chart(px.pie(tc, names="property_type", values="count", title="Property type share"), use_container_width=True)
            else:
                bc = d["bhk_label"].value_counts().reset_index()
                bc.columns = ["bhk", "count"]
                st.plotly_chart(px.bar(bc, x="bhk", y="count", title="BHK distribution", text_auto=True), use_container_width=True)
        else:
            st.info("No rows for share chart.")

    if st.checkbox("Show description word cloud", value=False, key="wc_show"):
        wc_text = " ".join(d["description"].fillna("").astype(str).tolist()) if "description" in d.columns else ""
        if wc_text.strip():
            try:
                from wordcloud import WordCloud
            except ImportError:
                st.error("Word cloud requires `wordcloud` in requirements.txt. Reboot the app after deploy.")
            else:
                with st.spinner("Building word cloud…"):
                    wc = WordCloud(width=900, height=320, background_color="white").generate(wc_text)
                st.image(wc.to_array(), caption="Description keywords (filtered slice)")
        else:
            st.info("No description text for the current filters.")

with tabs[2]:
    st.subheader("Recommendations near a location")
    st.caption("Pick a city, drill into street or zipcode, set radius, then get similar nearby listings.")

    loc_list = sorted(df["location"].dropna().unique().tolist())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        reco_location = st.selectbox("City / Location", loc_list, key="reco_loc")
    with c2:
        micro_mode = st.selectbox("Drill-down by", ["Street", "Zipcode"], key="reco_micro_mode")
    with c3:
        radius_label = st.selectbox("Nearby radius", ["1 mile", "3 miles", "5 miles", "10 miles"], key="reco_rad")
    with c4:
        top_k = st.slider("How many recommendations", 3, 20, 8, key="reco_k")

    miles = float(radius_label.split()[0])
    sub_loc = df[df["location"] == reco_location]
    if len(sub_loc) == 0:
        st.error("No rows for that location.")
    else:
        micro_col = "street" if micro_mode == "Street" else "zipcode"
        micro_vals = sub_loc[micro_col].fillna("(unknown)").astype(str).value_counts().head(100).index.tolist()
        if not micro_vals:
            st.error(f"No {micro_mode.lower()} values found inside {reco_location}.")
        else:
            reco_micro = st.selectbox(f"{micro_mode} in {reco_location}", micro_vals, key="reco_micro_value")
            sub_micro = sub_loc[sub_loc[micro_col].fillna("(unknown)").astype(str) == reco_micro]
            lat0 = float(sub_micro["latitude"].median())
            lon0 = float(sub_micro["longitude"].median())
            dist_all = _haversine_miles(df["latitude"], df["longitude"], lat0, lon0)
            same_loc = (df["location"] == reco_location).values
            same_micro = (df[micro_col].fillna("(unknown)").astype(str) == reco_micro).values
            near_mask = (dist_all <= miles) & same_loc & same_micro
            if near_mask.sum() < 2:
                near_mask = (dist_all <= miles) & same_loc
                st.warning("Few listings in this micro-area — expanded to same city within radius.")
            if near_mask.sum() < 2:
                near_mask = dist_all <= miles
                st.warning("Expanded to any listing within radius.")
            near_pos = np.where(near_mask)[0]
            if len(near_pos) == 0:
                st.error("No listings within that radius — try a larger radius or another location.")
            else:
                st.metric("Center (median lat/lon)", f"{lat0:.4f}, {lon0:.4f}")
                st.metric("Selected drill-down", f"{micro_mode}: {reco_micro}")
                st.metric("Listings within radius", int(len(near_pos)))

                cand_lat = df["latitude"].values[near_pos]
                cand_lon = df["longitude"].values[near_pos]
                d_cent = _haversine_miles(pd.Series(cand_lat), pd.Series(cand_lon), lat0, lon0)
                anchor_pos = int(near_pos[np.argmin(d_cent)])

                preview_cols = [c for c in ["price", "bedrooms", "bathrooms", "sqft", "location", "street", "property_type", "listing_url"] if c in df.columns]
                with st.expander("Anchor listing (nearest to area center within radius)", expanded=False):
                    st.dataframe(df.iloc[[anchor_pos]][preview_cols], use_container_width=True)

                others = np.array([i for i in near_pos if i != anchor_pos], dtype=int)

                if st.button("Get recommendations", type="primary", key="reco_run"):
                    if len(others) == 0:
                        st.warning("Not enough listings in radius to compare.")
                    else:
                        with st.spinner("Scoring similarity within radius…"):
                            blocks = _recommender_blocks_cached(
                                DATA_PATH.stat().st_mtime,
                                len(df),
                            )
                            scores = similarity_scores_for_row(blocks, anchor_pos)
                            order_local = np.argsort(-scores[others])
                            picked_pos = others[order_local[: min(top_k, len(others))]]
                            recs = df.iloc[picked_pos][preview_cols].copy()
                            recs["similarity"] = scores[picked_pos]
                            recs["distance_mi"] = dist_all[picked_pos]
                            st.session_state["reco_near_table"] = recs

                if "reco_near_table" in st.session_state:
                    st.subheader("Recommended nearby listings")
                    st.dataframe(st.session_state["reco_near_table"], use_container_width=True)

                map_d = df.iloc[near_pos].dropna(subset=["latitude", "longitude"])
                if len(map_d) > 0:
                    samp = map_d.sample(min(400, len(map_d)))
                    st.plotly_chart(
                        px.scatter_mapbox(
                            samp,
                            lat="latitude",
                            lon="longitude",
                            color="price",
                            hover_name="location",
                            zoom=10,
                            mapbox_style="carto-positron",
                            title=f"Near {reco_location} / {reco_micro} (~{miles:g} mi)",
                        ),
                        use_container_width=True,
                    )
