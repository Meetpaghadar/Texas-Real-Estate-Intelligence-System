from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommender import encode_similarity_blocks, similarity_scores_for_row

# Earth radius in miles (mean)
_R_MI = 3958.7613


def _haversine_miles(lat: pd.Series, lon: pd.Series, lat0: float, lon0: float) -> np.ndarray:
    """Great-circle distance from each row to (lat0, lon0)."""
    lat1 = np.radians(lat.astype(float).values)
    lon1 = np.radians(lon.astype(float).values)
    lat2 = np.radians(lat0)
    lon2 = np.radians(lon0)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * _R_MI * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


st.set_page_config(page_title="Texas Real Estate Intelligence", layout="wide")
st.title("Texas & Houston Real Estate Intelligence Dashboard")

DATA_PATH = ROOT / "data/processed/texas_houston_features.csv"
MODEL_PATH = ROOT / "models/pipeline.pkl"


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("Feature dataset not found. Run the training pipeline first.")
        st.stop()
    return pd.read_csv(DATA_PATH, low_memory=False).reset_index(drop=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model artifact not found. Run training first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Preparing recommender encodings (one-time per data refresh)…")
def _recommender_blocks_cached(path_str: str, mtime: float) -> dict:
    df_enc = pd.read_csv(Path(path_str), low_memory=False).reset_index(drop=True)
    return encode_similarity_blocks(df_enc)


df = load_data()
package = load_model()
pipeline = package["pipeline"]
feature_columns = package["feature_columns"]

# Derived columns for Analysis
if "address" in df.columns:
    df["street"] = df["address"].fillna("").astype(str).str.split(",").str[0].str.strip()
    df.loc[df["street"].eq(""), "street"] = "(no address)"
else:
    df["street"] = "(no address)"

df["bhk_label"] = df["bedrooms"].apply(
    lambda x: "5+ BHK" if pd.notna(x) and float(x) >= 5 else (f"{int(x)} BHK" if pd.notna(x) else "Unknown")
)

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

    input_row = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft": sqft,
        "lot_size": lot_size,
        "year_built": year_built,
        "location": location,
        "property_type": property_type,
        "latitude": latitude,
        "longitude": longitude,
        "price_per_sqft": price_per_sqft,
        "property_age": 2026 - year_built,
        "bed_bath_ratio": bedrooms / max(bathrooms, 1),
        "location_target_enc": float(df[df["location"] == location]["price"].median()),
        "location_cluster": int(df[df["location"] == location]["location_cluster"].mode().iloc[0]),
        "amenity_count": amenity_count,
        "luxury_score": luxury_score,
    }
    pred_df = pd.DataFrame([input_row])[feature_columns]

    if st.button("Predict Price", type="primary"):
        pred_log = pipeline.predict(pred_df)[0]
        prediction = float(np.expm1(pred_log))
        st.success(f"Estimated property price: ${prediction:,.0f}")

with tabs[1]:
    st.subheader("Market Analytics")
    st.caption("Use the filters below, then adjust each chart’s dropdown to change how it is grouped or colored.")

    street_counts = df["street"].value_counts()
    top_streets = ["All"] + street_counts.head(80).index.tolist()
    loc_options = ["All"] + sorted(df["location"].dropna().unique().tolist())
    cluster_options = ["All"] + [str(int(c)) for c in sorted(df["location_cluster"].dropna().unique())]

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        filt_location = st.selectbox("Location", loc_options, key="flt_loc")
    with f2:
        filt_sector = st.selectbox("Sector (area cluster)", cluster_options, key="flt_sec")
    with f3:
        filt_street = st.selectbox("Street", top_streets, key="flt_street")
    with f4:
        filt_bhk = st.selectbox("BHK (bedrooms)", ["All", "1", "2", "3", "4", "5+"], key="flt_bhk")
    with f5:
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
    if filt_sector != "All":
        d = d[d["location_cluster"] == int(filt_sector)]
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
            ["location", "location_cluster", "property_type", "bedrooms", "street"],
            key="chart_box_g",
            help="Street is capped to top frequent streets in the filtered slice for readability.",
        )
        plot_box = d.copy()
        if box_group == "location":
            vc = plot_box["location"].value_counts().head(18).index
            plot_box = plot_box[plot_box["location"].isin(vc)]
            xcol = "location"
        elif box_group == "location_cluster":
            plot_box["location_cluster"] = plot_box["location_cluster"].astype(str)
            xcol = "location_cluster"
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
            ["(none)", "property_type", "location_cluster", "bedrooms", "location"],
            key="chart_hist_c",
        )
        hc = None if hist_color == "(none)" else hist_color
        if hc == "location":
            vc = d["location"].value_counts().head(8).index
            plot_h = d[d["location"].isin(vc)]
        else:
            plot_h = d
        if len(plot_h) > 0:
            if hc:
                fig_h = px.histogram(
                    plot_h,
                    x="price",
                    nbins=45,
                    color=hc,
                    title="Price distribution",
                )
            else:
                fig_h = px.histogram(plot_h, x="price", nbins=45, title="Price distribution")
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("No rows for histogram.")

    r3, r4 = st.columns(2)
    with r3:
        map_color = st.selectbox(
            "Map — point color",
            ["price", "property_type", "bedrooms", "location_cluster", "location"],
            key="chart_map_c",
        )
        map_df = d.dropna(subset=["latitude", "longitude"])
        if len(map_df) > 0:
            sample = map_df.sample(min(1500, len(map_df)))
            if map_color == "bedrooms":
                sample = sample.copy()
                sample["bedrooms"] = sample["bedrooms"].astype(str)
            if map_color == "location_cluster":
                sample = sample.copy()
                sample["location_cluster"] = sample["location_cluster"].astype(str)
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
        share_mode = st.selectbox(
            "Share chart",
            ["Property type", "BHK (bedrooms)"],
            key="chart_share",
        )
        if len(d) > 0:
            if share_mode == "Property type":
                tc = d["property_type"].value_counts().reset_index()
                tc.columns = ["property_type", "count"]
                st.plotly_chart(
                    px.pie(tc, names="property_type", values="count", title="Property type share"),
                    use_container_width=True,
                )
            else:
                bc = d["bhk_label"].value_counts().reset_index()
                bc.columns = ["bhk", "count"]
                st.plotly_chart(
                    px.bar(bc, x="bhk", y="count", title="BHK distribution", text_auto=True),
                    use_container_width=True,
                )
        else:
            st.info("No rows for share chart.")

    wc_text = " ".join(d["description"].fillna("").astype(str).tolist())
    if wc_text.strip():
        wc = WordCloud(width=900, height=320, background_color="white").generate(wc_text)
        st.image(wc.to_array(), caption="Description keywords (filtered slice)")

with tabs[2]:
    st.subheader("Recommendations near a location")
    st.caption(
        "Pick a city, then drill into a smaller area (street / zipcode / cluster), "
        "apply a nearby radius, and rank similar homes from that micro-area."
    )

    loc_list = sorted(df["location"].dropna().unique().tolist())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        reco_location = st.selectbox("City / Location", loc_list, key="reco_loc")
    with c2:
        micro_mode = st.selectbox(
            "Drill-down by",
            ["Street", "Zipcode", "Area cluster"],
            key="reco_micro_mode",
        )
    with c3:
        radius_label = st.selectbox("Nearby radius", ["1 mile", "3 miles", "5 miles", "10 miles"], key="reco_rad")
    with c4:
        top_k = st.slider("How many recommendations", 3, 20, 8, key="reco_k")

    miles = float(radius_label.split()[0])

    sub_loc = df[df["location"] == reco_location]
    if len(sub_loc) == 0:
        st.error("No rows for that location.")
    else:
        if micro_mode == "Street":
            micro_col = "street"
        elif micro_mode == "Zipcode":
            micro_col = "zipcode"
        else:
            micro_col = "location_cluster"

        micro_vals = (
            sub_loc[micro_col]
            .fillna("(unknown)")
            .astype(str)
            .value_counts()
            .head(100)
            .index.tolist()
        )
        if not micro_vals:
            st.error(f"No {micro_mode.lower()} values found inside {reco_location}.")
        else:
            reco_micro = st.selectbox(
                f"{micro_mode} in {reco_location}",
                micro_vals,
                key="reco_micro_value",
            )

            sub_micro = sub_loc[sub_loc[micro_col].fillna("(unknown)").astype(str) == reco_micro]
            lat0 = float(sub_micro["latitude"].median())
            lon0 = float(sub_micro["longitude"].median())
            dist_all = _haversine_miles(df["latitude"], df["longitude"], lat0, lon0)
            same_loc = (df["location"] == reco_location).values
            same_micro = (df[micro_col].fillna("(unknown)").astype(str) == reco_micro).values
            near_mask = (dist_all <= miles) & same_loc & same_micro
            if near_mask.sum() < 2:
                near_mask = (dist_all <= miles) & same_loc
                st.warning(
                    "Few listings in this micro-area within radius — expanded to same city within radius."
                )
            if near_mask.sum() < 2:
                near_mask = dist_all <= miles
                st.warning(
                    "Still too few in city-radius — expanded to any listing within radius."
                )
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

                preview_cols = [
                    c
                    for c in [
                        "price",
                        "bedrooms",
                        "bathrooms",
                        "sqft",
                        "location",
                        "street",
                        "property_type",
                        "listing_url",
                    ]
                    if c in df.columns
                ]
                with st.expander("Anchor listing (nearest to area center within radius)", expanded=False):
                    st.dataframe(df.iloc[[anchor_pos]][preview_cols], use_container_width=True)

                mtime = DATA_PATH.stat().st_mtime
                blocks = _recommender_blocks_cached(str(DATA_PATH), mtime)

                others = np.array([i for i in near_pos if i != anchor_pos], dtype=int)
                if len(others) == 0:
                    st.info("Only one listing in this radius — increase radius to get recommendations.")

                if st.button("Get recommendations", type="primary", key="reco_run"):
                    if len(others) == 0:
                        st.warning("Not enough listings in radius to compare.")
                    else:
                        with st.spinner("Scoring similarity within radius…"):
                            scores = similarity_scores_for_row(blocks, anchor_pos)
                            cand_scores = scores[others]
                            order_local = np.argsort(-cand_scores)
                            take = min(top_k, len(order_local))
                            picked_pos = others[order_local[:take]]
                            recs = df.iloc[picked_pos][preview_cols].copy()
                            recs["similarity"] = scores[picked_pos]
                            recs["distance_mi"] = dist_all[picked_pos]
                            st.session_state["reco_near_table"] = recs
                            st.session_state["reco_near_params"] = (
                                reco_location,
                                micro_mode,
                                reco_micro,
                                radius_label,
                                top_k,
                            )

                if "reco_near_table" in st.session_state:
                    st.subheader("Recommended nearby listings")
                    st.dataframe(st.session_state["reco_near_table"], use_container_width=True)

                map_d = df.iloc[near_pos].dropna(subset=["latitude", "longitude"])
                if len(map_d) > 0:
                    st.caption("Map: listings in radius (sample). Area center shown in metrics above.")
                    samp = map_d.sample(min(400, len(map_d)))
                    figm = px.scatter_mapbox(
                        samp,
                        lat="latitude",
                        lon="longitude",
                        color="price",
                        hover_name="location",
                        zoom=10,
                        mapbox_style="carto-positron",
                        title=f"Near {reco_location} / {reco_micro} (~{miles:g} mi)",
                    )
                    st.plotly_chart(figm, use_container_width=True)
