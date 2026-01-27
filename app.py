import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Hyperlocal Grid Quality Dashboard",
    layout="wide"
)

st.title("🗺️ Hyperlocal Grid Quality Assessment")
st.caption("Unsupervised spatial scoring using static map-derived features")

@st.cache_data
def load_data():
    df = pd.read_csv("data/delhi_grids_enriched.csv")
    return df

df = load_data()
st.sidebar.header("Segmentation")

segment_mode = st.sidebar.radio(
    "Segmentation Method",
    options=[
        "Score-based (Business)",
        "KMeans (Unsupervised)"
    ],
    index=0
)

SEGMENT_COL = (
    "grid_segment"
    if segment_mode == "Score-based (Business)"
    else "grid_segment_kmeans"
)


st.sidebar.header("Filters")

segment_filter = st.sidebar.multiselect(
    "Grid Segment",
    options=df[SEGMENT_COL].unique(),
    default=df[SEGMENT_COL].unique()
)


score_range = st.sidebar.slider(
    "Grid Score Range",
    min_value=float(df.grid_score.min()),
    max_value=float(df.grid_score.max()),
    value=(float(df.grid_score.min()), float(df.grid_score.max()))
)

filtered_df = df[
    (df[SEGMENT_COL].isin(segment_filter)) &
    (df["grid_score"].between(score_range[0], score_range[1]))
]

st.subheader("📊 Grid Score Distribution")

fig = px.histogram(
    filtered_df,
    x="grid_score",
    color=SEGMENT_COL,
    nbins=40,
    title="Distribution of Grid Scores by Segment",
    opacity=0.7
)

st.plotly_chart(fig, use_container_width=True)



st.subheader("🔍 Grid-level Explainability")

selected_grid = st.selectbox(
    "Select Grid ID",
    options=filtered_df.grid_id.unique()
)

grid_row = filtered_df[filtered_df.grid_id == selected_grid].iloc[0]

col1, col2 = st.columns(2)

with col1:
    st.metric("Grid Score", round(grid_row.grid_score, 2))
    st.metric("Segment", grid_row[SEGMENT_COL])

with col2:
    feature_df = pd.DataFrame({
        "feature": [
            "building_density",
            "yellow_building_density",
            "total_road_density",
            "empty",
            "waterbody"
        ],
        "value": [
            grid_row.building_density,
            grid_row.yellow_building_density,
            grid_row.total_road_density,
            grid_row.empty,
            grid_row.waterbody
        ]
    })

    fig = px.bar(
        feature_df,
        x="value",
        y="feature",
        orientation="h",
        title="Feature Contribution Snapshot"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 Validation & Sanity Checks")

col3, col4 = st.columns(2)

with col3:
    fig1 = px.scatter(
        filtered_df,
        x="building_density",
        y="grid_score",
        color=SEGMENT_COL,
        title="Grid Score vs Building Density",
        opacity=0.6
    )
    st.plotly_chart(fig1, use_container_width=True)

with col4:
    fig2 = px.scatter(
        filtered_df,
        x="empty",
        y="grid_score",
        color=SEGMENT_COL,
        title="Grid Score vs Empty Area %",
        opacity=0.6
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("📌 Segment Profiles")

summary = (
    filtered_df
    .groupby(SEGMENT_COL)[
        [
            "building_density",
            "yellow_building_density",
            "total_road_density",
            "empty",
            "waterbody"
        ]
    ]
    .mean()
    .round(2)
)


st.dataframe(summary)
