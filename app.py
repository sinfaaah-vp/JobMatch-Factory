
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "jobmatch_kerala_synthetic_dataset.csv"
MODEL_PATH = BASE_DIR / "jobmatch_model.joblib"

st.set_page_config(
    page_title="JobMatch Factory AI",
    page_icon="🏭",
    layout="wide",
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🏭 JobMatch Factory ")
st.subheader("Employment-driven industrial planning for Kerala")

st.info(
    "Demo prototype: the dataset is synthetic. "
    "Recommendations are not verified investment advice."
)

try:
    df = load_data()
    model = load_model()
except Exception as error:
    st.error(f"Could not load the dataset or model: {error}")
    st.stop()

features = [
    "locality",
    "district",
    "factory_type",
    "unemployed_workers",
    "skill_match_pct",
    "local_demand_score",
    "raw_material_score",
    "transport_score",
    "infrastructure_score",
    "investment_lakh",
    "training_needed_pct",
    "estimated_jobs",
]

# Sidebar selection
st.sidebar.header("Choose a location")

localities = sorted(df["locality"].dropna().unique())
locality = st.sidebar.selectbox("Kerala locality", localities)

results = df[
    df["locality"].str.casefold() == locality.casefold()
].copy()

results["predicted_suitability"] = model.predict(
    results[features]
).round(1)

results = results.sort_values(
    "predicted_suitability",
    ascending=False,
).reset_index(drop=True)

# Summary metrics
district = results["district"].iloc[0]
worker_count = int(results["unemployed_workers"].iloc[0])

st.write(f"**Locality:** {locality}  |  **District:** {district}")

col1, col2, col3 = st.columns(3)

col1.metric("Demo job-seeker count", f"{worker_count:,}")
col2.metric("Factory types compared", len(results))
col3.metric(
    "Highest suitability score",
    f"{results['predicted_suitability'].max():.1f}/100",
)

st.divider()

st.header("Top factory recommendations")

top = results.head(5).copy()

display = top[
    [
        "factory_type",
        "predicted_suitability",
        "estimated_jobs",
        "skill_match_pct",
        "training_needed_pct",
        "investment_lakh",
    ]
].rename(
    columns={
        "factory_type": "Factory type",
        "predicted_suitability": "Predicted score",
        "estimated_jobs": "Demo job estimate",
        "skill_match_pct": "Skill match (%)",
        "training_needed_pct": "Training needed (%)",
        "investment_lakh": "Investment (₹ lakh)",
    }
)

st.dataframe(display, hide_index=True, use_container_width=True)

st.caption(
    "Job counts, investments, workforce values and scores are synthetic "
    "examples. They are not verified local statistics or forecasts."
)

# Suitability chart
st.header("Compare factory types")

chart_data = results.set_index("factory_type")[
    ["predicted_suitability"]
].sort_values("predicted_suitability")

st.bar_chart(chart_data)

# Training needs
st.header("Potential skills-training needs")

training = results[
    ["factory_type", "skill_match_pct", "training_needed_pct"]
].sort_values("training_needed_pct", ascending=False)

st.bar_chart(
    training.set_index("factory_type")[
        ["skill_match_pct", "training_needed_pct"]
    ]
)

st.write(
    "Training priorities are illustrative. Real training plans require "
    "worker-level skill assessments and factory job requirements."
)

# Download recommendations
csv_bytes = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download recommendations as CSV",
    data=csv_bytes,
    file_name=f"{locality.lower()}_factory_recommendations.csv",
    mime="text/csv",
)
