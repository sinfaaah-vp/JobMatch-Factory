
from pathlib import Path

import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "jobmatch_model.joblib"
DATA_PATH = BASE_DIR / "jobmatch_kerala_synthetic_dataset.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

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

locality = input("Enter a locality: ").strip()

results = df[
    df["locality"].str.casefold() == locality.casefold()
].copy()

if results.empty:
    print("Locality not found in the dataset.")
    print("Available localities:")
    print(", ".join(sorted(df["locality"].unique())))
    raise SystemExit

results["predicted_suitability"] = model.predict(
    results[features]
).round(1)

results = results.sort_values(
    "predicted_suitability",
    ascending=False,
)

print(f"\nFactory recommendations for {locality}")
print("(Synthetic demonstration results only)\n")

columns_to_show = [
    "factory_type",
    "predicted_suitability",
    "estimated_jobs",
    "skill_match_pct",
    "training_needed_pct",
    "investment_lakh",
]

print(results[columns_to_show].head(5).to_string(index=False))

print(
    "\nInvestment is in lakh rupees. "
    "Job counts and scores are illustrative, not verified forecasts."
)
