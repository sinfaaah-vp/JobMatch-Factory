
from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Locate the project folder
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "jobmatch_kerala_synthetic_dataset.csv"
MODEL_PATH = BASE_DIR / "jobmatch_model.joblib"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# Input features
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

target = "suitability_score"

# Ensure required columns exist
missing = set(features + [target]) - set(df.columns)
if missing:
    raise ValueError(f"Missing CSV columns: {sorted(missing)}")

X = df[features]
y = df[target]

categorical_features = [
    "locality",
    "district",
    "factory_type",
]

numeric_features = [
    col for col in features
    if col not in categorical_features
]

# Preprocess text and numerical columns
preprocessor = ColumnTransformer([
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical_features,
    ),
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_features,
    ),
])

# Machine learning model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
        ),
    ),
])

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\nModel training completed!")
print("Mean Absolute Error:", round(
    mean_absolute_error(y_test, predictions), 2
))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

# Save model
joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
print("WARNING: Evaluation uses synthetic demonstration data.")
