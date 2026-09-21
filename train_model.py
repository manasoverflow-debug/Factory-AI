import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = "data/AI_data.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("FACTORY AI - PREDICTIVE MAINTENANCE MODEL")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")

# ============================================================
# 2. BASIC VALIDATION
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["Target"].value_counts())

# ============================================================
# 3. SELECT FEATURES
# ============================================================

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target = "Target"

X = df[features]
y = df[target]

# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================
# 5. PREPROCESSING
# ============================================================

categorical_features = ["Type"]

numeric_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# ============================================================
# 6. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")

# ============================================================
# 8. EVALUATE MODEL
# ============================================================

y_pred = pipeline.predict(X_test)
y_probability = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# 9. SAVE MODEL
# ============================================================

MODEL_FILE = "factory_ai_model.pkl"

joblib.dump(pipeline, MODEL_FILE)

print("\n" + "=" * 60)
print(f"Model saved as: {MODEL_FILE}")
print("=" * 60)