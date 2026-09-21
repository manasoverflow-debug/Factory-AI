import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/AI_data.csv")

features = [
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
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# TRAIN PREDICTIVE MAINTENANCE MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ============================================================
# MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("PREDICTIVE MAINTENANCE MODEL")
print("========================================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Failure", "Failure"]
    )
)

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, "pm_model.pkl")

print("\nPM model saved as: pm_model.pkl")

# ============================================================
# FAILURE PROBABILITY FOR ALL MACHINES
# ============================================================

df["Failure Probability"] = model.predict_proba(X)[:, 1]

# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(probability):

    if probability < 0.30:
        return "Low"

    elif probability < 0.70:
        return "Medium"

    else:
        return "High"


df["Risk Level"] = (
    df["Failure Probability"]
    .apply(classify_risk)
)

# ============================================================
# MAINTENANCE RECOMMENDATION
# ============================================================

def maintenance_recommendation(risk):

    if risk == "High":
        return "Immediate inspection recommended"

    elif risk == "Medium":
        return "Schedule maintenance inspection"

    else:
        return "Continue normal operation"


df["Maintenance Recommendation"] = (
    df["Risk Level"]
    .apply(maintenance_recommendation)
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("PREDICTIVE MAINTENANCE RESULTS")
print("========================================")

print(
    df[
        features
        + [
            "Target",
            "Failure Probability",
            "Risk Level",
            "Maintenance Recommendation"
        ]
    ].head(20)
)

# ============================================================
# RISK SUMMARY
# ============================================================

print("\n========================================")
print("RISK SUMMARY")
print("========================================")

print(
    df["Risk Level"].value_counts()
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print(importance)

# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    "predictive_maintenance_results.csv",
    index=False
)

print(
    "\nSaved results to "
    "predictive_maintenance_results.csv"
)

print("\n========================================")
print("PREDICTIVE MAINTENANCE COMPLETE")
print("========================================")