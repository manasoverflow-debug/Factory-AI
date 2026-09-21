import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# FACTORY AI - QUALITY CONTROL AGENT
# ==========================================

print("\n==========================================")
print("FACTORY AI - QUALITY CONTROL AGENT")
print("==========================================")

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/AI_data.csv")

print(f"\nDataset loaded successfully!")
print(f"Total records: {len(df)}")


# ==========================================
# SENSOR FEATURES
# ==========================================

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


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# CREATE QUALITY MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining Quality Control model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# TEST MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# ==========================================
# MODEL PERFORMANCE
# ==========================================

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# PREDICT QUALITY
# ==========================================

df["Quality Prediction"] = model.predict(X)

df["Quality Status"] = df["Quality Prediction"].map({
    0: "Good",
    1: "Defective"
})


# ==========================================
# QUALITY SUMMARY
# ==========================================

print("\n==========================================")
print("QUALITY SUMMARY")
print("==========================================")

quality_counts = df["Quality Status"].value_counts()

print(quality_counts)

good_count = quality_counts.get("Good", 0)
defective_count = quality_counts.get("Defective", 0)

defect_rate = defective_count / len(df)

print(f"\nGood Products      : {good_count}")
print(f"Defective Products : {defective_count}")
print(f"Defect Rate        : {defect_rate:.2%}")


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n==========================================")
print("QUALITY FEATURE IMPORTANCE")
print("==========================================")

print(importance.to_string(index=False))


# ==========================================
# SAMPLE PREDICTIONS
# ==========================================

print("\n==========================================")
print("SAMPLE QUALITY PREDICTIONS")
print("==========================================")

sample_columns = features + [
    "Target",
    "Quality Prediction",
    "Quality Status"
]

print(
    df[sample_columns]
    .head(20)
    .to_string(index=False)
)


# ==========================================
# SAVE MODEL
# ==========================================

model_file = "quality_model.pkl"

joblib.dump(model, model_file)

print("\nQuality model saved as:")
print(model_file)


# ==========================================
# SAVE RESULTS
# ==========================================

output_file = "quality_control_results.csv"

df.to_csv(output_file, index=False)

print("\nResults saved as:")
print(output_file)


# ==========================================
# REAL-TIME QUALITY PREDICTION FUNCTION
# ==========================================

def predict_quality(
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
):

    input_data = pd.DataFrame([{
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    confidence = max(probability)

    if prediction == 0:
        status = "Good"
    else:
        status = "Defective"

    return {
        "prediction": int(prediction),
        "status": status,
        "confidence": float(confidence)
    }


# ==========================================
# AGENT TEST
# ==========================================

print("\n==========================================")
print("QUALITY AGENT TEST")
print("==========================================")

test_result = predict_quality(
    air_temperature=300,
    process_temperature=310,
    rotational_speed=1500,
    torque=40,
    tool_wear=100
)

print(f"Quality Status : {test_result['status']}")
print(f"Confidence     : {test_result['confidence']:.2%}")


# ==========================================
# COMPLETE
# ==========================================

print("\n==========================================")
print("QUALITY CONTROL AGENT READY!")
print("==========================================")