import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the factory dataset
df = pd.read_csv("data/AI_data.csv")

# Select sensor features for anomaly detection
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]

# Create the anomaly detection model
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

# Detect anomalies
df["Anomaly"] = model.fit_predict(X)

# Convert the result into readable labels
df["Anomaly"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})
# Save the results
df.to_csv("anomaly_results.csv", index=False)

print("Saved results to anomaly_results.csv")

# Display results
print("Dataset shape:", df.shape)
print("\nAnomaly results:")
print(df[features + ["Anomaly"]].head(20))

# Count normal vs anomalous records
print("\nSummary:")
print(df["Anomaly"].value_counts())