import pandas as pd
import joblib

# ==========================================
# FACTORY AI - MULTI-AGENT COORDINATOR
# ==========================================

print("\n==========================================")
print("FACTORY AI - MULTI-AGENT COORDINATOR")
print("==========================================")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/AI_data.csv")

print(f"\nFactory dataset loaded: {len(df)} records")


# ==========================================
# LOAD QUALITY MODEL
# ==========================================

quality_model = joblib.load("quality_model.pkl")

quality_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[quality_features]


# ==========================================
# QUALITY AGENT
# ==========================================

print("\nRunning Quality Control Agent...")

df["Quality Prediction"] = quality_model.predict(X)

df["Quality Status"] = df["Quality Prediction"].map({
    0: "Good",
    1: "Defective"
})

quality_probability = quality_model.predict_proba(X)

df["Quality Confidence"] = quality_probability.max(axis=1)


# ==========================================
# ANOMALY AGENT
# ==========================================

print("Running Anomaly Detection Agent...")

anomaly_results = pd.read_csv("anomaly_results.csv")

if "Anomaly" not in anomaly_results.columns:
    raise ValueError(
        "Anomaly column not found in anomaly_results.csv"
    )

df["Anomaly"] = anomaly_results["Anomaly"]


# ==========================================
# PREDICTIVE MAINTENANCE AGENT
# ==========================================

print("Running Predictive Maintenance Agent...")

pm_results = pd.read_csv(
    "predictive_maintenance_results.csv"
)

required_pm_columns = [
    "Failure Probability",
    "Risk Level",
    "Maintenance Recommendation"
]

for column in required_pm_columns:
    if column not in pm_results.columns:
        raise ValueError(
            f"Missing column in PM results: {column}"
        )

df["Failure Probability"] = pm_results[
    "Failure Probability"
]

df["Risk Level"] = pm_results[
    "Risk Level"
]

df["Maintenance Recommendation"] = pm_results[
    "Maintenance Recommendation"
]


# ==========================================
# MULTI-AGENT DECISION ENGINE
# ==========================================

def make_decision(row):

    quality = row["Quality Status"]
    anomaly = row["Anomaly"]
    risk = str(row["Risk Level"])

    failure_probability = row["Failure Probability"]

    # --------------------------------------
    # CRITICAL CONDITION
    # --------------------------------------

    if (
        quality == "Defective"
        and anomaly == "Anomaly"
        and risk == "High"
    ):
        return (
            "CRITICAL: Hold product and stop machine "
            "for immediate inspection"
        )

    # --------------------------------------
    # HIGH PRIORITY
    # --------------------------------------

    elif (
        quality == "Defective"
        and risk == "High"
    ):
        return (
            "HIGH PRIORITY: Hold defective product "
            "and perform immediate machine inspection"
        )

    elif (
        anomaly == "Anomaly"
        and risk == "High"
    ):
        return (
            "HIGH PRIORITY: Machine anomaly detected; "
            "perform immediate maintenance inspection"
        )

    # --------------------------------------
    # QUALITY ISSUE
    # --------------------------------------

    elif quality == "Defective":
        return (
            "QUALITY ALERT: Hold product for "
            "quality inspection"
        )

    # --------------------------------------
    # ANOMALY
    # --------------------------------------

    elif anomaly == "Anomaly":
        return (
            "ANOMALY ALERT: Investigate abnormal "
            "machine behavior"
        )

    # --------------------------------------
    # MAINTENANCE RISK
    # --------------------------------------

    elif risk == "High":
        return (
            "MAINTENANCE ALERT: Schedule immediate "
            "machine inspection"
        )

    elif risk == "Medium":
        return (
            "MAINTENANCE NOTICE: Schedule preventive "
            "maintenance"
        )

    # --------------------------------------
    # NORMAL
    # --------------------------------------

    else:
        return (
            "NORMAL: Continue operation and monitor "
            "machine"
        )


# ==========================================
# AGENT COORDINATION
# ==========================================

print("\nCoordinating agent decisions...")

df["AI Recommendation"] = df.apply(
    make_decision,
    axis=1
)


# ==========================================
# PRIORITY LEVEL
# ==========================================

def determine_priority(row):

    recommendation = row["AI Recommendation"]

    if recommendation.startswith("CRITICAL"):
        return "Critical"

    elif recommendation.startswith("HIGH PRIORITY"):
        return "High"

    elif (
        row["Quality Status"] == "Defective"
        or row["Anomaly"] == "Anomaly"
        or row["Risk Level"] == "High"
    ):
        return "High"

    elif row["Risk Level"] == "Medium":
        return "Medium"

    else:
        return "Low"


df["Priority"] = df.apply(
    determine_priority,
    axis=1
)


# ==========================================
# AGENT STATUS
# ==========================================

def overall_status(row):

    if row["Priority"] == "Critical":
        return "CRITICAL"

    elif row["Priority"] == "High":
        return "ATTENTION REQUIRED"

    elif row["Priority"] == "Medium":
        return "MONITOR"

    else:
        return "NORMAL"


df["Overall Status"] = df.apply(
    overall_status,
    axis=1
)


# ==========================================
# DISPLAY COORDINATED RESULTS
# ==========================================

print("\n==========================================")
print("MULTI-AGENT RESULTS")
print("==========================================")

display_columns = [
    "Quality Status",
    "Quality Confidence",
    "Anomaly",
    "Failure Probability",
    "Risk Level",
    "Priority",
    "Overall Status",
    "AI Recommendation"
]

print(
    df[display_columns]
    .head(20)
    .to_string(index=False)
)


# ==========================================
# PRIORITY SUMMARY
# ==========================================

print("\n==========================================")
print("PRIORITY SUMMARY")
print("==========================================")

print(
    df["Priority"]
    .value_counts()
)


# ==========================================
# OVERALL SYSTEM SUMMARY
# ==========================================

print("\n==========================================")
print("FACTORY AI SYSTEM SUMMARY")
print("==========================================")

print(
    f"Total Machines/Records : {len(df)}"
)

print(
    f"Defective Products     : "
    f"{(df['Quality Status'] == 'Defective').sum()}"
)

print(
    f"Anomalies Detected     : "
    f"{(df['Anomaly'] == 'Anomaly').sum()}"
)

print(
    f"High Risk Machines     : "
    f"{(df['Risk Level'].astype(str) == 'High').sum()}"
)

print(
    f"Critical Cases         : "
    f"{(df['Priority'] == 'Critical').sum()}"
)

print(
    f"High Priority Cases    : "
    f"{(df['Priority'] == 'High').sum()}"
)


# ==========================================
# SAVE COORDINATED RESULTS
# ==========================================

output_file = "factory_ai_decisions.csv"

df.to_csv(
    output_file,
    index=False
)


# ==========================================
# COMPLETE
# ==========================================

print("\n==========================================")
print("MULTI-AGENT COORDINATION COMPLETE!")
print("==========================================")

print(
    f"\nResults saved to: {output_file}"
)

print("\nAgents successfully coordinated:")
print("  [✓] Quality Control Agent")
print("  [✓] Anomaly Detection Agent")
print("  [✓] Predictive Maintenance Agent")
print("  [✓] Decision Engine")