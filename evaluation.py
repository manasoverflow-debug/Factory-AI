import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# FACTORY AI - EVALUATION & MONITORING
# ==========================================

print("\n==========================================")
print("FACTORY AI - EVALUATION & MONITORING")
print("==========================================")


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/AI_data.csv")

print(f"\nDataset loaded: {len(df)} records")


# ==========================================
# FEATURES
# ==========================================

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Target"]


# ==========================================
# LOAD QUALITY MODEL
# ==========================================

print("\nLoading Quality Control model...")

quality_model = joblib.load(
    "quality_model.pkl"
)

quality_predictions = quality_model.predict(X)


# ==========================================
# QUALITY CONTROL EVALUATION
# ==========================================

quality_accuracy = accuracy_score(
    y,
    quality_predictions
)

quality_precision = precision_score(
    y,
    quality_predictions,
    zero_division=0
)

quality_recall = recall_score(
    y,
    quality_predictions,
    zero_division=0
)

quality_f1 = f1_score(
    y,
    quality_predictions,
    zero_division=0
)

quality_cm = confusion_matrix(
    y,
    quality_predictions
)


print("\n==========================================")
print("QUALITY CONTROL EVALUATION")
print("==========================================")

print(
    f"\nAccuracy  : {quality_accuracy:.2%}"
)

print(
    f"Precision : {quality_precision:.2%}"
)

print(
    f"Recall    : {quality_recall:.2%}"
)

print(
    f"F1 Score  : {quality_f1:.2%}"
)

print("\nConfusion Matrix:")

print(quality_cm)

print("\nClassification Report:")

print(
    classification_report(
        y,
        quality_predictions,
        zero_division=0
    )
)


# ==========================================
# QUALITY FALSE POSITIVES / NEGATIVES
# ==========================================

quality_false_positive = (
    (y == 0) &
    (quality_predictions == 1)
).sum()

quality_false_negative = (
    (y == 1) &
    (quality_predictions == 0)
).sum()


print("\nQuality False Positives:",
      quality_false_positive)

print("Quality False Negatives:",
      quality_false_negative)


# ==========================================
# PREDICTIVE MAINTENANCE EVALUATION
# ==========================================

print("\n==========================================")
print("PREDICTIVE MAINTENANCE EVALUATION")
print("==========================================")


pm_model = None

try:

    pm_model = joblib.load(
        "pm_model.pkl"
    )

    pm_predictions = pm_model.predict(X)

    pm_accuracy = accuracy_score(
        y,
        pm_predictions
    )

    pm_precision = precision_score(
        y,
        pm_predictions,
        zero_division=0
    )

    pm_recall = recall_score(
        y,
        pm_predictions,
        zero_division=0
    )

    pm_f1 = f1_score(
        y,
        pm_predictions,
        zero_division=0
    )

    pm_cm = confusion_matrix(
        y,
        pm_predictions
    )

    print(
        f"\nAccuracy  : {pm_accuracy:.2%}"
    )

    print(
        f"Precision : {pm_precision:.2%}"
    )

    print(
        f"Recall    : {pm_recall:.2%}"
    )

    print(
        f"F1 Score  : {pm_f1:.2%}"
    )

    print("\nConfusion Matrix:")
    print(pm_cm)

except FileNotFoundError:

    print(
        "\nPM model file not found."
    )

    print(
        "Using predictive maintenance results "
        "for evaluation."
    )


# ==========================================
# LOAD PM RESULTS
# ==========================================

pm_results = pd.read_csv(
    "predictive_maintenance_results.csv"
)


# ==========================================
# FAILURE RISK DISTRIBUTION
# ==========================================

print("\n==========================================")
print("FAILURE RISK DISTRIBUTION")
print("==========================================")

risk_distribution = (
    pm_results["Risk Level"]
    .astype(str)
    .value_counts()
)

print(risk_distribution)


# ==========================================
# HIGH-RISK ANALYSIS
# ==========================================

high_risk = (
    pm_results["Risk Level"]
    .astype(str) == "High"
).sum()

medium_risk = (
    pm_results["Risk Level"]
    .astype(str) == "Medium"
).sum()

low_risk = (
    pm_results["Risk Level"]
    .astype(str) == "Low"
).sum()


print(
    f"\nHigh Risk   : {high_risk}"
)

print(
    f"Medium Risk : {medium_risk}"
)

print(
    f"Low Risk    : {low_risk}"
)


# ==========================================
# ANOMALY EVALUATION
# ==========================================

print("\n==========================================")
print("ANOMALY DETECTION EVALUATION")
print("==========================================")


anomaly_results = pd.read_csv(
    "anomaly_results.csv"
)


anomaly_counts = (
    anomaly_results["Anomaly"]
    .value_counts()
)


normal_count = anomaly_counts.get(
    "Normal",
    0
)

anomaly_count = anomaly_counts.get(
    "Anomaly",
    0
)


print(
    f"\nNormal Records  : {normal_count}"
)

print(
    f"Anomalies       : {anomaly_count}"
)

print(
    f"Anomaly Rate    : "
    f"{anomaly_count / len(anomaly_results):.2%}"
)


# ==========================================
# HUMAN APPROVAL EVALUATION
# ==========================================

print("\n==========================================")
print("HUMAN APPROVAL EVALUATION")
print("==========================================")


approval_results = pd.read_csv(
    "human_approval_results.csv"
)


approved = (
    approval_results["Human Decision"]
    == "Approved"
).sum()

rejected = (
    approval_results["Human Decision"]
    == "Rejected"
).sum()

skipped = (
    approval_results["Human Decision"]
    == "Skipped"
).sum()

pending = (
    approval_results["Human Decision"]
    == "Pending Human Review"
).sum()


print(
    f"\nApproved : {approved}"
)

print(
    f"Rejected : {rejected}"
)

print(
    f"Skipped  : {skipped}"
)

print(
    f"Pending  : {pending}"
)


# ==========================================
# HUMAN REJECTION RATE
# ==========================================

completed_reviews = (
    approved +
    rejected +
    skipped
)

if completed_reviews > 0:

    rejection_rate = (
        rejected /
        completed_reviews
    )

    print(
        f"\nHuman Rejection Rate: "
        f"{rejection_rate:.2%}"
    )

else:

    print(
        "\nHuman Rejection Rate: "
        "No completed reviews yet"
    )


# ==========================================
# OVERALL FACTORY AI METRICS
# ==========================================

print("\n==========================================")
print("OVERALL FACTORY AI METRICS")
print("==========================================")


print(
    f"\nTotal Records       : {len(df)}"
)

print(
    f"Defective Records   : "
    f"{(y == 1).sum()}"
)

print(
    f"Anomalies Detected  : "
    f"{anomaly_count}"
)

print(
    f"High Risk Machines  : "
    f"{high_risk}"
)

print(
    f"Critical Cases      : "
    f"{(
        approval_results['Priority']
        == 'Critical'
    ).sum()}"
)


# ==========================================
# CREATE EVALUATION REPORT
# ==========================================

evaluation_summary = {
    "Total Records": len(df),

    "Quality Accuracy": quality_accuracy,

    "Quality Precision": quality_precision,

    "Quality Recall": quality_recall,

    "Quality F1 Score": quality_f1,

    "Quality False Positives":
        quality_false_positive,

    "Quality False Negatives":
        quality_false_negative,

    "Anomalies Detected":
        anomaly_count,

    "Anomaly Rate":
        anomaly_count / len(anomaly_results),

    "High Risk Machines":
        high_risk,

    "Medium Risk Machines":
        medium_risk,

    "Low Risk Machines":
        low_risk,

    "Human Approved":
        approved,

    "Human Rejected":
        rejected,

    "Human Skipped":
        skipped,

    "Human Pending":
        pending
}


if pm_model is not None:

    evaluation_summary.update({

        "PM Accuracy":
            pm_accuracy,

        "PM Precision":
            pm_precision,

        "PM Recall":
            pm_recall,

        "PM F1 Score":
            pm_f1
    })


evaluation_df = pd.DataFrame(
    list(evaluation_summary.items()),
    columns=[
        "Metric",
        "Value"
    ]
)


# ==========================================
# SAVE EVALUATION REPORT
# ==========================================

output_file = "factory_ai_evaluation.csv"

evaluation_df.to_csv(
    output_file,
    index=False
)


# ==========================================
# COMPLETE
# ==========================================

print("\n==========================================")
print("EVALUATION COMPLETE!")
print("==========================================")

print(
    f"\nEvaluation report saved to:"
)

print(
    output_file
)

print(
    "\nFactory AI monitoring metrics are ready."
)