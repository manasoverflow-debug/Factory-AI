import pandas as pd
from datetime import datetime


# ==========================================
# FACTORY AI - HUMAN APPROVAL AGENT
# ==========================================

print("\n==========================================")
print("FACTORY AI - HUMAN APPROVAL AGENT")
print("==========================================")


# ==========================================
# LOAD COORDINATOR RESULTS
# ==========================================

input_file = "factory_ai_decisions.csv"

try:
    df = pd.read_csv(input_file)

except FileNotFoundError:
    print("\nERROR: factory_ai_decisions.csv not found.")
    print("Please run agent_coordinator.py first.")
    exit()


print(f"\nLoaded {len(df)} factory records.")


# ==========================================
# CREATE HUMAN REVIEW COLUMNS
# ==========================================

df["Human Decision"] = "Pending"

df["Approval Time"] = ""

df["Approved By"] = ""


# ==========================================
# IDENTIFY CASES REQUIRING HUMAN REVIEW
# ==========================================

review_mask = df["Priority"].isin(
    ["Critical", "High"]
)

df.loc[
    review_mask,
    "Human Decision"
] = "Pending Human Review"


# ==========================================
# REVIEW STATISTICS
# ==========================================

total_cases = len(df)

critical_cases = (
    df["Priority"] == "Critical"
).sum()

high_cases = (
    df["Priority"] == "High"
).sum()

review_cases = review_mask.sum()


# ==========================================
# DISPLAY REVIEW QUEUE
# ==========================================

print("\n==========================================")
print("HUMAN REVIEW QUEUE")
print("==========================================")

print(
    f"\nTotal factory records : {total_cases}"
)

print(
    f"Critical cases        : {critical_cases}"
)

print(
    f"High-priority cases   : {high_cases}"
)

print(
    f"Cases requiring review: {review_cases}"
)


# ==========================================
# SHOW FIRST 10 CASES
# ==========================================

print("\n==========================================")
print("PENDING HUMAN REVIEWS")
print("==========================================")

display_columns = [
    "Quality Status",
    "Anomaly",
    "Failure Probability",
    "Risk Level",
    "Priority",
    "Overall Status",
    "AI Recommendation",
    "Human Decision"
]

print(
    df.loc[
        review_mask,
        display_columns
    ].head(10).to_string(index=False)
)


# ==========================================
# SYSTEM MESSAGE
# ==========================================

print("\n==========================================")
print("HUMAN APPROVAL STATUS")
print("==========================================")

print(
    "\nAI recommendations have NOT been"
    " automatically approved."
)

print(
    "Critical and high-priority cases are"
    " waiting for human review."
)

print(
    "\nThe dashboard will provide:"
)

print("  [ APPROVE ]")
print("  [ REJECT  ]")
print("  [ SKIP    ]")


# ==========================================
# SAVE RESULTS
# ==========================================

output_file = "human_approval_results.csv"

df.to_csv(
    output_file,
    index=False
)


# ==========================================
# COMPLETE
# ==========================================

print("\n==========================================")
print("HUMAN APPROVAL AGENT READY!")
print("==========================================")

print(
    f"\nResults saved to: {output_file}"
)

print(
    "\nHuman-in-the-loop system is ready."
)