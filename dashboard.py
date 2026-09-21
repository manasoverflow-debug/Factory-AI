import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Try to load the existing AI coordinator.
try:
    from factory_ai_coordinator import run_ai_coordination
    COORDINATOR_AVAILABLE = True
except Exception:
    COORDINATOR_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Factory AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = os.path.join(
    BASE_DIR,
    "factory_ai_model.pkl"
)

AI_DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "AI_data.csv"
)

DECISIONS_FILE = os.path.join(
    BASE_DIR,
    "factory_ai_decisions.csv"
)

APPROVAL_FILE = os.path.join(
    BASE_DIR,
    "human_approval_results.csv"
)

EVALUATION_FILE = os.path.join(
    BASE_DIR,
    "factory_ai_evaluation.csv"
)

RIGOROUS_EVALUATION_FILE = os.path.join(
    BASE_DIR,
    "factory_ai_rigorous_evaluation.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

ANOMALY_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(file_path):
    """Safely load a CSV file."""
    if not os.path.exists(file_path):
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def find_column(df, keywords):
    """Find the first column containing any keyword."""
    if df.empty:
        return None

    for column in df.columns:
        column_lower = str(column).lower()

        for keyword in keywords:
            if keyword.lower() in column_lower:
                return column

    return None


def safe_numeric(series):
    """Convert a pandas series to numeric safely."""
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def percentage(part, total):
    """Calculate a safe percentage."""
    if total == 0:
        return 0.0

    return (part / total) * 100


def get_target_column(df):
    """Find the machine failure target column."""
    possible = [
        "Target",
        "target",
        "Machine failure",
        "Machine Failure"
    ]

    for column in possible:
        if column in df.columns:
            return column

    return find_column(
        df,
        ["target", "machine failure"]
    )


def get_quality_column(df):
    """Find a quality-related column."""
    return find_column(
        df,
        [
            "quality prediction",
            "quality",
            "defect",
            "quality status"
        ]
    )


def get_failure_type_column(df):
    """Find failure type column."""
    return find_column(
        df,
        ["failure type"]
    )


def normalize_binary_target(series):
    """
    Convert common binary target representations
    to numeric 0/1 where possible.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce"
        )

    mapping = {
        "0": 0,
        "1": 1,
        "normal": 0,
        "no failure": 0,
        "failure": 1,
        "failed": 1,
        "yes": 1,
        "no": 0,
        "false": 0,
        "true": 1
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


def save_approval_data(df):
    """Save approval data safely."""
    try:
        df.to_csv(
            APPROVAL_FILE,
            index=False
        )
        return True

    except Exception as error:
        st.error(
            f"Unable to save approval data: {error}"
        )
        return False


def ensure_approval_columns(df):
    """Ensure required approval columns exist."""
    required_columns = [
        "Case ID",
        "Human Decision",
        "Approval Time",
        "Approved By"
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = ""

    return df


def create_case_ids(df):
    """Create unique Case IDs for existing approval records."""
    df = ensure_approval_columns(df)

    for index in df.index:

        existing_id = str(
            df.loc[index, "Case ID"]
        ).strip()

        if (
            existing_id == ""
            or existing_id.lower() == "nan"
        ):
            df.loc[index, "Case ID"] = (
                f"CASE-{index + 1:05d}"
            )

    return df


def risk_level(probability):
    """Convert probability into a risk level."""
    if probability < 0.30:
        return "LOW"

    if probability < 0.60:
        return "MEDIUM"

    return "HIGH"


def risk_recommendation(level):
    """Return decision-support recommendation."""
    if level == "HIGH":
        return (
            "Immediate maintenance review is recommended."
        )

    if level == "MEDIUM":
        return (
            "Schedule a maintenance inspection "
            "and continue monitoring."
        )

    return (
        "Continue routine monitoring."
    )


def get_failure_rate(df):
    """Calculate failure rate from Target."""
    target_column = get_target_column(df)

    if target_column is None:
        return None

    target = normalize_binary_target(
        df[target_column]
    ).dropna()

    if target.empty:
        return None

    return float(
        target.mean() * 100
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_ml_model():
    if not os.path.exists(MODEL_FILE):
        return None

    try:
        return joblib.load(
            MODEL_FILE
        )
    except Exception:
        return None


ml_model = load_ml_model()

ai_data = load_csv(
    AI_DATA_FILE
)

decisions = load_csv(
    DECISIONS_FILE
)

approvals = load_csv(
    APPROVAL_FILE
)

evaluation = load_csv(
    EVALUATION_FILE
)

rigorous_evaluation = load_csv(
    RIGOROUS_EVALUATION_FILE
)


# ============================================================
# PREPARE APPROVAL DATA
# ============================================================

if not approvals.empty:

    approvals = create_case_ids(
        approvals
    )

    # Save generated IDs if required.
    if os.path.exists(APPROVAL_FILE):
        try:
            approvals.to_csv(
                APPROVAL_FILE,
                index=False
            )
        except Exception:
            pass


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 Factory AI")

st.sidebar.caption(
    "Smart Manufacturing Quality Control "
    "& Predictive Maintenance System"
)

st.sidebar.divider()

st.sidebar.subheader(
    "System Status"
)

if ml_model is not None:
    st.sidebar.success(
        "🟢 Predictive Model: ONLINE"
    )
else:
    st.sidebar.error(
        "🔴 Predictive Model: OFFLINE"
    )

if not ai_data.empty:
    st.sidebar.success(
        "🟢 AI Dataset: AVAILABLE"
    )
else:
    st.sidebar.error(
        "🔴 AI Dataset: NOT FOUND"
    )

if COORDINATOR_AVAILABLE:
    st.sidebar.success(
        "🟢 AI Coordinator: AVAILABLE"
    )
else:
    st.sidebar.warning(
        "🟡 AI Coordinator: FALLBACK MODE"
    )

if not approvals.empty:
    st.sidebar.success(
        "🟢 Human Approval: ACTIVE"
    )
else:
    st.sidebar.info(
        "Human Approval: WAITING"
    )

st.sidebar.divider()

st.sidebar.info(
    "Use the sections below to monitor factory "
    "data, AI predictions, anomalies, quality, "
    "recommendations and human decisions."
)


# ============================================================
# TITLE
# ============================================================

st.title("🏭 Factory AI")

st.subheader(
    "Smart Manufacturing Quality Control "
    "& Predictive Maintenance System"
)

st.caption(
    "AI-powered factory monitoring, anomaly detection, "
    "quality control, predictive maintenance, "
    "multi-agent coordination and human-in-the-loop decisions."
)


# ============================================================
# FACTORY OVERVIEW
# ============================================================

st.header("📊 Factory Overview")

total_machines = len(ai_data)

failure_rate = get_failure_rate(
    ai_data
)

if failure_rate is None:
    failure_rate = 0.0


# Calculate anomaly count.
anomaly_count = 0

if not ai_data.empty:

    try:
        valid_features = [
            column
            for column in ANOMALY_FEATURES
            if column in ai_data.columns
        ]

        if len(valid_features) == len(
            ANOMALY_FEATURES
        ):

            anomaly_data = ai_data[
                valid_features
            ].apply(
                pd.to_numeric,
                errors="coerce"
            ).dropna()

            if len(anomaly_data) > 10:

                anomaly_model = IsolationForest(
                    n_estimators=100,
                    contamination=0.05,
                    random_state=42
                )

                anomaly_predictions = (
                    anomaly_model.fit_predict(
                        anomaly_data
                    )
                )

                anomaly_count = int(
                    (anomaly_predictions == -1).sum()
                )

    except Exception:
        anomaly_count = 0


anomaly_rate = percentage(
    anomaly_count,
    total_machines
)


# High-risk cases from approvals.
pending_count = 0

if not approvals.empty:

    pending_count = int(
        approvals[
            "Human Decision"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("pending human review")
        .sum()
    )


# High-risk machine count.
high_risk_count = 0

if not decisions.empty:

    priority_column = find_column(
        decisions,
        ["priority", "risk"]
    )

    if priority_column is not None:

        high_risk_count = int(
            decisions[
                priority_column
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "high",
                    "critical",
                    "urgent"
                ]
            )
            .sum()
        )


overview_col1, overview_col2, overview_col3, overview_col4 = (
    st.columns(4)
)

with overview_col1:
    st.metric(
        "🏭 Total Machines",
        total_machines
    )

with overview_col2:
    st.metric(
        "⚠️ Failure Rate",
        f"{failure_rate:.2f}%"
    )

with overview_col3:
    st.metric(
        "🚨 Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )

with overview_col4:
    st.metric(
        "👤 Pending Reviews",
        pending_count
    )


overview_col5, overview_col6 = (
    st.columns(2)
)

with overview_col5:
    st.metric(
        "🔴 High-Risk Cases",
        high_risk_count
    )

with overview_col6:

    if not evaluation.empty:

        numeric_columns = (
            evaluation
            .select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )

        if numeric_columns:

            try:
                model_score = float(
                    evaluation[
                        numeric_columns[0]
                    ].iloc[0]
                )

                st.metric(
                    "📈 Model Metric",
                    f"{model_score:.4f}"
                )

            except Exception:
                st.metric(
                    "📈 Model Metric",
                    "N/A"
                )

        else:
            st.metric(
                "📈 Model Metric",
                "N/A"
            )

    else:
        st.metric(
            "📈 Model Evaluation",
            "Available after evaluation"
        )


# ============================================================
# DATA ANALYSIS
# ============================================================

st.divider()

st.header(
    "📈 Factory Data Analysis"
)

if ai_data.empty:

    st.error(
        "❌ AI_data.csv could not be loaded."
    )

    st.code(
        "Expected location:\n"
        "Factory AI/data/AI_data.csv"
    )

else:

    data_col1, data_col2, data_col3 = (
        st.columns(3)
    )

    with data_col1:
        st.metric(
            "Dataset Rows",
            len(ai_data)
        )

    with data_col2:
        st.metric(
            "Dataset Columns",
            len(ai_data.columns)
        )

    with data_col3:
        missing_values = int(
            ai_data.isnull()
            .sum()
            .sum()
        )

        st.metric(
            "Missing Values",
            missing_values
        )


    st.subheader(
        "🏭 Machine Type Distribution"
    )

    if "Type" in ai_data.columns:

        type_counts = (
            ai_data["Type"]
            .astype(str)
            .value_counts()
        )

        st.bar_chart(
            type_counts
        )

    else:

        st.info(
            "Machine Type column not available."
        )


    st.subheader(
        "⚙️ Machine Operating Statistics"
    )

    statistics_columns = [
        column
        for column in [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
        if column in ai_data.columns
    ]

    if statistics_columns:

        st.dataframe(
            ai_data[
                statistics_columns
            ].describe().round(2),
            use_container_width=True
        )

    else:

        st.info(
            "Operating-condition columns were not found."
        )


    st.subheader(
        "🔥 Failure Analysis"
    )

    target_column = get_target_column(
        ai_data
    )

    if target_column is not None:

        target_values = (
            ai_data[target_column]
            .astype(str)
            .value_counts()
        )

        st.bar_chart(
            target_values
        )

        failure_count = int(
            normalize_binary_target(
                ai_data[target_column]
            )
            .eq(1)
            .sum()
        )

        normal_count = int(
            normalize_binary_target(
                ai_data[target_column]
            )
            .eq(0)
            .sum()
        )

        failure_col1, failure_col2 = (
            st.columns(2)
        )

        with failure_col1:
            st.metric(
                "Normal Machines",
                normal_count
            )

        with failure_col2:
            st.metric(
                "Failure Cases",
                failure_count
            )

    else:

        st.info(
            "Target column was not found."
        )


    st.subheader(
        "🛠️ Failure Type Distribution"
    )

    failure_type_column = (
        get_failure_type_column(
            ai_data
        )
    )

    if failure_type_column is not None:

        failure_types = (
            ai_data[
                failure_type_column
            ]
            .astype(str)
            .value_counts()
        )

        st.bar_chart(
            failure_types
        )

    else:

        st.info(
            "Failure Type column was not found."
        )


    st.subheader(
        "🌡️ Temperature Analysis"
    )

    temperature_columns = [
        column
        for column in [
            "Air temperature [K]",
            "Process temperature [K]"
        ]
        if column in ai_data.columns
    ]

    if temperature_columns:

        st.line_chart(
            ai_data[
                temperature_columns
            ].head(500)
        )

    else:

        st.info(
            "Temperature data unavailable."
        )


    st.subheader(
        "📋 Dataset Preview"
    )

    st.dataframe(
        ai_data.head(10),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# PREDICTIVE MAINTENANCE
# ============================================================

st.divider()

st.header(
    "🔧 Predictive Maintenance"
)

if ml_model is None:

    st.error(
        "❌ Predictive maintenance model is unavailable."
    )

    st.code(
        "Expected file:\n"
        "Factory AI/factory_ai_model.pkl\n\n"
        "If the model has not been created yet, run:\n"
        "python train_model.py"
    )

else:

    st.success(
        "🟢 Predictive maintenance model is ONLINE."
    )

    st.write(
        "Enter current machine operating conditions "
        "to estimate machine failure risk."
    )

    input_col1, input_col2 = st.columns(2)

    with input_col1:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
            key="prediction_machine_type"
        )

        air_temperature = st.number_input(
            "Air Temperature [K]",
            min_value=250.0,
            max_value=350.0,
            value=298.0,
            step=0.1,
            key="prediction_air_temperature"
        )

        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=250.0,
            max_value=400.0,
            value=308.0,
            step=0.1,
            key="prediction_process_temperature"
        )

    with input_col2:

        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=0,
            max_value=5000,
            value=1500,
            step=10,
            key="prediction_rotational_speed"
        )

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.5,
            key="prediction_torque"
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=300,
            value=100,
            step=1,
            key="prediction_tool_wear"
        )


    predict_button = st.button(
        "🔮 Predict Machine Failure",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        input_data = pd.DataFrame(
            {
                "Type": [machine_type],

                "Air temperature [K]": [
                    air_temperature
                ],

                "Process temperature [K]": [
                    process_temperature
                ],

                "Rotational speed [rpm]": [
                    rotational_speed
                ],

                "Torque [Nm]": [
                    torque
                ],

                "Tool wear [min]": [
                    tool_wear
                ]
            }
        )

        try:

            prediction = ml_model.predict(
                input_data
            )[0]

            if hasattr(
                ml_model,
                "predict_proba"
            ):

                probability = float(
                    ml_model.predict_proba(
                        input_data
                    )[0][1]
                )

            else:

                probability = float(
                    prediction
                )

            level = risk_level(
                probability
            )

            recommendation = (
                risk_recommendation(
                    level
                )
            )


            st.subheader(
                "🎯 Prediction Result"
            )

            result_col1, result_col2, result_col3 = (
                st.columns(3)
            )

            with result_col1:

                st.metric(
                    "Failure Probability",
                    f"{probability * 100:.2f}%"
                )

            with result_col2:

                st.metric(
                    "Risk Level",
                    level
                )

            with result_col3:

                if prediction == 1:

                    st.error(
                        "⚠️ FAILURE RISK DETECTED"
                    )

                else:

                    st.success(
                        "✅ MACHINE STATUS: NORMAL"
                    )


            if level == "HIGH":

                st.error(
                    f"🔴 HIGH RISK\n\n"
                    f"{recommendation}"
                )

            elif level == "MEDIUM":

                st.warning(
                    f"🟡 MEDIUM RISK\n\n"
                    f"{recommendation}"
                )

            else:

                st.success(
                    f"🟢 LOW RISK\n\n"
                    f"{recommendation}"
                )


            st.caption(
                "This recommendation is decision support "
                "for human review and does not automatically "
                "control factory machinery."
            )


            st.subheader(
                "Machine Input"
            )

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )


# ============================================================
# QUALITY CONTROL
# ============================================================

st.divider()

st.header(
    "🔍 Quality Control"
)

quality_column = get_quality_column(
    ai_data
)

if quality_column is None:

    st.info(
        "No explicit quality classification column "
        "was found in AI_data.csv."
    )

else:

    quality_values = (
        ai_data[
            quality_column
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    quality_counts = (
        quality_values
        .value_counts()
    )

    st.subheader(
        "Quality Prediction Distribution"
    )

    st.bar_chart(
        quality_counts
    )

    total_quality_records = len(
        quality_values
    )

    defective_keywords = [
        "defective",
        "defect",
        "bad",
        "fail",
        "failed",
        "poor",
        "ng"
    ]

    defective_count = sum(
        quality_values.str.contains(
            keyword,
            regex=False
        ).sum()
        for keyword in defective_keywords
    )

    defective_count = min(
        defective_count,
        total_quality_records
    )

    good_count = (
        total_quality_records
        - defective_count
    )

    quality_col1, quality_col2, quality_col3 = (
        st.columns(3)
    )

    with quality_col1:

        st.metric(
            "Good / Normal",
            good_count
        )

    with quality_col2:

        st.metric(
            "Defective / Risk",
            defective_count
        )

    with quality_col3:

        st.metric(
            "Defect Percentage",
            f"{percentage(defective_count, total_quality_records):.2f}%"
        )


# ============================================================
# QUALITY MODEL EVALUATION
# ============================================================

st.subheader(
    "📊 Quality Model Metrics"
)

quality_target = get_quality_column(
    ai_data
)

quality_prediction_column = find_column(
    ai_data,
    [
        "quality prediction",
        "quality predicted",
        "predicted quality"
    ]
)

if (
    quality_target is not None
    and quality_prediction_column is not None
    and quality_target != quality_prediction_column
):

    actual_quality = (
        ai_data[
            quality_target
        ]
        .astype(str)
        .str.lower()
    )

    predicted_quality = (
        ai_data[
            quality_prediction_column
        ]
        .astype(str)
        .str.lower()
    )

    valid_quality = (
        actual_quality.notna()
        & predicted_quality.notna()
    )

    if valid_quality.sum() > 1:

        actual = actual_quality[
            valid_quality
        ]

        predicted = predicted_quality[
            valid_quality
        ]

        try:

            q_accuracy = accuracy_score(
                actual,
                predicted
            )

            q_precision = precision_score(
                actual,
                predicted,
                average="weighted",
                zero_division=0
            )

            q_recall = recall_score(
                actual,
                predicted,
                average="weighted",
                zero_division=0
            )

            q_f1 = f1_score(
                actual,
                predicted,
                average="weighted",
                zero_division=0
            )

            qc1, qc2, qc3, qc4 = (
                st.columns(4)
            )

            with qc1:
                st.metric(
                    "Accuracy",
                    f"{q_accuracy:.2%}"
                )

            with qc2:
                st.metric(
                    "Precision",
                    f"{q_precision:.2%}"
                )

            with qc3:
                st.metric(
                    "Recall",
                    f"{q_recall:.2%}"
                )

            with qc4:
                st.metric(
                    "F1 Score",
                    f"{q_f1:.2%}"
                )

            st.subheader(
                "Confusion Matrix"
            )

            labels = sorted(
                list(
                    set(actual)
                    | set(predicted)
                )
            )

            matrix = confusion_matrix(
                actual,
                predicted,
                labels=labels
            )

            matrix_df = pd.DataFrame(
                matrix,
                index=[
                    f"Actual: {x}"
                    for x in labels
                ],
                columns=[
                    f"Predicted: {x}"
                    for x in labels
                ]
            )

            st.dataframe(
                matrix_df,
                use_container_width=True
            )

        except Exception as error:

            st.info(
                f"Quality metrics could not be calculated: {error}"
            )

else:

    st.info(
        "Separate actual and predicted quality columns "
        "are required to calculate quality metrics."
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.divider()

st.header(
    "🚨 Anomaly Detection"
)

if ai_data.empty:

    st.info(
        "AI dataset is unavailable."
    )

else:

    available_anomaly_features = [
        column
        for column in ANOMALY_FEATURES
        if column in ai_data.columns
    ]

    if len(
        available_anomaly_features
    ) < len(ANOMALY_FEATURES):

        st.warning(
            "Some anomaly detection features are missing."
        )

        st.write(
            "Required features:"
        )

        for feature in ANOMALY_FEATURES:
            st.write(
                f"• {feature}"
            )

    else:

        try:

            anomaly_input = (
                ai_data[
                    ANOMALY_FEATURES
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )
            )

            valid_anomaly_rows = (
                anomaly_input
                .dropna()
            )

            if len(
                valid_anomaly_rows
            ) > 10:

                anomaly_detector = (
                    IsolationForest(
                        n_estimators=100,
                        contamination=0.05,
                        random_state=42
                    )
                )

                anomaly_predictions = (
                    anomaly_detector
                    .fit_predict(
                        valid_anomaly_rows
                    )
                )

                anomaly_scores = (
                    anomaly_detector
                    .decision_function(
                        valid_anomaly_rows
                    )
                )

                anomaly_result = (
                    ai_data.loc[
                        valid_anomaly_rows.index
                    ].copy()
                )

                anomaly_result[
                    "Anomaly Status"
                ] = np.where(
                    anomaly_predictions == -1,
                    "Anomaly",
                    "Normal"
                )

                anomaly_result[
                    "Anomaly Score"
                ] = anomaly_scores


                anomaly_count = int(
                    (
                        anomaly_predictions
                        == -1
                    ).sum()
                )

                normal_count = int(
                    (
                        anomaly_predictions
                        == 1
                    ).sum()
                )

                total_analyzed = (
                    len(anomaly_predictions)
                )

                anomaly_percentage = (
                    percentage(
                        anomaly_count,
                        total_analyzed
                    )
                )


                anomaly_col1, anomaly_col2, anomaly_col3 = (
                    st.columns(3)
                )

                with anomaly_col1:

                    st.metric(
                        "🚨 Total Anomalies",
                        anomaly_count
                    )

                with anomaly_col2:

                    st.metric(
                        "✅ Normal Machines",
                        normal_count
                    )

                with anomaly_col3:

                    st.metric(
                        "📊 Anomaly Percentage",
                        f"{anomaly_percentage:.2f}%"
                    )


                st.subheader(
                    "Anomaly Distribution"
                )

                anomaly_distribution = (
                    anomaly_result[
                        "Anomaly Status"
                    ]
                    .value_counts()
                )

                st.bar_chart(
                    anomaly_distribution
                )


                st.subheader(
                    "Highest-Risk Anomalies"
                )

                high_risk_anomalies = (
                    anomaly_result
                    .sort_values(
                        "Anomaly Score"
                    )
                    .head(20)
                )

                display_columns = [
                    column
                    for column in [
                        "Product ID",
                        "Type",
                        "Air temperature [K]",
                        "Process temperature [K]",
                        "Rotational speed [rpm]",
                        "Torque [Nm]",
                        "Tool wear [min]",
                        "Anomaly Status",
                        "Anomaly Score"
                    ]
                    if column in high_risk_anomalies.columns
                ]

                st.dataframe(
                    high_risk_anomalies[
                        display_columns
                    ],
                    use_container_width=True,
                    hide_index=True
                )


                st.subheader(
                    "Risk Summary"
                )

                high_risk_anomaly_count = (
                    anomaly_count
                )

                medium_risk_anomaly_count = int(
                    total_analyzed * 0.05
                )

                low_risk_anomaly_count = max(
                    normal_count
                    - medium_risk_anomaly_count,
                    0
                )

                risk_col1, risk_col2, risk_col3 = (
                    st.columns(3)
                )

                with risk_col1:

                    st.metric(
                        "🔴 High Risk",
                        high_risk_anomaly_count
                    )

                with risk_col2:

                    st.metric(
                        "🟡 Medium Risk",
                        medium_risk_anomaly_count
                    )

                with risk_col3:

                    st.metric(
                        "🟢 Low Risk",
                        low_risk_anomaly_count
                    )

            else:

                st.info(
                    "Not enough valid records for anomaly detection."
                )

        except Exception as error:

            st.error(
                f"Anomaly detection error: {error}"
            )


# ============================================================
# AI AGENT COORDINATION
# ============================================================

st.divider()

st.header(
    "🤖 AI Agent Coordination"
)

st.write(
    "Select a machine from the factory dataset. "
    "The AI agents will analyze the selected machine "
    "and produce a coordinated recommendation."
)


if ai_data.empty:

    st.warning(
        "AI dataset is unavailable."
    )

else:

    # --------------------------------------------------------
    # MACHINE SELECTION
    # --------------------------------------------------------

    machine_id_column = find_column(
        ai_data,
        [
            "product id",
            "machine id",
            "machine",
            "id"
        ]
    )

    if machine_id_column is not None:

        machine_options = (
            ai_data[
                machine_id_column
            ]
            .astype(str)
            .tolist()
        )

        selected_machine_id = st.selectbox(
            "🏭 Select Machine / Case",
            machine_options,
            key="agent_machine_selector"
        )

        selected_machine_index = (
            ai_data[
                machine_id_column
            ]
            .astype(str)
            .eq(
                str(selected_machine_id)
            )
            .idxmax()
        )

    else:

        selected_machine_index = st.selectbox(
            "🏭 Select Machine Record",
            ai_data.index.tolist(),
            key="agent_record_selector"
        )


    machine_row = ai_data.loc[
        selected_machine_index
    ]


    # --------------------------------------------------------
    # MACHINE DATA
    # --------------------------------------------------------

    machine_data = {}

    for feature in FEATURES:

        if feature in machine_row.index:

            machine_data[feature] = (
                machine_row[feature]
            )


    if machine_id_column is not None:

        machine_data[
            "Machine ID"
        ] = str(
            machine_row[
                machine_id_column
            ]
        )

    else:

        machine_data[
            "Machine ID"
        ] = str(
            selected_machine_index
        )


    st.subheader(
        "Selected Machine"
    )

    st.dataframe(
        pd.DataFrame(
            [machine_data]
        ),
        use_container_width=True,
        hide_index=True
    )


    run_agents_button = st.button(
        "🤖 Run AI Agent Coordination",
        type="primary",
        use_container_width=True
    )


    if run_agents_button:

        # ----------------------------------------------------
        # DEFAULT VALUES
        # ----------------------------------------------------

        quality_risk = "UNKNOWN"
        anomaly_risk = "UNKNOWN"
        maintenance_risk = "UNKNOWN"
        overall_risk = "UNKNOWN"
        maintenance_probability = 0.0
        recommendation = (
            "Further human review is recommended."
        )


        # ----------------------------------------------------
        # PREDICTIVE MAINTENANCE AGENT
        # ----------------------------------------------------

        if ml_model is not None:

            try:

                prediction_input = pd.DataFrame(
                    [{
                        feature:
                        machine_data.get(
                            feature,
                            0
                        )
                        for feature in FEATURES
                    }]
                )

                prediction = ml_model.predict(
                    prediction_input
                )[0]

                if hasattr(
                    ml_model,
                    "predict_proba"
                ):

                    maintenance_probability = float(
                        ml_model.predict_proba(
                            prediction_input
                        )[0][1]
                    )

                else:

                    maintenance_probability = float(
                        prediction
                    )

                maintenance_risk = risk_level(
                    maintenance_probability
                )

            except Exception:

                maintenance_risk = "UNKNOWN"


        # ----------------------------------------------------
        # ANOMALY AGENT
        # ----------------------------------------------------

        try:

            anomaly_values = pd.DataFrame(
                [{
                    feature:
                    pd.to_numeric(
                        machine_data.get(
                            feature,
                            np.nan
                        ),
                        errors="coerce"
                    )
                    for feature in ANOMALY_FEATURES
                }]
            )

            if not anomaly_values.isnull().any().any():

                anomaly_detector = (
                    IsolationForest(
                        n_estimators=100,
                        contamination=0.05,
                        random_state=42
                    )
                )

                # Fit on complete dataset.
                training_data = (
                    ai_data[
                        ANOMALY_FEATURES
                    ]
                    .apply(
                        pd.to_numeric,
                        errors="coerce"
                    )
                    .dropna()
                )

                if len(
                    training_data
                ) > 10:

                    anomaly_detector.fit(
                        training_data
                    )

                    anomaly_prediction = (
                        anomaly_detector
                        .predict(
                            anomaly_values
                        )[0]
                    )

                    if anomaly_prediction == -1:
                        anomaly_risk = "HIGH"
                    else:
                        anomaly_risk = "LOW"

        except Exception:

            anomaly_risk = "UNKNOWN"


        # ----------------------------------------------------
        # QUALITY AGENT
        # ----------------------------------------------------

        quality_column = (
            get_quality_column(
                ai_data
            )
        )

        if quality_column is not None:

            quality_value = str(
                machine_row[
                    quality_column
                ]
            ).lower()

            if any(
                keyword in quality_value
                for keyword in [
                    "defect",
                    "bad",
                    "fail",
                    "poor",
                    "ng"
                ]
            ):

                quality_risk = "HIGH"

            else:

                quality_risk = "LOW"

        else:

            quality_risk = "UNKNOWN"


        # ----------------------------------------------------
        # EXISTING AI COORDINATOR
        # ----------------------------------------------------

        if COORDINATOR_AVAILABLE:

            try:

                coordinator_result = (
                    run_ai_coordination(
                        machine_data
                    )
                )

                if isinstance(
                    coordinator_result,
                    dict
                ):

                    quality_risk = (
                        coordinator_result.get(
                            "quality_risk",
                            quality_risk
                        )
                    )

                    anomaly_risk = (
                        coordinator_result.get(
                            "anomaly_risk",
                            anomaly_risk
                        )
                    )

                    maintenance_risk = (
                        coordinator_result.get(
                            "maintenance_risk",
                            maintenance_risk
                        )
                    )

                    overall_risk = (
                        coordinator_result.get(
                            "overall_risk",
                            overall_risk
                        )
                    )

                    maintenance_probability = (
                        coordinator_result.get(
                            "maintenance_probability",
                            maintenance_probability
                        )
                    )

                    recommendation = (
                        coordinator_result.get(
                            "recommendation",
                            recommendation
                        )
                    )

            except Exception:

                pass


        # ----------------------------------------------------
        # DECISION COORDINATOR FALLBACK
        # ----------------------------------------------------

        if overall_risk == "UNKNOWN":

            risks = [
                str(quality_risk).upper(),
                str(anomaly_risk).upper(),
                str(maintenance_risk).upper()
            ]

            if "HIGH" in risks:

                overall_risk = "HIGH"

            elif "MEDIUM" in risks:

                overall_risk = "MEDIUM"

            elif "LOW" in risks:

                overall_risk = "LOW"

            else:

                overall_risk = "UNKNOWN"


        if (
            recommendation
            == "Further human review is recommended."
        ):

            if overall_risk == "HIGH":

                recommendation = (
                    "High-risk condition detected. "
                    "Immediate maintenance review "
                    "by a human reviewer is recommended."
                )

            elif overall_risk == "MEDIUM":

                recommendation = (
                    "Medium-risk condition detected. "
                    "Schedule inspection and continue monitoring."
                )

            elif overall_risk == "LOW":

                recommendation = (
                    "Low-risk condition detected. "
                    "Continue routine monitoring."
                )


        # ----------------------------------------------------
        # DISPLAY AGENT RESULTS
        # ----------------------------------------------------

        st.subheader(
            "🧠 AI Agent Results"
        )

        agent_col1, agent_col2, agent_col3 = (
            st.columns(3)
        )

        with agent_col1:

            st.metric(
                "Quality Control Agent",
                str(
                    quality_risk
                )
            )

        with agent_col2:

            st.metric(
                "Anomaly Detection Agent",
                str(
                    anomaly_risk
                )
            )

        with agent_col3:

            st.metric(
                "Predictive Maintenance Agent",
                str(
                    maintenance_risk
                )
            )


        # ----------------------------------------------------
        # DECISION COORDINATOR
        # ----------------------------------------------------

        st.subheader(
            "🎯 Decision Coordinator"
        )

        decision_col1, decision_col2 = (
            st.columns(2)
        )

        with decision_col1:

            st.metric(
                "Overall Risk",
                str(
                    overall_risk
                )
            )

        with decision_col2:

            st.metric(
                "Maintenance Probability",
                f"{float(maintenance_probability) * 100:.2f}%"
            )


        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        st.subheader(
            "💡 AI Recommendation"
        )

        if overall_risk == "HIGH":

            st.error(
                recommendation
            )

        elif overall_risk == "MEDIUM":

            st.warning(
                recommendation
            )

        else:

            st.success(
                recommendation
            )

        st.caption(
            "AI recommendations are decision-support information "
            "and require human review before operational action."
        )


        # ----------------------------------------------------
        # COORDINATION TABLE
        # ----------------------------------------------------

        st.subheader(
            "📋 Agent Coordination Summary"
        )

        coordination_table = pd.DataFrame(
            {
                "AI Agent": [
                    "Quality Control Agent",
                    "Anomaly Detection Agent",
                    "Predictive Maintenance Agent",
                    "Decision Coordinator"
                ],

                "Result": [
                    quality_risk,
                    anomaly_risk,
                    maintenance_risk,
                    overall_risk
                ]
            }
        )

        st.dataframe(
            coordination_table,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # SAVE CURRENT COORDINATION RESULT
        # ----------------------------------------------------

        if st.button(
            "💾 Save AI Decision",
            use_container_width=True
        ):

            try:

                decision_record = {
                    "Case ID":
                        f"AI-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",

                    "Machine ID":
                        machine_data.get(
                            "Machine ID",
                            ""
                        ),

                    "Quality Risk":
                        quality_risk,

                    "Anomaly Risk":
                        anomaly_risk,

                    "Maintenance Risk":
                        maintenance_risk,

                    "Overall Risk":
                        overall_risk,

                    "Maintenance Probability":
                        maintenance_probability,

                    "Recommendation":
                        recommendation,

                    "Decision Time":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                }

                new_decision = pd.DataFrame(
                    [decision_record]
                )

                if os.path.exists(
                    DECISIONS_FILE
                ):

                    existing_decisions = (
                        pd.read_csv(
                            DECISIONS_FILE
                        )
                    )

                    updated_decisions = (
                        pd.concat(
                            [
                                existing_decisions,
                                new_decision
                            ],
                            ignore_index=True
                        )
                    )

                else:

                    updated_decisions = (
                        new_decision
                    )

                updated_decisions.to_csv(
                    DECISIONS_FILE,
                    index=False
                )

                st.success(
                    "AI decision saved successfully."
                )

            except Exception as error:

                st.error(
                    f"Unable to save AI decision: {error}"
                )


        # ----------------------------------------------------
        # SEND TO HUMAN APPROVAL
        # ----------------------------------------------------

        st.subheader(
            "👤 Human-in-the-Loop Approval"
        )

        st.write(
            "Send this AI recommendation to a human reviewer "
            "before any operational decision is made."
        )

        send_for_approval = st.button(
            "📤 Send Recommendation for Human Approval",
            use_container_width=True
        )


        if send_for_approval:

            try:

                approval_record = {}

                # Create a unique case ID.
                approval_record[
                    "Case ID"
                ] = (
                    "CASE-"
                    + datetime.now().strftime(
                        "%Y%m%d%H%M%S%f"
                    )
                )

                for feature in FEATURES:

                    approval_record[
                        feature
                    ] = machine_data.get(
                        feature,
                        ""
                    )

                approval_record[
                    "Machine ID"
                ] = machine_data.get(
                    "Machine ID",
                    ""
                )

                approval_record[
                    "AI Recommendation"
                ] = recommendation

                approval_record[
                    "AI Overall Risk"
                ] = overall_risk

                approval_record[
                    "Maintenance Probability"
                ] = maintenance_probability

                approval_record[
                    "Quality Risk"
                ] = quality_risk

                approval_record[
                    "Anomaly Risk"
                ] = anomaly_risk

                approval_record[
                    "Maintenance Risk"
                ] = maintenance_risk

                approval_record[
                    "Human Decision"
                ] = "Pending Human Review"

                approval_record[
                    "Approval Time"
                ] = ""

                approval_record[
                    "Approved By"
                ] = ""

                approval_record[
                    "Created Time"
                ] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                new_approval = pd.DataFrame(
                    [approval_record]
                )

                if os.path.exists(
                    APPROVAL_FILE
                ):

                    existing_approvals = (
                        pd.read_csv(
                            APPROVAL_FILE
                        )
                    )

                    existing_approvals = (
                        ensure_approval_columns(
                            existing_approvals
                        )
                    )

                    updated_approvals = (
                        pd.concat(
                            [
                                existing_approvals,
                                new_approval
                            ],
                            ignore_index=True
                        )
                    )

                else:

                    updated_approvals = (
                        new_approval
                    )

                updated_approvals = (
                    create_case_ids(
                        updated_approvals
                    )
                )

                if save_approval_data(
                    updated_approvals
                ):

                    st.success(
                        "✅ AI recommendation sent "
                        "to Human Approval Center."
                    )

                    st.rerun()

            except Exception as error:

                st.error(
                    f"Unable to send recommendation: {error}"
                )




# ============================================================
# HUMAN APPROVAL CENTER
# ============================================================

st.divider()

st.header("👤 Human Approval Center")

st.write(
    "Review AI-generated recommendations and record a human "
    "decision before any operational action is taken."
)


# ------------------------------------------------------------
# LOAD APPROVAL DATA
# ------------------------------------------------------------

if os.path.exists(APPROVAL_FILE):

    try:

        approval_data = pd.read_csv(
            APPROVAL_FILE
        )

        approval_data = ensure_approval_columns(
            approval_data
        )

        approval_data = create_case_ids(
            approval_data
        )

        save_approval_data(
            approval_data
        )

    except Exception as error:

        st.error(
            f"Unable to load approval data: {error}"
        )

        approval_data = pd.DataFrame()

else:

    approval_data = pd.DataFrame()


# ------------------------------------------------------------
# APPROVAL SUMMARY
# ------------------------------------------------------------

if not approval_data.empty:

    approval_data[
        "Human Decision"
    ] = approval_data[
        "Human Decision"
    ].fillna(
        "Pending Human Review"
    )

    pending_count = int(
        (
            approval_data[
                "Human Decision"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            == "pending human review"
        ).sum()
    )

    approved_count = int(
        (
            approval_data[
                "Human Decision"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            == "approved"
        ).sum()
    )

    rejected_count = int(
        (
            approval_data[
                "Human Decision"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            == "rejected"
        ).sum()
    )

    skipped_count = int(
        (
            approval_data[
                "Human Decision"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            == "skipped"
        ).sum()
    )

else:

    pending_count = 0
    approved_count = 0
    rejected_count = 0
    skipped_count = 0


summary_col1, summary_col2, summary_col3, summary_col4 = (
    st.columns(4)
)


with summary_col1:

    st.metric(
        "⏳ Pending Reviews",
        pending_count
    )


with summary_col2:

    st.metric(
        "✅ Approved",
        approved_count
    )


with summary_col3:

    st.metric(
        "❌ Rejected",
        rejected_count
    )


with summary_col4:

    st.metric(
        "⏭️ Skipped",
        skipped_count
    )


# ------------------------------------------------------------
# PENDING CASES
# ------------------------------------------------------------

st.subheader(
    "📋 Pending Human Reviews"
)


if approval_data.empty:

    st.info(
        "No human approval cases are currently available."
    )

else:

    pending_mask = (
        approval_data[
            "Human Decision"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        == "pending human review"
    )

    pending_cases = (
        approval_data[
            pending_mask
        ]
        .copy()
    )


    if pending_cases.empty:

        st.success(
            "🎉 There are no pending human reviews."
        )

    else:

        # ----------------------------------------------------
        # CASE SELECTION
        # ----------------------------------------------------

        case_options = (
            pending_cases[
                "Case ID"
            ]
            .astype(str)
            .tolist()
        )

        selected_case_id = st.selectbox(
            "🔎 Select Case to Review",
            case_options,
            key="human_approval_case_selector"
        )


        selected_case_mask = (
            pending_cases[
                "Case ID"
            ]
            .astype(str)
            .eq(
                str(selected_case_id)
            )
        )

        selected_case = (
            pending_cases[
                selected_case_mask
            ]
            .iloc[0]
        )


        # ----------------------------------------------------
        # CASE DETAILS
        # ----------------------------------------------------

        st.subheader(
            "🔍 Selected Case Details"
        )

        detail_col1, detail_col2 = (
            st.columns(2)
        )


        with detail_col1:

            st.write(
                f"**Case ID:** "
                f"{selected_case.get('Case ID', '')}"
            )

            st.write(
                f"**Machine ID:** "
                f"{selected_case.get('Machine ID', '')}"
            )

            st.write(
                f"**Machine Type:** "
                f"{selected_case.get('Type', '')}"
            )

            st.write(
                f"**Created Time:** "
                f"{selected_case.get('Created Time', '')}"
            )


        with detail_col2:

            st.write(
                f"**Overall Risk:** "
                f"{selected_case.get('AI Overall Risk', 'UNKNOWN')}"
            )

            probability = pd.to_numeric(
                selected_case.get(
                    "Maintenance Probability",
                    0
                ),
                errors="coerce"
            )

            if pd.isna(probability):

                probability = 0.0

            st.write(
                f"**Maintenance Probability:** "
                f"{float(probability) * 100:.2f}%"
            )

            st.write(
                f"**Quality Risk:** "
                f"{selected_case.get('Quality Risk', 'UNKNOWN')}"
            )

            st.write(
                f"**Anomaly Risk:** "
                f"{selected_case.get('Anomaly Risk', 'UNKNOWN')}"
            )


        # ----------------------------------------------------
        # MACHINE PARAMETERS
        # ----------------------------------------------------

        st.subheader(
            "⚙️ Machine Parameters"
        )

        parameter_columns = [
            feature
            for feature in FEATURES
            if feature in selected_case.index
        ]

        if parameter_columns:

            parameter_data = pd.DataFrame(
                [
                    {
                        column:
                        selected_case[column]
                        for column in parameter_columns
                    }
                ]
            )

            st.dataframe(
                parameter_data,
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # AI RECOMMENDATION
        # ----------------------------------------------------

        st.subheader(
            "💡 AI Recommendation"
        )

        selected_recommendation = str(
            selected_case.get(
                "AI Recommendation",
                "No recommendation available."
            )
        )

        selected_overall_risk = str(
            selected_case.get(
                "AI Overall Risk",
                "UNKNOWN"
            )
        ).upper()


        if selected_overall_risk == "HIGH":

            st.error(
                selected_recommendation
            )

        elif selected_overall_risk == "MEDIUM":

            st.warning(
                selected_recommendation
            )

        else:

            st.info(
                selected_recommendation
            )


        st.caption(
            "The AI recommendation is decision-support information. "
            "A human reviewer must make the final decision."
        )


        # ----------------------------------------------------
        # REVIEWER INFORMATION
        # ----------------------------------------------------

        st.subheader(
            "👨‍💼 Human Reviewer"
        )

        reviewer_name = st.text_input(
            "Reviewer Name",
            key="approval_reviewer_name"
        )


        # ----------------------------------------------------
        # HUMAN DECISION BUTTONS
        # ----------------------------------------------------

        st.subheader(
            "⚖️ Human Decision"
        )

        approve_col, reject_col, skip_col = (
            st.columns(3)
        )


        # ----------------------------------------------------
        # APPROVE
        # ----------------------------------------------------

        with approve_col:

            approve_button = st.button(
                "✅ Approve",
                type="primary",
                use_container_width=True,
                key="approve_selected_case"
            )


        # ----------------------------------------------------
        # REJECT
        # ----------------------------------------------------

        with reject_col:

            reject_button = st.button(
                "❌ Reject",
                use_container_width=True,
                key="reject_selected_case"
            )


        # ----------------------------------------------------
        # SKIP
        # ----------------------------------------------------

        with skip_col:

            skip_button = st.button(
                "⏭️ Skip",
                use_container_width=True,
                key="skip_selected_case"
            )


        # ----------------------------------------------------
        # PROCESS APPROVE
        # ----------------------------------------------------

        if approve_button:

            if not reviewer_name.strip():

                st.warning(
                    "Please enter the reviewer name before "
                    "approving the case."
                )

            else:

                try:

                    case_mask = (
                        approval_data[
                            "Case ID"
                        ]
                        .astype(str)
                        .eq(
                            str(selected_case_id)
                        )
                    )

                    approval_data.loc[
                        case_mask,
                        "Human Decision"
                    ] = "Approved"

                    approval_data.loc[
                        case_mask,
                        "Approval Time"
                    ] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    approval_data.loc[
                        case_mask,
                        "Approved By"
                    ] = reviewer_name.strip()

                    approval_data = (
                        ensure_approval_columns(
                            approval_data
                        )
                    )

                    if save_approval_data(
                        approval_data
                    ):

                        st.success(
                            f"✅ Case {selected_case_id} "
                            f"approved successfully."
                        )

                        st.rerun()

                except Exception as error:

                    st.error(
                        f"Unable to approve case: {error}"
                    )


        # ----------------------------------------------------
        # PROCESS REJECT
        # ----------------------------------------------------

        if reject_button:

            if not reviewer_name.strip():

                st.warning(
                    "Please enter the reviewer name before "
                    "rejecting the case."
                )

            else:

                try:

                    case_mask = (
                        approval_data[
                            "Case ID"
                        ]
                        .astype(str)
                        .eq(
                            str(selected_case_id)
                        )
                    )

                    approval_data.loc[
                        case_mask,
                        "Human Decision"
                    ] = "Rejected"

                    approval_data.loc[
                        case_mask,
                        "Approval Time"
                    ] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    approval_data.loc[
                        case_mask,
                        "Approved By"
                    ] = reviewer_name.strip()

                    approval_data = (
                        ensure_approval_columns(
                            approval_data
                        )
                    )

                    if save_approval_data(
                        approval_data
                    ):

                        st.success(
                            f"❌ Case {selected_case_id} "
                            f"rejected successfully."
                        )

                        st.rerun()

                except Exception as error:

                    st.error(
                        f"Unable to reject case: {error}"
                    )


        # ----------------------------------------------------
        # PROCESS SKIP
        # ----------------------------------------------------

        if skip_button:

            if not reviewer_name.strip():

                st.warning(
                    "Please enter the reviewer name before "
                    "skipping the case."
                )

            else:

                try:

                    case_mask = (
                        approval_data[
                            "Case ID"
                        ]
                        .astype(str)
                        .eq(
                            str(selected_case_id)
                        )
                    )

                    approval_data.loc[
                        case_mask,
                        "Human Decision"
                    ] = "Skipped"

                    approval_data.loc[
                        case_mask,
                        "Approval Time"
                    ] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    approval_data.loc[
                        case_mask,
                        "Approved By"
                    ] = reviewer_name.strip()

                    approval_data = (
                        ensure_approval_columns(
                            approval_data
                        )
                    )

                    if save_approval_data(
                        approval_data
                    ):

                        st.success(
                            f"⏭️ Case {selected_case_id} "
                            f"skipped successfully."
                        )

                        st.rerun()

                except Exception as error:

                    st.error(
                        f"Unable to skip case: {error}"
                    )


# ============================================================
# HUMAN DECISION HISTORY
# ============================================================

st.divider()

st.header(
    "📜 Human Decision History"
)


if approval_data.empty:

    st.info(
        "No human decision history is available yet."
    )

else:

    history_mask = (
        approval_data[
            "Human Decision"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "approved",
                "rejected",
                "skipped"
            ]
        )
    )

    history_data = (
        approval_data[
            history_mask
        ]
        .copy()
    )


    if history_data.empty:

        st.info(
            "No approved, rejected, or skipped cases yet."
        )

    else:

        history_columns = [
            "Case ID",
            "Machine ID",
            "AI Overall Risk",
            "Maintenance Probability",
            "AI Recommendation",
            "Human Decision",
            "Approval Time",
            "Approved By"
        ]

        history_columns = [
            column
            for column in history_columns
            if column in history_data.columns
        ]

        history_display = (
            history_data[
                history_columns
            ]
            .sort_values(
                by="Approval Time",
                ascending=False,
                na_position="last"
            )
            .copy()
        )


        if (
            "Maintenance Probability"
            in history_display.columns
        ):

            history_display[
                "Maintenance Probability"
            ] = pd.to_numeric(
                history_display[
                    "Maintenance Probability"
                ],
                errors="coerce"
            )

            history_display[
                "Maintenance Probability"
            ] = (
                history_display[
                    "Maintenance Probability"
                ]
                .fillna(0)
                .mul(100)
                .round(2)
                .astype(str)
                + "%"
            )


        st.dataframe(
            history_display,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # DOWNLOAD HISTORY
        # ----------------------------------------------------

        history_csv = (
            history_data.to_csv(
                index=False
            ).encode(
                "utf-8"
            )
        )

        st.download_button(
            "⬇️ Download Human Decision History",
            history_csv,
            file_name="human_decision_history.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# HUMAN DECISION SUMMARY
# ============================================================

st.divider()

st.header(
    "📊 Human Decision Summary"
)

if not approvals.empty:

    if "Human Decision" in approvals.columns:

        decisions_series = (
            approvals[
                "Human Decision"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        approved_count = int(
            decisions_series.eq(
                "approved"
            ).sum()
        )

        rejected_count = int(
            decisions_series.eq(
                "rejected"
            ).sum()
        )

        skipped_count = int(
            decisions_series.eq(
                "skipped"
            ).sum()
        )

        pending_count = int(
            decisions_series.eq(
                "pending human review"
            ).sum()
        )

        summary_col1, summary_col2, summary_col3, summary_col4 = (
            st.columns(4)
        )

        with summary_col1:

            st.metric(
                "✅ Approved",
                approved_count
            )

        with summary_col2:

            st.metric(
                "❌ Rejected",
                rejected_count
            )

        with summary_col3:

            st.metric(
                "⏭️ Skipped",
                skipped_count
            )

        with summary_col4:

            st.metric(
                "⏳ Pending",
                pending_count
            )


        decision_summary = (
            approvals[
                "Human Decision"
            ]
            .astype(str)
            .value_counts()
        )

        st.subheader(
            "Decision Distribution"
        )

        st.bar_chart(
            decision_summary
        )

else:

    st.info(
        "No human approval data available."
    )


# ============================================================
# AI DECISION RECORDS
# ============================================================

st.divider()

st.header(
    "🧾 AI Decision Records"
)

latest_decisions = load_csv(
    DECISIONS_FILE
)

if not latest_decisions.empty:

    st.dataframe(
        latest_decisions,
        use_container_width=True,
        height=400,
        hide_index=True
    )

else:

    st.info(
        "No AI decision records available yet."
    )


# ============================================================
# PREDICTIVE MAINTENANCE MODEL INFORMATION
# ============================================================

st.divider()

st.header(
    "🧠 Predictive Maintenance Model"
)

if ml_model is not None:

    st.success(
        "🟢 Predictive maintenance model loaded successfully."
    )

    model_col1, model_col2 = (
        st.columns(2)
    )

    with model_col1:

        st.write(
            "**Model File**"
        )

        st.code(
            "factory_ai_model.pkl"
        )

        st.write(
            "**Algorithm**"
        )

        st.write(
            "Random Forest Classifier"
        )

    with model_col2:

        st.write(
            "**Prediction Target**"
        )

        st.write(
            "Machine Failure"
        )

        st.write(
            "**Prediction Type**"
        )

        st.write(
            "Binary Classification"
        )

else:

    st.error(
        "Predictive maintenance model is unavailable."
    )


st.subheader(
    "Model Input Features"
)

for feature in FEATURES:

    st.write(
        f"• {feature}"
    )


# ============================================================
# AI MODEL EVALUATION
# ============================================================

st.divider()

st.header(
    "📈 AI Model Evaluation"
)


# ============================================================
# STANDARD EVALUATION
# ============================================================

st.subheader(
    "📊 Standard AI Evaluation"
)

if evaluation.empty:

    st.info(
        "Standard AI evaluation results are not available yet."
    )

else:

    st.success(
        "AI evaluation data loaded successfully."
    )

    st.dataframe(
        evaluation,
        use_container_width=True,
        hide_index=True
    )

    numeric_columns = (
        evaluation
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    if numeric_columns:

        st.subheader(
            "Evaluation Metrics"
        )

        metric_count = min(
            len(numeric_columns),
            4
        )

        metric_columns = st.columns(
            metric_count
        )

        for index, column in enumerate(
            numeric_columns[:4]
        ):

            try:

                value = float(
                    evaluation[
                        column
                    ].iloc[0]
                )

                with metric_columns[index]:

                    st.metric(
                        column,
                        f"{value:.4f}"
                    )

            except Exception:
                pass


# ============================================================
# RIGOROUS EVALUATION
# ============================================================

st.subheader(
    "🧪 Rigorous AI Model Evaluation"
)

if rigorous_evaluation.empty:

    st.info(
        "Rigorous evaluation results are not available yet."
    )

else:

    st.success(
        "Rigorous evaluation results loaded successfully."
    )

    st.dataframe(
        rigorous_evaluation,
        use_container_width=True,
        hide_index=True
    )

    rigorous_numeric_columns = (
        rigorous_evaluation
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    if rigorous_numeric_columns:

        st.subheader(
            "Rigorous Metrics"
        )

        metric_count = min(
            len(rigorous_numeric_columns),
            4
        )

        metric_columns = st.columns(
            metric_count
        )

        for index, column in enumerate(
            rigorous_numeric_columns[:4]
        ):

            try:

                value = float(
                    rigorous_evaluation[
                        column
                    ].iloc[0]
                )

                with metric_columns[index]:

                    st.metric(
                        column,
                        f"{value:.4f}"
                    )

            except Exception:
                pass


# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

st.header(
    "📂 Factory AI Dataset"
)

if os.path.exists(
    AI_DATA_FILE
):

    try:

        factory_data = pd.read_csv(
            AI_DATA_FILE
        )

        dataset_col1, dataset_col2, dataset_col3 = (
            st.columns(3)
        )

        with dataset_col1:

            st.metric(
                "Rows",
                len(factory_data)
            )

        with dataset_col2:

            st.metric(
                "Columns",
                len(factory_data.columns)
            )

        with dataset_col3:

            st.metric(
                "Missing Values",
                int(
                    factory_data
                    .isnull()
                    .sum()
                    .sum()
                )
            )


        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            factory_data.head(10),
            use_container_width=True,
            hide_index=True
        )


    except Exception as error:

        st.error(
            f"Unable to load dataset: {error}"
        )

else:

    st.error(
        "❌ AI_data.csv was not found."
    )

    st.code(
        "Expected location:\n"
        "Factory AI/data/AI_data.csv"
    )


# ============================================================
# SYSTEM FILE STATUS
# ============================================================

st.divider()

st.header(
    "⚙️ Factory AI System Status"
)

file_status = {
    "AI Dataset":
        AI_DATA_FILE,

    "Predictive Maintenance Model":
        MODEL_FILE,

    "AI Decisions":
        DECISIONS_FILE,

    "Human Approvals":
        APPROVAL_FILE,

    "Standard Evaluation":
        EVALUATION_FILE,

    "Rigorous Evaluation":
        RIGOROUS_EVALUATION_FILE
}


for file_name, file_path in file_status.items():

    if os.path.exists(
        file_path
    ):

        try:

            file_size = (
                os.path.getsize(
                    file_path
                )
                / 1024
            )

            st.success(
                f"✅ {file_name}: Available "
                f"({file_size:.1f} KB)"
            )

        except Exception:

            st.success(
                f"✅ {file_name}: Available"
            )

    else:

        st.warning(
            f"⚠️ {file_name}: Not Found"
        )


# ============================================================
# AI MODULE STATUS
# ============================================================

st.subheader(
    "🤖 AI Module Status"
)

status_col1, status_col2, status_col3 = (
    st.columns(3)
)

with status_col1:

    if ml_model is not None:

        st.success(
            "🟢 Predictive Maintenance\n\nONLINE"
        )

    else:

        st.error(
            "🔴 Predictive Maintenance\n\nOFFLINE"
        )


with status_col2:

    if not ai_data.empty:

        st.success(
            "🟢 Factory Data Analysis\n\nONLINE"
        )

    else:

        st.error(
            "🔴 Factory Data Analysis\n\nOFFLINE"
        )


with status_col3:

    if COORDINATOR_AVAILABLE:

        st.success(
            "🟢 AI Agent Coordinator\n\nONLINE"
        )

    else:

        st.warning(
            "🟡 AI Agent Coordinator\n\nFALLBACK"
        )


# ============================================================
# HUMAN-IN-THE-LOOP WORKFLOW
# ============================================================

st.divider()

st.header(
    "🔄 Human-in-the-Loop AI Workflow"
)

workflow = pd.DataFrame(
    {
        "Stage": [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7"
        ],

        "Process": [
            "Factory Data",
            "AI Agents",
            "Risk Analysis",
            "AI Recommendation",
            "Human Approval Center",
            "Approve / Reject / Skip",
            "Decision Recorded & Evaluated"
        ],

        "Status": [
            "ACTIVE" if not ai_data.empty else "WAITING",
            "ACTIVE",
            "ACTIVE",
            "ACTIVE",
            "ACTIVE",
            "ACTIVE",
            "ACTIVE"
        ]
    }
)

st.dataframe(
    workflow,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SYSTEM SUMMARY
# ============================================================

st.divider()

st.header(
    "📋 Factory AI System Summary"
)

summary_col1, summary_col2 = (
    st.columns(2)
)

with summary_col1:

    st.subheader(
        "🤖 AI Capabilities"
    )

    st.write(
        "✅ Factory Data Analysis"
    )

    st.write(
        "✅ Quality Control"
    )

    st.write(
        "✅ Anomaly Detection"
    )

    st.write(
        "✅ Predictive Maintenance"
    )

    st.write(
        "✅ Machine Failure Prediction"
    )

    st.write(
        "✅ Multi-Agent Coordination"
    )


with summary_col2:

    st.subheader(
        "👤 Decision Support"
    )

    st.write(
        "✅ AI Recommendations"
    )

    st.write(
        "✅ Human Approval"
    )

    st.write(
        "✅ Approve / Reject / Skip"
    )

    st.write(
        "✅ Decision Tracking"
    )

    st.write(
        "✅ Model Evaluation"
    )

    st.write(
        "✅ Human-in-the-Loop Workflow"
    )


# ============================================================
# FINAL PROJECT ARCHITECTURE
# ============================================================

st.divider()

st.header(
    "🏭 Factory AI Architecture"
)

st.code(
    """
Factory Dataset
       ↓
Data Analysis
       ↓
 ┌───────────────┬────────────────┬────────────────────┐
 ↓               ↓                ↓
Quality Agent   Anomaly Agent   Predictive Maintenance
 ↓               ↓                ↓
 └───────────────┴────────────────┘
                  ↓
          Decision Coordinator
                  ↓
            Overall Risk
                  ↓
          AI Recommendation
                  ↓
        Human Approval Center
                  ↓
       ┌──────────┼──────────┐
       ↓          ↓          ↓
    APPROVE     REJECT      SKIP
       ↓          ↓          ↓
       └──────────┼──────────┘
                  ↓
          Decision Recorded
                  ↓
          AI Model Evaluation
    """,
    language="text"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏭 Factory AI | Smart Manufacturing Quality Control "
    "& Predictive Maintenance System"
)

st.caption(
    "AI provides decision-support recommendations. "
    "Human review remains part of the operational workflow."
)