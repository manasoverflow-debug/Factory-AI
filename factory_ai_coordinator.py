import os
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest


# ============================================================
# FACTORY AI COORDINATOR
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


# ============================================================
# SENSOR FEATURES
# ============================================================

SENSOR_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# ============================================================
# LOAD PREDICTIVE MAINTENANCE MODEL
# ============================================================

def load_predictive_model():

    if not os.path.exists(MODEL_FILE):
        return None

    try:

        return joblib.load(
            MODEL_FILE
        )

    except Exception:

        return None


# ============================================================
# CREATE ISOLATION FOREST MODEL
# ============================================================

def create_anomaly_model():

    if not os.path.exists(AI_DATA_FILE):
        return None

    try:

        df = pd.read_csv(
            AI_DATA_FILE
        )

        # Check that all required sensor columns exist
        missing_columns = [
            column
            for column in SENSOR_FEATURES
            if column not in df.columns
        ]

        if missing_columns:
            return None

        X = df[SENSOR_FEATURES].copy()

        # Remove invalid rows
        X = X.dropna()

        if X.empty:
            return None

        model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )

        model.fit(X)

        return model

    except Exception:

        return None


# ============================================================
# LOAD ANOMALY MODEL
# ============================================================

def load_anomaly_model():

    return create_anomaly_model()


# ============================================================
# PREDICTIVE MAINTENANCE AGENT
# ============================================================

def predictive_maintenance_agent(
    machine_data
):

    model = load_predictive_model()

    if model is None:

        return {
            "risk": "UNKNOWN",
            "probability": 0.0
        }

    try:

        input_data = pd.DataFrame(
            [machine_data]
        )

        probability = model.predict_proba(
            input_data
        )[0][1]

        probability = float(
            probability
        )

        # Risk thresholds
        if probability >= 0.70:

            risk = "HIGH"

        elif probability >= 0.30:

            risk = "MEDIUM"

        else:

            risk = "LOW"

        return {
            "risk": risk,
            "probability": probability
        }

    except Exception:

        return {
            "risk": "UNKNOWN",
            "probability": 0.0
        }


# ============================================================
# QUALITY CONTROL AGENT
# ============================================================

def quality_control_agent(
    machine_data
):

    try:

        torque = float(
            machine_data[
                "Torque [Nm]"
            ]
        )

        tool_wear = float(
            machine_data[
                "Tool wear [min]"
            ]
        )

        # Quality risk rules
        if (
            torque >= 70
            or tool_wear >= 200
        ):

            return "HIGH"

        elif (
            torque >= 55
            or tool_wear >= 150
        ):

            return "MEDIUM"

        else:

            return "LOW"

    except Exception:

        return "UNKNOWN"


# ============================================================
# ANOMALY DETECTION AGENT
# ============================================================

def anomaly_detection_agent(
    machine_data
):

    anomaly_model = load_anomaly_model()

    if anomaly_model is None:

        return {
            "risk": "UNKNOWN",
            "label": "Unknown"
        }

    try:

        input_data = pd.DataFrame(
            [machine_data]
        )

        # Isolation Forest prediction
        prediction = anomaly_model.predict(
            input_data[SENSOR_FEATURES]
        )[0]

        # Isolation Forest score
        anomaly_score = anomaly_model.decision_function(
            input_data[SENSOR_FEATURES]
        )[0]

        anomaly_score = float(
            anomaly_score
        )

        if prediction == -1:

            label = "Anomaly"
            risk = "HIGH"

        else:

            label = "Normal"

            # A normal observation with a relatively
            # low decision score is treated as medium risk.
            if anomaly_score < 0.05:

                risk = "MEDIUM"

            else:

                risk = "LOW"

        return {
            "risk": risk,
            "label": label,
            "score": anomaly_score
        }

    except Exception:

        return {
            "risk": "UNKNOWN",
            "label": "Unknown",
            "score": 0.0
        }


# ============================================================
# DECISION COORDINATOR
# ============================================================

def coordinate_decision(
    quality_risk,
    anomaly_risk,
    maintenance_risk
):

    risks = [
        quality_risk,
        anomaly_risk,
        maintenance_risk
    ]

    high_count = risks.count(
        "HIGH"
    )

    medium_count = risks.count(
        "MEDIUM"
    )

    unknown_count = risks.count(
        "UNKNOWN"
    )


    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    if high_count >= 2:

        overall_risk = "HIGH"

        recommendation = (
            "HIGH PRIORITY: Hold potentially "
            "defective production and perform "
            "immediate machine inspection."
        )


    elif high_count == 1:

        overall_risk = "HIGH"

        recommendation = (
            "HIGH PRIORITY: Schedule maintenance "
            "and closely monitor the machine."
        )


    # --------------------------------------------------------
    # MEDIUM RISK
    # --------------------------------------------------------

    elif medium_count >= 2:

        overall_risk = "MEDIUM"

        recommendation = (
            "MEDIUM PRIORITY: Increase machine "
            "monitoring and inspect operating "
            "conditions."
        )


    elif medium_count == 1:

        overall_risk = "MEDIUM"

        recommendation = (
            "MEDIUM PRIORITY: Continue production "
            "with increased monitoring."
        )


    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    elif unknown_count > 0:

        overall_risk = "UNKNOWN"

        recommendation = (
            "AI analysis is incomplete. "
            "Human inspection is recommended "
            "before making a maintenance decision."
        )


    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    else:

        overall_risk = "LOW"

        recommendation = (
            "LOW PRIORITY: Machine operating "
            "within expected conditions. "
            "Continue production and routine monitoring."
        )


    return {
        "overall_risk": overall_risk,
        "recommendation": recommendation
    }


# ============================================================
# COMPLETE AI COORDINATION PIPELINE
# ============================================================

def run_ai_coordination(
    machine_data
):

    # --------------------------------------------------------
    # Validate machine data
    # --------------------------------------------------------

    if not isinstance(
        machine_data,
        dict
    ):

        return {
            "quality_risk": "UNKNOWN",
            "anomaly_risk": "UNKNOWN",
            "maintenance_risk": "UNKNOWN",
            "maintenance_probability": 0.0,
            "anomaly_label": "Unknown",
            "anomaly_score": 0.0,
            "overall_risk": "UNKNOWN",
            "recommendation": (
                "Invalid machine data. "
                "Human inspection required."
            )
        }


    # --------------------------------------------------------
    # AGENT 1
    # Quality Control
    # --------------------------------------------------------

    quality_risk = (
        quality_control_agent(
            machine_data
        )
    )


    # --------------------------------------------------------
    # AGENT 2
    # Anomaly Detection
    # --------------------------------------------------------

    anomaly_result = (
        anomaly_detection_agent(
            machine_data
        )
    )


    anomaly_risk = (
        anomaly_result["risk"]
    )


    # --------------------------------------------------------
    # AGENT 3
    # Predictive Maintenance
    # --------------------------------------------------------

    maintenance_result = (
        predictive_maintenance_agent(
            machine_data
        )
    )


    maintenance_risk = (
        maintenance_result["risk"]
    )


    maintenance_probability = (
        maintenance_result[
            "probability"
        ]
    )


    # --------------------------------------------------------
    # AGENT 4
    # DECISION COORDINATOR
    # --------------------------------------------------------

    decision = coordinate_decision(
        quality_risk,
        anomaly_risk,
        maintenance_risk
    )


    # --------------------------------------------------------
    # FINAL COORDINATED RESULT
    # --------------------------------------------------------

    return {

        # Agent results
        "quality_risk":
            quality_risk,

        "anomaly_risk":
            anomaly_risk,

        "maintenance_risk":
            maintenance_risk,

        # Anomaly information
        "anomaly_label":
            anomaly_result[
                "label"
            ],

        "anomaly_score":
            anomaly_result[
                "score"
            ],

        # Predictive maintenance
        "maintenance_probability":
            maintenance_probability,

        # Final decision
        "overall_risk":
            decision[
                "overall_risk"
            ],

        "recommendation":
            decision[
                "recommendation"
            ]
    }