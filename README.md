# 🏭 Factory AI

## Smart Manufacturing Quality Control & Predictive Maintenance System

Factory AI is an AI-powered smart manufacturing system designed to analyze factory machine data, detect anomalies, predict maintenance risks, evaluate product quality, and support human decision-making.

The system combines machine learning, anomaly detection, AI agent coordination, and human-in-the-loop approval into a single Streamlit dashboard.

---

## 🎯 Problem Statement

Modern manufacturing systems generate large amounts of machine and production data.

Identifying machine failures, abnormal operating conditions, and product-quality problems early can help support better maintenance planning and quality monitoring.

Factory AI analyzes machine parameters such as temperature, rotational speed, torque, and tool wear to provide AI-assisted insights.

---

## 💡 Solution

Factory AI provides a centralized dashboard that performs:

- Predictive maintenance
- Quality control
- Anomaly detection
- AI agent coordination
- Risk analysis
- Human approval
- Model evaluation
- Factory data analysis

The system is designed as a **decision-support system**. AI recommendations require human review before operational action.

---

## 🤖 AI Agent Architecture

The system uses multiple AI agents:

### 1. Quality Control Agent
Analyzes product-quality information and identifies potential quality risks.

### 2. Anomaly Detection Agent
Identifies unusual machine operating conditions using anomaly detection.

### 3. Predictive Maintenance Agent
Estimates machine maintenance/failure risk using machine-learning predictions.

### 4. Decision Coordinator
Combines the outputs of the AI agents and produces an overall risk assessment and recommendation.

### Workflow

Factory Data  
↓  
Quality Control Agent  
↓  
Anomaly Detection Agent  
↓  
Predictive Maintenance Agent  
↓  
Decision Coordinator  
↓  
Overall Risk  
↓  
AI Recommendation  
↓  
Human Approval  
↓  
Approved / Rejected / Skipped  
↓  
Decision History & Evaluation

---

## 📊 Dataset

The system uses manufacturing machine data containing parameters such as:

- Product ID
- Product Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Target
- Failure Type

The primary dataset is located at:

`data/AI_data.csv`

---

## 🔧 Predictive Maintenance

The predictive maintenance component analyzes machine operating parameters and estimates maintenance risk.

Risk levels:

- **LOW:** Continue routine monitoring
- **MEDIUM:** Schedule inspection and continue monitoring
- **HIGH:** Immediate maintenance review recommended

The system provides recommendations for human decision-making rather than automatically controlling factory equipment.

---

## 🚨 Anomaly Detection

The anomaly detection component uses machine operating parameters to identify unusual conditions.

Analyzed parameters include:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The system reports:

- Total anomalies
- Normal machines
- Anomaly rate
- Risk levels
- High-risk machines
- Anomaly distribution

---

## ✅ Quality Control

The quality-control component evaluates manufacturing quality using machine-learning classification.

The dashboard reports:

- Quality accuracy
- Precision
- Recall
- F1 score
- False positives
- False negatives
- Quality predictions
- Confusion matrix
- Classification results

---

## 👤 Human-in-the-Loop

Factory AI includes a Human Approval Center.

AI recommendations are sent to human reviewers before an operational decision is made.

Reviewers can:

- ✅ Approve
- ❌ Reject
- ⏭️ Skip

Each decision records:

- Case ID
- Machine ID
- Human decision
- Reviewer
- Approval time

A decision-history table provides an audit trail of reviewed cases.

---

## 📈 Model Evaluation

The dashboard includes AI model evaluation metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score
- False Positive
- False Negative
- Anomaly Detection Accuracy
- Anomaly Rate
- High-Risk Machine Count

The project also includes a rigorous evaluation section using a separate test dataset.

---

## 📊 Current Evaluation Results

The dashboard evaluates the models using the available factory dataset and test data.

### Standard Evaluation

- Total Records: 10,000
- Quality Accuracy: 0.93
- Quality Precision: 0.93
- Quality Recall: 0.93
- Quality F1 Score: 0.93
- Anomaly Detection Accuracy: 0.93
- Anomaly Rate: 0.05
- High-Risk Machines: 315

### Rigorous Evaluation

The rigorous evaluation uses a separate test dataset containing 2,000 records.

The dashboard reports quality and predictive-maintenance evaluation metrics along with false-positive and false-negative results.

---

## 🖥️ Dashboard Modules

The Streamlit dashboard contains:

1. Factory Overview
2. Data Analysis
3. Predictive Maintenance
4. Quality Control
5. Anomaly Detection
6. AI Agent Coordination
7. Human Approval Center
8. AI Model Evaluation
9. Dataset Preview
10. System Status

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Machine Learning
- Isolation Forest
- Random Forest
- AI Agent Coordination
- GitHub

---

## 📁 Project Structure

```text
Factory-AI/
│
├── dashboard.py
├── factory_ai_coordinator.py
├── train_model.py
├── factory_ai_model.pkl
├── agent_coordinator.py
├── quality.py
├── anomaly.py
├── pm.py
├── evaluation.py
├── human_approval.py
├── check.py
│
└── data/
    └── AI_data.csv
