# FairLens — Bias Auditing Toolkit

FairLens is an end-to-end machine learning bias auditing toolkit designed to help users identify, understand, and mitigate fairness issues in machine learning models.

The platform allows users to upload a dataset, preprocess the data, train machine learning models, evaluate model performance, analyze fairness across sensitive groups, apply bias mitigation techniques, and understand model predictions using explainability techniques.

---

## Features

### 1. Dataset Upload & Preprocessing

- Upload CSV datasets
- Preview uploaded data
- Automatically analyze dataset structure
- Select the target variable
- Select one or more sensitive attributes
- Handle missing values
- Encode categorical variables
- Scale numerical features
- Train/test data splitting

### 2. Automatic Task Detection

FairLens automatically determines whether the selected target represents:

- Classification
- Regression

The appropriate evaluation and fairness metrics are then selected based on the task.

### 3. Machine Learning Model Training

FairLens supports multiple machine learning algorithms.

#### Classification

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

#### Regression

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

Users can compare models based on their performance and fairness characteristics.

### 4. Fairness Auditing

FairLens evaluates model predictions across different sensitive groups.

#### Classification Metrics

- Demographic Parity Difference
- Disparate Impact
- Equal Opportunity Difference
- Group Accuracy
- Equalized Odds

#### Regression Metrics

- Mean Prediction Difference
- Group MAE
- Group RMSE

These metrics help identify whether a model behaves differently for different demographic groups.

### 5. Bias Mitigation

FairLens provides mitigation techniques to reduce detected unfairness.

Implemented techniques include:

- Reweighing
- Exponentiated Gradient
- Threshold-based mitigation where applicable

The toolkit allows users to compare model fairness before and after mitigation.

### 6. Explainable AI

FairLens uses SHAP (SHapley Additive exPlanations) to help users understand model predictions.

The explainability module provides:

- Global feature importance
- SHAP summary plots
- Local explanations
- Feature contribution analysis

This helps users understand which features have the greatest influence on model predictions.

### 7. Fairness Recommendation Engine

FairLens analyzes the detected fairness problems and provides recommendations for appropriate mitigation strategies.

The recommendation system considers:

- Group imbalance
- Detected fairness metrics
- Model behavior
- Classification or regression task

### 8. Model Saving

After evaluating the available models, users can select the model they want to keep.

FairLens supports saving the selected model for future use.

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │      User           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  React Frontend     │
                    │  Vite + JavaScript  │
                    └──────────┬──────────┘
                               │
                         REST API
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Flask Backend     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
       │Preprocessing│  │Model Training│ │Fairness Engine│
       └─────────────┘  └─────────────┘  └──────────────┘
                               │                │
                               ▼                ▼
                        ┌─────────────┐  ┌─────────────┐
                        │ Evaluation  │  │ Mitigation  │
                        └─────────────┘  └─────────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │   SHAP      │
                        │Explainability│
                        └─────────────┘
```


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bhoomiy/FairLens.git
cd FairLens
```

### 2. Backend setup

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Start the Flask server:

```powershell
python app.py
```

### 3. Frontend setup

Open a **new terminal**:

```powershell
npm install
npm run dev
```

Then open the Vite URL shown in the terminal, usually:

```text
http://localhost:5173
```

### 4. Backend

The Flask API will normally run at:

```text
http://127.0.0.1:5000
```

---

