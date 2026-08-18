import shap
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


# ============================================================
# GET SHAP EXPLAINER
# ============================================================

def get_shap_explainer(model, X_background):

    tree_models = (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        XGBClassifier,
        XGBRegressor
    )

    linear_models = (
        LogisticRegression,
        LinearRegression
    )

    if isinstance(model, tree_models):
        return shap.TreeExplainer(model)

    elif isinstance(model, linear_models):
        return shap.LinearExplainer(
            model,
            X_background
        )

    else:
        return shap.Explainer(
            model,
            X_background
        )


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

def calculate_shap_values(model, X_background, X_test):

    explainer = get_shap_explainer(
        model,
        X_background
    )

    shap_values = explainer(X_test)

    return explainer, shap_values


# ============================================================
# CLEAN SHAP VALUES
# ============================================================

def clean_shap_values(values):

    values = np.asarray(values, dtype=float)

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return values


# ============================================================
# SHAP FEATURE IMPORTANCE
# ============================================================

def get_shap_feature_importance(
    shap_values,
    feature_names
):

    values = clean_shap_values(
        shap_values.values
    )

    # Multi-output classification
    if values.ndim == 3:

        values = np.mean(
            np.abs(values),
            axis=2
        )

    else:

        values = np.abs(values)

    importance = np.mean(
        values,
        axis=0
    )

    importance = clean_shap_values(
        importance
    )

    importance_df = pd.DataFrame({
        "Feature": list(feature_names),
        "SHAP Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="SHAP Importance",
        ascending=False
    )

    return importance_df.reset_index(drop=True)


# ============================================================
# MAIN EXPLAINABILITY FUNCTION
# ============================================================

def explain_model(
    model,
    X_train,
    X_test
):

    explainer, shap_values = calculate_shap_values(
        model,
        X_train,
        X_test
    )

    feature_importance = get_shap_feature_importance(
        shap_values,
        X_test.columns
    )

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "feature_importance": feature_importance
    }