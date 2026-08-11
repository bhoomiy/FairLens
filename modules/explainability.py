import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import (LogisticRegression,LinearRegression)
from sklearn.tree import (DecisionTreeClassifier,DecisionTreeRegressor)
from sklearn.ensemble import (RandomForestClassifier,RandomForestRegressor)
from xgboost import (XGBClassifier,XGBRegressor)

# ============================================================
# GET SHAP EXPLAINER
# ============================================================

def get_shap_explainer(model, X_background):
    """
    Selects an appropriate SHAP explainer based on the model.
    """

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
    """
    Calculates SHAP values for the test dataset.
    """
    explainer = get_shap_explainer(
        model,
        X_background
    )
    shap_values = explainer(X_test)
    return explainer, shap_values


# ============================================================
# SHAP FEATURE IMPORTANCE
# ============================================================

def get_shap_feature_importance(
    shap_values,
    feature_names
):
    """
    Calculates mean absolute SHAP importance
    for every feature.
    """

    values = shap_values.values
    # Handle classification models where SHAP
    # may return multiple output dimensions.
    if values.ndim == 3:
        values = np.mean(
            np.abs(values),
            axis=2
        )

    importance = np.mean(
        np.abs(values),
        axis=0
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="SHAP Importance",
        ascending=False
    )

    return importance_df.reset_index(drop=True)


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

def create_shap_summary_plot(
    shap_values,
    X_test
):
    """
    Creates a SHAP summary plot.
    """
    values = shap_values

    # Handle multi-output classification
    if values.values.ndim == 3:

        values = shap.Explanation(
            values=np.mean(
                values.values,
                axis=2
            ),
            base_values=values.base_values,
            data=values.data,
            feature_names=values.feature_names
        )

    fig, ax = plt.subplots()

    shap.summary_plot(
        values,
        X_test,
        show=False
    )

    plt.tight_layout()

    return fig


# ============================================================
# SHAP LOCAL EXPLANATION
# ============================================================

def create_local_explanation(
    shap_values,
    X_test,
    sample_index
):
    """
    Creates a SHAP waterfall plot for one prediction.
    """

    sample = shap_values[sample_index]

    # Handle multi-output classification
    if sample.values.ndim > 1:

        sample_values = np.mean(
            sample.values,
            axis=1
        )

        base_value = np.mean(
            sample.base_values
        )

        sample = shap.Explanation(
            values=sample_values,
            base_values=base_value,
            data=sample.data,
            feature_names=sample.feature_names
        )

    fig = plt.figure()

    shap.plots.waterfall(
        sample,
        show=False
    )

    plt.tight_layout()

    return fig


# ============================================================
# MAIN EXPLAINABILITY FUNCTION
# ============================================================

def explain_model(
    model,
    X_train,
    X_test
):
    """
    Main function for model explainability.

    Returns:
        SHAP explainer
        SHAP values
        feature importance
    """

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