import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# CLASSIFICATION FAIRNESS METRICS
# ============================================================

def demographic_parity_difference(y_pred, sensitive_feature):
    #Checks whether different sensitive groups receive positive predictions at similar rates.
    data = pd.DataFrame({
        "prediction": y_pred,
        "group": sensitive_feature
    })

    positive_rates = {}
    for group in data["group"].unique():
        group_data = data[data["group"] == group]
        positive_rate = np.mean(group_data["prediction"] == 1)
        positive_rates[group] = positive_rate

    if len(positive_rates) < 2:
        return 0.0

    return min(positive_rates.values()) - max(positive_rates.values())


def disparate_impact(y_pred, sensitive_feature):
    #Instead of calculating the difference, this calculates the ratio of positive prediction rates.
    data = pd.DataFrame({
        "prediction": y_pred,
        "group": sensitive_feature
    })

    positive_rates = {}

    for group in data["group"].unique():
        group_data = data[data["group"] == group]
        positive_rate = np.mean(group_data["prediction"] == 1)
        positive_rates[group] = positive_rate

    if len(positive_rates) < 2:
        return 1.0

    max_rate = max(positive_rates.values())
    min_rate = min(positive_rates.values())

    if max_rate == 0:
        return 1.0

    return min_rate / max_rate


def equal_opportunity_difference(y_true, y_pred, sensitive_feature):
    #Among the people who actually belong to the positive class, how many did the model correctly identify?
    data = pd.DataFrame({
        "actual": y_true,
        "prediction": y_pred,
        "group": sensitive_feature
    })

    tpr_values = {}

    for group in data["group"].unique():

        group_data = data[data["group"] == group]

        actual_positive = (group_data["actual"] == 1).sum()

        if actual_positive == 0:
            tpr = 0.0
        else:
            true_positive = (
                (group_data["actual"] == 1) &
                (group_data["prediction"] == 1)
            ).sum()

            tpr = true_positive / actual_positive

        tpr_values[group] = tpr

    if len(tpr_values) < 2:
        return 0.0

    return min(tpr_values.values()) - max(tpr_values.values())


def group_accuracy(y_true, y_pred, sensitive_feature):
    #Instead of calculating one overall accuracy, it calculates accuracy for every sensitive group.
    data = pd.DataFrame({
        "actual": y_true,
        "prediction": y_pred,
        "group": sensitive_feature
    })

    results = {}

    for group in data["group"].unique():

        group_data = data[data["group"] == group]

        results[group] = accuracy_score(
            group_data["actual"],
            group_data["prediction"]
        )

    return results


def classification_fairness_report(
        #This function combines all the classification metrics.
    y_true,
    y_pred,
    sensitive_feature
):

    return {
        "Demographic Parity Difference":
            demographic_parity_difference(
                y_pred,
                sensitive_feature
            ),

        "Disparate Impact":
            disparate_impact(
                y_pred,
                sensitive_feature
            ),

        "Equal Opportunity Difference":
            equal_opportunity_difference(
                y_true,
                y_pred,
                sensitive_feature
            ),

        "Group Accuracy":
            group_accuracy(
                y_true,
                y_pred,
                sensitive_feature
            )
    }


# ============================================================
# REGRESSION FAIRNESS METRICS
# ============================================================

def group_mae(y_true, y_pred, sensitive_feature):
    #Calculates Mean Absolute Error separately for each sensitive group.
    data = pd.DataFrame({
        "actual": y_true,
        "prediction": y_pred,
        "group": sensitive_feature
    })

    results = {}

    for group in data["group"].unique():

        group_data = data[data["group"] == group]

        results[group] = mean_absolute_error(
            group_data["actual"],
            group_data["prediction"]
        )

    return results


def group_rmse(y_true, y_pred, sensitive_feature):
    #similar as group_mae()
    data = pd.DataFrame({
        "actual": y_true,
        "prediction": y_pred,
        "group": sensitive_feature
    })

    results = {}

    for group in data["group"].unique():

        group_data = data[data["group"] == group]

        results[group] = np.sqrt(
            mean_squared_error(
                group_data["actual"],
                group_data["prediction"]
            )
        )

    return results


def mean_prediction_difference(y_pred, sensitive_feature):
    #Checks whether the model's average predicted values differ between groups.
    data = pd.DataFrame({
        "prediction": y_pred,
        "group": sensitive_feature
    })

    group_means = data.groupby("group")["prediction"].mean()

    if len(group_means) < 2:
        return 0.0

    return group_means.min() - group_means.max()


def regression_fairness_report(
    y_true,
    y_pred,
    sensitive_feature
):

    return {
        "Group MAE":
            group_mae(
                y_true,
                y_pred,
                sensitive_feature
            ),

        "Group RMSE":
            group_rmse(
                y_true,
                y_pred,
                sensitive_feature
            ),

        "Mean Prediction Difference":
            mean_prediction_difference(
                y_pred,
                sensitive_feature
            )
    }


# ============================================================
# BIAS DETECTION
# ============================================================

def detect_classification_bias(report):

    bias_detected = False
    reasons = []

    dp = abs(
        report["Demographic Parity Difference"]
    )

    di = report["Disparate Impact"]

    eo = abs(
        report["Equal Opportunity Difference"]
    )

    if dp > 0.10:
        bias_detected = True
        reasons.append(
            "Significant difference in positive prediction rates."
        )

    if di < 0.80:
        bias_detected = True
        reasons.append(
            "Disparate impact indicates unequal treatment between groups."
        )

    if eo > 0.10:
        bias_detected = True
        reasons.append(
            "Significant difference in true positive rates."
        )

    return {
        "bias_detected": bias_detected,
        "reasons": reasons
    }


def detect_regression_bias(report):

    bias_detected = False
    reasons = []

    mae_values = list(
        report["Group MAE"].values()
    )

    rmse_values = list(
        report["Group RMSE"].values()
    )

    if len(mae_values) >= 2:

        mae_difference = (
            max(mae_values) -
            min(mae_values)
        )

        if mae_difference > 0.10:
            bias_detected = True
            reasons.append(
                "Prediction error differs significantly between groups."
            )

    if len(rmse_values) >= 2:

        rmse_difference = (
            max(rmse_values) -
            min(rmse_values)
        )

        if rmse_difference > 0.10:
            bias_detected = True
            reasons.append(
                "RMSE differs significantly between groups."
            )

    return {
        "bias_detected": bias_detected,
        "reasons": reasons
    }


# ============================================================
# MULTIPLE SENSITIVE FEATURES
# ============================================================

def evaluate_fairness(
    y_true,
    y_pred,
    dataframe,
    sensitive_features,
    task_type
):
    """
    Evaluate fairness for multiple sensitive features.
    Example:

        sensitive_features = [
            "gender",
            "age_group",
            "education"
        ]
    Each sensitive feature is evaluated independently.
    """

    results = {}

    for feature in sensitive_features:
        # Get one sensitive feature
        sensitive_feature = dataframe[feature]
        # Classification
        if task_type == "classification":
            report = classification_fairness_report(
                y_true,
                y_pred,
                sensitive_feature
            )
            bias_result = detect_classification_bias(
                report
            )
        # Regression
        elif task_type == "regression":
            report = regression_fairness_report(
                y_true,
                y_pred,
                sensitive_feature
            )
            bias_result = detect_regression_bias(
                report
            )
        else:
            raise ValueError(
                "task_type must be 'classification' "
                "or 'regression'"
            )
        results[feature] = {
            "metrics": report,
            "bias_detection": bias_result
        }

    return results