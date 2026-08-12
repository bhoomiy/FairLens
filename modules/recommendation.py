def detect_group_imbalance(
    dataframe,
    sensitive_feature,
    threshold=0.20
):
    """
    Detect whether a sensitive feature has a significantly
    underrepresented group.

    threshold = minimum acceptable proportion for a group.
    """

    proportions = (
        dataframe[sensitive_feature]
        .value_counts(normalize=True)
    )

    if len(proportions) < 2:
        return False, proportions

    imbalance = proportions.min() < threshold

    return imbalance, proportions


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommend_classification_strategy(
    metrics,
    bias_result,
    group_imbalance=False,
    can_retrain=True,
    predictions_only=False
):
    """
    Recommend a suitable mitigation strategy
    for classification bias.
    """

    # If only predictions are available,
    # post-processing is the most suitable option.
    if predictions_only:
        return {
            "strategy": "Threshold Optimizer",
            "reason": (
                "Only predictions are available, so a "
                "post-processing strategy can be used "
                "without retraining the model."
            )
        }

    # If the protected group is underrepresented,
    # recommend reweighing.
    if group_imbalance:
        return {
            "strategy": "Reweighing",
            "reason": (
                "The protected group is underrepresented "
                "in the dataset."
            )
        }

    # If equal opportunity is violated and
    # retraining is possible.
    if abs(
        metrics["Equal Opportunity Difference"]
    ) > 0.10 and can_retrain:

        return {
            "strategy": "Exponentiated Gradient",
            "reason": (
                "There is a significant difference in "
                "true positive rates between groups. "
                "Since the model can be retrained, "
                "Exponentiated Gradient can optimize "
                "fairness constraints during training."
            )
        }

    # If demographic parity is violated.
    if abs(
        metrics["Demographic Parity Difference"]
    ) > 0.10:

        if can_retrain:
            return {
                "strategy": "Exponentiated Gradient",
                "reason": (
                    "Positive prediction rates differ "
                    "significantly between groups. "
                    "Retraining allows fairness constraints "
                    "to be incorporated into the model."
                )
            }

        return {
            "strategy": "Threshold Optimizer",
            "reason": (
                "Positive prediction rates differ "
                "between groups, so prediction thresholds "
                "can be adjusted to improve fairness."
            )
        }

    # Default classification recommendation.
    return {
        "strategy": "Reweighing",
        "reason": (
            "Reweighing can reduce the influence of "
            "representation differences during model training."
        )
    }


# ============================================================
# REGRESSION RECOMMENDATION
# ============================================================

def recommend_regression_strategy(
    metrics,
    bias_result,
    group_imbalance=False,
    can_retrain=True,
    predictions_only=False
):
    """
    Recommend a suitable mitigation strategy
    for regression bias.
    """

    # Threshold optimization is mainly designed
    # for classification, so for regression we don't
    # recommend it.
    if predictions_only:
        return {
            "strategy": "Reweighing",
            "reason": (
                "Only predictions are available. For this "
                "regression task, reweighing is the most "
                "appropriate available strategy when "
                "retraining becomes possible."
            )
        }

    # Dataset imbalance.
    if group_imbalance:
        return {
            "strategy": "Reweighing",
            "reason": (
                "The protected group is underrepresented "
                "in the dataset, so reweighing can give "
                "greater importance to that group during training."
            )
        }

    # Check whether group MAE differs significantly.
    mae_values = list(
        metrics["Group MAE"].values()
    )

    if len(mae_values) >= 2:

        mae_difference = (
            max(mae_values) -
            min(mae_values)
        )

        if mae_difference > 0.10:

            if can_retrain:
                return {
                    "strategy": "Reweighing",
                    "reason": (
                        "Prediction error differs significantly "
                        "between protected groups. Reweighing "
                        "can give greater importance to "
                        "underperforming groups during training."
                    )
                }

    # Check prediction difference.
    prediction_difference = abs(
        metrics["Mean Prediction Difference"]
    )

    if prediction_difference > 0.10:

        return {
            "strategy": "Reweighing",
            "reason": (
                "Average predictions differ significantly "
                "between protected groups. Reweighing can "
                "help reduce this disparity during training."
            )
        }

    # Default regression recommendation.
    return {
        "strategy": "Reweighing",
        "reason": (
            "Reweighing is a suitable training-time strategy "
            "for reducing group-related differences in regression."
        )
    }


# ============================================================
# MAIN RECOMMENDATION FUNCTION
# ============================================================

def recommend_mitigation(
    feature_result,
    task_type,
    group_imbalance=False,
    can_retrain=True,
    predictions_only=False
):
    """
    Generate a mitigation recommendation for one
    sensitive feature.
    """

    metrics = feature_result["metrics"]
    bias_result = feature_result["bias_detection"]

    # No bias detected.
    if not bias_result["bias_detected"]:
        return {
            "bias_detected": False,
            "strategy": None,
            "reason": "No significant bias was detected."
        }

    # Classification.
    if task_type == "classification":

        recommendation = recommend_classification_strategy(
            metrics,
            bias_result,
            group_imbalance,
            can_retrain,
            predictions_only
        )

    # Regression.
    elif task_type == "regression":

        recommendation = recommend_regression_strategy(
            metrics,
            bias_result,
            group_imbalance,
            can_retrain,
            predictions_only
        )

    else:
        raise ValueError(
            "task_type must be 'classification' "
            "or 'regression'"
        )

    return {
        "bias_detected": True,
        "strategy": recommendation["strategy"],
        "reason": recommendation["reason"]
    }


# ============================================================
# MULTIPLE SENSITIVE FEATURES
# ============================================================

def generate_recommendations(
    fairness_results,
    sensitive_dataframe,
    task_type,
    can_retrain=True,
    predictions_only=False
):
    """
    Generate recommendations for all sensitive features
    and automatically detect group imbalance.
    """

    recommendations = {}

    for feature, feature_result in fairness_results.items():

        # Automatically detect group imbalance
        group_imbalance, proportions = detect_group_imbalance(
            sensitive_dataframe,
            feature
        )

        # Generate recommendation
        recommendation = recommend_mitigation(
            feature_result,
            task_type,
            group_imbalance=group_imbalance,
            can_retrain=can_retrain,
            predictions_only=predictions_only
        )

        # Store imbalance information
        recommendation["group_imbalance"] = group_imbalance

        # Store group distribution
        recommendation["group_distribution"] = proportions

        # Store recommendation for this feature
        recommendations[feature] = recommendation

    return recommendations