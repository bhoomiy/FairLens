import numpy as np
import pandas as pd
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.preprocessing import LabelEncoder
from fairlearn.postprocessing import ThresholdOptimizer


def calculate_reweighing_weights(
    sensitive_feature,
    target
):
    """
    Calculate instance weights for reweighing.

    The weights are based on the relationship between
    sensitive groups and target values.
    """

    data = pd.DataFrame({
        "sensitive": sensitive_feature,
        "target": target
    })

    # Probability of each sensitive group
    group_probability = (
        data["sensitive"]
        .value_counts(normalize=True)
    )

    # Probability of each target value
    target_probability = (
        data["target"]
        .value_counts(normalize=True)
    )

    # Joint probability of group + target
    joint_probability = (
        data.groupby(
            ["sensitive", "target"]
        )
        .size()
        .div(len(data))
    )

    weights = []

    for _, row in data.iterrows():

        group = row["sensitive"]
        target_value = row["target"]

        joint = joint_probability.get(
            (group, target_value),
            0
        )

        if joint == 0:
            weight = 1.0
        else:
            weight = (
                group_probability[group]
                * target_probability[target_value]
            ) / joint

        weights.append(weight)

    return np.array(weights)

def apply_reweighing(
    model,
    X_train,
    y_train,
    sensitive_feature
):
    """
    Retrain a model using reweighing.
    """

    # Calculate sample weights
    weights = calculate_reweighing_weights(
        sensitive_feature,
        y_train
    )

    # Train model using the calculated weights
    model.fit(
        X_train,
        y_train,
        sample_weight=weights
    )

    return model


def apply_exponentiated_gradient(
    model,
    X_train,
    y_train,
    sensitive_feature
):
    """
    Apply Exponentiated Gradient fairness mitigation
    for binary classification.
    """

    # Convert target labels to 0/1
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    # Exponentiated Gradient currently expects
    # binary labels: 0 and 1
    if len(label_encoder.classes_) != 2:

        raise ValueError(
            "Exponentiated Gradient currently supports "
            "binary classification only. "
            f"Detected {len(label_encoder.classes_)} classes."
        )

    mitigator = ExponentiatedGradient(
        estimator=model,
        constraints=DemographicParity()
    )

    mitigator.fit(
        X_train,
        y_encoded,
        sensitive_features=sensitive_feature
    )

    return mitigator, label_encoder


def apply_threshold_optimizer(
    model,
    X_train,
    y_train,
    sensitive_feature
):
    """
    Apply Threshold Optimizer for binary classification.
    """

    # Encode target labels to 0/1
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    # Threshold Optimizer supports binary classification
    if len(label_encoder.classes_) != 2:
        raise ValueError(
            "Threshold Optimizer currently supports "
            "binary classification only. "
            f"Detected {len(label_encoder.classes_)} classes."
        )

    # Create Threshold Optimizer
    threshold_optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity",
        prefit=True
    )

    # Fit using encoded labels
    threshold_optimizer.fit(
        X_train,
        y_encoded,
        sensitive_features=sensitive_feature
    )

    return threshold_optimizer, label_encoder