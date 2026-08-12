import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error

from modules.dataloader import upload_dataset
from modules.preprocessing import (detect_missing_values,handle_missing_values,remove_duplicates,encode_categorical,scale_features,
                                    split_dataset)
from modules.model_training import (build_decision_tree_classifier,build_decision_tree_regressor,build_linear_regression,build_logistic_regression,
                                    build_random_forest_classifier,build_random_forest_regressor,build_xgboost_classifier,build_xgboost_regressor,
                                    detect_task_type)
from modules.evaluation import evaluate_classification_models,evaluate_regression_models
from modules.bias_detection import evaluate_fairness
from modules.explainability import explain_model,create_shap_summary_plot,create_local_explanation
from modules.recommendation import generate_recommendations
from modules.bias_mitigation import (calculate_reweighing_weights,apply_reweighing,apply_exponentiated_gradient,apply_threshold_optimizer)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

st.set_page_config(page_title="FairLens", layout="wide")

st.title("FairLens: Bias Auditing Toolkit")


# Upload Dataset
df = upload_dataset()

if df is not None:

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select Target & Sensitive Feature
    target = st.selectbox(
        "Select Target Column",
        df.columns
    )

    sensitive = st.multiselect(
        "Select Sensitive Feature",
        [col for col in df.columns if col != target]
    )

    
    # Preprocessing Options
    st.subheader("Preprocessing")

    missing_strategy = st.selectbox(
        "Missing Value Strategy",
        ["Mean", "Median", "Mode", "Drop Rows"]
    )

    encoding_method = st.selectbox(
        "Categorical Encoding",
        ["Label Encoding", "One-Hot Encoding"]
    )

    scaling_method = st.selectbox(
        "Feature Scaling",
        ["None", "StandardScaler", "MinMaxScaler"]
    )

    test_size = st.slider(
        "Test Size",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )

    # Preprocess Button
    if st.button("Preprocess Dataset"):

        # Show missing values before handling
        st.subheader("Missing Values Before Handling")
        st.dataframe(detect_missing_values(df))

        # Handle missing values
        strategy_map = {
            "Mean": "mean",
            "Median": "median",
            "Mode": "mode",
            "Drop Rows": "drop"
        }

        df = handle_missing_values(
            df,
            strategy_map[missing_strategy]
        )

        # Show missing values after handling
        st.subheader("Missing Values After Handling")
        st.dataframe(detect_missing_values(df))

        # Remove duplicates
        before = len(df)
        df = remove_duplicates(df)
        st.session_state["processed_df"] = df.copy()
        after = len(df)

        st.write(f"Duplicate rows removed: {before-after}")

        # Split features & target
        X = df.drop(columns=[target])
        y = df[target]

        #detect task type
        detected_task, detection_reason = detect_task_type(y)
        st.session_state["detected_task"] = detected_task
        st.session_state["detection_reason"] = detection_reason

        # Encode categorical columns
        encoding_map = {
            "Label Encoding": "label",
            "One-Hot Encoding": "one-hot"
        }
        X, encoders = encode_categorical(X,strategy=encoding_map[encoding_method])

        # Scale features
        scale_map = {
            "None": "none",
            "StandardScaler": "standard",
            "MinMaxScaler": "minmax"
        }

        X, scaler = scale_features(
            X,
            scale_map[scaling_method]
        )

        # Train-test split
        X_train, X_test, y_train, y_test = split_dataset(
            X,
            y,
            test_size
        )

        st.success("Preprocessing Completed Successfully!")

        st.subheader("Preprocessing Summary")
        st.write(f"Rows after preprocessing: {len(df)}")
        st.write(f"Features after encoding: {X.shape[1]}")
        st.write(f"Training Samples: {X_train.shape[0]}")
        st.write(f"Testing Samples: {X_test.shape[0]}")
        st.write(f"Sensitive Attributes: {', '.join(sensitive)}")

        # Store for later modules
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["sensitive_features"] = sensitive

        # Store sensitive feature values for training data
        sensitive_train = df.loc[
            X_train.index,
            sensitive
        ]

        st.session_state["sensitive_train"] = sensitive_train

    # Model Training
    if "X_train" in st.session_state and "detected_task" in st.session_state:
        st.header("Model Training")
        detected_task = st.session_state["detected_task"]
        detection_reason = st.session_state["detection_reason"]
        st.info(
            f"FairLens detected: "
            f"**{detected_task.capitalize()}**"
        )
        st.caption(f"Reason: {detection_reason}")
        task_type = detected_task

        if task_type == "classification":

            algorithms = {
                "Logistic Regression": build_logistic_regression,
                "Decision Tree Classifier": build_decision_tree_classifier,
                "Random Forest Classifier": build_random_forest_classifier,
                "XGBoost Classifier": build_xgboost_classifier
            }

        else:

            algorithms = {
                "Linear Regression": build_linear_regression,
                "Decision Tree Regressor": build_decision_tree_regressor,
                "Random Forest Regressor": build_random_forest_regressor,
                "XGBoost Regressor": build_xgboost_regressor
            }

        selected_algorithm = st.selectbox(
            "Choose Algorithm",
            list(algorithms.keys())
        )

        if st.button("Train Model"):

            selected_function = algorithms[selected_algorithm]

            model = selected_function(
                st.session_state["X_train"],
                st.session_state["y_train"]
            )

            st.session_state["trained_model"] = model
            st.session_state["model_name"] = selected_algorithm
            st.session_state["task_type"] = task_type

            st.success(
                f"{selected_algorithm} trained successfully!"
            )

        # Model Evaluation
        if ("trained_model" in st.session_state and "X_test" in st.session_state and "y_test" in st.session_state):
            st.header("Model Evaluation")
            model = st.session_state["trained_model"]
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]
            task_type = st.session_state["task_type"]

            if st.button("Evaluate Model"):
                if task_type == "classification":
                    results = evaluate_classification_models(
                        model,
                        X_test,
                        y_test
                    )

                    st.session_state["evaluation_results"] = results
                    st.success(
                        "Classification evaluation completed successfully!"
                    )
                    # ------------------------------------------
                    # Main Metrics
                    # ------------------------------------------

                    st.subheader("Classification Performance")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Accuracy", f"{results['accuracy']:.4f}")
                    col2.metric("Precision",f"{results['precision']:.4f}")
                    col3.metric("Recall",f"{results['recall_score']:.4f}")
                    col4.metric("F1 Score", f"{results['f1_score']:.4f}")
                    if results["ROC-AUC"] is not None:
                        col5.metric(
                            "ROC-AUC",
                            f"{results['ROC-AUC']:.4f}"
                        )
                    else:
                        col5.metric(
                            "ROC-AUC",
                            "N/A"
                        )
                    # ------------------------------------------
                    # Confusion Matrix + Classification Report
                    # ------------------------------------------
                    left, right = st.columns(2)
                    # ==========================================
                    # LEFT: Confusion Matrix
                    # ==========================================
                    with left:
                        st.subheader("Confusion Matrix")
                        cm = results["confusion_matrix"]
                        labels = results["Labels"]
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            cbar=False,
                            xticklabels=labels,
                            yticklabels=labels,
                            annot_kws={"size": 11},
                            ax=ax
                        )

                        ax.set_xlabel("Predicted",fontsize=9)
                        ax.set_ylabel( "Actual",fontsize=9)
                        ax.set_title("Confusion Matrix",fontsize=11)
                        ax.tick_params(axis="both",labelsize=8)
                        st.pyplot(
                            fig,
                            use_container_width=False
                        )
                        plt.close(fig)

                    # ==========================================
                    # RIGHT: Classification Report
                    # ==========================================

                    with right:

                        st.subheader("Classification Report")

                        report = results["classification_report"]

                        report_df = pd.DataFrame(report).T

                        # Keep only useful columns
                        report_df = report_df[
                            ["precision", "recall", "f1-score", "support"]
                        ]

                        # Rename columns for display
                        report_df.columns = [
                            "Precision",
                            "Recall",
                            "F1 Score",
                            "Support"
                        ]

                        # Format values
                        report_df["Precision"] = report_df[
                            "Precision"
                        ].apply(
                            lambda x: f"{x:.3f}"
                        )

                        report_df["Recall"] = report_df[
                            "Recall"
                        ].apply(
                            lambda x: f"{x:.3f}"
                        )

                        report_df["F1 Score"] = report_df[
                            "F1 Score"
                        ].apply(
                            lambda x: f"{x:.3f}"
                        )

                        report_df["Support"] = report_df[
                            "Support"
                        ].astype(int)

                        st.dataframe(
                            report_df,
                            use_container_width=True
                        )
                # -----------------------------
                # Regression Evaluation
                # -----------------------------

                else:
                    results = evaluate_regression_models( model,X_test,y_test)
                    st.session_state["evaluation_results"] = results
                    st.success("Regression evaluation completed successfully!")
                    st.subheader("Regression Performance")
                    metric_names = [
                        "mae",
                        "mse",
                        "rmse",
                        "r2_score"
                    ]
                    cols = st.columns(
                        len(metric_names)
                    )

                    for col, metric in zip(cols,metric_names):
                        col.metric(
                            label=metric,
                            value=f"{float(results[metric]):.4f}"
                        )

        # ============================================================
        # FAIRNESS EVALUATION
        # ============================================================

        if (
            "trained_model" in st.session_state
            and "X_test" in st.session_state
            and "y_test" in st.session_state
            and "processed_df" in st.session_state
            and "sensitive_features" in st.session_state
        ):

            st.header("Fairness Evaluation")

            # Get stored data
            model = st.session_state["trained_model"]
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]
            processed_df = st.session_state["processed_df"]
            sensitive_features = st.session_state["sensitive_features"]
            task_type = st.session_state["task_type"]

            # Check whether sensitive features were selected
            if not sensitive_features:

                st.warning(
                    "No sensitive features selected. "
                    "Please select at least one sensitive feature."
                )

            else:

                st.write(
                    "**Sensitive Features:**",
                    ", ".join(sensitive_features)
                )

                if st.button("Evaluate Fairness"):

                    # ==================================================
                    # Generate Predictions
                    # ==================================================

                    y_pred = model.predict(X_test)

                    # ==================================================
                    # Get Original Sensitive Feature Values
                    # ==================================================

                    sensitive_test = processed_df.loc[
                        X_test.index,
                        sensitive_features
                    ]

                    st.session_state["sensitive_test"] = sensitive_test

                    # ==================================================
                    # Run Fairness Engine
                    # ==================================================

                    fairness_results = evaluate_fairness(
                        y_true=y_test,
                        y_pred=y_pred,
                        dataframe=sensitive_test,
                        sensitive_features=sensitive_features,
                        task_type=task_type
                    )

                    # Store results
                    st.session_state["fairness_results"] = fairness_results

                    st.success(
                        "Fairness evaluation completed successfully!"
                    )

                    # ==================================================
                    # DISPLAY RESULTS FOR EACH SENSITIVE FEATURE
                    # ==================================================

                    for feature, result in fairness_results.items():

                        st.subheader(
                            f"Fairness Analysis: {feature}"
                        )

                        metrics = result["metrics"]
                        bias_result = result["bias_detection"]

                        # ==================================================
                        # CLASSIFICATION
                        # ==================================================

                        if task_type == "classification":

                            st.write("### Fairness Metrics")

                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "Demographic Parity Difference",
                                f"{metrics['Demographic Parity Difference']:.4f}"
                            )

                            col2.metric(
                                "Disparate Impact",
                                f"{metrics['Disparate Impact']:.4f}"
                            )

                            col3.metric(
                                "Equal Opportunity Difference",
                                f"{metrics['Equal Opportunity Difference']:.4f}"
                            )

                            # ----------------------------------------------
                            # Accuracy by Group
                            # ----------------------------------------------

                            st.write("### Accuracy by Group")

                            accuracy_data = []

                            for group, accuracy in metrics[
                                "Group Accuracy"
                            ].items():

                                accuracy_data.append({
                                    "Group": group,
                                    "Accuracy": round(accuracy, 4)
                                })

                            accuracy_df = pd.DataFrame(
                                accuracy_data
                            )

                            st.dataframe(
                                accuracy_df,
                                use_container_width=True
                            )

                        # ==================================================
                        # REGRESSION
                        # ==================================================

                        else:

                            st.write("### Fairness Metrics")

                            col1, col2, col3 = st.columns(3)

                            # ----------------------------------------------
                            # Mean Prediction Difference
                            # ----------------------------------------------

                            col1.metric(
                                "Mean Prediction Difference",
                                f"{metrics['Mean Prediction Difference']:.4f}"
                            )

                            # ----------------------------------------------
                            # MAE
                            # ----------------------------------------------

                            mae_values = list(
                                metrics["Group MAE"].values()
                            )

                            if len(mae_values) >= 2:

                                mae_difference = (
                                    max(mae_values)
                                    - min(mae_values)
                                )

                            else:

                                mae_difference = 0.0

                            col2.metric(
                                "MAE Difference",
                                f"{mae_difference:.4f}"
                            )

                            # ----------------------------------------------
                            # RMSE
                            # ----------------------------------------------

                            rmse_values = list(
                                metrics["Group RMSE"].values()
                            )

                            if len(rmse_values) >= 2:

                                rmse_difference = (
                                    max(rmse_values)
                                    - min(rmse_values)
                                )

                            else:

                                rmse_difference = 0.0

                            col3.metric(
                                "RMSE Difference",
                                f"{rmse_difference:.4f}"
                            )

                            # ----------------------------------------------
                            # Error by Group
                            # ----------------------------------------------

                            st.write("### Error by Group")

                            regression_data = []

                            groups = metrics["Group MAE"].keys()

                            for group in groups:

                                regression_data.append({
                                    "Group": group,
                                    "MAE": round(
                                        metrics["Group MAE"][group],
                                        4
                                    ),
                                    "RMSE": round(
                                        metrics["Group RMSE"][group],
                                        4
                                    )
                                })

                            regression_df = pd.DataFrame(
                                regression_data
                            )

                            st.dataframe(
                                regression_df,
                                use_container_width=True
                            )

                        # ==================================================
                        # BIAS DETECTION RESULT
                        # ==================================================

                        st.write("### Bias Detection")

                        if bias_result["bias_detected"]:

                            st.error(
                                "Potential Bias Detected"
                            )

                            st.write("**Reasons:**")

                            for reason in bias_result["reasons"]:

                                st.write(
                                    f"- {reason}"
                                )

                        else:

                            st.success(
                                "No Significant Bias Detected"
                            )

                        st.divider()

        # ============================================================
        # EXPLAINABILITY
        # ============================================================

        if (
            "trained_model" in st.session_state
            and "X_train" in st.session_state
            and "X_test" in st.session_state
        ):

            st.header("Model Explainability")

            model = st.session_state["trained_model"]
            X_train = st.session_state["X_train"]
            X_test = st.session_state["X_test"]

            if st.button("Generate SHAP Explanation"):

                with st.spinner("Generating SHAP explanations..."):

                    try:

                        explanation_results = explain_model(
                            model,
                            X_train,
                            X_test
                        )

                        st.session_state[
                            "explanation_results"
                        ] = explanation_results

                        st.success(
                            "SHAP explanation generated successfully!"
                        )

                    except Exception as e:

                        st.error(
                            f"Unable to generate SHAP explanation: {e}"
                        )

            # ========================================================
            # DISPLAY EXPLANATIONS
            # ========================================================

            if "explanation_results" in st.session_state:

                results = st.session_state[
                    "explanation_results"
                ]

                shap_values = results["shap_values"]
                feature_importance = results[
                    "feature_importance"
                ]

                # ====================================================
                # SHAP FEATURE IMPORTANCE
                # ====================================================

                st.subheader("SHAP Feature Importance")

                st.dataframe(
                    feature_importance,
                    use_container_width=True
                )

                # ====================================================
                # SHAP SUMMARY PLOT
                # ====================================================

                st.subheader("SHAP Summary Plot")

                try:

                    summary_fig = create_shap_summary_plot(
                        shap_values,
                        X_test
                    )

                    st.pyplot(
                        summary_fig,
                        use_container_width=True
                    )

                    plt.close(summary_fig)

                except Exception as e:

                    st.error(
                        f"Unable to generate SHAP summary plot: {e}"
                    )

                # ====================================================
                # LOCAL EXPLANATION
                # ====================================================

                st.subheader("SHAP Local Explanation")

                sample_index = st.number_input(
                    "Select test sample",
                    min_value=0,
                    max_value=len(X_test) - 1,
                    value=0,
                    step=1
                )

                if st.button("Explain Selected Prediction"):

                    try:

                        local_fig = create_local_explanation(
                            shap_values,
                            X_test,
                            int(sample_index)
                        )

                        st.pyplot(
                            local_fig,
                            use_container_width=True
                        )

                        plt.close(local_fig)

                    except Exception as e:

                        st.error(
                            f"Unable to generate local explanation: {e}"
                        )

        # ============================================================
        # RECOMMENDATION ENGINE
        # ============================================================

        if "fairness_results" in st.session_state:

            fairness_results = st.session_state["fairness_results"]
            
            task_type = st.session_state["task_type"]

            sensitive_test = st.session_state["sensitive_test"]
            st.session_state["sensitive_test"] = sensitive_test

            recommendations = generate_recommendations(
                fairness_results,
                sensitive_test,
                task_type,
                can_retrain=True,
                predictions_only=False
            )

            # Store recommendations
            st.session_state["recommendations"] = recommendations

            st.header("Recommendation Engine")

            for feature, recommendation in recommendations.items():

                st.subheader(
                    f"Sensitive Feature: {feature}"
                )

                # -------------------------------
                # Group Distribution
                # -------------------------------

                st.markdown("### Group Distribution")

                distribution = recommendation[
                    "group_distribution"
                ]

                distribution_df = pd.DataFrame({
                    "Group": distribution.index,
                    "Proportion": [
                        f"{value * 100:.2f}%"
                        for value in distribution.values
                    ]
                })

                st.dataframe(
                    distribution_df,
                    use_container_width=True
                )

                # -------------------------------
                # Prediction Bias
                # -------------------------------

                if recommendation["bias_detected"]:

                    st.error("Bias detected.")

                    st.markdown("### Recommended Strategy")

                    st.success(
                        f"✓ {recommendation['strategy']}"
                    )

                    st.markdown("### Reason")

                    st.write(
                        recommendation["reason"]
                    )

                # -------------------------------
                # Dataset Imbalance
                # -------------------------------

                elif recommendation["group_imbalance"]:

                    st.warning(
                        "Dataset imbalance detected."
                    )

                    st.markdown("### Recommended Strategy")

                    st.success("✓ Reweighing")

                    st.markdown("### Reason")

                    st.write(
                        "One or more sensitive groups are "
                        "underrepresented in the dataset."
                    )

                else:

                    st.success(
                        "No significant prediction bias "
                        "or group imbalance detected."
                    )
        # ============================================================
        # BIAS MITIGATION
        # ============================================================

        st.header("Bias Mitigation")

        if "recommendations" in st.session_state:

            recommendations = st.session_state["recommendations"]

            # Find features where mitigation is recommended
            mitigation_features = []

            for feature, recommendation in recommendations.items():

                if (
                    recommendation.get("group_imbalance", False)
                    or recommendation.get("bias_detected", False)
                ):
                    mitigation_features.append(feature)

            if mitigation_features:

                st.warning(
                    "Bias mitigation is recommended for the following "
                    "sensitive feature(s):"
                )

                st.write(
                    ", ".join(mitigation_features)
                )

                selected_feature = st.selectbox(
                    "Select sensitive feature for mitigation",
                    mitigation_features
                )

                task_type = st.session_state["task_type"]

                # --------------------------------------------------------
                # Classification
                # --------------------------------------------------------

                if task_type == "classification":

                    mitigation_method = st.selectbox(
                        "Select Mitigation Method",
                        [
                            "Reweighing",
                            "Exponentiated Gradient",
                            "Threshold Optimizer"
                        ]
                    )

                    # ====================================================
                    # REWEIGHING
                    # ====================================================

                    if mitigation_method == "Reweighing":

                        if st.button("Apply Reweighing"):

                            try:

                                from sklearn.base import clone

                                model = st.session_state[
                                    "trained_model"
                                ]

                                X_train = st.session_state[
                                    "X_train"
                                ]

                                y_train = st.session_state[
                                    "y_train"
                                ]

                                X_test = st.session_state[
                                    "X_test"
                                ]

                                sensitive_train = st.session_state[
                                    "sensitive_train"
                                ]

                                sensitive_values = (
                                    sensitive_train[
                                        selected_feature
                                    ]
                                )

                                # Create separate copy
                                mitigated_model = clone(model)

                                # Apply Reweighing
                                mitigated_model = apply_reweighing(
                                    mitigated_model,
                                    X_train,
                                    y_train,
                                    sensitive_values
                                )

                                # Generate predictions
                                mitigated_predictions = (
                                    mitigated_model.predict(
                                        X_test
                                    )
                                )

                                # Store results
                                st.session_state[
                                    "mitigated_model"
                                ] = mitigated_model

                                st.session_state[
                                    "mitigated_predictions"
                                ] = mitigated_predictions

                                st.session_state[
                                    "mitigation_feature"
                                ] = selected_feature

                                st.session_state[
                                    "mitigation_method"
                                ] = "Reweighing"

                                st.success(
                                    "Reweighing applied successfully!"
                                )

                            except Exception as e:

                                st.error(
                                    f"Reweighing failed: {e}"
                                )

                    # ====================================================
                    # EXPONENTIATED GRADIENT
                    # ====================================================

                    elif mitigation_method == "Exponentiated Gradient":

                        if st.button(
                            "Apply Exponentiated Gradient"
                        ):

                            try:

                                model = st.session_state[
                                    "trained_model"
                                ]

                                X_train = st.session_state[
                                    "X_train"
                                ]

                                y_train = st.session_state[
                                    "y_train"
                                ]

                                X_test = st.session_state[
                                    "X_test"
                                ]

                                sensitive_train = st.session_state[
                                    "sensitive_train"
                                ]

                                sensitive_values = (
                                    sensitive_train[
                                        selected_feature
                                    ]
                                )

                                with st.spinner(
                                    "Applying Exponentiated "
                                    "Gradient..."
                                ):

                                    mitigated_model, label_encoder = (
                                        apply_exponentiated_gradient(
                                            model,
                                            X_train,
                                            y_train,
                                            sensitive_values
                                        )
                                    )

                                # Generate encoded predictions
                                mitigated_predictions_encoded = (
                                    mitigated_model.predict(
                                        X_test
                                    )
                                )

                                # Convert predictions back to original labels
                                mitigated_predictions = (
                                    label_encoder.inverse_transform(
                                        mitigated_predictions_encoded
                                    )
                                )

                                # Store results
                                st.session_state[
                                    "mitigated_model"
                                ] = mitigated_model

                                st.session_state[
                                    "mitigated_predictions"
                                ] = mitigated_predictions

                                st.session_state[
                                    "mitigation_feature"
                                ] = selected_feature

                                st.session_state[
                                    "mitigation_method"
                                ] = "Exponentiated Gradient"

                                st.session_state[
                                    "eg_label_encoder"
                                ] = label_encoder

                                st.success(
                                    "Exponentiated Gradient "
                                    "applied successfully!"
                                )

                            except Exception as e:

                                st.error(
                                    f"Exponentiated Gradient "
                                    f"failed: {e}"
                                )

                    # ====================================================
                    # THRESHOLD OPTIMIZER
                    # ====================================================

                    else:

                        if st.button("Apply Threshold Optimizer"):

                            try:

                                model = st.session_state[
                                    "trained_model"
                                ]

                                X_train = st.session_state[
                                    "X_train"
                                ]

                                y_train = st.session_state[
                                    "y_train"
                                ]

                                X_test = st.session_state[
                                    "X_test"
                                ]

                                sensitive_train = st.session_state[
                                    "sensitive_train"
                                ]

                                # Sensitive feature used during fitting
                                sensitive_values = (
                                    sensitive_train[
                                        selected_feature
                                    ]
                                )

                                # Get test sensitive values
                                processed_df = st.session_state[
                                    "processed_df"
                                ]

                                sensitive_test = processed_df.loc[
                                    X_test.index,
                                    st.session_state["sensitive_features"]
                                ]

                                with st.spinner(
                                    "Applying Threshold Optimizer..."
                                ):

                                    mitigated_model,label_encoder = (
                                        apply_threshold_optimizer(
                                            model,
                                            X_train,
                                            y_train,
                                            sensitive_values
                                        )
                                    )

                                # Generate fairness-aware predictions
                                mitigated_predictions_encoded = (
                                    mitigated_model.predict(
                                        X_test,
                                        sensitive_features=sensitive_test
                                    )
                                )

                                mitigated_predictions = (
                                    label_encoder.inverse_transform(
                                        mitigated_predictions_encoded
                                    )
                                )

                                # Store results
                                st.session_state[
                                    "mitigated_model"
                                ] = mitigated_model

                                st.session_state[
                                    "mitigated_predictions"
                                ] = mitigated_predictions

                                st.session_state[
                                    "mitigation_feature"
                                ] = selected_feature

                                st.session_state[
                                    "mitigation_method"
                                ] = "Threshold Optimizer"

                                # Store sensitive test values
                                st.session_state[
                                    "sensitive_test"
                                ] = sensitive_test

                                st.success(
                                    "Threshold Optimizer "
                                    "applied successfully!"
                                )

                            except Exception as e:

                                st.error(
                                    f"Threshold Optimizer failed: {e}"
                                )


                # --------------------------------------------------------
                # Regression
                # --------------------------------------------------------

                else:

                    st.info(
                        "Regression mitigation currently uses "
                        "Reweighing."
                    )

                    if st.button("Apply Reweighing"):

                        try:

                            from sklearn.base import clone

                            model = st.session_state[
                                "trained_model"
                            ]

                            X_train = st.session_state[
                                "X_train"
                            ]

                            y_train = st.session_state[
                                "y_train"
                            ]

                            X_test = st.session_state[
                                "X_test"
                            ]

                            sensitive_train = st.session_state[
                                "sensitive_train"
                            ]

                            sensitive_values = (
                                sensitive_train[
                                    selected_feature
                                ]
                            )

                            mitigated_model = clone(model)

                            mitigated_model = apply_reweighing(
                                mitigated_model,
                                X_train,
                                y_train,
                                sensitive_values
                            )

                            mitigated_predictions = (
                                mitigated_model.predict(
                                    X_test
                                )
                            )

                            st.session_state[
                                "mitigated_model"
                            ] = mitigated_model

                            st.session_state[
                                "mitigated_predictions"
                            ] = mitigated_predictions

                            st.session_state[
                                "mitigation_feature"
                            ] = selected_feature

                            st.session_state[
                                "mitigation_method"
                            ] = "Reweighing"

                            st.success(
                                "Reweighing applied successfully!"
                            )

                        except Exception as e:

                            st.error(
                                f"Reweighing failed: {e}"
                            )

            else:

                st.info(
                    "No mitigation strategy is currently recommended."
                )

        else:

            st.info(
                "Run the Recommendation Engine first."
            )

        # ============================================================
        # MITIGATION EVALUATION
        # ============================================================

        if (
            "mitigated_predictions" in st.session_state
            and "trained_model" in st.session_state
            and "X_test" in st.session_state
            and "y_test" in st.session_state
            and "sensitive_test" in st.session_state
        ):

            st.header("Mitigation Evaluation")

            original_model = st.session_state["trained_model"]
            mitigated_predictions = st.session_state[
                "mitigated_predictions"
            ]

            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]
            sensitive_test = st.session_state["sensitive_test"]
            sensitive_features = st.session_state["sensitive_features"]
            task_type = st.session_state["task_type"]

            # Original predictions
            original_predictions = original_model.predict(X_test)

            # ========================================================
            # FAIRNESS COMPARISON
            # ========================================================

            st.subheader("Fairness Comparison")

            original_results = evaluate_fairness(
                y_test,
                original_predictions,
                sensitive_test,
                sensitive_features,
                task_type
            )

            mitigated_results = evaluate_fairness(
                y_test,
                mitigated_predictions,
                sensitive_test,
                sensitive_features,
                task_type
            )

            for feature in original_results:

                st.markdown(
                    f"### Sensitive Feature: {feature}"
                )

                original_metrics = original_results[
                    feature
                ]["metrics"]

                mitigated_metrics = mitigated_results[
                    feature
                ]["metrics"]

                # ----------------------------------------------------
                # Classification
                # ----------------------------------------------------

                if task_type == "classification":

                    comparison = pd.DataFrame({

                        "Metric": [
                            "Demographic Parity Difference",
                            "Disparate Impact",
                            "Equal Opportunity Difference"
                        ],

                        "Before Mitigation": [
                            original_metrics[
                                "Demographic Parity Difference"
                            ],

                            original_metrics[
                                "Disparate Impact"
                            ],

                            original_metrics[
                                "Equal Opportunity Difference"
                            ]
                        ],

                        "After Mitigation": [
                            mitigated_metrics[
                                "Demographic Parity Difference"
                            ],

                            mitigated_metrics[
                                "Disparate Impact"
                            ],

                            mitigated_metrics[
                                "Equal Opportunity Difference"
                            ]
                        ]
                    })

                # ----------------------------------------------------
                # Regression
                # ----------------------------------------------------

                else:

                    comparison = pd.DataFrame({

                        "Metric": [
                            "Group MAE",
                            "Group RMSE",
                            "Mean Prediction Difference"
                        ],

                        "Before Mitigation": [
                            original_metrics["Group MAE"],
                            original_metrics["Group RMSE"],
                            original_metrics[
                                "Mean Prediction Difference"
                            ]
                        ],

                        "After Mitigation": [
                            mitigated_metrics["Group MAE"],
                            mitigated_metrics["Group RMSE"],
                            mitigated_metrics[
                                "Mean Prediction Difference"
                            ]
                        ]
                    })

                st.dataframe(
                    comparison,
                    use_container_width=True
                )

            # ========================================================
            # MODEL PERFORMANCE COMPARISON
            # ========================================================

            st.subheader("Model Performance Comparison")

            if task_type == "classification":

                original_accuracy = accuracy_score(
                    y_test,
                    original_predictions
                )

                mitigated_accuracy = accuracy_score(
                    y_test,
                    mitigated_predictions
                )

                performance_df = pd.DataFrame({

                    "Metric": [
                        "Accuracy"
                    ],

                    "Before Mitigation": [
                        original_accuracy
                    ],

                    "After Mitigation": [
                        mitigated_accuracy
                    ]
                })

            else:

                original_mae = mean_absolute_error(
                    y_test,
                    original_predictions
                )

                mitigated_mae = mean_absolute_error(
                    y_test,
                    mitigated_predictions
                )

                original_rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        original_predictions
                    )
                )

                mitigated_rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        mitigated_predictions
                    )
                )

                performance_df = pd.DataFrame({

                    "Metric": [
                        "MAE",
                        "RMSE"
                    ],

                    "Before Mitigation": [
                        original_mae,
                        original_rmse
                    ],

                    "After Mitigation": [
                        mitigated_mae,
                        mitigated_rmse
                    ]
                })

            st.dataframe(
                performance_df,
                use_container_width=True
            )

        else:

            st.info(
                "Apply a mitigation strategy to view "
                "the before vs after comparison."
            )