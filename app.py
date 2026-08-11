import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules.dataloader import upload_dataset

from modules.preprocessing import (detect_missing_values,handle_missing_values,remove_duplicates,encode_categorical,scale_features,
                                    split_dataset)

from modules.model_training import (build_decision_tree_classifier,build_decision_tree_regressor,build_linear_regression,build_logistic_regression,
                                    build_random_forest_classifier,build_random_forest_regressor,build_xgboost_classifier,build_xgboost_regressor,
                                    detect_task_type)

from modules.evaluation import evaluate_classification_models,evaluate_regression_models

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