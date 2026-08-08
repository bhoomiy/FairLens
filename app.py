import streamlit as st
import pandas as pd

from modules.dataloader import upload_dataset

from modules.preprocessing import (detect_missing_values,handle_missing_values,remove_duplicates,encode_categorical,scale_features,
                                    split_dataset)

from modules.model_training import (build_decision_tree_classifier,build_decision_tree_regressor,build_linear_regression,build_logistic_regression,
                                    build_random_forest_classifier,build_random_forest_regressor,build_xgboost_classifier,build_xgboost_regressor,
                                    detect_task_type)

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
        task_type = st.radio(
            "Problem Type",
            ["Classification", "Regression"],
            index=(
                0 if detected_task == "classification"
                else 1
            )
        )

        if task_type == "Classification":

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