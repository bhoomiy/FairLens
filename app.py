import streamlit as st
import pandas as pd

from modules.dataloader import upload_dataset

from modules.preprocessing import (
    detect_missing_values,
    handle_missing_values,
    remove_duplicates,
    encode_categorical,
    scale_features,
    split_dataset
)

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

        # Show missing values
        st.subheader("Missing Values")
        st.write(detect_missing_values(df))

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

        # Remove duplicates
        before = len(df)
        df = remove_duplicates(df)
        after = len(df)

        st.write(f"Duplicate rows removed: {before-after}")

        # Split features & target
        X = df.drop(columns=[target])
        y = df[target]

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