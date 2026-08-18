from flask import Flask, jsonify, request,send_file
from flask_cors import CORS
import uuid
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
    is_categorical_dtype,
    is_bool_dtype,
)
import joblib
from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER

from modules.bias_detection import evaluate_fairness
from modules.recommendation import generate_recommendations
from modules.explainability import explain_model
from modules.bias_mitigation import (
    calculate_reweighing_weights,
    apply_reweighing,
    apply_exponentiated_gradient,
    apply_threshold_optimizer,
)
from modules.report_generator import generate_pdf_report
from modules.model_training import (
    build_linear_regression,
    build_logistic_regression,
    build_decision_tree_regressor,
    build_decision_tree_classifier,
    build_random_forest_regressor,
    build_random_forest_classifier,
    build_xgboost_classifier,
    build_xgboost_regressor,
)

from modules.evaluation import (
    evaluate_classification_models,
    evaluate_classification_predictions,
    evaluate_regression_models,
)

def pdf_format_value(value):
    """Convert Python/NumPy values into clean PDF-friendly text."""

    if value is None:
        return "-"

    # NumPy scalar
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    # Boolean
    if isinstance(value, bool):
        return "Yes" if value else "No"

    # Numbers
    if isinstance(value, (int, float)):
        try:
            if np.isnan(value):
                return "-"
        except Exception:
            pass

        return f"{float(value):.4f}"

    # Dictionary
    if isinstance(value, dict):
        parts = []

        for key, val in value.items():

            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    pass

            if isinstance(val, (int, float)):
                try:
                    formatted = f"{float(val):.4f}"
                except Exception:
                    formatted = str(val)

                parts.append(
                    f"{key}: {formatted}"
                )

            else:
                parts.append(
                    f"{key}: {val}"
                )

        return "<br/>".join(parts)

    # List / tuple
    if isinstance(value, (list, tuple)):
        return "<br/>".join(
            pdf_format_value(v)
            for v in value
        )

    return str(value)


def pdf_cell(value, font_size=8):
    """Create a wrapped ReportLab table cell."""

    return Paragraph(
        pdf_format_value(value),
        ParagraphStyle(
            f"PDFCell{font_size}",
            fontSize=font_size,
            leading=font_size + 2,
            wordWrap="CJK",
            spaceAfter=0,
            spaceBefore=0
        )
    )

def make_json_safe(obj):

    # Dictionary
    if isinstance(obj, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in obj.items()
        }

    # List / tuple / set
    if isinstance(obj, (list, tuple, set)):
        return [
            make_json_safe(v)
            for v in obj
        ]

    # Pandas Series
    if isinstance(obj, pd.Series):
        return [
            make_json_safe(v)
            for v in obj.tolist()
        ]

    # Pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        return [
            make_json_safe(row)
            for row in obj.to_dict(orient="records")
        ]

    # NumPy array
    if isinstance(obj, np.ndarray):
        return [
            make_json_safe(v)
            for v in obj.tolist()
        ]

    # NumPy integer
    if isinstance(obj, np.integer):
        return int(obj)

    # NumPy float
    if isinstance(obj, np.floating):
        return float(obj)

    # NumPy boolean
    if isinstance(obj, np.bool_):
        return bool(obj)

    # Python boolean
    if isinstance(obj, bool):
        return obj

    # None
    if obj is None:
        return None

    # NaN / NaT
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass

    return obj

def create_sensitive_groups(series, max_groups=5):
    """
    Convert a sensitive feature into meaningful fairness groups.

    Categorical features:
        Keep their original categories.

    Numerical features:
        Automatically divide values into up to `max_groups`
        quantile-based ranges.
    """

    series = pd.Series(series).copy()

    # --------------------------------------------------------
    # CATEGORICAL FEATURE
    # --------------------------------------------------------

    if (
        is_object_dtype(series)
        or is_string_dtype(series)
        or is_categorical_dtype(series)
        or is_bool_dtype(series)
    ):
        return series.astype(str)

    # --------------------------------------------------------
    # NUMERICAL FEATURE
    # --------------------------------------------------------

    numeric = pd.to_numeric(series, errors="coerce")

    unique_values = numeric.dropna().nunique()

    # If there are very few unique values, treat them as groups
    if unique_values <= max_groups:

        return numeric.map(
            lambda x: str(x)
            if pd.notna(x)
            else "Missing"
        )

    # Number of groups cannot exceed number of unique values
    n_groups = min(max_groups, unique_values)

    try:

        # Quantile-based bins
        bins = pd.qcut(
            numeric,
            q=n_groups,
            duplicates="drop"
        )

        # Convert intervals into readable labels
        def format_interval(interval):

            if pd.isna(interval):
                return "Missing"

            left = interval.left
            right = interval.right

            # Integers look cleaner without decimals
            if float(left).is_integer():
                left = int(left)

            if float(right).is_integer():
                right = int(right)

            return f"{left}–{right}"

        grouped = bins.map(format_interval)

        grouped = grouped.astype(object)

        grouped[numeric.isna()] = "Missing"

        return grouped

    except Exception:

        # Safe fallback
        return numeric.map(
            lambda x: str(x)
            if pd.notna(x)
            else "Missing"
        )

def group_sensitive_dataframe(sensitive_dataframe):
    """
    Convert sensitive features into meaningful fairness groups.

    Categorical features remain unchanged.
    Numerical features are grouped into quantile-based ranges.
    """

    grouped = pd.DataFrame(
        index=sensitive_dataframe.index
    )

    for feature in sensitive_dataframe.columns:
        grouped[feature] = create_sensitive_groups(
            sensitive_dataframe[feature]
        )

    return grouped


def analyze_class_imbalance(y, sensitive_dataframe=None):
    """
    Analyze overall target class distribution and
    class distribution within meaningful sensitive groups.
    """

    result = {
        "detected": False,
        "distribution": {},
        "groups": {}
    }

    # ========================================================
    # OVERALL CLASS DISTRIBUTION
    # ========================================================

    distribution = (
        pd.Series(y)
        .value_counts(normalize=True)
        .sort_index()
    )

    result["distribution"] = {
        str(label): round(float(proportion), 4)
        for label, proportion in distribution.items()
    }

    # ========================================================
    # OVERALL IMBALANCE RATIO
    # ========================================================

    if len(distribution) > 1:

        min_ratio = float(distribution.min())
        max_ratio = float(distribution.max())

        imbalance_ratio = (
            max_ratio / min_ratio
            if min_ratio > 0
            else float("inf")
        )

        result["imbalance_ratio"] = (
            round(imbalance_ratio, 4)
            if np.isfinite(imbalance_ratio)
            else None
        )

        result["detected"] = (
            imbalance_ratio >= 1.5
        )

    # ========================================================
    # PER-SENSITIVE-FEATURE ANALYSIS
    # ========================================================

    if sensitive_dataframe is not None:

        for feature in sensitive_dataframe.columns:

            feature_groups = {}

            # ------------------------------------------------
            # CREATE MEANINGFUL GROUPS
            # ------------------------------------------------

            grouped_sensitive = create_sensitive_groups(
                sensitive_dataframe[feature]
            )

            combined = pd.DataFrame({
                "target": pd.Series(y).values,
                "sensitive": grouped_sensitive.values
            })

            # ------------------------------------------------
            # ANALYZE EACH GROUP
            # ------------------------------------------------

            for group, group_df in combined.groupby(
                "sensitive",
                dropna=False
            ):

                group_distribution = (
                    group_df["target"]
                    .value_counts(normalize=True)
                    .sort_index()
                )

                distribution_dict = {
                    str(label): round(
                        float(proportion),
                        4
                    )
                    for label, proportion
                    in group_distribution.items()
                }

                group_detected = False
                group_ratio = None

                if len(group_distribution) > 1:

                    min_ratio = float(
                        group_distribution.min()
                    )

                    max_ratio = float(
                        group_distribution.max()
                    )

                    group_ratio = (
                        max_ratio / min_ratio
                        if min_ratio > 0
                        else float("inf")
                    )

                    group_detected = (
                        group_ratio >= 1.5
                    )

                feature_groups[str(group)] = {

                    "distribution": distribution_dict,

                    "imbalance_ratio": (
                        round(group_ratio, 4)
                        if group_ratio is not None
                        and np.isfinite(group_ratio)
                        else None
                    ),

                    "detected": group_detected,

                    "samples": int(len(group_df))
                }

            result["groups"][feature] = feature_groups

    return result

from modules.dataloader import upload_dataset, dataset_preview
from modules.preprocessing import (
    dataset_summary,
    handle_missing_values,
    remove_duplicates,
    encode_categorical,
    scale_features,
    split_dataset,
)

app = Flask(__name__)
CORS(app)

# Temporary in-memory storage for uploaded datasets.
# This keeps the dataset available between API requests.
DATASETS = {}


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "status": "ok",
        "message": "FairLens backend is running"
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({
                "ok": False,
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "ok": False,
                "error": "No file selected"
            }), 400

        if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
            return jsonify({
                "ok": False,
                "error": "Only CSV and Excel files are supported"
            }), 400

        # Read dataset
        df = upload_dataset(file)

        if df is None or df.empty:
            return jsonify({
                "ok": False,
                "error": "The uploaded dataset is empty"
            }), 400

        # Create session ID
        session_id = str(uuid.uuid4())

        # Store dataset
        DATASETS[session_id] = {
            "df": df,
            "filename": file.filename
        }

        summary = dataset_summary(df)
        preview = dataset_preview(df)

        # dtype information
        dtypes = {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        }

        missing = [
            {
                "column": column,
                "missing": int(df[column].isna().sum())
            }
            for column in df.columns
        ]

        return jsonify(make_json_safe({
            "ok": True,
            "success": True,
            "session_id": session_id,
            "dataset_name": file.filename,
            "filename": file.filename,
            "column_names": list(df.columns),
            "columns": len(df.columns),
            "rows": len(df),
            "duplicates": int(df.duplicated().sum()),
            "dtypes": dtypes,
            "missing": missing,
            "summary": summary,
            "preview": preview
        }))

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route("/api/schema/<session_id>", methods=["GET"])
def schema(session_id):
    try:
        if session_id not in DATASETS:
            return jsonify({
                "ok": False,
                "error": "Dataset session not found"
            }), 404

        df = DATASETS[session_id]["df"]

        # Basic quality information
        total_cells = df.shape[0] * df.shape[1]

        missing_cells = int(df.isna().sum().sum())

        completeness = (
            100
            if total_cells == 0
            else round((1 - missing_cells / total_cells) * 100, 2)
        )

        # Simple quality score
        duplicate_penalty = (
            df.duplicated().sum() / len(df) * 100
            if len(df) > 0
            else 0
        )

        quality_score = max(
            0,
            round(completeness - duplicate_penalty, 2)
        )

        numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Schema
        schema_data = []

        for column in df.columns:
            schema_data.append({
                "column": column,
                "dtype": str(df[column].dtype),
                "missing": int(df[column].isna().sum()),
                "unique": int(df[column].nunique(dropna=True))
            })

        # Missing values
        missing_by_column = {
            column: int(df[column].isna().sum())
            for column in df.columns
        }

        # Numeric statistics
        numeric_stats = []

        if numeric_columns:
            stats = df[numeric_columns].describe().T

            for column in numeric_columns:
                numeric_stats.append({
                    "column": column,
                    "count": float(stats.loc[column, "count"]),
                    "mean": float(stats.loc[column, "mean"]),
                    "std": float(stats.loc[column, "std"])
                    if stats.loc[column, "std"] == stats.loc[column, "std"]
                    else 0,
                    "min": float(stats.loc[column, "min"]),
                    "max": float(stats.loc[column, "max"])
                })

        # Categorical statistics
        categorical_stats = []

        for column in categorical_columns:
            categorical_stats.append({
                "column": column,
                "unique": int(df[column].nunique(dropna=True)),
                "missing": int(df[column].isna().sum()),
                "top": (
                    str(df[column].mode().iloc[0])
                    if not df[column].mode().empty
                    else ""
                )
            })

        # Quality issues
        issues = []

        for column in df.columns:
            missing_count = int(df[column].isna().sum())

            if missing_count > 0:
                issues.append({
                    "column": column,
                    "severity": "warning",
                    "message": f"{missing_count} missing values detected."
                })

            if df[column].nunique(dropna=True) <= 1:
                issues.append({
                    "column": column,
                    "severity": "warning",
                    "message": "Column contains only one unique value."
                })

        if df.duplicated().sum() > 0:
            issues.append({
                "column": "Dataset",
                "severity": "warning",
                "message": f"{df.duplicated().sum()} duplicate rows detected."
            })

        return jsonify(make_json_safe({
            "ok": True,
            "quality_score": quality_score,
            "completeness": completeness,
            "missing_cells": missing_cells,
            "duplicates": int(df.duplicated().sum()),
            "memory_kb": round(
                df.memory_usage(deep=True).sum() / 1024,
                2
            ),
            "schema": schema_data,
            "missing_by_column": missing_by_column,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "issues": issues
        }))

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route("/api/preprocess", methods=["POST"])
def preprocess():
    try:
        data = request.get_json()

        session_id = data.get("session_id")
        target = data.get("target")
        sensitive = data.get("sensitive", [])
        missing_strategy = data.get(
            "missing_strategy",
            "Mean"
        )
        encoding_method = data.get(
            "encoding_method",
            "Label Encoding"
        )
        scaling_method = data.get(
            "scaling_method",
            "StandardScaler"
        )
        test_size = float(
            data.get("test_size", 0.2)
        )

        if session_id not in DATASETS:
            return jsonify({
                "ok": False,
                "error": "Dataset session not found"
            }), 404

        if not target:
            return jsonify({
                "ok": False,
                "error": "Target column is required"
            }), 400

        if not sensitive:
            return jsonify({
                "ok": False,
                "error": "At least one sensitive feature is required"
            }), 400

        df = DATASETS[session_id]["df"].copy()

        if target not in df.columns:
            return jsonify({
                "ok": False,
                "error": f"Target column '{target}' not found"
            }), 400

        # ---------------------------------------------------------
        # Missing values
        # ---------------------------------------------------------

        missing_before = [
            {
                "column": column,
                "missing": int(df[column].isna().sum())
            }
            for column in df.columns
        ]

        strategy_map = {
            "Mean": "mean",
            "Median": "median",
            "Mode": "mode",
            "Drop Rows": "drop"
        }

        strategy = strategy_map.get(
            missing_strategy,
            "mean"
        )

        df = handle_missing_values(
            df,
            strategy=strategy
        )

        # ---------------------------------------------------------
        # Remove duplicates
        # ---------------------------------------------------------

        duplicates_before = int(
            df.duplicated().sum()
        )

        df = remove_duplicates(df)

        duplicates_removed = duplicates_before - int(
            df.duplicated().sum()
        )

        # ---------------------------------------------------------
        # Separate target
        # ---------------------------------------------------------

        y = df[target]

        X = df.drop(columns=[target])

        # Preserve sensitive columns separately
        sensitive_data = {}

        for feature in sensitive:
            if feature in X.columns:
                sensitive_data[feature] = X[feature].copy()

        

        # ---------------------------------------------------------
        # Encoding
        # ---------------------------------------------------------

        encoding_map = {
            "Label Encoding": "label",
            "One-Hot Encoding": "one-hot"
        }

        encoding = encoding_map.get(
            encoding_method,
            "label"
        )

        X_encoded, encoders = encode_categorical(
            X,
            strategy=encoding
        )

        # ---------------------------------------------------------
        # Scaling
        # ---------------------------------------------------------

        scaler = None

        if scaling_method == "StandardScaler":
            X_processed, scaler = scale_features(
                X_encoded,
                method="standard"
            )

        elif scaling_method == "MinMaxScaler":
            X_processed, scaler = scale_features(
                X_encoded,
                method="minmax"
            )

        else:
            X_processed = X_encoded

        # ---------------------------------------------------------
        # Task detection
        # ---------------------------------------------------------

        if (
            y.dtype == "object"
            or str(y.dtype) == "category"
            or y.nunique() <= 10
        ):
            task_type = "classification"
            detection_reason = (
                "Target is categorical or contains a small "
                "number of unique classes."
            )
        else:
            task_type = "regression"
            detection_reason = (
                "Target is numerical with more than 10 unique values."
            )

        # ---------------------------------------------------------
        # Train/test split
        # ---------------------------------------------------------

        X_train, X_test, y_train, y_test = split_dataset(
            X_processed,
            y,
            test_size=test_size
        )
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # Sensitive features for test set
        # ---------------------------------------------------------
        
        sensitive_test = {}

        for feature in sensitive:
            if feature in df.columns:

                values = df.loc[
                    X_test.index,
                    feature
                ].copy()

                # Preserve the actual sensitive values.
                # Do NOT convert them to groups here.
                sensitive_test[feature] = values

        # ---------------------------------------------------------
        # Target distribution / regression statistics
        # ---------------------------------------------------------

        if task_type == "classification":

            target_distribution = {
                str(key): int(value)
                for key, value in y.value_counts().items()
            }

            target_statistics = {}

        else:

            numeric_target = pd.to_numeric(
                y,
                errors="coerce"
            )

            target_statistics = {
                "min": float(numeric_target.min()),
                "max": float(numeric_target.max()),
                "mean": float(numeric_target.mean()),
                "median": float(numeric_target.median()),
                "std": float(numeric_target.std())
                if not pd.isna(numeric_target.std())
                else 0.0
            }

            target_distribution = {}

        missing_after = [
            {
                "column": column,
                "missing": int(df[column].isna().sum())
            }
            for column in df.columns
        ]

        # Store processed state for later steps
        DATASETS[session_id].update({
            "processed_df": df,
            "X": X_processed,
            "y": y,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "target": target,
            "sensitive": sensitive,
            "sensitive_data": sensitive_data,
            "sensitive_test": sensitive_test,
            "task_type": task_type,
            "encoders": encoders,
            "scaler": scaler
        })

        return jsonify(make_json_safe({
            "ok": True,
            "rows": len(df),
            "duplicates_removed": duplicates_removed,
            "features": X_processed.shape[1],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "task_type": task_type,
            "detection_reason": detection_reason,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "target_distribution": target_distribution,
            "target_statistics": target_statistics
        }))
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

@app.route("/api/train", methods=["POST"])
def train():
    try:
        data = request.get_json()

        session_id = data.get("session_id")
        model_name = data.get("model")

        if not session_id:
            return jsonify({
                "ok": False,
                "error": "Session ID is required"
            }), 400

        if session_id not in DATASETS:
            return jsonify({
                "ok": False,
                "error": "Dataset session not found"
            }), 404

        if not model_name:
            return jsonify({
                "ok": False,
                "error": "Model selection is required"
            }), 400

        dataset = DATASETS[session_id]

        required = [
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "task_type"
        ]

        if not all(key in dataset for key in required):
            return jsonify({
                "ok": False,
                "error": "Please complete preprocessing before training"
            }), 400

        X_train = dataset["X_train"]
        X_test = dataset["X_test"]
        y_train = dataset["y_train"]
        y_test = dataset["y_test"]
        task_type = dataset["task_type"]

        # ========================================================
        # CLASSIFICATION
        # ========================================================

        if task_type == "classification":

            if model_name == "Logistic Regression":
                model = build_logistic_regression(
                    X_train,
                    y_train
                )

                results = evaluate_classification_models(
                    model,
                    X_test,
                    y_test
                )

            elif model_name == "Decision Tree":
                model = build_decision_tree_classifier(
                    X_train,
                    y_train
                )

                results = evaluate_classification_models(
                    model,
                    X_test,
                    y_test
                )

            elif model_name == "Random Forest":
                model = build_random_forest_classifier(
                    X_train,
                    y_train
                )

                results = evaluate_classification_models(
                    model,
                    X_test,
                    y_test
                )

            elif model_name == "XGBoost":
                model, encoder = build_xgboost_classifier(
                    X_train,
                    y_train
                )

                # XGBoost uses encoded target labels.
                y_pred_encoded = model.predict(X_test)

                y_pred = encoder.inverse_transform(
                    y_pred_encoded.astype(int)
                )

                results = evaluate_classification_predictions(
                    y_test,
                    y_pred
                )

                dataset["label_encoder"] = encoder

            else:
                return jsonify({
                    "ok": False,
                    "error": f"Unsupported classification model: {model_name}"
                }), 400

        # ========================================================
        # REGRESSION
        # ========================================================

        elif task_type == "regression":

            if model_name == "Linear Regression":
                model = build_linear_regression(
                    X_train,
                    y_train
                )

            elif model_name == "Decision Tree":
                model = build_decision_tree_regressor(
                    X_train,
                    y_train
                )

            elif model_name == "Random Forest":
                model = build_random_forest_regressor(
                    X_train,
                    y_train
                )

            elif model_name == "XGBoost":
                model = build_xgboost_regressor(
                    X_train,
                    y_train
                )

            else:
                return jsonify({
                    "ok": False,
                    "error": f"Unsupported regression model: {model_name}"
                }), 400

            results = evaluate_regression_models(
                model,
                X_test,
                y_test
            )

        else:
            return jsonify({
                "ok": False,
                "error": f"Unsupported task type: {task_type}"
            }), 400

        # ========================================================
        # STORE TRAINED MODEL
        # ========================================================

        dataset["trained_model"] = model
        dataset["model_name"] = model_name
        dataset["evaluation_results"] = results

        return jsonify(make_json_safe({
            "ok": True,
            "success": True,
            "session_id": session_id,
            "model_name": model_name,
            "task_type": task_type,
            "metrics": results
        }))

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

# ============================================================
# STEP 03 - BIAS DETECTION & EXPLAINABILITY
# ============================================================

@app.route("/api/bias-detection", methods=["POST"])
def bias_detection():
    try:
        data = request.get_json() or {}

        session_id = data.get("session_id")

        if not session_id:
            return jsonify({
                "ok": False,
                "error": "Session ID is required"
            }), 400

        if session_id not in DATASETS:
            return jsonify({
                "ok": False,
                "error": "Dataset session not found"
            }), 404

        dataset = DATASETS[session_id]

        # --------------------------------------------------------
        # Check that preprocessing and training are complete
        # --------------------------------------------------------

        required = [
            "X_train",
            "X_test",
            "y_test",
            "trained_model",
            "task_type",
            "sensitive_test",
            "sensitive"
        ]

        missing = [
            key for key in required
            if key not in dataset
        ]

        if missing:
            return jsonify({
                "ok": False,
                "error": (
                    "Please complete preprocessing and training "
                    "before detecting bias."
                ),
                "missing": missing
            }), 400

        model = dataset["trained_model"]
        X_train = dataset["X_train"]
        X_test = dataset["X_test"]
        y_test = dataset["y_test"]
        task_type = dataset["task_type"]
        sensitive_features = dataset["sensitive"]
        sensitive_test = dataset["sensitive_test"]

        # --------------------------------------------------------
        # Generate predictions
        # --------------------------------------------------------

        if task_type == "classification":

            y_pred = model.predict(X_test)

            # XGBoost uses encoded target labels.
            if "label_encoder" in dataset:
                encoder = dataset["label_encoder"]

                y_pred = encoder.inverse_transform(
                    np.asarray(y_pred).astype(int)
                )

        elif task_type == "regression":

            y_pred = model.predict(X_test)

        else:
            return jsonify({
                "ok": False,
                "error": f"Unsupported task type: {task_type}"
            }), 400

        # --------------------------------------------------------
        # Build sensitive dataframe
        # --------------------------------------------------------

        sensitive_dataframe = pd.DataFrame(
            sensitive_test,
            index=X_test.index
        )

        for feature_name in sensitive_features:
            sensitive_dataframe[feature_name] = create_sensitive_groups(
                sensitive_dataframe[feature_name]
            )

        print("\n===== SENSITIVE DATA DEBUG =====")
        print(sensitive_dataframe.dtypes)

        for column in sensitive_dataframe.columns:
            print(f"\n{column}:")
            print(sensitive_dataframe[column].value_counts(dropna=False))

        print("===============================\n")

        # Convert numerical sensitive features into meaningful groups
        sensitive_dataframe = group_sensitive_dataframe(
            sensitive_dataframe
        )

        # --------------------------------------------------------
        # Class imbalance analysis
        # --------------------------------------------------------

        

        if task_type == "classification":

            class_imbalance = analyze_class_imbalance(
                y_test,
                sensitive_dataframe
            )

        else:

            class_imbalance = {
                "detected": False,
                "distribution": {},
                "groups": {}
            }

        # --------------------------------------------------------
        # Fairness evaluation
        # --------------------------------------------------------

        fairness_results = evaluate_fairness(
            y_test,
            y_pred,
            sensitive_dataframe,
            sensitive_features,
            task_type
        )

        # --------------------------------------------------------
        # Mitigation recommendations
        # --------------------------------------------------------

        recommendations = generate_recommendations(
            fairness_results,
            sensitive_dataframe,
            task_type,
            can_retrain=True,
            predictions_only=False
        )

        # --------------------------------------------------------
        # Explainability
        # --------------------------------------------------------

        explainability = explain_model(
            model,
            X_train,
            X_test
        )

        feature_importance = (
            explainability["feature_importance"]
            .to_dict(orient="records")
        )

        # --------------------------------------------------------
        # Overall bias status
        # --------------------------------------------------------

        bias_detected = any(
            feature_result["bias_detection"]["bias_detected"]
            for feature_result in fairness_results.values()
        )

        

        # --------------------------------------------------------
        # Store Step 03 results
        # --------------------------------------------------------

        dataset["y_pred"] = y_pred
        dataset["fairness_results"] = fairness_results
        dataset["recommendations"] = recommendations
        dataset["explainability"] = {
    "feature_importance": feature_importance
}
        dataset["class_imbalance"] = class_imbalance
        dataset["bias_detected"] = bias_detected

        # --------------------------------------------------------
        # Return JSON-safe response
        # --------------------------------------------------------

        return jsonify(make_json_safe({
            "ok": True,
            "success": True,
            "session_id": session_id,
            "task_type": task_type,

            # Overall bias
            "bias_detected": bool(bias_detected),

            # Sensitive features
            "sensitive_features": sensitive_features,

            # Fairness
            "fairness_results": fairness_results,

            # Recommendations
            "recommendations": recommendations,

            # Class imbalance
            "class_imbalance": class_imbalance,

            # Explainability
            "explainability": {
                "feature_importance": feature_importance
            }
        }))
      

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route("/api/mitigation", methods=["POST"])
def mitigation():
    try:
        data = request.get_json() or {}

        session_id = data.get("session_id")
        technique = data.get("technique")
        feature = data.get("feature")

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        if not session_id:
            return jsonify({
                "ok": False,
                "error": "Session ID is required"
            }), 400

        if session_id not in DATASETS:
            return jsonify({
                "ok": False,
                "error": "Dataset session not found"
            }), 404

        if not technique:
            return jsonify({
                "ok": False,
                "error": "Mitigation technique is required"
            }), 400

        dataset = DATASETS[session_id]

        required = [
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "task_type",
            "trained_model",
            "sensitive",
            "sensitive_test"
        ]

        missing = [
            key
            for key in required
            if key not in dataset
        ]

        if missing:
            return jsonify({
                "ok": False,
                "error": (
                    "Please complete preprocessing, "
                    "training and bias detection first."
                ),
                "missing": missing
            }), 400

        sensitive_features = dataset["sensitive"]

        if not feature:
            if len(sensitive_features) == 1:
                feature = sensitive_features[0]
            else:
                return jsonify({
                    "ok": False,
                    "error": "Please select a sensitive feature."
                }), 400

        if feature not in sensitive_features:
            return jsonify({
                "ok": False,
                "error": (
                    f"Sensitive feature '{feature}' "
                    "was not selected."
                )
            }), 400

        # --------------------------------------------------------
        # DATA
        # --------------------------------------------------------

        model = dataset["trained_model"]

        X_train = dataset["X_train"]
        X_test = dataset["X_test"]

        y_train = dataset["y_train"]
        y_test = dataset["y_test"]

        task_type = dataset["task_type"]

        sensitive_test = dataset["sensitive_test"]

        # Sensitive training values must use the same
        # indices as X_train.
        processed_df = dataset["processed_df"]

        sensitive_train = processed_df.loc[
            X_train.index,
            feature
        ].copy()

        # Group numerical sensitive feature values
        sensitive_train = create_sensitive_groups(
            sensitive_train
        )

        # --------------------------------------------------------
        # ORIGINAL PREDICTIONS
        # --------------------------------------------------------

        original_predictions = model.predict(X_test)

        # XGBoost predictions are encoded.
        if "label_encoder" in dataset:
            encoder = dataset["label_encoder"]

            original_predictions = encoder.inverse_transform(
                np.asarray(original_predictions).astype(int)
            )

        # --------------------------------------------------------
        # APPLY MITIGATION
        # --------------------------------------------------------

        if technique == "Reweighing":

            from sklearn.base import clone

            mitigated_model = clone(model)

            # --------------------------------------------------------
            # XGBoost was originally trained using encoded labels.
            # Reweighing must therefore use the same encoded labels.
            # --------------------------------------------------------

            if "label_encoder" in dataset:

                encoder = dataset["label_encoder"]

                y_train_encoded = encoder.transform(
                    np.asarray(y_train)
                )

                mitigated_model = apply_reweighing(
                    mitigated_model,
                    X_train,
                    y_train_encoded,
                    sensitive_train
                )

                mitigated_predictions_encoded = (
                    mitigated_model.predict(X_test)
                )

                mitigated_predictions = (
                    encoder.inverse_transform(
                        np.asarray(
                            mitigated_predictions_encoded
                        ).astype(int)
                    )
                )

            else:

                # Logistic Regression, Decision Tree,
                # Random Forest, etc. can use the original labels.

                mitigated_model = apply_reweighing(
                    mitigated_model,
                    X_train,
                    y_train,
                    sensitive_train
                )

                mitigated_predictions = (
                    mitigated_model.predict(X_test)
                )

        elif technique == "Exponentiated Gradient":

            if task_type != "classification":
                return jsonify({
                    "ok": False,
                    "error": (
                        "Exponentiated Gradient is "
                        "available for classification only."
                    )
                }), 400

            mitigated_model, label_encoder = (
                apply_exponentiated_gradient(
                    model,
                    X_train,
                    y_train,
                    sensitive_train
                )
            )

            mitigated_predictions_encoded = (
                mitigated_model.predict(X_test)
            )

            mitigated_predictions = (
                label_encoder.inverse_transform(
                    np.asarray(
                        mitigated_predictions_encoded
                    ).astype(int)
                )
            )

            dataset["mitigation_label_encoder"] = (
                label_encoder
            )

        elif technique == "Threshold Optimizer":

            if task_type != "classification":
                return jsonify({
                    "ok": False,
                    "error": (
                        "Threshold Optimizer is "
                        "available for classification only."
                    )
                }), 400

            mitigated_model, label_encoder = (
                apply_threshold_optimizer(
                    model,
                    X_train,
                    y_train,
                    sensitive_train
                )
            )

            sensitive_test_dataframe = pd.DataFrame(
                sensitive_test,
                index=X_test.index
            )

            sensitive_test_dataframe = group_sensitive_dataframe(
                sensitive_test_dataframe
            )

            mitigated_predictions_encoded = (
                mitigated_model.predict(
                    X_test,
                    sensitive_features=sensitive_test_dataframe
                )
            )

            mitigated_predictions = (
                label_encoder.inverse_transform(
                    np.asarray(
                        mitigated_predictions_encoded
                    ).astype(int)
                )
            )

            dataset["mitigation_label_encoder"] = (
                label_encoder
            )

        else:
            return jsonify({
                "ok": False,
                "error": (
                    f"Unsupported mitigation technique: "
                    f"{technique}"
                )
            }), 400

        # --------------------------------------------------------
        # FAIRNESS: BEFORE
        # --------------------------------------------------------

        sensitive_dataframe = pd.DataFrame(
            sensitive_test,
            index=X_test.index
        )

        for feature_name in sensitive_features:

            sensitive_dataframe[feature_name] = create_sensitive_groups(
                sensitive_dataframe[feature_name]
            )

        # Group numerical sensitive features
        sensitive_dataframe = group_sensitive_dataframe(
            sensitive_dataframe
        )

        before_fairness = evaluate_fairness(
            y_test,
            original_predictions,
            sensitive_dataframe,
            sensitive_features,
            task_type
        )

        # --------------------------------------------------------
        # FAIRNESS: AFTER
        # --------------------------------------------------------

        after_fairness = evaluate_fairness(
            y_test,
            mitigated_predictions,
            sensitive_dataframe,
            sensitive_features,
            task_type
        )

        # --------------------------------------------------------
        # STORE MITIGATION RESULTS
        # --------------------------------------------------------

        dataset["mitigated_model"] = mitigated_model
        dataset["mitigated_predictions"] = (
            mitigated_predictions
        )
        dataset["mitigation_feature"] = feature
        dataset["mitigation_method"] = technique

        dataset["before_after_fairness"] = {
            "before": before_fairness,
            "after": after_fairness
        }

        # --------------------------------------------------------
        # RESPONSE
        # --------------------------------------------------------

        return jsonify(make_json_safe({
            "ok": True,
            "success": True,

            "session_id": session_id,

            "task_type": task_type,

            "technique": technique,

            "feature": feature,

            "before_fairness": before_fairness,

            "after_fairness": after_fairness,

            "message": (
                f"{technique} applied successfully."
            )
        }))

    except Exception as e:

        import traceback
        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


# ============================================================
# PDF REPORT
# ============================================================

@app.route("/api/report/<session_id>", methods=["GET"])
def generate_report(session_id):
    print("DEBUG TA_CENTER:", TA_CENTER)

    try:

        # ====================================================
        # VALIDATE SESSION
        # ====================================================

        if session_id not in DATASETS:
            return jsonify({
                "ok": False,
                "error": "Dataset session not found"
            }), 404

        dataset = DATASETS[session_id]

        if "trained_model" not in dataset:
            return jsonify({
                "ok": False,
                "error": "Model training has not been completed."
            }), 400

        # ====================================================
        # PDF BUFFER
        # ====================================================

        buffer = BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=42,
            bottomMargin=42
        )

        styles = getSampleStyleSheet()

        # ====================================================
        # CUSTOM STYLES
        # ====================================================

        title_style = ParagraphStyle(
            "FairLensTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=8
        )

        subtitle_style = ParagraphStyle(
            "FairLensSubtitle",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=20
        )

        heading_style = ParagraphStyle(
            "FairLensHeading",
            parent=styles["Heading2"],
            fontSize=15,
            leading=18,
            spaceBefore=8,
            spaceAfter=10,
            textColor=colors.HexColor("#243B53")
        )

        subheading_style = ParagraphStyle(
            "FairLensSubHeading",
            parent=styles["Heading3"],
            fontSize=11,
            leading=14,
            spaceBefore=8,
            spaceAfter=6,
            textColor=colors.HexColor("#334E68")
        )

        normal_style = ParagraphStyle(
            "FairLensNormal",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            spaceAfter=6
        )

        small_style = ParagraphStyle(
            "FairLensSmall",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#555555")
        )

        footer_style = ParagraphStyle(
            "FairLensFooter",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#777777")
        )

        # ====================================================
        # HELPER FUNCTIONS
        # ====================================================

        def make_table(
            data,
            widths,
            header=True,
            font_size=8
        ):

            converted = []

            for row_index, row in enumerate(data):

                converted_row = []

                for value in row:

                    if isinstance(value, Paragraph):
                        converted_row.append(value)
                    else:
                        converted_row.append(
                            pdf_cell(
                                value,
                                font_size
                            )
                        )

                converted.append(
                    converted_row
                )

            table = Table(
                converted,
                colWidths=widths,
                repeatRows=1 if header else 0,
                hAlign="LEFT"
            )

            style_commands = [

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#B8C2CC")
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ]

            if header:

                style_commands.extend([

                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#243B53")
                    ),

                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),

                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    )
                ])

            if len(converted) > 1:

                style_commands.append(
                    (
                        "ROWBACKGROUNDS",
                        (0, 1 if header else 0),
                        (-1, -1),
                        [
                            colors.white,
                            colors.HexColor("#F7F9FB")
                        ]
                    )
                )

            table.setStyle(
                TableStyle(style_commands)
            )

            return table

        def add_section_title(number, title):

            elements.append(
                Paragraph(
                    f"{number}. {title}",
                    heading_style
                )
            )

        def extract_metrics(data):

            rows = []

            if not isinstance(data, dict):
                return rows

            for feature, result in data.items():

                if not isinstance(result, dict):
                    continue

                metrics = result.get(
                    "metrics",
                    {}
                )

                if not isinstance(metrics, dict):
                    continue

                for metric_name, value in metrics.items():

                    rows.append([
                        str(feature),
                        str(metric_name),
                        value
                    ])

            return rows

        # ====================================================
        # DOCUMENT ELEMENTS
        # ====================================================

        elements = []

        # ====================================================
        # TITLE
        # ====================================================

        elements.append(
            Spacer(1, 10)
        )

        elements.append(
            Paragraph(
                "FairLens Bias Audit Report",
                title_style
            )
        )

        elements.append(
            Paragraph(
                "Fairness, Bias Detection, Mitigation and "
                "Explainability Analysis",
                subtitle_style
            )
        )

        # ====================================================
        # DATASET INFORMATION
        # ====================================================

        add_section_title(
            1,
            "Dataset Information"
        )

        df = dataset.get("processed_df")

        dataset_name = dataset.get(
            "filename",
            "Unknown"
        )

        task_type = dataset.get(
            "task_type",
            "Unknown"
        )

        target = dataset.get(
            "target",
            "Unknown"
        )

        sensitive_features = dataset.get(
            "sensitive",
            []
        )

        dataset_info = [
            ["Dataset", dataset_name],
            [
                "Rows",
                len(df) if df is not None else "N/A"
            ],
            [
                "Features",
                dataset["X_train"].shape[1]
                if "X_train" in dataset
                else "N/A"
            ],
            ["Target", target],
            ["Task Type", task_type],
            [
                "Sensitive Features",
                ", ".join(
                    map(str, sensitive_features)
                )
                if sensitive_features
                else "None"
            ]
        ]

        elements.append(
            make_table(
                dataset_info,
                [150, 350],
                header=False,
                font_size=9
            )
        )

        elements.append(
            Spacer(1, 18)
        )

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        add_section_title(
            2,
            "Model Information"
        )

        model_name = dataset.get(
            "model_name",
            "Unknown"
        )

        model_info = [
            ["Original Model", model_name],
            ["Task Type", task_type]
        ]

        if dataset.get("mitigation_method"):

            model_info.append([
                "Mitigation Technique",
                dataset.get(
                    "mitigation_method"
                )
            ])

        if dataset.get("mitigation_feature"):

            model_info.append([
                "Sensitive Feature",
                dataset.get(
                    "mitigation_feature"
                )
            ])

        elements.append(
            make_table(
                model_info,
                [180, 320],
                header=False,
                font_size=9
            )
        )

        elements.append(
            Spacer(1, 18)
        )

        # ====================================================
        # ORIGINAL MODEL EVALUATION
        # ====================================================

        add_section_title(
            3,
            "Original Model Evaluation"
        )

        evaluation_results = dataset.get(
            "evaluation_results",
            {}
        )

        # ----------------------------------------------------
        # Classification evaluation
        # ----------------------------------------------------

        if task_type == "classification" and isinstance(
            evaluation_results,
            dict
        ):

            # ================================================
            # CLASSIFICATION REPORT
            # ================================================

            classification_report_data = evaluation_results.get(
                "classification_report"
            )

            if isinstance(classification_report_data, dict):

                elements.append(
                    Paragraph(
                        "Classification Report",
                        ParagraphStyle(
                            "SubHeading",
                            parent=heading_style,
                            fontSize=11,
                            spaceAfter=8
                        )
                    )
                )

                classification_rows = [
                    [
                        "Class",
                        "Precision",
                        "Recall",
                        "F1-Score",
                        "Support"
                    ]
                ]

                for label, metrics in classification_report_data.items():

                    # Skip non-class dictionary values if necessary
                    if not isinstance(metrics, dict):
                        continue

                    classification_rows.append([
                        pdf_cell(label),
                        pdf_cell(
                            metrics.get("precision", "-")
                        ),
                        pdf_cell(
                            metrics.get("recall", "-")
                        ),
                        pdf_cell(
                            metrics.get("f1-score", "-")
                        ),
                        pdf_cell(
                            metrics.get("support", "-")
                        )
                    ])

                if len(classification_rows) > 1:

                    classification_table = Table(
                        classification_rows,
                        colWidths=[
                            100,
                            100,
                            100,
                            100,
                            100
                        ],
                        repeatRows=1
                    )

                    classification_table.setStyle(
                        TableStyle([

                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor("#2F4F4F")
                            ),

                            (
                                "TEXTCOLOR",
                                (0, 0),
                                (-1, 0),
                                colors.white
                            ),

                            (
                                "FONTNAME",
                                (0, 0),
                                (-1, 0),
                                "Helvetica-Bold"
                            ),

                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.5,
                                colors.grey
                            ),

                            (
                                "VALIGN",
                                (0, 0),
                                (-1, -1),
                                "MIDDLE"
                            ),

                            (
                                "ALIGN",
                                (1, 1),
                                (-1, -1),
                                "CENTER"
                            ),

                            (
                                "LEFTPADDING",
                                (0, 0),
                                (-1, -1),
                                5
                            ),

                            (
                                "RIGHTPADDING",
                                (0, 0),
                                (-1, -1),
                                5
                            ),

                            (
                                "TOPPADDING",
                                (0, 0),
                                (-1, -1),
                                5
                            ),

                            (
                                "BOTTOMPADDING",
                                (0, 0),
                                (-1, -1),
                                5
                            ),

                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [
                                    colors.white,
                                    colors.HexColor("#F5F5F5")
                                ]
                            )
                        ])
                    )

                    elements.append(
                        classification_table
                    )

            # ================================================
            # OTHER CLASSIFICATION METRICS
            # ================================================

            classification_summary = []

            for key, value in evaluation_results.items():

                if key == "classification_report":
                    continue

                if isinstance(value, dict):
                    continue

                classification_summary.append([
                    pdf_cell(key),
                    pdf_cell(value)
                ])

            if classification_summary:

                elements.append(
                    Spacer(1, 12)
                )

                elements.append(
                    Paragraph(
                        "Overall Classification Metrics",
                        ParagraphStyle(
                            "SubHeading2",
                            parent=heading_style,
                            fontSize=11,
                            spaceAfter=8
                        )
                    )
                )

                summary_rows = [
                    [
                        "Metric",
                        "Value"
                    ]
                ]

                summary_rows.extend(
                    classification_summary
                )

                summary_table = Table(
                    summary_rows,
                    colWidths=[
                        250,
                        250
                    ],
                    repeatRows=1
                )

                summary_table.setStyle(
                    TableStyle([

                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor("#2F4F4F")
                        ),

                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white
                        ),

                        (
                            "FONTNAME",
                            (0, 0),
                            (-1, 0),
                            "Helvetica-Bold"
                        ),

                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey
                        ),

                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP"
                        ),

                        (
                            "LEFTPADDING",
                            (0, 0),
                            (-1, -1),
                            5
                        ),

                        (
                            "RIGHTPADDING",
                            (0, 0),
                            (-1, -1),
                            5
                        ),

                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            5
                        ),

                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            5
                        ),

                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [
                                colors.white,
                                colors.HexColor("#F5F5F5")
                            ]
                        )
                    ])
                )

                elements.append(
                    summary_table
                )

        # ----------------------------------------------------
        # Regression evaluation
        # ----------------------------------------------------

        elif task_type == "regression" and isinstance(
            evaluation_results,
            dict
        ):

            regression_rows = [
                [
                    "Metric",
                    "Value"
                ]
            ]

            for key, value in evaluation_results.items():

                if isinstance(value, (dict, list)):
                    continue

                regression_rows.append([
                    pdf_cell(key),
                    pdf_cell(value)
                ])

            if len(regression_rows) > 1:

                regression_table = Table(
                    regression_rows,
                    colWidths=[
                        250,
                        250
                    ],
                    repeatRows=1
                )

                regression_table.setStyle(
                    TableStyle([

                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor("#2F4F4F")
                        ),

                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white
                        ),

                        (
                            "FONTNAME",
                            (0, 0),
                            (-1, 0),
                            "Helvetica-Bold"
                        ),

                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey
                        ),

                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP"
                        ),

                        (
                            "PADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        ),

                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [
                                colors.white,
                                colors.HexColor("#F5F5F5")
                            ]
                        )
                    ])
                )

                elements.append(
                    regression_table
                )

        else:

            elements.append(
                Paragraph(
                    "No evaluation metrics available.",
                    normal_style
                )
            )

        elements.append(
            Spacer(1, 20)
        )

        # ====================================================
        # BIAS DETECTION
        # ====================================================

        add_section_title(
            4,
            "Bias Detection"
        )

        bias_detected = bool(
            dataset.get(
                "bias_detected",
                False
            )
        )

        status_text = (
            "Bias was detected in at least one "
            "selected sensitive feature."
            if bias_detected
            else
            "No significant bias was detected "
            "using the selected fairness criteria."
        )

        elements.append(
            Paragraph(
                f"<b>Overall Bias Status:</b> "
                f"{'Bias Detected' if bias_detected else 'No Significant Bias Detected'}",
                normal_style
            )
        )

        elements.append(
            Paragraph(
                status_text,
                normal_style
            )
        )

        fairness_results = dataset.get(
            "fairness_results",
            {}
        )

        fairness_rows = [
            [
                "Sensitive Attribute",
                "Fairness Metric",
                "Value"
            ]
        ]

        for feature, metric, value in extract_metrics(
            fairness_results
        ):

            fairness_rows.append([
                feature,
                metric,
                value
            ])

        if len(fairness_rows) > 1:

            elements.append(
                make_table(
                    fairness_rows,
                    [130, 230, 140],
                    font_size=8
                )
            )

        else:

            elements.append(
                Paragraph(
                    "No fairness results are available.",
                    normal_style
                )
            )

        elements.append(
            Spacer(1, 18)
        )

        # ====================================================
        # CLASS IMBALANCE
        # ====================================================

        class_imbalance = dataset.get(
            "class_imbalance"
        )

        if class_imbalance:

            elements.append(
                Paragraph(
                    "Class Imbalance Analysis",
                    subheading_style
                )
            )

            imbalance_detected = class_imbalance.get(
                "detected",
                False
            )

            distribution = class_imbalance.get(
                "distribution",
                {}
            )

            imbalance_rows = [
                ["Class", "Proportion"]
            ]

            for label, proportion in distribution.items():

                imbalance_rows.append([
                    label,
                    proportion
                ])

            if len(imbalance_rows) > 1:

                elements.append(
                    make_table(
                        imbalance_rows,
                        [250, 250],
                        font_size=8
                    )
                )

            elements.append(
                Spacer(1, 10)
            )

            elements.append(
                Paragraph(
                    f"<b>Overall Class Imbalance:</b> "
                    f"{'Detected' if imbalance_detected else 'Not Detected'}",
                    normal_style
                )
            )

        # ====================================================
        # MITIGATION ANALYSIS
        # ====================================================

        add_section_title(
            5,
            "Mitigation Analysis"
        )

        before_after = dataset.get(
            "before_after_fairness",
            {}
        )

        before = before_after.get(
            "before",
            {}
        )

        after = before_after.get(
            "after",
            {}
        )

        before_rows = extract_metrics(
            before
        )

        after_rows = extract_metrics(
            after
        )

        after_lookup = {}

        for feature, metric, value in after_rows:

            after_lookup[
                (feature, metric)
            ] = value

        comparison_data = [
            [
                "Sensitive Attribute",
                "Fairness Metric",
                "Before",
                "After"
            ]
        ]

        for feature, metric, before_value in before_rows:

            after_value = after_lookup.get(
                (feature, metric),
                "-"
            )

            comparison_data.append([
                feature,
                metric,
                before_value,
                after_value
            ])

        if len(comparison_data) > 1:

            elements.append(
                make_table(
                    comparison_data,
                    [120, 190, 95, 95],
                    font_size=7
                )
            )

        else:

            elements.append(
                Paragraph(
                    "No before-and-after fairness results "
                    "are available.",
                    normal_style
                )
            )

        elements.append(
            Spacer(1, 12)
        )

        mitigation_method = dataset.get(
            "mitigation_method"
        )

        mitigation_feature = dataset.get(
            "mitigation_feature"
        )

        if mitigation_method:

            mitigation_text = (
                f"<b>Applied Mitigation:</b> "
                f"{mitigation_method}"
            )

            if mitigation_feature:

                mitigation_text += (
                    f"<br/><b>Target Sensitive Feature:</b> "
                    f"{mitigation_feature}"
                )

        else:

            mitigation_text = (
                "<b>Applied Mitigation:</b> "
                "No mitigation technique has been applied."
            )

        elements.append(
            Paragraph(
                mitigation_text,
                normal_style
            )
        )

        elements.append(
            Spacer(1, 18)
        )

        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        add_section_title(
            6,
            "Recommendations"
        )

        recommendations = dataset.get(
            "recommendations",
            {}
        )

        recommendation_rows = [
            [
                "Sensitive Attribute",
                "Bias",
                "Recommended Strategy",
                "Reason"
            ]
        ]

        if isinstance(
            recommendations,
            dict
        ):

            for feature, result in recommendations.items():

                if not isinstance(result, dict):
                    continue

                recommendation_bias = result.get(
                    "bias_detected",
                    False
                )

                strategy = result.get(
                    "strategy"
                )

                reason = result.get(
                    "reason",
                    "No significant bias was detected."
                )

                if strategy is None:
                    strategy = (
                        "No mitigation required"
                    )

                recommendation_rows.append([
                    feature,
                    "Yes"
                    if recommendation_bias
                    else "No",
                    strategy,
                    reason
                ])

        if len(recommendation_rows) > 1:

            elements.append(
                make_table(
                    recommendation_rows,
                    [105, 55, 130, 210],
                    font_size=7
                )
            )

        else:

            elements.append(
                Paragraph(
                    "No specific mitigation recommendation "
                    "was generated.",
                    normal_style
                )
            )

        elements.append(
            Spacer(1, 18)
        )

        # ====================================================
        # EXPLAINABILITY
        # ====================================================

        add_section_title(
            7,
            "Explainability"
        )

        explainability = dataset.get(
            "explainability",
            {}
        )

        feature_importance = explainability.get(
            "feature_importance",
            []
        )

        importance_rows = [
            [
                "Feature",
                "SHAP Importance"
            ]
        ]

        cleaned_importance = []

        if isinstance(
            feature_importance,
            list
        ):

            for item in feature_importance:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                feature_name = item.get(
                    "feature",
                    item.get(
                        "Feature",
                        ""
                    )
                )

                importance = item.get(
                    "importance",
                    item.get(
                        "Importance",
                        item.get(
                            "SHAP Importance",
                            0
                        )
                    )
                )

                try:
                    importance = float(
                        importance
                    )
                except Exception:
                    importance = 0.0

                cleaned_importance.append([
                    str(feature_name),
                    importance
                ])

        cleaned_importance.sort(
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for feature_name, importance in (
            cleaned_importance[:10]
        ):

            importance_rows.append([
                feature_name,
                f"{importance:.4f}"
            ])

        if len(importance_rows) > 1:

            elements.append(
                make_table(
                    importance_rows,
                    [330, 170],
                    font_size=8
                )
            )

            elements.append(
                Spacer(1, 8)
            )

            elements.append(
                Paragraph(
                    "SHAP importance represents the relative "
                    "influence of each feature on the model's "
                    "predictions. Higher absolute values indicate "
                    "greater contribution to the model's output.",
                    small_style
                )
            )

        else:

            elements.append(
                Paragraph(
                    "No feature importance information "
                    "is available for this model.",
                    normal_style
                )
            )

        elements.append(
            Spacer(1, 18)
        )

        # ====================================================
        # CONCLUSION
        # ====================================================

        add_section_title(
            8,
            "Conclusion"
        )

        bias_status = (
            "bias was detected"
            if bias_detected
            else
            "no significant bias was detected"
        )

        if mitigation_method:

            conclusion = (
                f"The {model_name} model was evaluated "
                f"for fairness across the selected "
                f"sensitive features. The audit indicated "
                f"that {bias_status}. The "
                f"{mitigation_method} technique was applied"
            )

            if mitigation_feature:

                conclusion += (
                    f" to the sensitive feature "
                    f"{mitigation_feature}"
                )

            conclusion += (
                ". The before-and-after fairness metrics "
                "are presented in this report to support "
                "comparison of the model's fairness "
                "performance."
            )

        else:

            conclusion = (
                f"The {model_name} model was evaluated "
                f"for fairness across the selected "
                f"sensitive features. The audit indicated "
                f"that {bias_status}. No mitigation "
                f"technique was applied during this audit."
            )

        elements.append(
            Paragraph(
                conclusion,
                normal_style
            )
        )

        elements.append(
            Spacer(1, 25)
        )

        # ====================================================
        # FOOTER
        # ====================================================

        elements.append(
            Paragraph(
                "Generated by FairLens - Bias Auditing Toolkit",
                footer_style
            )
        )

        # ====================================================
        # BUILD PDF
        # ====================================================

        doc.build(
            elements
        )

        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=(
                "fairlens_bias_audit_report.pdf"
            ),
            mimetype="application/pdf"
        )

    except Exception as e:

        import traceback

        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

# ============================================================
# MODEL DOWNLOADS
# ============================================================

@app.route("/api/download/original-model/<session_id>", methods=["GET"])
def download_original_model(session_id):

    try:
        if session_id not in DATASETS:
            return jsonify({
                "ok": False,
                "error": "Dataset session not found"
            }), 404

        dataset = DATASETS[session_id]

        if "trained_model" not in dataset:
            return jsonify({
                "ok": False,
                "error": "No trained model available."
            }), 400

        model_package = {
            "model": dataset["trained_model"],
            "model_name": dataset.get("model_name"),
            "task_type": dataset.get("task_type"),
            "target": dataset.get("target"),
            "sensitive_features": dataset.get(
                "sensitive", []
            ),
            "encoders": dataset.get("encoders"),
            "scaler": dataset.get("scaler"),
            "label_encoder": dataset.get(
                "label_encoder"
            )
        }

        buffer = BytesIO()

        joblib.dump(model_package, buffer)

        buffer.seek(0)

        model_name = dataset.get(
            "model_name",
            "original_model"
        )

        filename = (
            f"FairLens_{model_name}"
            f"_original.joblib"
        )

        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype="application/octet-stream"
        )

    except Exception as e:

        import traceback
        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500
    

@app.route("/api/download/mitigated-model/<session_id>", methods=["GET"])
def download_mitigated_model(session_id):

    try:
        if session_id not in DATASETS:
            return jsonify({
                "ok": False,
                "error": "Dataset session not found"
            }), 404

        dataset = DATASETS[session_id]

        if "mitigated_model" not in dataset:
            return jsonify({
                "ok": False,
                "error": (
                    "No mitigated model available. "
                    "Apply a mitigation technique first."
                )
            }), 400

        model_package = {
            "model": dataset["mitigated_model"],
            "model_name": dataset.get(
                "model_name"
            ),
            "task_type": dataset.get(
                "task_type"
            ),
            "target": dataset.get(
                "target"
            ),
            "sensitive_features": dataset.get(
                "sensitive", []
            ),
            "mitigation_method": dataset.get(
                "mitigation_method"
            ),
            "mitigation_feature": dataset.get(
                "mitigation_feature"
            ),
            "encoders": dataset.get(
                "encoders"
            ),
            "scaler": dataset.get(
                "scaler"
            ),
            "label_encoder": dataset.get(
                "mitigation_label_encoder",
                dataset.get("label_encoder")
            )
        }

        buffer = BytesIO()

        joblib.dump(
            model_package,
            buffer
        )

        buffer.seek(0)

        technique = dataset.get(
            "mitigation_method",
            "mitigated"
        )

        filename = (
            f"FairLens_{technique}"
            f"_mitigated.joblib"
        )

        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype="application/octet-stream"
        )

    except Exception as e:

        import traceback
        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(
        debug=False,
        port=5000
    )