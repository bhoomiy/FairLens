from sklearn.linear_model import (LogisticRegression,LinearRegression)
from sklearn.tree import (DecisionTreeClassifier,DecisionTreeRegressor)
from sklearn.ensemble import (RandomForestClassifier,RandomForestRegressor)
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import pandas as pd

def detect_task_type(y):
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 10:
            return "classification", (
                f"Target is numerical but contains only "
                f"{y.nunique()} unique values."
            )
        return "regression", (
            f"Target is numerical and contains "
            f"{y.nunique()} unique values."
        )
    return "classification", (
        "Target contains categorical values."
    )

def build_linear_regression(X_train,y_train):
    model=LinearRegression()
    model.fit(X_train,y_train)
    return model

def build_logistic_regression(X_train,y_train):
    model=LogisticRegression()
    model.fit(X_train,y_train)
    return model

def build_decision_tree_regressor(X_train,y_train):
    model=DecisionTreeRegressor(max_depth=4,random_state=42)
    model.fit(X_train,y_train)
    return model

def build_decision_tree_classifier(X_train,y_train):
    model=DecisionTreeClassifier(max_depth=4,random_state=42)
    model.fit(X_train,y_train)
    return model

def build_random_forest_regressor(X_train,y_train):
    model=RandomForestRegressor(max_depth=4,random_state=42)
    model.fit(X_train,y_train)
    return model

def build_random_forest_classifier(X_train,y_train):
    model=RandomForestClassifier(max_depth=4,random_state=42)
    model.fit(X_train,y_train)
    return model

def build_xgboost_classifier(X_train,y_train):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)
    model=xgb.XGBClassifier(n_estimators=100,max_depth=3,eval_metric="logloss",random_state=42,learning_rate=0.1)
    model.fit(X_train,y_encoded)
    return model,encoder

def build_xgboost_regressor(X_train,y_train):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)
    model=xgb.XGBRegressor(n_estimators=100,max_depth=3,eval_metric="logloss",random_state=42,learning_rate=0.1)
    model.fit(X_train,y_encoded)
    return model,encoder



