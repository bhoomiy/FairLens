import numpy as np
from sklearn.metrics import (accuracy_score,precision_score,r2_score,f1_score,mean_absolute_error,mean_squared_error,recall_score,
                             roc_auc_score,confusion_matrix,classification_report)

def evaluate_classification_models(model,X_test,y_test):
    y_pred=model.predict(X_test)
    # Get class labels
    labels = np.unique(y_test)
    results={
        "accuracy": accuracy_score(y_test,y_pred),
        "precision":precision_score(y_test,y_pred,average="weighted", zero_division=0),
        "confusion_matrix":confusion_matrix(y_test,y_pred),
        "recall_score":recall_score(y_test,y_pred,average="weighted", zero_division=0),
        "classification_report":classification_report(y_test,y_pred,output_dict=True,zero_division=0),
        "f1_score":f1_score(y_test,y_pred,average="weighted", zero_division=0),
        "Labels":labels
    }
    # ROC-AUC only when probability estimates are available
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if len(labels) == 2:
                results["ROC-AUC"] = roc_auc_score(
                    y_test,
                    y_prob[:, 1]
                )
            else:
                results["ROC-AUC"] = roc_auc_score(
                    y_test,
                    y_prob,
                    multi_class="ovr",
                    average="weighted"
                )

        else:
            results["ROC-AUC"] = None

    except (ValueError, TypeError, IndexError):
        results["ROC-AUC"] = None

    return results

def evaluate_regression_models(model,X_test,y_test):
     y_pred=model.predict(X_test)
     mse=mean_squared_error(y_test,y_pred)
     results={
          "mae":mean_absolute_error(y_test,y_pred),
          "mse":mse,
          "r2_score":r2_score(y_test,y_pred),
          "rmse": np.sqrt(mse),
        }

     return results