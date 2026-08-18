import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler

def dataset_summary(df):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum(),
        "Categorical Columns": list(df.select_dtypes(include=["object", "category"]).columns),
        "Numerical Columns": list(df.select_dtypes(include=["number"]).columns),
    }

def detect_missing_values(df):
    return df.isnull().sum()

def handle_missing_values(df,strategy="mean"):
    df=df.copy()

    numerical=df.select_dtypes(include=["int64","float64"]).columns
    categorical=df.select_dtypes(include=["object","category"]).columns

    if strategy=="drop":
        df=df.dropna()

    elif strategy=="mean":
        for col in numerical:
            df[col]=df[col].fillna(df[col].mean())
        for col in categorical:
            df[col]=df[col].fillna(df[col].mode()[0])

    elif strategy=="median":
        for col in numerical:
                    df[col]=df[col].fillna(df[col].median())
        for col in categorical:
                    df[col]=df[col].fillna(df[col].mode()[0])

    elif strategy == "mode":
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

    return df

def remove_duplicates(df):
     return df.drop_duplicates()

def encode_categorical(df,strategy="label"):
    df=df.copy()
    if strategy=="label":
        label_encoders={}
        categorical=df.select_dtypes(include=["object","category"]).columns
        for col in categorical:
            encoder=LabelEncoder()
            df[col]=encoder.fit_transform(df[col].astype(str))
            label_encoders[col]=encoder
        return df,label_encoders
    
    elif strategy=="one-hot":
         df = pd.get_dummies(df, drop_first=True)
         return df,None
     

def scale_features(df,method="standard"):
    df=df.copy()
    numerical=df.select_dtypes(include=["int64", "float64"]).columns
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        return df, None
    
    df[numerical] = scaler.fit_transform(df[numerical])
    return df, scaler

def split_dataset(
    X,
    y,
    test_size=0.2,
    random_state=42
):

    # Check whether stratification is possible
    class_counts = y.value_counts()

    if class_counts.min() >= 2:
        stratify_value = y
    else:
        stratify_value = None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_value
    )