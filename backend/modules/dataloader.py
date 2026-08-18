import pandas as pd


def upload_dataset(file):
    if file is None:
        return None

    filename = file.filename.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file)

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    raise ValueError("Unsupported file format")


def dataset_preview(df, rows=5):
    """Return the first rows as JSON-compatible records."""
    return df.head(rows).to_dict(orient="records")