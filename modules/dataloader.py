import streamlit as st
import pandas as pd

def upload_dataset():
    uploaded_file=st.file_uploader(
        "Upload your dataset here",
        type=["csv"]
    )

    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
        return df

    return None