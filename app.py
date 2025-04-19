import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Smart City Traffic Patterns", layout="wide")
st.title("🚦 Smart City Traffic Patterns Dashboard")

# File Upload
st.sidebar.header("Upload CSV Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Basic Statistics
    st.subheader("📈 Data Summary")
    st.write(df.describe(include='all'))

    # Feature Exploration (If columns exist)
    time_col = st.selectbox("Select Time Column (for Time Series)", df.columns)
    traffic_col = st.selectbox("Select Traffic Volume Column", df.columns)

    if time_col and traffic_col:
        st.subheader("📉 Traffic Volume Over Time")
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])
        df_sorted = df.sort_values(by=time_col)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_sorted[time_col], df_sorted[traffic_col], color='blue')
        ax.set_xlabel("Time")
        ax.set_ylabel("Traffic Volume")
        ax.set_title("Traffic Volume Over Time")
        st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("📌 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include='number')
    if not numeric_cols.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
else:
    st.info("Please upload a dataset to begin analysis.")
