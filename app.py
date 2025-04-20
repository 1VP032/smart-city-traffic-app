import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Smart City Traffic Patterns", layout="wide")
st.title("🚦 Smart City Traffic Patterns Dashboard")

# Upload CSV
df = None
st.sidebar.header("Upload CSV Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Descriptive Statistics")
    st.write(df.describe(include='all'))

    # DateTime conversion and feature engineering
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['date'] = df['DateTime'].dt.date
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        # Create is_holiday column (weekends as proxy)
        df['is_holiday'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Optional Filters
    if 'Junction' in df.columns:
        junctions = df['Junction'].unique().tolist()
        selected_junction = st.sidebar.selectbox("🛣 Select Junction", ['All'] + junctions)
        if selected_junction != 'All':
            df = df[df['Junction'] == selected_junction]

    if 'is_holiday' in df.columns:
        holiday_filter = st.sidebar.radio("📅 Filter by Holiday", ["All", "Holiday", "Non-Holiday"])
        if holiday_filter == "Holiday":
            df = df[df['is_holiday'] == 1]
        elif holiday_filter == "Non-Holiday":
            df = df[df['is_holiday'] == 0]

    # Column selectors
    time_col = st.selectbox("🕒 Select Timestamp Column", df.columns)
    value_col = st.selectbox("📦 Select Traffic Volume Column", [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'ID'])

    if time_col and value_col:
        df.sort_values(by=time_col, inplace=True)

        # Time-Series Chart
        st.subheader("📉 Traffic Volume Over Time")
        line_chart = alt.Chart(df).mark_line().encode(
            x=alt.X(time_col, title='Time'),
            y=alt.Y(value_col, title='Traffic Volume'),
            tooltip=[time_col, value_col]
        ).properties(width=900, height=400).interactive()
        st.altair_chart(line_chart, use_container_width=True)

        # Aggregated Daily Trend
        st.subheader("📊 Daily Average Traffic Volume")
        if 'date' in df.columns:
            daily_avg = df.groupby('date')[value_col].mean().reset_index()
            daily_chart = alt.Chart(daily_avg).mark_area(opacity=0.5).encode(
                x='date:T',
                y=value_col,
                tooltip=['date', value_col]
            ).properties(width=900, height=300)
            st.altair_chart(daily_chart, use_container_width=True)

        # Peak Hour Analysis
        if 'hour' in df.columns:
            st.subheader("⏰ Peak Hour Insight")
            hour_avg = df.groupby('hour')[value_col].mean().reset_index()
            bar_chart = alt.Chart(hour_avg).mark_bar().encode(
                x='hour:O',
                y=value_col,
                tooltip=['hour', value_col]
            ).properties(width=800)
            st.altair_chart(bar_chart, use_container_width=True)

    # Correlation Heatmap
    st.subheader("📌 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include='number')
    if not numeric_cols.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Forecasting using Linear Regression
    st.subheader("🔮 Simple Traffic Prediction (Linear Regression)")
    if 'Vehicles' in df.columns and all(col in df.columns for col in ['hour', 'day_of_week', 'is_holiday']):
        df_model = df[['hour', 'day_of_week', 'is_holiday', 'Vehicles']].dropna()
        X = df_model[['hour', 'day_of_week', 'is_holiday']]
        y = df_model['Vehicles']

        model = LinearRegression()
        model.fit(X, y)

        st.success("✅ Model trained. Predicting traffic volume...")

        test_data = pd.DataFrame({
            'hour': range(0, 24),
            'day_of_week': [0]*24,       # Monday
            'is_holiday': [0]*24         # Non-Holiday
        })
        test_data['Predicted_Vehicles'] = model.predict(test_data)

        pred_chart = alt.Chart(test_data).mark_line().encode(
            x='hour:O',
            y='Predicted_Vehicles:Q'
        ).properties(title="Predicted Vehicles for a Non-Holiday Monday", width=800)
        st.altair_chart(pred_chart, use_container_width=True)

else:
    st.info("📥 Please upload a dataset to get started.")
