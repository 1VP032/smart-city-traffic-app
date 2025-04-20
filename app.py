import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
import datetime
import io
import openpyxl
from sklearn.linear_model import LinearRegression
import streamlit_authenticator as stauth
from datetime import datetime as dt

# -------- AUTHENTICATION --------
st.set_page_config(page_title="Smart City Dashboard", layout="wide")
authenticator = stauth.Authenticate(
    credentials={"usernames": {"admin": {"name": "Admin", "password": "adminpass"}}},
    cookie_name="smartcityauth", key="key", cookie_expiry_days=1
)

name, auth_status, username = authenticator.login("Login", "main")
if not auth_status:
    st.warning("Please login to access the dashboard.")
    st.stop()
elif auth_status is None:
    st.info("Enter your credentials.")
    st.stop()

authenticator.logout("Logout", "sidebar")

# -------- PAGE NAVIGATION --------
page = st.sidebar.radio("Navigate", ["Dashboard", "Prediction", "Export Anomalies", "About"])

# -------- LOAD DATA --------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=["DateTime"])
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['is_holiday'] = df['DateTime'].dt.date.apply(lambda x: x in holiday_list)
    return df

# Simulated holiday list (you can replace with real holidays)
holiday_list = [dt(2017, 10, 2).date(), dt(2017, 12, 25).date(), dt(2018, 1, 1).date()]

st.sidebar.header("Upload Files")
train_file = st.sidebar.file_uploader("Upload train.csv", type=["csv"])
test_file = st.sidebar.file_uploader("Upload test.csv", type=["csv"])

if train_file:
    train = load_data(train_file)

    if page == "Dashboard":
        st.title("🚦 Smart City Traffic Dashboard")
        st.subheader("Traffic Overview by Junction")

        junction = st.selectbox("Select Junction", sorted(train["Junction"].unique()))
        holiday_filter = st.checkbox("Show only holidays")

        filtered = train[train["Junction"] == junction]
        if holiday_filter:
            filtered = filtered[filtered["is_holiday"] == True]

        st.line_chart(filtered.groupby("DateTime")["Vehicles"].sum())

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.boxplot(data=filtered, x="hour", y="Vehicles", ax=ax)
            st.pyplot(fig)

        with col2:
            heatmap_data = filtered.pivot_table(index="day_of_week", columns="hour", values="Vehicles", aggfunc="mean")
            st.write("Traffic Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
            st.pyplot(fig)

    elif page == "Prediction":
        st.title("🔮 Predict Traffic Volume")

        if test_file:
            test = load_data(test_file)

            X_train = train[["hour", "day_of_week", "is_holiday"]].astype(int)
            y_train = train["Vehicles"]
            model = LinearRegression().fit(X_train, y_train)

            X_test = test[["hour", "day_of_week", "is_holiday"]].astype(int)
            test["Predicted"] = model.predict(X_test).round()

            st.write("Prediction Results")
            st.dataframe(test[["DateTime", "Junction", "Predicted"]])

            st.line_chart(test.set_index("DateTime")[["Predicted"]])

    elif page == "Export Anomalies":
        st.title("📁 Export Anomalous Traffic Points")

        threshold = st.slider("Set anomaly threshold (vehicles)", 100, 500, 300)
        anomalies = train[train["Vehicles"] > threshold]

        if st.button("📥 Download as Excel"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                anomalies.to_excel(writer, index=False, sheet_name="Anomalies")
            st.download_button("Download Excel", data=buffer.getvalue(), file_name="anomalies.xlsx")

        st.write(anomalies.head(10))

    elif page == "About":
        st.title("ℹ️ About This Project")
        st.markdown("""
        This dashboard is built to empower smart cities by visualizing, predicting, and detecting anomalies in urban traffic.
        
        **Features**:
        - Upload training and test datasets
        - Analyze patterns by junction and time
        - Build and apply ML predictions
        - Export anomaly reports
        - Deploy with Docker, GitHub CI, or Streamlit Cloud

        _Developed by Vedant Patel_
        """)
else:
    st.info("Please upload the training dataset from sidebar to continue.")
