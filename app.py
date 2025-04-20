
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import openpyxl

st.set_page_config(page_title="Smart City Traffic Dashboard", layout="wide")

st.title("🚦 Smart City Traffic Intelligence Dashboard")

# Upload Section
st.sidebar.header("Upload Datasets")
train_file = st.sidebar.file_uploader("Upload Training CSV", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=["DateTime"])
    df["hour"] = df["DateTime"].dt.hour
    df["day_of_week"] = df["DateTime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["is_holiday"] = df["DateTime"].dt.date.astype(str).isin([
        "2015-10-02", "2015-12-25", "2016-01-01", "2016-08-15", "2016-10-02"
    ]).astype(int)
    return df

if train_file:
    train_df = load_data(train_file)
    st.subheader("📊 Training Dataset Preview")
    st.dataframe(train_df.head())

    st.markdown("### 🚗 Traffic Volume by Junction")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x="Junction", y="Vehicles", data=train_df, ax=ax1)
    st.pyplot(fig1)

    st.markdown("### 📈 Average Traffic Volume by Hour")
    avg_hourly = train_df.groupby("hour")["Vehicles"].mean()
    fig2, ax2 = plt.subplots()
    avg_hourly.plot(kind="line", ax=ax2)
    st.pyplot(fig2)

    st.markdown("### 🔍 Holiday vs Non-Holiday Traffic")
    fig3, ax3 = plt.subplots()
    sns.barplot(x="is_holiday", y="Vehicles", data=train_df, ax=ax3)
    ax3.set_xticklabels(["Non-Holiday", "Holiday"])
    st.pyplot(fig3)

    # ML Section
    st.subheader("🤖 Predict Traffic Volume")
    if test_file:
        test_df = load_data(test_file)
        features = ["hour", "day_of_week", "is_weekend", "is_holiday", "Junction"]
        model = LinearRegression()
        model.fit(train_df[features], train_df["Vehicles"])
        test_df["Predicted_Vehicles"] = model.predict(test_df[features])

        st.markdown("### 🧪 Test Data with Predictions")
        st.dataframe(test_df[["DateTime", "Junction", "Predicted_Vehicles"]].head())

        # Download predictions
        if st.button("⬇️ Export Predictions to Excel"):
            export_df = test_df[["DateTime", "Junction", "Predicted_Vehicles"]]
            export_path = "traffic_predictions.xlsx"
            export_df.to_excel(export_path, index=False)
            with open(export_path, "rb") as f:
                st.download_button("Download Excel", f, file_name="predictions.xlsx")
