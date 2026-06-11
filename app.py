import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import boto3, os
from pathlib import Path


# --------------
# Config
# --------------

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
S3_BUCKET = os.getenv("S3_BUCKET", "wind-power-forecasting-data")
REGION = os.getenv("AWS_REGION", "eu-central-1")

s3 = boto3.client("s3", region_name=REGION)

def load_from_s3(key, local_path):
    """
    Download files from S3 if not already cached locally
    """
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        st.info(f"Downloading {key} from S3")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)

# Paths (ensure available locally by fetching from S3 if missing)
HOLDOUT_ENGINEERED_PATH = load_from_s3(
    "processed/holdout.csv",
    "data/processed/holdout.csv"
)
HOLDOUT_RAW_PATH = load_from_s3(
    "raw/cleaned_holdout.csv",
    "data/raw/cleaned_holdout.csv"
)

# --------------
# Data Loading
# --------------

@st.cache_data
def load_data():
    fe_df = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    raw_df = pd.read_csv(HOLDOUT_RAW_PATH)

    if len(fe_df) != len(raw_df):
        st.warning("Engineered and raw holdout lengths differ. Aligning by index")
        min_len = min(len(fe_df), len(raw_df))
        fe_df = fe_df.iloc[:min_len].copy()
        raw_df = raw_df.iloc[:min_len].copy()

    disp_df = pd.DataFrame(index=fe_df.index)
    disp_df["time"] = raw_df["time"]
    disp_df["time"] = pd.to_datetime(disp_df["time"])
    disp_df["year"] = disp_df["time"].dt.year
    disp_df["month"] = disp_df["time"].dt.month
    disp_df["day"] = disp_df["time"].dt.day
    #disp_df["temperature_2m"] = raw_df["temperature_2m"]
    #disp_df["relativehumidity_2m"] = raw_df["relativehumidity_2m"]
    disp_df["windspeed_100m"] = raw_df["windspeed_100m"]
    disp_df["actual_power"] = fe_df["power"]

    return fe_df, raw_df, disp_df

fe_df, raw_df, disp_df = load_data()

# --------------
# UI
# --------------

st.title("Wind Power Forecasting")

days = sorted(disp_df["day"].unique())
months = sorted(disp_df["month"].unique())
years = sorted(disp_df["year"].unique())

col1, col2, col3 = st.columns(3)

with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    day = st.selectbox("Select Day", days, index=0)

if st.button("Show Predictions"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month) & (disp_df["day"] == day)
    idx = disp_df.index[mask]
    if len(idx) == 0:
        st.warning("No data found for these filters")
    else:
        st.write(f"Running predictions for **{year}-{month:02d}-{day:02d}**")
        payload = raw_df.loc[idx].to_dict(orient="records")

        try:
            resp = requests.post(API_URL, json = payload, timeout=60)
            print("Status:", resp.status_code)
            print("Response:", resp.text)
            resp.raise_for_status()
            output = resp.json()
            preds = output.get("predictions", [])
            actuals = output.get("actuals", None)

            view = disp_df.loc[idx, ["time", "actual_power"]].copy()
            view = view.sort_values("time")
            view["prediction"] = pd.Series(preds, index=view.index).astype(float)

            if actuals is not None and len(actuals) == len(view):
                view["actual_power"] = pd.Series(actuals, index=view.index).astype(float)

            # Metrics
            mae = (view["prediction"] - view["actual_power"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_power"]) ** 2).mean() ** 0.5
            avg_pct_error = ((view["prediction"] - view["actual_power"]).abs() / view["actual_power"]).mean()

            st.subheader("Predictions vs Actuals")
            st.dataframe(
                view[["time", "actual_power", "prediction"]].reset_index(drop=True),
                use_container_width=True
            )
            c1, c2, c3 =st.columns(3)
            with c1:
                st.metric("MAE", f"{mae:,.2f}")
            with c2:
                st.metric("RMSE", f"{rmse:,.2f}")
            with c3:
                st.metric("Avg % Error", f"{avg_pct_error:.2f}%")

            # -------------
            # Trend Charts
            # -------------

            yearly_data = disp_df[disp_df["year"] == year].copy()
            idx_all = yearly_data.index
            payload_all = raw_df.loc[idx_all].to_dict(orient="records")

            resp_all = requests.post(API_URL, json=payload_all, timeout=60)
            resp_all.raise_for_status()
            preds_all = resp_all.json().get("predictions", [])

            yearly_data["prediction"] = pd.Series(preds_all, index=yearly_data.index).astype(float)
            
            # Daily Aggregation
            daily_avg = yearly_data.groupby("day")[["actual_power", "prediction"]].mean().reset_index()

            # Highlight selected day
            daily_avg["highlight"] = daily_avg["day"].apply(lambda d: "Selected" if d == day else "")

            fig = px.line(
                daily_avg,
                x = "day",
                y = ["actual_power", "prediction"],
                markers = True,
                labels = {"value": "power", "day": "day"},
                title = f"Yearly Trend - {year}"
            )
            # Add highlight with background shading
            highlight_day = day
            fig.add_vrect(
                x0 = highlight_day - 0.5,
                x1 = highlight_day + 0.5,
                fillcolor = "red",
                opacity=0.1,
                layer="below",
                line_width=0
                )
            
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"API call failed: {e}")
            st.exception(e)
    
else:
    st.info("Choose filters and click **Show Predictions** to compute.")