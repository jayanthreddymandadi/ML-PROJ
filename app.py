import streamlit as st
import pandas as pd
import random
import time
from datetime import date

# --- Placeholder classes ---
class CustomData:
    def __init__(self, weather, road_condition, time_of_day, traffic, accident_type,
                 latitude, longitude, accident_date):
        self.weather = weather
        self.road_condition = road_condition
        self.time_of_day = time_of_day
        self.traffic = traffic
        self.accident_type = accident_type
        self.latitude = latitude
        self.longitude = longitude
        self.accident_date = accident_date

    def get_data_as_data_frame(self):
        return pd.DataFrame({
            'weather': [self.weather],
            'road_condition': [self.road_condition],
            'time_of_day': [self.time_of_day],
            'traffic': [self.traffic],
            'accident_type': [self.accident_type],
            'latitude': [self.latitude],
            'longitude': [self.longitude],
            'date': [self.accident_date]
        })


class PredictPipeline:
    def predict(self, features):
        severities = ["Low", "Medium", "High"]
        return [random.choice(severities)]


# --- Streamlit Page Config ---
st.set_page_config(page_title="Accident Severity Predictor", page_icon="🚦", layout="centered")

st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#FFDD57;">🚦 Accident Severity Prediction</h1>
        <p style="font-size:18px; color:white;">Enter accident details below.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Input Form ---
st.subheader("📋 Enter Accident Details")
with st.form("prediction_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        weather = st.selectbox("🌤 Weather", ["Select Weather", "Clear", "Rainy", "Foggy", "Snowy"])
        time_of_day = st.selectbox("⏰ Time of Day", ["Select Time", "Morning", "Afternoon", "Evening", "Night"])
        traffic = st.selectbox("🚗 Traffic Level", ["Select Traffic", "Low", "Medium", "High"])
        latitude = st.number_input("📍 Latitude", min_value=-90.0, max_value=90.0, value=20.5937)

    with col2:
        road_condition = st.selectbox("🛣 Road Condition", ["Select Road Condition", "Dry", "Wet", "Icy", "Snowy"])
        accident_type = st.selectbox("💥 Accident Type", ["Select Type", "Rear-end", "Head-on", "Side-impact", "Rollover"])
        longitude = st.number_input("📍 Longitude", min_value=-180.0, max_value=180.0, value=78.9629)
        accident_date = st.date_input("📅 Date of Accident", value=date.today())

    submitted = st.form_submit_button("🔮 Predict Severity")

# --- Prediction Logic ---
if submitted:
    if (
        weather.startswith("Select") or
        road_condition.startswith("Select") or
        time_of_day.startswith("Select") or
        traffic.startswith("Select") or
        accident_type.startswith("Select")
    ):
        st.warning("⚠ Please select all fields before predicting.")
    else:
        with st.spinner("Analyzing accident details... 🔍"):
            time.sleep(2)

        # Prepare data and predict
        data = CustomData(weather, road_condition, time_of_day, traffic, accident_type, latitude, longitude, accident_date)
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)[0]

        # Colors & enhanced awareness captions
        severity_color = {"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E74C3C"}
        severity_messages = {
            "Low": "✅ Minor Accident: Stay alert but no immediate danger.",
            "Medium": "⚠ Medium Severity: Take safety precautions, drive carefully.",
            "High": "🚨 High Severity: Emergency response may be needed immediately!"
        }
        severity_icons = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}

        # Display result with enhanced visuals
        st.markdown(
            f"""
            <div style="background-color:{severity_color[results]};
                        padding:25px; border-radius:15px; text-align:center;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.4);">
                <h2 style="color:white;">{severity_icons[results]} Predicted Severity: {results}</h2>
                <p style="color:white; font-size:16px;">{severity_messages[results]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("📊 Input Data")
        st.dataframe(pred_df)
