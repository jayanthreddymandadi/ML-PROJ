import streamlit as st
import pandas as pd
import random
import time
import folium
from streamlit_folium import st_folium
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

# --- Title ---
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#FFDD57;">🚦 Accident Severity Prediction</h1>
        <p style="font-size:18px; color:white;">Select accident location & details below.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Map Selection ---
st.subheader("📍 Select Location on Custom Map")

m = folium.Map(location=[20.5937, 78.9629], zoom_start=3, tiles=None)

# Custom tile (your map background image)
folium.raster_layers.TileLayer(
    tiles="https://offloadmedia.feverup.com/secretnyc.co/wp-content/uploads/2018/04/20032112/map.jpg",
    attr="Custom Map",
    name="Custom Map",
    overlay=True,
    control=True
).add_to(m)

m.add_child(folium.LatLngPopup())  # clicking gives lat/lon

map_data = st_folium(m, height=500, width=700)

# Default lat/lon
lat, lon = 20.5937, 78.9629

# If clicked on map update lat/lon
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

# Display selected coords
st.success(f"✅ Selected Location: Latitude {lat:.4f}, Longitude {lon:.4f}")

# --- Date Picker ---
st.subheader("📅 Select Date of Accident")
accident_date = st.date_input("Date", value=date.today())

# --- Input Form ---
with st.form("prediction_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        weather = st.selectbox("🌤 Weather", ["Select Weather", "Clear", "Rainy", "Foggy", "Snowy"])
        time_of_day = st.selectbox("⏰ Time of Day", ["Select Time", "Morning", "Afternoon", "Evening", "Night"])
        traffic = st.selectbox("🚗 Traffic Level", ["Select Traffic", "Low", "Medium", "High"])

    with col2:
        road_condition = st.selectbox("🛣 Road Condition", ["Select Road Condition", "Dry", "Wet", "Icy", "Snowy"])
        accident_type = st.selectbox("💥 Accident Type", ["Select Type", "Rear-end", "Head-on", "Side-impact", "Rollover"])

    # Show updated lat/lon inside form (readonly style)
    st.markdown(f"**📍 Latitude:** `{lat:.4f}` &nbsp;&nbsp; **Longitude:** `{lon:.4f}`")

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

        data = CustomData(weather, road_condition, time_of_day, traffic, accident_type, lat, lon, accident_date)
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)[0]

        severity_color = {"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E74C3C"}
        st.markdown(
            f"""
            <div style="background-color:{severity_color[results]};
                        padding:25px; border-radius:15px; text-align:center;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.4);">
                <h2 style="color:white;">🚨 Predicted Severity: {results}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(pred_df)
