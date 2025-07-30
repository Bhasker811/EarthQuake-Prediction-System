import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
from geopy.geocoders import Nominatim

# Suppress warnings and set safe matplotlib backend
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')

# Configuration
st.set_page_config(
    page_title="Earthquake Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data

def load_data():
    try:
        possible_paths = [
            "dataset.csv",
            "data/dataset.csv",
            r"C:\\Users\\bhasker avusali\\OneDrive\\Desktop\\MAJOR\\dataset.csv"
        ]
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        if df is None:
            st.warning("Dataset not found. Using sample data for demonstration.")
            np.random.seed(42)
            df = pd.DataFrame({
                'Lat': np.random.uniform(-90, 90, 1000),
                'Long': np.random.uniform(-180, 180, 1000),
                'Depth': np.random.uniform(0, 700, 1000),
                'Magnitude': np.random.uniform(1, 9, 1000)
            })
        if 'Magnitude' in df.columns:
            df['Magnitude'] = df['Magnitude'].astype(str)
            df['Magnitude'] = df['Magnitude'].str.extract(r'(\d+\.?\d*)', expand=False)
            df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
        numeric_cols = ['Lat', 'Long', 'Depth', 'Magnitude']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=numeric_cols)
        df = df[np.isfinite(df[numeric_cols]).all(axis=1)]
        df = df[
            (df['Lat'].between(-90, 90)) &
            (df['Long'].between(-180, 180)) &
            (df['Depth'] >= 0) &
            (df['Magnitude'].between(0, 10))
        ]
        df['Lat_Long'] = df['Lat'] * df['Long']
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource

def load_models():
    models = {}
    model_files = {
        'rf_depth': 'rf_depth_model.pkl',
        'rf_mag': 'rf_mag_model.pkl',
        'xgb_depth': 'xgb_depth_model.pkl',
        'xgb_mag': 'xgb_mag_model.pkl',
        'scaler': 'scaler.pkl',
        'features': 'features.pkl'
    }
    for model_name, filename in model_files.items():
        try:
            if os.path.exists(filename):
                models[model_name] = joblib.load(filename)
            else:
                st.warning(f"Model file {filename} not found. Using mock predictions.")
                models[model_name] = None
        except Exception as e:
            st.error(f"Error loading {filename}: {str(e)}")
            models[model_name] = None
    try:
        if os.path.exists("lstm_model.keras"):
            from keras.models import load_model
            models['lstm'] = load_model("lstm_model.keras", compile=False)
        else:
            models['lstm'] = None
    except Exception as e:
        st.warning(f"LSTM model not available: {str(e)}")
        models['lstm'] = None
    return models

@st.cache_data(show_spinner=False)
def get_coordinates_from_name(location_name):
    try:
        geolocator = Nominatim(user_agent="earthquake_dashboard")
        location = geolocator.geocode(location_name, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        return None, None

def mock_prediction(lat, long, model_type="magnitude"):
    np.random.seed(int(lat * long * 1000) % 2**32)
    return np.random.uniform(2, 8) if model_type == "magnitude" else np.random.uniform(10, 500)

def make_predictions(lat, long, models, model_choice):
    try:
        lat_long_input = pd.DataFrame([[lat, long]], columns=["Lat", "Long"])
        if model_choice == "Random Forest":
            if models['rf_depth'] and models['rf_mag']:
                pred_depth = models['rf_depth'].predict(lat_long_input)[0]
                pred_mag = models['rf_mag'].predict(lat_long_input)[0]
            else:
                pred_depth = mock_prediction(lat, long, "depth")
                pred_mag = mock_prediction(lat, long, "magnitude")
        elif model_choice == "XGBoost":
            if models['xgb_depth'] and models['xgb_mag']:
                pred_depth = models['xgb_depth'].predict(lat_long_input)[0]
                pred_mag = models['xgb_mag'].predict(lat_long_input)[0]
            else:
                pred_depth = mock_prediction(lat, long, "depth")
                pred_mag = mock_prediction(lat, long, "magnitude")
        else:
            if models['lstm'] and models['scaler']:
                input_scaled = models['scaler'].transform([[lat, long]])
                input_lstm = input_scaled.reshape((1, 1, 2))
                pred_mag = models['lstm'].predict(input_lstm, verbose=0)[0][0]
                pred_depth = None
            else:
                pred_mag = mock_prediction(lat, long, "magnitude")
                pred_depth = None
        return pred_depth, pred_mag
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return mock_prediction(lat, long, "depth"), mock_prediction(lat, long, "magnitude")

def create_heatmap(df):
    try:
        heatmap_df = df[['Lat', 'Long', 'Magnitude']].dropna()
        heatmap_df = heatmap_df[np.isfinite(heatmap_df).all(axis=1)]
        heat_data = [[row['Lat'], row['Long'], row['Magnitude']] for _, row in heatmap_df.iterrows()]
        if not heat_data:
            return None
        center_lat = heatmap_df['Lat'].mean()
        center_long = heatmap_df['Long'].mean()
        m = folium.Map(location=[center_lat, center_long], zoom_start=2)
        HeatMap(heat_data, radius=15).add_to(m)
        return m
    except Exception as e:
        st.error(f"Heatmap generation error: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🌍 Earthquake Prediction Dashboard</h1>', unsafe_allow_html=True)
    df = load_data()
    models = load_models()
    if df.empty:
        st.error("No valid data available. Please check your dataset.")
        return

    st.sidebar.header("🎯 Prediction Settings")
    location_input = st.sidebar.text_input("Enter Location Name", value="New York")
    lat, long = get_coordinates_from_name(location_input)

    if lat is None or long is None:
        st.sidebar.error("Could not find coordinates for the given location.")
        return

    st.sidebar.markdown(f"**📌 Coordinates:**\n\nLatitude: `{lat:.4f}`\n\nLongitude: `{long:.4f}`")
    model_choice = st.sidebar.selectbox("Choose Prediction Model", ["Random Forest", "XGBoost", "LSTM"])
    pred_depth, pred_mag = make_predictions(lat, long, models, model_choice)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🗺️ Global Earthquake Heatmap")
        heatmap = create_heatmap(df)
        if heatmap:
            st_folium(heatmap, width=700, height=400)
        else:
            st.warning("No heatmap could be generated.")
    with col2:
        st.subheader("📍 Prediction Results")
        if pred_depth is not None:
            st.metric("Predicted Depth", f"{pred_depth:.2f} km")
        st.metric("Predicted Magnitude", f"{pred_mag:.2f}")
        if pred_mag < 3:
            risk_level = "🟢 Low Risk"
            risk_color = "#28a745"
        elif pred_mag < 6:
            risk_level = "🟡 Moderate Risk"
            risk_color = "#ffc107"
        else:
            risk_level = "🔴 High Risk"
            risk_color = "#dc3545"

        st.markdown(f"""
        <div style="background-color: {risk_color}20; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
            <h4 style="color: {risk_color}; margin: 0;">{risk_level}</h4>
            <p style="margin: 0.5rem 0 0 0;">Based on predicted magnitude</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
