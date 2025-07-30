import pandas as pd
import numpy as np
import joblib
import re
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv(r"C:\Users\bhasker avusali\OneDrive\Desktop\MAJOR\dataset.csv")

# Clean Magnitude column
def extract_first_float(text):
    match = re.search(r'\d+(\.\d+)?', str(text))
    return float(match.group()) if match else None

df['Magnitude'] = df['Magnitude'].apply(extract_first_float)
df = df.dropna(subset=['Lat', 'Long', 'Magnitude', 'Depth'])

# Features and Targets
features = df[['Lat', 'Long']]
y_mag = df['Magnitude']
y_depth = df['Depth']

# Save features to features.pkl
joblib.dump(features, 'features.pkl')

# Scale the features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
joblib.dump(scaler, 'scaler.pkl')

# Split data
X_train, X_test, y_mag_train, y_mag_test = train_test_split(features_scaled, y_mag, test_size=0.2, random_state=42)
_, _, y_depth_train, y_depth_test = train_test_split(features_scaled, y_depth, test_size=0.2, random_state=42)

# === Random Forest Models ===
rf_mag = RandomForestRegressor(n_estimators=100, random_state=42)
rf_depth = RandomForestRegressor(n_estimators=100, random_state=42)

rf_mag.fit(X_train, y_mag_train)
rf_depth.fit(X_train, y_depth_train)

joblib.dump(rf_mag, 'rf_mag_model.pkl')
joblib.dump(rf_depth, 'rf_depth_model.pkl')

# === XGBoost Models ===
xgb_mag = XGBRegressor(n_estimators=100, random_state=42)
xgb_depth = XGBRegressor(n_estimators=100, random_state=42)

xgb_mag.fit(X_train, y_mag_train)
xgb_depth.fit(X_train, y_depth_train)

joblib.dump(xgb_mag, 'xgb_mag_model.pkl')
joblib.dump(xgb_depth, 'xgb_depth_model.pkl')
