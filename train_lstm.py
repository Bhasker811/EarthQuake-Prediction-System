import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\bhasker avusali\OneDrive\Desktop\MAJOR\dataset.csv")

# Clean 'Magnitude'
def extract_first_float(text):
    match = re.search(r'\d+(\.\d+)?', str(text))
    return float(match.group()) if match else None

df['Magnitude'] = df['Magnitude'].apply(extract_first_float)
df = df.dropna(subset=['Lat', 'Long', 'Magnitude'])

# Select features and target
X = df[['Lat', 'Long']]
y = df['Magnitude']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM: [samples, time_steps, features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(1, X.shape[1])))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer for regression

model.compile(optimizer='adam', loss='mse')

# Train model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

# Save model and scaler
model.save("lstm_model.keras")
joblib.dump(scaler, "scaler.pkl")
