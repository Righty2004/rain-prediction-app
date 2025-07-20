import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title("Rain Prediction App")

df = pd.read_csv("weather_forecast_data.csv")

label_encoder = LabelEncoder()
df['Rain'] = label_encoder.fit_transform(df['Rain'])

X = df[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]
y = df['Rain']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

temp = st.number_input("Temperature (°C)", value=25.0)
hum = st.number_input("Humidity (%)", value=60.0)
wind = st.number_input("Wind Speed (km/h)", value=8.0)
cloud = st.number_input("Cloud Cover (%)", value=50.0)
press = st.number_input("Pressure (hPa)", value=1013.0)

if st.button("Predict"):
    input_df = pd.DataFrame([[temp, hum, wind, cloud, press]], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    result = label_encoder.inverse_transform([prediction])[0]
    result = result.title()
    st.success(f"Prediction: {result}")
