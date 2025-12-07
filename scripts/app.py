from fastapi import FastAPI
from pathlib import Path
import joblib
import pandas as pd
import os

app = FastAPI()

# Ruta del modelo
MODEL_PATH = os.getenv("MODEL_PATH")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")

# Cargar el modelo al iniciar la API
model = joblib.load(MODEL_PATH)


# Endpoint de salud
@app.get("/")
def health():
    return {"status": "ok"}

# Endpoint de predicción
@app.post("/predict")
def predict(data: dict):
    # manejo de errores simplificado
    if not data:
        return {"error": "No se proporcionaron datos para la predicción"}
    if not isinstance(data, dict):
        return {"error": "Los datos deben estar en formato JSON"}
    
    try:
        df = pd.DataFrame([data])
    except Exception as e:
        return {"error": f"Error al convertir los datos a DataFrame: {str(e)}"}
    
    expected_columns = [
        "open", "open_lag1", "high_lag1", "low_lag1", "close_lag1",
        "volume_lag1", "return_prev_close_lag1", "return_close_open_lag1",
        "volatility_7_days", "volatility_30_days", "year", "month",
        "day_of_week", "is_monday", "is_friday", "is_earning_day", "ticker"
    ]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        return {"error": f"Faltan columnas requeridas: {', '.join(missing_cols)}"}
    
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
