from fastapi import FastAPI
from pathlib import Path
import joblib
import pandas as pd

app = FastAPI()

# Ruta del modelo
MODEL_PATH = Path("/app/model.joblib")

# Cargar el modelo al iniciar la API
model = joblib.load(MODEL_PATH)

# Endpoint de salud
@app.get("/")
def health():
    return {"status": "ok"}

# Endpoint de predicción
@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
