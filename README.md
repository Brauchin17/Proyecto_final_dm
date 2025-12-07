Pipeline de datos para trading algorítmico con mercado (Yahoo Finance). Modelo de clasificación binaria para predecir si el activo cierra arriba o abajo respecto al open (target_up). Incluye simulación de inversión de USD 10,000 en 2025 y API REST.## Requisitos

```bash
Python >= 3.10
pandas
numpy
scikit-learn
fastapi
uvicorn
yfinance
matplotlib
seaborn
```

Se instalan:

```bash
pip install -r requirements.txt
```

# Estructura del proyecto
```text
├── docker-compose.yml     # Configuración de servicios
├── .env.example           # Ejemplo de variables de ambiente
├── notebooks/             # Notebooks de ingesta y ML
│   ├── 01_ingesta_prices_raw.ipynb
│   └── ml_trading_classifier.ipynb  # Entrenamiento, evaluación y simulación
├── scripts/
│   ├── build_features.py  # Script CLI para analytics.daily_features
│   └── app.py             # FastAPI para el modelo
└── README.md
```
# Columnas principales de analytics.daily_features

- Identificación: date, ticker, year, month, day_of_week
- Mercado: open, close, high, low, volume
- Features derivadas: return_close_open = (close - open) / open, return_prev_close = close / close_lag1 - 1, volatility_n_days (std de retornos últimos N días)
- Flags: is_monday, is_friday (opcional: is_earnings_day)
- Metadatos: run_id, ingested_at_utc
- Target (agregado en ML): target_up = 1 si close > open, else 0

# Justificación del modelo

Al evaluar los modelos, observamos que muchas métricas tradicionales como el F1-Score podían ser engañosas debido al comportamiento del mercado, el baseline que asume siempre la clase positiva (1) predecía "bien" debido a los mercados en alza, ya que ciertas acciones (como NVIDIA, Google o Apple) hayan estado en alza en el periodo de test, esto no garantiza que un modelo tenga buen desempeño, pero puede influir en métricas como F1 si hay un sesgo de clase hacia subidas. Esto hace que el F1-Score aparente ser relativamente alto incluso en modelos que no capturan patrones reales de subida o bajada.

Por esta razón, se priorizó ROC-AUC como métrica principal. ROC-AUC mide la capacidad del modelo para diferenciar correctamente entre subidas y no subidas a lo largo de todos los posibles umbrales de decisión, lo cual es crucial en un mercado altamente variable donde la proporción de subidas y bajadas puede cambiar día a día. De hecho, aunque algunos modelos tenían un F1-Score alto en entrenamiento, su ROC-AUC era mucho más bajo en validación, reflejando que su capacidad real de discriminar correctamente los movimientos del mercado era limitada.

Es por esto que el modelo que se escoge es el de mayor ROC-AUC en validación, siendo para el caso `Random Forest` (0.541075), que también a su vez tiene un F1-macro en validación (0.539861) superior a todos los otros modelos

## Comparación con metricas de ML despues de simulación

Se evaluó el Random Forest tanto con métricas de ML como con resultados financieros en 2025.

F1-Score Test: 0.6360

ROC-AUC Test: 0.5070

Capital final del portafolio: $14,225.52

- AAPL: $10,086.26
- AMZN: $10,304.61
- GOOGL: $16,826.35
- NVDA: $19,684.85

Aunque el **F1-Score es relativamente alto, esto no garantiza buenos retornos**, ya que puede verse inflado si el mercado está en alza (como AAPL, GOOGL y NVDA) y el modelo predice mayormente “subida”. Por otro lado, la ROC-AUC refleja la capacidad real del modelo para distinguir subidas y bajadas, siendo más confiable para evaluar desempeño en mercados variables. En este caso, la ROC-AUC cercana a 0.5 indica que el modelo tiene capacidad limitada de discriminación, y los retornos positivos se deben en parte a la tendencia general del mercado, no solo a predicciones precisas, con NVDA se puede ver ya que es el que mas esta en alza, se ve como predecir gran cantidad de 1 (inversiones) te da un mayor retorno.


# 1. Cómo levantar el entorno con Docker Compose
Copia .env.example a .env y completa las variables (credenciales Postgres, TICKERS=AAPL, START_DATE=2020-01-01, END_DATE=2025-12-31, etc.).

```bash
docker compose up -d
```
## Servicios:

- jupyter-notebook: http://localhost:8888 (token en logs)
- postgres: BD para raw y analytics
- feature-builder: Worker para build_features.py
- model-api: API REST (después de entrenar el modelo)
- (Opcional) pgadmin: http://localhost:5050 para inspeccionar tablas.

# 2. Comandos de ingesta

Desde jupyter-notebook, ejecuta:

- **Ingesta precios**:notebooks/01_ingesta_prices_raw.ipynb
  - Lee TICKERS, START_DATE, END_DATE de env.
  - Descarga OHLCV de yfinance.
  - Carga a raw.prices_daily con metadatos.
# 3. Comando para construir analytics.daily_features

Se usa el script CLI

```bash
docker compose run feature-builder \
    --mode full \
    --ticker AAPL \
    --start-date 2020-01-01 \
    --end-date 2025-12-31 \
    --run-id my_run_1 \
    --overwrite false
```

- --mode full: (Re)crea la tabla completa.
- --mode by-date-range: Procesa subconjunto de fechas.
- Logs: Filas creadas/actualizadas, min/max date, duración.

# 4. Entrenamiento de los modelos
Ejecuta notebooks/ml_trading_classifier.ipynb en jupyter-notebook.

- Carga analytics.daily_features (de Postgres o exporta a CSV).
- EDA: Balance de target_up, distribuciones, correlaciones.
- Features: Lags de retornos, volatility, etc. (sin leakage).
- Split temporal: Train (2021-2023), Val (2024), Test (2025).
- Preprocesamiento: Imputación, StandardScaler, one-hot para categóricas.
- Modelos (mínimo 7): LogisticRegression, DecisionTree, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost.
- Tuning: GridSearchCV con TimeSeriesSplit para hiperparámetros (ej. C para logistic, max_depth para trees).
- Métricas: Accuracy, Precision, Recall, F1, ROC-AUC en Train/Val/Test.
- Baseline: Siempre predice clase mayoritaria.
- Selecciona mejor: Mayor F1 en Val + simplicidad.
- Reentrena en Train+Val, evalúa en Test.

El notebook muestra matriz de confusión, comparación tabular y análisis de errores.

# 5. Guardar el mejor modelo
En el notebook ml_trading_classifier.ipynb, al final:
- Guarda pipeline completo (preprocesamiento + modelo) con joblib:
```python
import joblib
joblib.dump(pipeline, 'models/best_model.joblib')
```

# 6. Cómo levantar la API y probar /predict
## Iniciar la API (FastAPI)
Asegúrate de que el modelo esté guardado
```Bash
docker compose up -d model-api
```
O localmente:
```Bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
- API disponible en http://localhost:8000
## Endpoint: POST /predict
El body esperado sigue el siguiente formato:
```JSON
{
  "open": 150.2,
  "open_lag1": 149.8,
  "high_lag1": 151.0,
  "low_lag1": 148.9,
  "close_lag1": 150.1,
  "volume_lag1": 1200000,
  "return_prev_close_lag1": 0.002,
  "return_close_open_lag1": -0.001,
  "volatility_7_days": 0.015,
  "volatility_30_days": 0.02,
  "year": 2024,
  "month": 12,
  "day_of_week": 3,
  "is_monday": 0,
  "is_friday": 0,
  "is_earning_day": 0,
  "ticker": "AAPL"
}
```
Con curl:
```Bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{ "open": 150.2, "open_lag1": 149.8, "high_lag1": 151.0, "low_lag1": 148.9, "close_lag1": 150.1, "volume_lag1": 1200000, "return_prev_close_lag1": 0.002, "return_close_open_lag1": -0.001, "volatility_7_days": 0.015, "volatility_30_days": 0.02,"year": 2024,"month": 12,"day_of_week": 3,"is_monday": 0,"is_friday": 0,"is_earning_day": 0,"ticker": "AAPL"
}'
```

Respuesta (ejemplo):

```JSON
{
  "prediction": 1,
}
```
## Ejemplo con Postman:

- Method: POST
- URL: http://localhost:8000/predict
- Body → raw → JSON → Pega el JSON de ejemplo
- Send → Ver prediction (0/1).

# 7. Cómo correr la simulación de inversión de USD 10,000 en 2025

En ml_trading_classifier.ipynb, sección de simulación:

- Usa datos de Test (2025).
- Regla: Si prediction == 1, compra al open, vende al close (largo intradía).
- Si 0, queda en efectivo.
- Capital inicial: 10000 USD.
- Sin costos (o agrega 0.1% comisión opcional).
- Outputs: Capital final, retorno total (%), trades, curva de equity (plot).

Ejecuta la celda de simulación en el notebook.
Comparación: Retorno vs. métricas ML (F1/ROC-AUC en Test). 

# 8. Flujo completo
```Bash
# 1. Levantar entorno
docker compose up -d

# 2. Ingesta (en jupyter-notebook)
# Ejecuta 01_ingesta_prices_raw.ipynb

# 3. Construir features
docker compose run feature-builder --mode full --ticker AAPL --start-date 2020-01-01 --end-date 2025-12-31 --run-id test_run

# 4. Entrenar y guardar modelo (en jupyter-notebook)
# Ejecuta ml_trading_classifier.ipynb completo → guarda models/best_model.joblib

# 5. Levantar API
docker compose up -d model-api

# 6. Probar API (curl ejemplo arriba)

# 7. Simulación: Re-ejecuta sección de simulación en ml_trading_classifier.ipynb
```

# 9. Checklist

- [x] Se usa `analytics.daily_features` como base.
- [x] `target_up` definido sin leakage.
- [x] ≥ 7 modelos entrenados, tuneados y comparados.
- [x] Baseline implementado.
- [x] Modelo ganador reentrenado en Train+Val y evaluado en Test.
- [x] Simulación de inversión con USD 10,000 en 2025 realizada y documentada.
- [x] Modelo serializado y cargado por la API.
- [x] `model-api` responde correctamente a `/predict`.



