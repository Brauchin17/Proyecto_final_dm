FROM python:3.10

WORKDIR /app

# Instalar dependencias del sistema necesarias para psycopg2 (driver de postgres)
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Copiamos requirements e instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copiamos el resto del código
COPY . .
EXPOSE 8000

# Comando por defecto para levantar la API
CMD ["uvicorn", "scripts.app:app", "--host", "0.0.0.0", "--port", "8000"]

