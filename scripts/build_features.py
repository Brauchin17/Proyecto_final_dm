import os
import argparse
import time
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from datetime import datetime

def get_conn():
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD")
    )

def ensure_schema_and_table(conn):
    cur = conn.cursor()

    # Crear schema SI NO EXISTE
    cur.execute("""
        CREATE SCHEMA IF NOT EXISTS analytics;
    """)

    # Crear tabla SI NO EXISTE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analytics.prices_daily_features (
            date DATE NOT NULL,
            ticker TEXT NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume BIGINT,
            year INTEGER,
            month INTEGER,
            day_of_week INTEGER,
            is_monday INTEGER,
            is_friday INTEGER,
            is_earning_day INTEGER,
            return_close_open DOUBLE PRECISION,
            return_prev_close DOUBLE PRECISION,
            volatility_7_days DOUBLE PRECISION,
            volatility_30_days DOUBLE PRECISION,
            close_lag1 DOUBLE PRECISION,
            ingested_at_utc TIMESTAMP,
            run_id TEXT,
            PRIMARY KEY (date, ticker)
        );
    """)

    conn.commit()
    cur.close()

    

SQL_query = """
WITH base AS (
    SELECT
        p.date,
        p.ticker,
        p.open,
        p.high,
        p.low,
        p.close,
        p.volume,
        p.ingested_at_utc,
        p.run_id,

        -- returns and lags
        LAG(p.close, 1) OVER (PARTITION BY p.ticker ORDER BY p.date) AS close_lag1,

        (p.close - p.open) / NULLIF(p.open, 0) AS return_close_open,

        p.close / NULLIF(LAG(p.close, 1) OVER (PARTITION BY p.ticker ORDER BY p.date), 0) - 1
            AS return_prev_close,


        -- earning days
        CASE WHEN e.earnings_date IS NOT NULL THEN 1 ELSE 0 END AS is_earning_day

    FROM raw.prices_daily p
    LEFT JOIN raw.earnings_dates e
        ON p.ticker = e.ticker 
        AND p.date = DATE(e.earnings_date) -- comparar solo la parte de fecha
    WHERE p.ticker = %(ticker)s
        AND p.date >= %(start_date)s
        AND p.date <= %(end_date)s
),
feat AS (
    SELECT
        *,
        EXTRACT(YEAR FROM date) AS year,
        EXTRACT(MONTH FROM date) AS month,
        EXTRACT(DAY FROM date) AS day_of_week,

        CASE WHEN EXTRACT(DOW FROM date) = 1 THEN 1 ELSE 0 END AS is_monday,
        CASE WHEN EXTRACT(DOW FROM date) = 5 THEN 1 ELSE 0 END AS is_friday,

        -- volatilities
        STDDEV(return_prev_close)
            OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)
            AS volatility_7_days,
        
        STDDEV(return_prev_close)
            OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING)
            AS volatility_30_days

    FROM base
)

INSERT INTO analytics.prices_daily_features (
    date, ticker, open, high, low, close, volume,
    year, month, day_of_week, is_monday, is_friday, is_earning_day,
    return_close_open, return_prev_close, volatility_7_days, volatility_30_days,
    close_lag1, ingested_at_utc, run_id
)
SELECT
    date, ticker, open, high, low, close, volume,
    year, month, day_of_week, is_monday, is_friday, is_earning_day,
    return_close_open, return_prev_close, volatility_7_days, volatility_30_days,
    close_lag1, NOW() AS ingested_at_utc, %(run_id)s
FROM feat
    {conflict_clause};        
"""

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', required=True, choices=['full', 'by-date-range'], help='Modo de ejecución')
    parser.add_argument('--ticker', required=False, help='Ticker a procesar (requerido si el modo es by-date-range)')
    parser.add_argument('--start-date', required=False, help='Fecha de inicio (YYYY-MM-DD) (requerido si el modo es by-date-range)')
    parser.add_argument('--end-date', required=False, help='Fecha de fin (YYYY-MM-DD) (requerido si el modo es by-date-range)') 
    parser.add_argument('--run-id', required=True, help='Identificador de la ejecucion actual')
    parser.add_argument('--overwrite', required=False, help='Si se especifica, sobrescribe los datos existentes en el rango dado')

    args = parser.parse_args()

    if args.overwrite:
        conflict_clause = f"""
        ON CONFLICT (date, ticker) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        year = EXCLUDED.year,
        month = EXCLUDED.month,
        day_of_week = EXCLUDED.day_of_week,
        is_monday = EXCLUDED.is_monday,
        is_friday = EXCLUDED.is_friday,
        return_close_open = EXCLUDED.return_close_open,
        return_prev_close = EXCLUDED.return_prev_close,
        volatility_7_days = EXCLUDED.volatility_7_days,
        volatility_30_days = EXCLUDED.volatility_30_days,
        close_lag1 = EXCLUDED.close_lag1,
        ingested_at_utc = NOW(),
        run_id = EXCLUDED.run_id; 
        """
    else:
        conflict_clause = "ON CONFLICT (date, ticker) DO NOTHING"
    
    # Asegurar que esquema 'analytics' existe
    conn = get_conn()
    ensure_schema_and_table(conn)
    cur = conn.cursor()

    # Determinar rango de fechas
    if args.mode == 'full':
        cur.execute("""SELECT MIN(date), MAX(date) FROM raw.prices_daily WHERE ticker = %s""", (args.ticker,))
        start_date, end_date = cur.fetchone()
    else:
        start_date = args.start_date
        end_date = args.end_date
    
    print(f"Procesando ticker {args.ticker} desde {start_date} hasta {end_date}")

    start_time = time.time()
    final_sql = SQL_query.format(conflict_clause=conflict_clause)
    # Ejecutar la query de construcción de features
    cur.execute(
        final_sql,
        {
            'ticker': args.ticker,
            'start_date': start_date,
            'end_date': end_date,
            'run_id': args.run_id,
        }
    )

    rows_affected = cur.rowcount
    conn.commit()

    duration = time.time() - start_time

    print(f"Filas insertadas/actualizadas: {rows_affected}")
    print(f"Duración: {duration:.2f} segundos")
    print(f"Rango de fechas procesado: {start_date} a {end_date}")
    
if __name__ == '__main__':
    main()