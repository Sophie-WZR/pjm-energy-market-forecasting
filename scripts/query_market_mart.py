from pathlib import Path
import os

import pandas as pd

try:
    import snowflake.connector
except ImportError as exc:
    raise ImportError(
        "Missing Snowflake dependency. Install it with: "
        "pip install snowflake-connector-python[pandas] python-dotenv"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_dotenv_if_available():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def snowflake_connection():
    load_dotenv_if_available()
    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "PJM_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "PJM_MARKET_DB"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "ANALYTICS"),
    )


def main():
    query = """
        SELECT
            DATE_PART(EPOCH_SECOND, "DATETIME") AS DATETIME_EPOCH_NS,
            CAST(TOTAL_LMP_DAY_AHEAD AS FLOAT) AS TOTAL_LMP_DAY_AHEAD,
            CAST(TOTAL_LMP_REAL_TIME AS FLOAT) AS TOTAL_LMP_REAL_TIME,
            CAST(CONGESTION_PRICE_DAY_AHEAD AS FLOAT) AS CONGESTION_PRICE_DAY_AHEAD,
            CAST(CONGESTION_PRICE_REAL_TIME AS FLOAT) AS CONGESTION_PRICE_REAL_TIME,
            CAST(MARGINAL_LOSS_PRICE_DAY_AHEAD AS FLOAT) AS MARGINAL_LOSS_PRICE_DAY_AHEAD,
            CAST(MARGINAL_LOSS_PRICE_REAL_TIME AS FLOAT) AS MARGINAL_LOSS_PRICE_REAL_TIME,
            CAST(RT_DA_SPREAD AS FLOAT) AS RT_DA_SPREAD,
            CAST(CONGESTION_SHARE_DA AS FLOAT) AS CONGESTION_SHARE_DA,
            CAST(CONGESTION_SHARE_RT AS FLOAT) AS CONGESTION_SHARE_RT,
            CAST(DA_PRICE_SPIKE AS INTEGER) AS DA_PRICE_SPIKE,
            CAST(RT_PRICE_SPIKE AS INTEGER) AS RT_PRICE_SPIKE,
            CAST(LOAD_MW AS FLOAT) AS LOAD_MW,
            CAST(TEMPERATURE_2M AS FLOAT) AS TEMPERATURE_2M,
            CAST(HOUR AS INTEGER) AS HOUR,
            CAST(DAYOFWEEK AS INTEGER) AS DAYOFWEEK,
            CAST(MONTH AS INTEGER) AS MONTH,
            CAST(IS_WEEKEND AS INTEGER) AS IS_WEEKEND,
            CAST(SIN_HOUR AS FLOAT) AS SIN_HOUR,
            CAST(COS_HOUR AS FLOAT) AS COS_HOUR
        FROM MART_AEP_MARKET_HOURLY
        ORDER BY "DATETIME"
    """
    output_path = OUTPUT_DIR / "master_market_df_from_snowflake.csv"

    conn = snowflake_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute(query)
            columns = [col[0].lower() for col in cur.description]
            rows = cur.fetchall()
        finally:
            cur.close()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=columns)
    # Snowflake returns this TIMESTAMP_NTZ epoch value through the connector at nanosecond scale.
    # Parse as ns and round to clean hourly timestamps used by the notebooks.
    df["datetime"] = pd.to_datetime(df["datetime_epoch_ns"], unit="ns", errors="coerce").dt.round("s")
    df = df.drop(columns=["datetime_epoch_ns"])
    ordered_cols = ["datetime"] + [col for col in df.columns if col != "datetime"]
    df = df[ordered_cols]
    numeric_cols = [col for col in df.columns if col != "datetime"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_csv(output_path, index=False)

    print(f"Saved Snowflake mart extract to {output_path}")
    print(f"Rows: {len(df):,}")
    if "datetime" in df.columns and not df.empty:
        print(f"Datetime range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
