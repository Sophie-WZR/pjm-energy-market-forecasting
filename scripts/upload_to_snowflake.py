from pathlib import Path
import os

import pandas as pd

try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas
except ImportError as exc:
    raise ImportError(
        "Missing Snowflake dependency. Install it with: "
        "pip install snowflake-connector-python[pandas] python-dotenv"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SQL_DIR = PROJECT_ROOT / "sql"


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


def execute_sql_file(conn, path):
    sql_text = path.read_text()
    statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]
    with conn.cursor() as cur:
        for statement in statements:
            cur.execute(statement)


def parse_pjm_datetime(series):
    return pd.to_datetime(series, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")


def prepare_aep_load():
    paths = [
        DATA_DIR / "pjm_aep_hourly_load_2024.csv",
        DATA_DIR / "pjm_aep_hourly_load_2025.csv",
    ]
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    df.columns = [col.upper() for col in df.columns]
    df["DATETIME_BEGINNING_UTC"] = parse_pjm_datetime(df["DATETIME_BEGINNING_UTC"])
    df["DATETIME_BEGINNING_EPT"] = parse_pjm_datetime(df["DATETIME_BEGINNING_EPT"])
    df["MW"] = pd.to_numeric(df["MW"], errors="coerce")
    df["IS_VERIFIED"] = df["IS_VERIFIED"].astype("boolean")
    return df[
        [
            "DATETIME_BEGINNING_UTC",
            "DATETIME_BEGINNING_EPT",
            "NERC_REGION",
            "MKT_REGION",
            "ZONE",
            "LOAD_AREA",
            "MW",
            "IS_VERIFIED",
        ]
    ]


def prepare_weather():
    # The source weather CSV contains metadata rows; notebook 01 reads it with skiprows=3.
    df = pd.read_csv(DATA_DIR / "columbus_hourly_weather_2024_2025.csv", skiprows=3)
    df = df.rename(columns={"temperature_2m (°C)": "TEMPERATURE_2M", "time": "TIME"})
    df["TIME"] = pd.to_datetime(df["TIME"], format="%Y-%m-%dT%H:%M", errors="coerce")
    df["TEMPERATURE_2M"] = pd.to_numeric(df["TEMPERATURE_2M"], errors="coerce")
    return df[["TIME", "TEMPERATURE_2M"]]


def prepare_lmp():
    paths = [
        DATA_DIR / "rt_da_monthly_lmps_2024.csv",
        DATA_DIR / "rt_da_monthly_lmps_2025.csv",
    ]
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    df.columns = [col.upper() for col in df.columns]
    df["DATETIME_BEGINNING_UTC"] = parse_pjm_datetime(df["DATETIME_BEGINNING_UTC"])
    df["DATETIME_BEGINNING_EPT"] = parse_pjm_datetime(df["DATETIME_BEGINNING_EPT"])

    numeric_cols = [
        "PNODE_ID",
        "SYSTEM_ENERGY_PRICE_RT",
        "TOTAL_LMP_RT",
        "CONGESTION_PRICE_RT",
        "MARGINAL_LOSS_PRICE_RT",
        "SYSTEM_ENERGY_PRICE_DA",
        "TOTAL_LMP_DA",
        "CONGESTION_PRICE_DA",
        "MARGINAL_LOSS_PRICE_DA",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[
        [
            "DATETIME_BEGINNING_UTC",
            "DATETIME_BEGINNING_EPT",
            "PNODE_ID",
            "PNODE_NAME",
            "VOLTAGE",
            "EQUIPMENT",
            "TYPE",
            "ZONE",
            "SYSTEM_ENERGY_PRICE_RT",
            "TOTAL_LMP_RT",
            "CONGESTION_PRICE_RT",
            "MARGINAL_LOSS_PRICE_RT",
            "SYSTEM_ENERGY_PRICE_DA",
            "TOTAL_LMP_DA",
            "CONGESTION_PRICE_DA",
            "MARGINAL_LOSS_PRICE_DA",
        ]
    ]


def upload_dataframe(conn, table_name, df):
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {table_name}")
    success, nchunks, nrows, _ = write_pandas(
        conn,
        df,
        table_name,
        quote_identifiers=False,
        auto_create_table=False,
        overwrite=False,
    )
    if not success:
        raise RuntimeError(f"write_pandas failed for {table_name}")
    print(f"{table_name}: uploaded {nrows:,} rows in {nchunks} chunk(s)")


def print_count(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cur.fetchone()[0]
    print(f"{table_name}: {count:,} rows available")


def main():
    conn = snowflake_connection()
    try:
        execute_sql_file(conn, SQL_DIR / "01_create_raw_tables.sql")

        raw_tables = {
            "RAW_AEP_LOAD_HOURLY": prepare_aep_load(),
            "RAW_WEATHER_HOURLY": prepare_weather(),
            "RAW_LMP_HOURLY": prepare_lmp(),
        }
        for table_name, df in raw_tables.items():
            upload_dataframe(conn, table_name, df)
            print_count(conn, table_name)

        execute_sql_file(conn, SQL_DIR / "02_create_market_mart.sql")
        print_count(conn, "MART_AEP_MARKET_HOURLY")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
