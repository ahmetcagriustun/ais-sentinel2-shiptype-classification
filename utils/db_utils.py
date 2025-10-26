import psycopg2
from utils.config_utils import load_config
import os
import zipfile
import io
import tempfile
import boto3
from datetime import datetime

def create_pg_connection(db_config):
    """
    Establishes a PostgreSQL database connection using the provided configuration.

    Args:
        db_config (dict): A dictionary containing PostgreSQL connection parameters
                          (host, port, dbname, user, password).

    Returns:
        connection: A psycopg2 connection object.

    Raises:
        psycopg2.OperationalError: If the connection fails.
    """
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            dbname=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"]
        )
        return conn
    except psycopg2.OperationalError as e:
        raise psycopg2.OperationalError(f"PostgreSQL connection failed: {e}")

def create_table(conn, table_name, schema_sql):
    """
    Creates a table in PostgreSQL using the given schema.

    Args:
        conn: An open psycopg2 connection object.
        table_name (str): Name of the table to be created.
        schema_sql (str): SQL CREATE TABLE statement.

    Returns:
        None

    Raises:
        psycopg2.DatabaseError: If table creation fails.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            cur.execute(schema_sql)
        conn.commit()
        print(f"Table '{table_name}' created successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error creating table '{table_name}': {e}")
        raise

def insert_dicts_to_table(conn, table_name, data):
    """
    Inserts data into the specified PostgreSQL table.
    Accepts either a list of dictionaries (each dict = row) or a Pandas DataFrame.

    Args:
        conn: An open psycopg2 connection object.
        table_name (str): Name of the table to insert data into.
        data (list of dict or pandas.DataFrame): Data to be inserted.

    Returns:
        None

    Raises:
        psycopg2.DatabaseError: If the insertion fails.
    """
    import pandas as pd

    # Convert DataFrame to list of dict if needed
    if isinstance(data, pd.DataFrame):
        data_list = data.to_dict(orient="records")
    else:
        data_list = data

    if not data_list:
        print("No data to insert.")
        return

    keys = data_list[0].keys()
    columns = ', '.join(keys)
    values_template = ', '.join(['%s'] * len(keys))
    insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({values_template})"

    rows = [tuple(d[k] for k in keys) for d in data_list]

    try:
        with conn.cursor() as cur:
            cur.executemany(insert_sql, rows)
        conn.commit()
        print(f"Inserted {len(rows)} rows into '{table_name}'.")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting data into '{table_name}': {e}")
        raise

def get_product_names_from_postgres(
    config_path="config.yaml",
    table_name="public.sentinel_metadata_select_deneme",
    column_name="api_id"
):
    """
    Reads product names from the specified PostgreSQL table and returns as a Python list.
    """
    config = load_config(config_path)
    db = config["database"]

    conn = psycopg2.connect(
        host=db["host"],
        port=db["port"],
        dbname=db["dbname"],
        user=db["user"],
        password=db["password"]
    )
    cur = conn.cursor()
    # Adjust column/table name as needed
    cur.execute(f"SELECT {column_name} FROM {table_name};")
    results = cur.fetchall()
    cur.close()
    conn.close()

    # Convert list of tuples to list of strings
    product_names = [row[0] for row in results if row[0]]
    return product_names

def get_date_list_from_db(db_config, table_name="ais_download_list", date_col="date"):
    import psycopg2
    conn = psycopg2.connect(
        host=db_config["host"],
        port=db_config["port"],
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"]
    )
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT {date_col} FROM {table_name} ORDER BY {date_col}")
    dates = [row[0].strftime('%Y-%m-%d') for row in cur.fetchall()]
    cur.close()
    conn.close()
    return dates

def _parse_ts(val: str):
    """
    Normalize AIS time strings to Python datetime for Postgres TIMESTAMP.
    Accepts:
      - '13/05/2024 00:00:00'  (DMY)
      - '2024-05-13 00:00:00'  (ISO)
      - '2024-05-13T00:00:00Z' (ISO Z)
    Returns datetime or None.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # DMY format
    try:
        if '/' in s:
            return datetime.strptime(s, '%d/%m/%Y %H:%M:%S')
    except Exception:
        pass
    # ISO with 'T' and optional 'Z'
    try:
        s2 = s.replace('T', ' ').replace('Z', '')
        return datetime.strptime(s2, '%Y-%m-%d %H:%M:%S')
    except Exception:
        pass
    # Final fallback: try fromisoformat without 'Z'
    try:
        return datetime.fromisoformat(s.replace('Z', ''))
    except Exception:
        return None

def insert_ais_records_to_pg(conn, records, table_name="ship_raw_data"):
    if not records:
        print("No records to insert.")
        return

    keys = [
        "# Timestamp", "Type of mobile", "MMSI", "Latitude", "Longitude",
        "Navigational status", "ROT", "SOG", "COG", "Heading", "IMO", "Callsign",
        "Name", "Ship type", "Cargo type", "Width", "Length",
        "Type of position fixing device", "Draught", "Destination", "ETA",
        "Data source type", "A", "B", "C", "D"
    ]
    columns = ', '.join([f'"{k}"' for k in keys])
    values_template = ', '.join(['%s'] * len(keys))
    insert_sql = f'INSERT INTO {table_name} ({columns}) VALUES ({values_template})'

    # Build sanitized rows:
    rows = []
    for rec in records:
        row = []
        for k in keys:
            v = rec.get(k, None)
            # Normalize empty strings to None
            if isinstance(v, str) and v.strip() == '':
                v = None
            # Parse timestamps for the two columns that are TIMESTAMP in schema
            if k in ("# Timestamp", "ETA"):
                v = _parse_ts(v)
            row.append(v)
        rows.append(tuple(row))

    cur = conn.cursor()
    try:
        cur.executemany(insert_sql, rows)
        conn.commit()
        print(f"Inserted {len(rows)} rows into '{table_name}'.")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting data into '{table_name}': {e}")
        raise
    finally:
        cur.close()


def csv_to_postgresql(folder_path, table_name, conn):
    log_file_path = os.path.join(folder_path, "import_log.txt")  
    with conn.cursor() as cur, open(log_file_path, 'a') as log_file:
        try:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if filename.endswith('.csv'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            copy_query = f"COPY {table_name} FROM STDIN WITH CSV HEADER DELIMITER ','"
                            cur.copy_expert(copy_query, f)
                        except Exception as e:  
                            log_file.write(f"Error in file {file_path}: {e}\n")
                    conn.commit()  

                elif filename.endswith('.zip'):
                    try:
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            for zipped_file in zip_ref.namelist():
                                if zipped_file.endswith('.csv'):
                                    with zip_ref.open(zipped_file) as f:
                                        try:
                                            copy_query = f"COPY {table_name} FROM STDIN WITH CSV HEADER DELIMITER ','"
                                            cur.copy_expert(copy_query, io.TextIOWrapper(f, encoding='utf-8'))
                                        except Exception as e:
                                            log_file.write(f"Error in file {zipped_file} from {file_path}: {e}\n")
                        conn.commit()  
                    except Exception as e:
                        log_file.write(f"Error processing zip file {file_path}: {e}\n")
        except Exception as e:
            log_file.write(f"Error processing files in folder {folder_path}: {e}\n")

def create_ship_raw_data_table(db_conf):
    sql = '''
CREATE TABLE IF NOT EXISTS ship_raw_data (
    "# Timestamp" TIMESTAMP,
    "Type of mobile" VARCHAR(255),
    "MMSI" BIGINT,
    "Latitude" DOUBLE PRECISION,
    "Longitude" DOUBLE PRECISION,
    "Navigational status" VARCHAR(255),
    "ROT" DOUBLE PRECISION,
    "SOG" DOUBLE PRECISION,
    "COG" DOUBLE PRECISION,
    "Heading" DOUBLE PRECISION,
    "IMO" VARCHAR(255),
    "Callsign" VARCHAR(255),
    "Name" VARCHAR(255),
    "Ship type" VARCHAR(255),
    "Cargo type" VARCHAR(255),
    "Width" DOUBLE PRECISION,
    "Length" DOUBLE PRECISION,
    "Type of position fixing device" VARCHAR(255),
    "Draught" DOUBLE PRECISION,
    "Destination" VARCHAR(255),
    "ETA" VARCHAR(255),
    "Data source type" VARCHAR(255),
    "A" DOUBLE PRECISION,
    "B" DOUBLE PRECISION,
    "C" DOUBLE PRECISION,
    "D" DOUBLE PRECISION
);
    '''
    import psycopg2
    conn = psycopg2.connect(**db_conf)
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()

def get_needed_dates_from_db(db_conf, table="ais_download_list", col="date"):
    import psycopg2
    conn = psycopg2.connect(**db_conf)
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT {col} FROM {table}")
    dates = {row[0].strftime("%Y-%m-%d") for row in cur.fetchall()}
    cur.close()
    conn.close()
    return dates

def bulk_copy_csv_to_ship_raw_data(db_conf, csvfile):
    conn = None
    try:
        conn = psycopg2.connect(**db_conf)
        cur = conn.cursor()
        cur.execute("SET datestyle TO 'ISO, DMY';")
        copy_sql = "COPY ship_raw_data FROM STDIN WITH CSV HEADER DELIMITER ','"
        cur.copy_expert(sql=copy_sql, file=csvfile)
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def execute_sql_file(conn, sql_path):
    """Executes a single .sql file on the given open psycopg2 connection."""
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print(f"Executed {os.path.basename(sql_path)} successfully.")
