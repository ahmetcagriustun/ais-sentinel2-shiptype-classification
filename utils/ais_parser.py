import zipfile
import io
import tempfile
import os
import boto3

def process_zip_s3_to_pg_disk(
    bucket, s3_key, conn, table_name, s3_kwargs
):
    import zipfile
    import io
    import tempfile
    import os
    import boto3

    from utils.db_utils import ensure_table_exists  # <-- changed import

    schema_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
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
    """
    # Create table once, safely
    ensure_table_exists(conn, table_name, schema_sql)  # <-- changed call

    # Ensure Postgres reads dates like "31/12/2023" properly
    with conn.cursor() as cur:
        cur.execute("SET datestyle = 'ISO, DMY';")
    conn.commit()

    # === Download ZIP from S3 ===
    s3 = boto3.client('s3', **(s3_kwargs or {}))
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        tmp_path = tmp.name
        s3.download_fileobj(bucket, s3_key, tmp)

    try:
        with zipfile.ZipFile(tmp_path, 'r') as z:
            for filename in z.namelist():
                # process only CSV files
                if not filename.lower().endswith('.csv'):
                    continue

                # Open zipped CSV as a text stream (no buffering to memory)
                with z.open(filename, 'r') as f_bin:
                    text_stream = io.TextIOWrapper(f_bin, encoding='utf-8', newline="")

                    # COPY directly from the text stream; use HEADER true
                    copy_sql = f"""
                        COPY {table_name} (
                            "# Timestamp",
                            "Type of mobile",
                            "MMSI",
                            "Latitude",
                            "Longitude",
                            "Navigational status",
                            "ROT",
                            "SOG",
                            "COG",
                            "Heading",
                            "IMO",
                            "Callsign",
                            "Name",
                            "Ship type",
                            "Cargo type",
                            "Width",
                            "Length",
                            "Type of position fixing device",
                            "Draught",
                            "Destination",
                            "ETA",
                            "Data source type",
                            "A",
                            "B",
                            "C",
                            "D"
                        )
                        FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER ',');
                    """
                    with conn.cursor() as cur:
                        cur.copy_expert(copy_sql, text_stream)
                    conn.commit()
                    print(f"[COPY] {s3_key} :: {filename} -> {table_name}")

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
