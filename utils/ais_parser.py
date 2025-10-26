import zipfile
import csv
import io
import tempfile
import os
import boto3

def process_zip_s3_to_pg_disk(
    bucket, s3_key, conn, table_name, s3_kwargs, batch_size=1000
):
    from utils.db_utils import create_table
    schema_sql = """
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
    """
    create_table(conn, table_name, schema_sql)

    # ZIP DOSYASINI İNDİR VE SATIRLARI EKLE
    s3 = boto3.client('s3', **s3_kwargs)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        tmp_path = tmp.name
        s3.download_fileobj(bucket, s3_key, tmp)
    batch = []
    try:
        with zipfile.ZipFile(tmp_path, 'r') as z:
            for filename in z.namelist():
                with z.open(filename) as f:
                    reader = csv.DictReader(io.TextIOWrapper(f, 'utf-8'))
                    for row in reader:
                        batch.append(row)
                        if len(batch) == batch_size:
                            from utils.db_utils import insert_ais_records_to_pg
                            insert_ais_records_to_pg(conn, batch, table_name)
                            batch.clear()
        if batch:
            from utils.db_utils import insert_ais_records_to_pg
            insert_ais_records_to_pg(conn, batch, table_name)
    finally:
        os.remove(tmp_path)
