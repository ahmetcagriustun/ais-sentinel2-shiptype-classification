import zipfile
import io
import tempfile
import os
import boto3

from datetime import datetime, date, timedelta
from typing import Iterable, List, Optional, Tuple

def parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def zip_date_range_from_key(key: str) -> Optional[Tuple[date, date]]:
    import os
    basename = os.path.basename(key)
    if not basename.startswith("aisdk-") or not basename.endswith(".zip"):
        return None

    core = basename[len("aisdk-"):-len(".zip")]  # "2023-05" or "2024-03-10"
    parts = core.split("-")

    # Monthly ZIP: aisdk-YYYY-MM.zip
    if len(parts) == 2:
        year, month = map(int, parts)
        start = date(year, month, 1)
        if month == 12:
            end = date(year + 1, 1, 1)
        else:
            end = date(year, month + 1, 1)
        return start, end

    # Daily ZIP: aisdk-YYYY-MM-DD.zip
    if len(parts) == 3:
        year, month, day = map(int, parts)
        start = date(year, month, day)
        end = start + timedelta(days=1)
        return start, end

    return None


def filter_zip_keys_by_date(
    keys: Iterable[str],
    start_date: Optional[date],
    end_date: Optional[date],
) -> List[str]:
    """
    Keep AIS ZIP keys whose date range intersects [start_date, end_date).

    end_date is exclusive (same logic as '# Timestamp' < end_date).
    """
    if not start_date and not end_date:
        return list(keys)

    filtered: List[str] = []
    for key in keys:
        rng = zip_date_range_from_key(key)
        if rng is None:
            # Skip keys that do not follow aisdk-YYYY-MM(.zip) naming
            continue

        zip_start, zip_end = rng

        if start_date and zip_end <= start_date:
            continue
        if end_date and zip_start >= end_date:
            continue

        filtered.append(key)

    return filtered


def filter_ais_zip_keys(
    keys: Iterable[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    key_contains: Optional[str] = None,
) -> List[str]:
    """
    Convenience wrapper used by the CLI:
    - start_date / end_date are YYYY-MM-DD strings (end_date is exclusive),
    - key_contains filters by substring on the basename.
    """
    import os

    keys_list = list(keys)

    if key_contains:
        keys_list = [
            k for k in keys_list
            if key_contains in os.path.basename(k)
        ]

    start = parse_iso_date(start_date) if isinstance(start_date, str) else start_date
    end = parse_iso_date(end_date) if isinstance(end_date, str) else end_date

    return filter_zip_keys_by_date(keys_list, start, end)

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
