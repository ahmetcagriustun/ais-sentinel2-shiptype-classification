from __future__ import annotations
"""
CLI for AIS–Sentinel-2 pipeline (uses existing utils as-is; no changes to utils)

Commands
--------
- s2.select-file                 : run SQL selection (sql/sentinel2_metadata_select.sql)
- s2.download                    : read product names from Postgres and call downloader
- s2.build-download-index        : scan S3 zips -> sentinel_download_index table (HTML/XML sensing_time)
- s2.build-download-index-geom   : run SQL to create/populate sentinel_download_index_geom
- ais.build-list                 : run SQL to create ais_download_list
- ais.download                   : download AIS zips to S3 by dates from ais_download_list
- ais.parse-s3                   : iterate S3 AIS ZIPs and insert rows into Postgres via COPY
- db.exec-sql                    : execute a single SQL file on the configured database
                                    python main.py db.exec-sql --sql-file sql/sql_create_point_geom.sql
                                    python main.py db.exec-sql --sql-file sql/sql_create_sensing_time_without_tz.sql
                                    python main.py db.exec-sql --sql-file sql/sql_create_index_sentinel.sql
                                    python main.py db.exec-sql --sql-file sql/sql_create_index_ais1.sql
                                    python main.py db.exec-sql --sql-file sql/sql_create_index_ais2.sql
                                    python main.py db.exec-sql --sql-file sql/sql_create_index_ais3.sql                                  
                                    python main.py db.exec-sql --sql-file sql/sql_create_closest_timestamp_with_time_filter.sql
                                    python main.py db.exec-sql --sql-file sql/sql_create_prediction_point_table.sql
                                    python main.py db.exec-sql --sql-file sql/sql_create_prediction_point_open_sea.sql

python main.py ais.parse-s3 \
  --table ship_raw_data \
  --start-date 2024-01-01 \
  --end-date 2025-01-01

python main.py ais.parse-s3 \
  --table ship_raw_data \
  --key-contains "aisdk-2023-12"

- db.build-closest              : build closest_mmsi_timestamps from ship_raw_data and sentinel_download_index_geom
                                    python main.py db.build-closest --time-window-minutes 10

                                    
- db.exec-sql-batch             : execute all SQL files in a folder (optionally filtered), in order
- patches.build                  : build Sentinel-2 patches by delegating to utils.patches_sentinel2_from_db
                                    python main.py --config config.yaml patches.build \
                                  --table public.ship_predicted_positions \
                                  --patch-size-px 128 \
                                  --bands B02 B03 B04 B08 \
                                  --save-tci \
                                  --log-level INFO

- patches.clean-cloudy           : move/remove cloudy patches by delegating to utils.clean_cloudy_s3

                                    python main.py --config config.yaml patches.clean-cloudy \
                                      --bright-thresh 0.85 \
                                      --max-cloud-ratio 0.20

- patches.rebucket : python main.py --config config.yaml patches.rebucket
                    python main.py --config config.yaml patches.rebucket --extra \
                      --source s3://ais-sentinel2-dataset/training/patches/ \
                      --dest   s3://ais-sentinel2-dataset/training/patches_rebucketed/ \
                      --map-file config/ship_type_map.yaml \
                      --dry-run
"""
import os
import glob
import sys
import subprocess
import argparse
import logging


level = os.getenv("LOG_LEVEL", "INFO").upper()
fmt = os.getenv("LOG_FMT", "[%(levelname)s] %(name)s: %(message)s")
datefmt = os.getenv("LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")

logging.basicConfig(
    level=getattr(logging, level, logging.INFO),
    format=fmt,
    datefmt=datefmt,
)

log_file = os.getenv("LOG_FILE")
if log_file:
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.getLogger().level)
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    logging.getLogger().addHandler(fh)

from utils.config_utils import load_config
from utils.db_utils import (
    insert_dicts_to_table,
    create_table,
    ensure_table_exists, 
    create_pg_connection,
    execute_sql_file,
)
from utils.db_utils import get_product_names_from_postgres  # if you still use it elsewhere
from utils.sentinel2_download import (
    download_and_upload_products_by_name,
    process_zip_files_in_s3_streaming,
)
from utils.ais_download import download_ais_zips_from_dates
from utils.s3_utils import list_s3_zip_files
from utils.ais_parser import process_zip_s3_to_pg_disk,filter_ais_zip_keys
from utils.closest_mmsi_timestamps_utils import batch_insert_closest_mmsi_timestamps

def cmd_s2_select_file(args):
    sql_file = args.sql_file
    if not os.path.isfile(sql_file):
        raise SystemExit(f"SQL file not found: {sql_file}")
    cfg = load_config(args.config)
    conn = create_pg_connection(cfg["database"])
    try:
        print(f"[s2.select-file] Executing: {sql_file}")
        execute_sql_file(conn, sql_file)
        print("[s2.select-file] DONE.")
    finally:
        conn.close()


def cmd_s2_download(args):
    product_names = get_product_names_from_postgres(
        config_path=args.config,
        table_name=args.table,
        column_name=args.column
    )
    print(f"[s2.download] Found {len(product_names)} product names from PostgreSQL.")
    if not product_names:
        print("[s2.download] No products found, exiting.")
        return

    download_and_upload_products_by_name(
        product_name_list=product_names,
        config_path=args.config,
        out_dir=args.out_dir
    )
    print("[s2.download] DONE.")


def cmd_s2_build_download_index(args):
    cfg = load_config(args.config)

    # S3 config
    s3_conf = cfg["s3"]
    s3_bucket = s3_conf["bucket"]
    s3_prefix = args.prefix or s3_conf.get("sentinel2_prefix", "sentinel2/")
    s3_kwargs = {}
    if s3_conf.get("region"):
        s3_kwargs["region_name"] = s3_conf["region"]
    if s3_conf.get("access_key_id") and s3_conf.get("secret_access_key"):
        s3_kwargs["aws_access_key_id"] = s3_conf["access_key_id"]
        s3_kwargs["aws_secret_access_key"] = s3_conf["secret_access_key"]

    conn = create_pg_connection(cfg["database"])
    try:

        table_name = args.table
        schema_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            file_name TEXT NOT NULL,
            sensing_time TIMESTAMPTZ
        );
        """
        create_table(conn, table_name, schema_sql)


        print(f"[s2.build-download-index] Scanning s3://{s3_bucket}/{s3_prefix} …")
        results = process_zip_files_in_s3_streaming(
            s3_bucket, s3_prefix, s3_kwargs, conn, table_name
        )


        if results:
            insert_dicts_to_table(conn, table_name, results)
            print(f"[s2.build-download-index] Inserted {len(results)} rows into {table_name}.")
        else:
            print("[s2.build-download-index] No results to insert.")

        print("[s2.build-download-index] DONE.")
    finally:
        conn.close()


def cmd_s2_build_download_index_geom(args):
    sql_file = args.sql_file
    if not os.path.isfile(sql_file):
        raise SystemExit(f"SQL file not found: {sql_file}")
    cfg = load_config(args.config)
    conn = create_pg_connection(cfg["database"])
    try:
        print(f"[s2.build-download-index-geom] Executing: {sql_file}")
        execute_sql_file(conn, sql_file)
        print("[s2.build-download-index-geom] DONE.")
    finally:
        conn.close()


def cmd_ais_build_list(args):
    sql_file = args.sql_file
    if not os.path.isfile(sql_file):
        raise SystemExit(f"SQL file not found: {sql_file}")
    cfg = load_config(args.config)
    conn = create_pg_connection(cfg["database"])
    try:
        print(f"[ais.build-list] Executing: {sql_file}")
        execute_sql_file(conn, sql_file)
        print("[ais.build-list] DONE.")
    finally:
        conn.close()


def cmd_ais_download(args):
    cfg = load_config(args.config)

    # S3 config
    s3_conf = cfg["s3"]
    s3_bucket = s3_conf["bucket"]
    ais_prefix = args.prefix or s3_conf.get("raw_data_prefix", "AIS/")
    s3_kwargs = {}
    if s3_conf.get("region"):
        s3_kwargs["region_name"] = s3_conf["region"]
    if s3_conf.get("access_key_id") and s3_conf.get("secret_access_key"):
        s3_kwargs["aws_access_key_id"] = s3_conf["access_key_id"]
        s3_kwargs["aws_secret_access_key"] = s3_conf["secret_access_key"]

    db_conf = cfg["database"]

    print(f"[ais.download] bucket={s3_bucket} prefix={ais_prefix} table={args.table} date_col={args.date_col}")
    download_ais_zips_from_dates(
        db_conf,
        s3_bucket,
        ais_prefix,
        s3_kwargs,
        date_table=args.table,
        date_col=args.date_col
    )
    print("[ais.download] DONE.")

def cmd_ais_parse_s3(args):
    cfg = load_config(args.config)
    db_conf = cfg["database"]
    conn = create_pg_connection(db_conf)

    s3_conf = cfg["s3"]
    s3_bucket = s3_conf["bucket"]
    ais_prefix = args.prefix or s3_conf.get("raw_data_prefix", "AIS/")
    s3_kwargs = {}
    if s3_conf.get("region"):
        s3_kwargs["region_name"] = s3_conf["region"]
    if s3_conf.get("access_key_id") and s3_conf.get("secret_access_key"):
        s3_kwargs["aws_access_key_id"] = s3_conf["access_key_id"]
        s3_kwargs["aws_secret_access_key"] = s3_conf["secret_access_key"]

    table_name = args.table

    try:
        all_zips = list_s3_zip_files(s3_bucket, ais_prefix, s3_kwargs)
        print(f"[ais.parse-s3] Found {len(all_zips)} ZIP files under s3://{s3_bucket}/{ais_prefix}")

        filtered = filter_ais_zip_keys(
            all_zips,
            start_date=args.start_date,
            end_date=args.end_date,
            key_contains=args.key_contains,
        )

        filtered = sorted(filtered)
        print(f"[ais.parse-s3] After filters: {len(filtered)} ZIP files to process.")

        for s3_key in filtered:
            print(f"[ais.parse-s3] Processing: s3://{s3_bucket}/{s3_key}")
            process_zip_s3_to_pg_disk(s3_bucket, s3_key, conn, table_name, s3_kwargs)

        print("[ais.parse-s3] All AIS records inserted!")
    finally:
        conn.close()

def _open_pg_conn_from_cfg(cfg_path: str):
    """Open a Postgres connection using database section in config.yaml."""
    cfg = load_config(cfg_path)
    return create_pg_connection(cfg["database"])


def cmd_db_exec_sql(args):
    """
    Execute a single SQL file on the configured Postgres database.
    """
    sql_file = args.sql_file
    if not os.path.isfile(sql_file):
        raise SystemExit(f"[db.exec-sql] SQL file not found: {sql_file}")

    conn = _open_pg_conn_from_cfg(args.config)
    try:
        print(f"[db.exec-sql] Executing: {sql_file}")
        execute_sql_file(conn, sql_file)
        print("[db.exec-sql] DONE.")
    finally:
        conn.close()


def cmd_db_exec_sql_batch(args):

    folder = args.folder
    if not os.path.isdir(folder):
        raise SystemExit(f"[db.exec-sql-batch] Folder not found: {folder}")

    if args.pattern:
        stems = [args.pattern] if isinstance(args.pattern, str) else args.pattern
        patterns = []
        for st in stems:
            patterns.append(os.path.join(folder, f"{st}.sql"))
            patterns.append(os.path.join(folder, f"{st}.txt"))
    else:
        patterns = [os.path.join(folder, "*.sql"), os.path.join(folder, "*.txt")]

    files = sorted({f for p in patterns for f in glob.glob(p) if os.path.isfile(f)})
    if not files:
        print(f"[db.exec-sql-batch] No SQL files matched in: {folder} (pattern='{args.pattern or '*'}')")
        return

    print(f"[db.exec-sql-batch] {len(files)} file(s) to execute:")
    for f in files:
        print(f"  - {f}")

    conn = _open_pg_conn_from_cfg(args.config)
    try:
        if args.single_transaction:
            print("[db.exec-sql-batch] Running in a single transaction...")
            with conn:
                for fpath in files:
                    print(f"[db.exec-sql-batch] Executing: {fpath}")
                    execute_sql_file(conn, fpath)
        else:
            print("[db.exec-sql-batch] Running files one-by-one (autocommit between files)...")
            for fpath in files:
                print(f"[db.exec-sql-batch] Executing: {fpath}")
                execute_sql_file(conn, fpath)

        print("[db.exec-sql-batch] DONE.")
    finally:
        conn.close()

def cmd_patches_build(args):

    cmd = [
        sys.executable, "-m", "utils.patches_sentinel2_from_db",
        "--config", args.config,
        "--table", args.table,
        "--patch-size-px", str(args.patch_size_px),
        "--log-level", args.log_level,
    ]
    if args.bands:
        cmd.extend(["--bands", *args.bands])
    if args.save_tci:
        cmd.append("--save-tci")
    if args.extra:
        cmd.extend(args.extra)

    print("[patches.build] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[patches.build] DONE.")

def cmd_patches_clean_cloudy(args):
    cmd = [
        sys.executable, "-m", "utils.clean_cloudy_s3",
        "--config", args.config,  # if the script reads config; harmless otherwise
        "--bright-thresh", str(args.bright_thresh),
        "--max-cloud-ratio", str(args.max_cloud_ratio),
    ]
    if args.extra:
        cmd.extend(args.extra)

    print("[patches.clean-cloudy] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[patches.clean-cloudy] DONE.")

def cmd_patches_rebucket(args):
    base_cmd = [sys.executable, "-m", "utils.rebucket_by_ship_type"]
    if args.config:
        base_cmd += ["--config", args.config]
    if args.extra:
        base_cmd += args.extra

    logging.getLogger(__name__).info("[patches.rebucket] Running: %s", " ".join(base_cmd))
    subprocess.run(base_cmd, check=True)
    logging.getLogger(__name__).info("[patches.rebucket] DONE.")

def cmd_db_build_closest(args):

    cfg = load_config(args.config)
    db_conf = cfg["database"]

    batch_insert_closest_mmsi_timestamps(
        db_conf=db_conf,
        verbose=not args.quiet,
        create_table_if_not_exists=not args.no_create,
        truncate_existing=args.truncate,
        time_window_minutes=args.time_window_minutes,
    )

def build_parser():
    p = argparse.ArgumentParser(
        description="AIS–Sentinel-2 pipeline CLI (uses existing utils, no changes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    # s2.select-file
    sf = sub.add_parser("s2.select-file", help="Run SQL file to build selected scenes tables.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sf.add_argument("--sql-file", default="sql/sentinel2_metadata_select.sql",
                    help="Path to the SQL script to execute.")
    sf.set_defaults(func=cmd_s2_select_file)

    # s2.download
    sd = sub.add_parser("s2.download", help="Fetch product names from Postgres and download/upload to S3.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sd.add_argument("--table", default="sentinel_metadata_selected",
                    help="Table name (in public schema) containing product names.")
    sd.add_argument("--column", default="api_id",
                    help="Column holding product names (WITHOUT .SAFE).")
    sd.add_argument("--out-dir", default="./downloads",
                    help="Local directory to store temporary downloads.")
    sd.set_defaults(func=cmd_s2_download)

    # s2.build-download-index
    bi = sub.add_parser("s2.build-download-index",
                        help="Scan S3 zips and populate sentinel_download_index table.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bi.add_argument("--table", default="sentinel_download_index",
                    help="Target table name to create/populate.")
    bi.add_argument("--prefix", default=None,
                    help="Override S3 prefix; default from config.s3.sentinel2_prefix or 'sentinel2/'.")
    bi.set_defaults(func=cmd_s2_build_download_index)

    # s2.build-download-index-geom
    big = sub.add_parser("s2.build-download-index-geom",
                         help="Run SQL to create sentinel_download_index_geom table.",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    big.add_argument("--sql-file", default="sql/create-sentinel_download_index_geom.sql",
                     help="Path to the SQL script that creates/populates *_geom.")
    big.set_defaults(func=cmd_s2_build_download_index_geom)

    # ais.build-list
    abl = sub.add_parser("ais.build-list",
                         help="Run SQL to create 'ais_download_list' from sentinel_download_index_geom.",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    abl.add_argument("--sql-file", default="sql/create_ais_download_list.sql",
                     help="Path to the SQL script that creates ais_download_list (your original spelling).")
    abl.set_defaults(func=cmd_ais_build_list)

    # ais.download
    ad = sub.add_parser("ais.download",
                        help="Download AIS zips to S3 by dates read from a Postgres table.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ad.add_argument("--table", default="ais_download_list",
                    help="Table with dates to download (matches your SQL name).")
    ad.add_argument("--date-col", default="date",
                    help="Date column name in --table.")
    ad.add_argument("--prefix", default=None,
                    help="Override S3 AIS prefix; default from config.s3.raw_data_prefix or 'AIS/'.")
    ad.set_defaults(func=cmd_ais_download)

    # ais.parse-s3
    ap = sub.add_parser(
        "ais.parse-s3",
        help="Parse AIS ZIPs from S3 and insert into Postgres using fast COPY.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--table",
        default="ship_raw_data",
        help="Target Postgres table to insert AIS rows.",
    )
    ap.add_argument(
        "--prefix",
        default=None,
        help="Override S3 prefix; default from config.s3.raw_data_prefix or 'AIS/'.",
    )
    ap.add_argument(
        "--start-date",
        default=None,
        help="Optional start date (YYYY-MM-DD). ZIPs outside [start-date, end-date) are skipped.",
    )
    ap.add_argument(
        "--end-date",
        default=None,
        help="Optional end date (YYYY-MM-DD, exclusive). For 'through 2023-12-31' use 2024-01-01.",
    )
    ap.add_argument(
        "--key-contains",
        default=None,
        help="Optional substring that S3 key basename must contain (e.g. 'aisdk-2023-12').",
    )
    ap.set_defaults(func=cmd_ais_parse_s3)


    # db.build-closest
    bc = sub.add_parser(
        "db.build-closest",
        help="Build closest_mmsi_timestamps from ship_raw_data and sentinel_download_index_geom.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bc.add_argument(
        "--time-window-minutes",
        type=int,
        default=10,
        help="Half-width of time window around sensing_time_without_tz.",
    )
    bc.add_argument(
        "--no-create",
        action="store_true",
        help="Skip CREATE TABLE step; assume closest_mmsi_timestamps already exists.",
    )
    bc.add_argument(
        "--truncate",
        action="store_true",
        help="TRUNCATE closest_mmsi_timestamps before inserting (rebuild from scratch).",
    )
    bc.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console logging.",
    )
    bc.set_defaults(func=cmd_db_build_closest)

    # db.exec-sql (single file)
    es = sub.add_parser("db.exec-sql",
                        help="Execute a single SQL file on the configured database.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    es.add_argument("--sql-file", required=True,
                    help="Path to the SQL file to execute (e.g., sql/sql_create_index.sql).")
    es.set_defaults(func=cmd_db_exec_sql)

    # db.exec-sql-batch (all files under a folder)
    eb = sub.add_parser("db.exec-sql-batch",
                        help="Execute all SQL files under a folder in alphanumeric order.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    eb.add_argument("--folder", default="sql",
                    help="Folder containing SQL files.")
    eb.add_argument("--pattern", default=None,
                    help="Optional filename stem to filter (no extension). Example: 'create_*' or 'idx_*'. "
                         "Both .sql and .txt are considered.")
    eb.add_argument("--single-transaction", action="store_true",
                    help="If set, run all files in a single transaction (all-or-nothing).")
    eb.set_defaults(func=cmd_db_exec_sql_batch)

    # patches.build
    pb = sub.add_parser("patches.build",
                        help="Build Sentinel-2 patches (delegates to utils.patches_sentinel2_from_db).",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pb.add_argument("--table", default="public.ship_predicted_positions",
                    help="PostgreSQL table to read predicted positions from.")
    pb.add_argument("--patch-size-px", type=int, default=128,
                    help="Patch size in pixels.")
    pb.add_argument("--bands", nargs="+", default=["B02", "B03", "B04", "B08"],
                    help="Band list, e.g., B02 B03 B04 B08")
    pb.add_argument("--save-tci", action="store_true",
                    help="If set, also save TCI images.")
    pb.add_argument("--log-level", default="INFO",
                    help="Logging level for the patch builder (e.g., INFO, DEBUG).")
    pb.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Any additional flags to pass through to the underlying module "
                         "(e.g., --limit 500 --overwrite). Everything after --extra is forwarded verbatim.")
    pb.set_defaults(func=cmd_patches_build)

    # patches.clean-cloudy
    cc = sub.add_parser("patches.clean-cloudy",
                        help="Clean/move cloudy patches (delegates to utils.clean_cloudy_s3).",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cc.add_argument("--bright-thresh", type=float, default=0.85,
                    help="Brightness threshold (e.g., 0.85).")
    cc.add_argument("--max-cloud-ratio", type=float, default=0.20,
                    help="Max cloud ratio to keep (e.g., 0.20).")
    cc.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra flags forwarded to the underlying module verbatim.")
    cc.set_defaults(func=cmd_patches_clean_cloudy)

    # patches.rebucket
    pr = sub.add_parser("patches.rebucket",
                        help="Re-bucket patches by ship type (delegates to utils.rebucket_by_ship_type; falls back to utils.rebucked_by_ship_type).",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pr.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="All flags after --extra are forwarded verbatim to the underlying module "
                         "(e.g., --source s3://... --dest s3://... --map-file class_map.yaml --dry-run).")
    pr.set_defaults(func=cmd_patches_rebucket)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
