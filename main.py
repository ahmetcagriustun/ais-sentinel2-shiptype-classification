from __future__ import annotations

# main.py
# -*- coding: utf-8 -*-
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
- ais.parse-s3                   : iterate S3 AIS ZIPs and insert rows into Postgres in batches (reads ais.batch_size)
- db.exec-sql                    : execute a single SQL file on the configured database
                                    python main.py db.exec-sql --sql-file sql/sql_create_point_geom.sql
                                    python main.py db.exec-sql --sql-file sql/sql_create_sensing_time_without_tz.sql
                                    python main.py db.exec-sql --sql-file sql/sql_create_index.sql
                                    python main.py db.exec-sql --sql-file sql/sql_create_closest_timestamp_with_time_filter.sql
                                    python main.py db.exec-sql --sql-file sql/sql_create_prediction_point_table.sql
                                    python main.py db.exec-sql --sql-file sql/sql_create_prediction_point_open_sea.sql
                                    
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

import os
import glob
import sys
import subprocess
import argparse

# === Use your existing utils (same imports as your old main.py) ===
from utils.config_utils import load_config
from utils.db_utils import (
    insert_dicts_to_table,
    create_table,
    create_pg_connection,
    execute_sql_file,
)
from utils.db_utils import get_product_names_from_postgres  # if you still use it elsewhere
from utils.sentinel2_download import (
    download_and_upload_products_by_name,
    process_zip_files_in_s3_streaming,   # <- same function you used before
)
from utils.ais_download import download_ais_zips_from_dates
from utils.s3_utils import list_s3_zip_files
from utils.ais_parser import process_zip_s3_to_pg_disk


# -----------------------------
# Commands
# -----------------------------
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
    # 1) Product names (WITHOUT .SAFE) from Postgres
    product_names = get_product_names_from_postgres(
        config_path=args.config,
        table_name=args.table,
        column_name=args.column
    )
    print(f"[s2.download] Found {len(product_names)} product names from PostgreSQL.")
    if not product_names:
        print("[s2.download] No products found, exiting.")
        return

    # 2) Download + upload to S3
    download_and_upload_products_by_name(
        product_name_list=product_names,
        config_path=args.config,
        out_dir=args.out_dir
    )
    print("[s2.download] DONE.")


def cmd_s2_build_download_index(args):
    """
    - Read S3 config
    - Create table if not exists
    - process_zip_files_in_s3_streaming(...) to extract sensing_time
    - insert rows into DB
    """
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

    # DB connect
    conn = create_pg_connection(cfg["database"])
    try:
        # Table + schema
        table_name = args.table
        schema_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            file_name TEXT NOT NULL,
            sensing_time TIMESTAMPTZ
        );
        """
        create_table(conn, table_name, schema_sql)

        # Scan S3
        print(f"[s2.build-download-index] Scanning s3://{s3_bucket}/{s3_prefix} …")
        results = process_zip_files_in_s3_streaming(
            s3_bucket, s3_prefix, s3_kwargs, conn, table_name
        )

        # Insert
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
    ais_prefix = args.prefix or s3_conf.get("ais_prefix", "AIS/")
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
    """
    Iterate over AIS ZIPs under S3 raw_data_prefix and insert CSV rows into Postgres
    using utils.ais_parser.process_zip_s3_to_pg_disk. Batch size is read from config.ais.batch_size,
    but can be overridden via --batch-size.
    """
    cfg = load_config(args.config)

    # ---- DB
    db_conf = cfg["database"]
    conn = create_pg_connection(db_conf)

    # ---- S3
    s3_conf = cfg["s3"]
    s3_bucket = s3_conf["bucket"]
    # Use raw_data_prefix from config as requested
    ais_prefix = args.prefix or s3_conf.get("raw_data_prefix", "raw-data/")
    s3_kwargs = {}
    if s3_conf.get("region"):
        s3_kwargs["region_name"] = s3_conf["region"]
    if s3_conf.get("access_key_id") and s3_conf.get("secret_access_key"):
        s3_kwargs["aws_access_key_id"] = s3_conf["access_key_id"]
        s3_kwargs["aws_secret_access_key"] = s3_conf["secret_access_key"]

    # ---- Batch size from config ais.batch_size (default 10000)
    cfg_batch = (cfg.get("ais") or {}).get("batch_size", 10000)
    batch_size = int(args.batch_size or cfg_batch)

    # ---- Target table
    table_name = args.table

    try:
        # List S3 ZIPs
        all_zips = list_s3_zip_files(s3_bucket, ais_prefix, s3_kwargs)
        print(f"[ais.parse-s3] Found {len(all_zips)} ZIP files under s3://{s3_bucket}/{ais_prefix}")

        for s3_key in all_zips:
            print(f"[ais.parse-s3] Processing: s3://{s3_bucket}/{s3_key} (batch_size={batch_size})")
            process_zip_s3_to_pg_disk(
                s3_bucket, s3_key, conn, table_name, s3_kwargs, batch_size=batch_size
            )

        print("[ais.parse-s3] All AIS records inserted!")
    finally:
        conn.close()


# -----------------------------
# Generic DB SQL executors
# -----------------------------
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
    """
    Execute all SQL files under a folder in alphanumeric order.
    - Supports .sql and .txt
    - Use --pattern to narrow (e.g., 'create_*', 'idx_*', '2025_*')
    - With --single-transaction, all files run atomically (if compatible)
    """
    folder = args.folder
    if not os.path.isdir(folder):
        raise SystemExit(f"[db.exec-sql-batch] Folder not found: {folder}")

    # Build glob patterns for .sql and .txt
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


# -----------------------------
# Patches builder (delegates to python -m utils.patches_sentinel2_from_db)
# -----------------------------
def cmd_patches_build(args):
    """
    Delegate to 'python -m utils.patches_sentinel2_from_db' to keep behavior identical
    to your previous workflow. We pass through common flags.
    """
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


# -----------------------------
# Clean cloudy patches (delegates to python -m utils.clean_cloudy_s3)
# -----------------------------
def cmd_patches_clean_cloudy(args):
    """
    Delegate to 'python -m utils.clean_cloudy_s3'.
    Passes through thresholds and optional extras.
    """
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

# -----------------------------
# Re-bucket by ship type (delegates to python -m utils.rebucket_by_ship_type)
# -----------------------------
def cmd_patches_rebucket(args):
    """
    Delegate to 'python -m utils.rebucket_by_ship_type' (and fall back to
    'utils.rebucked_by_ship_type' if needed). All flags after --extra are forwarded verbatim.
    """
    base_cmd = [sys.executable, "-m", "utils.rebucket_by_ship_type"]
    fallback_cmd = [sys.executable, "-m", "utils.rebucked_by_ship_type"]  # in case repo uses old name

    # Forward --config if script supports it (zararsızsa da sorun olmaz)
    forwarded = base_cmd + ["--config", args.config] if args.config else base_cmd
    if args.extra:
        forwarded += args.extra

    print("[patches.rebucket] Trying:", " ".join(forwarded))
    try:
        subprocess.run(forwarded, check=True)
        print("[patches.rebucket] DONE.")
        return
    except subprocess.CalledProcessError as e:
        print(f"[patches.rebucket] Module name 'utils.rebucket_by_ship_type' failed (exit {e.returncode}). Trying fallback...")

    # Fallback to old module name
    forwarded_fb = fallback_cmd + (["--config", args.config] if args.config else [])
    if args.extra:
        forwarded_fb += args.extra

    print("[patches.rebucket] Fallback running:", " ".join(forwarded_fb))
    subprocess.run(forwarded_fb, check=True)
    print("[patches.rebucket] DONE (fallback).")

# -----------------------------
# Argparse
# -----------------------------
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
                    help="Override S3 AIS prefix; default from config.s3.ais_prefix or 'AIS/'.")
    ad.set_defaults(func=cmd_ais_download)

    # ais.parse-s3
    ap = sub.add_parser("ais.parse-s3",
                        help="Parse AIS ZIPs from S3 raw_data_prefix and insert into Postgres with batching.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--table", default="ship_raw_data",
                    help="Target Postgres table to insert AIS rows.")
    ap.add_argument("--prefix", default=None,
                    help="Override S3 prefix; default from config.s3.raw_data_prefix or 'raw-data/'.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override config.ais.batch_size (if not set, defaults to 10000).")
    ap.set_defaults(func=cmd_ais_parse_s3)

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
