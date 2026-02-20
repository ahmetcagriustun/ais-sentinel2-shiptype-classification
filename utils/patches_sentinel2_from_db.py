import os
import re
import io
import sys
import json
import math
import time
import uuid
import shutil
import zipfile
import logging
import tempfile
import argparse
from pathlib import Path
from contextlib import ExitStack

# --- Keep GDAL/rasterio modest on memory & threads (must be BEFORE rasterio import!) ---
os.environ.setdefault("GDAL_CACHEMAX", "64")                    # MB
os.environ.setdefault("GDAL_NUM_THREADS", "1")                  # GDAL internal threads
os.environ.setdefault("RIO_MAX_THREADS", "1")                   # rasterio reader threads
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".jp2,.tif")
os.environ.setdefault("JP2KAK_THREADS", "1")                    # if Kakadu is present, keep it single-threaded

import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
import psycopg2
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import gc  # for explicit garbage collection


logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

TRANSFER_CFG = TransferConfig(
    max_concurrency=2,                  
    multipart_threshold=8 * 1024 * 1024,
    multipart_chunksize=8 * 1024 * 1024,
    use_threads=False                  
)

def _slug(text: str) -> str:

    if text is None:
        return "NA"
    t = re.sub(r"\s+", "_", str(text).strip())
    t = re.sub(r"[^A-Za-z0-9_\-]", "", t)
    return t or "NA"

def make_s3_client(region: str | None = None, access_key_id: str | None = None, secret_access_key: str | None = None):

    kwargs = {}
    if region:
        kwargs["region_name"] = region
    if access_key_id and secret_access_key:
        kwargs["aws_access_key_id"] = access_key_id
        kwargs["aws_secret_access_key"] = secret_access_key
    kwargs["config"] = Config(
        retries={'max_attempts': 5, 'mode': 'standard'},
        max_pool_connections=4,
        signature_version='s3v4'
    )
    return boto3.client("s3", **kwargs)


def s3_key_join(prefix: str, *parts: str) -> str:
    prefix = prefix.strip("/")
    suffix = "/".join(p.strip("/") for p in parts if p)
    if prefix and suffix:
        return f"{prefix}/{suffix}"
    return prefix or suffix

def open_pg(db):

    return psycopg2.connect(
        host=db["host"], port=db.get("port", 5432), dbname=db["dbname"], user=db["user"], password=db["password"],
    )


def fetch_api_ids(conn, table: str) -> list[str]:
    sql = f"SELECT DISTINCT api_id FROM {table} WHERE api_id IS NOT NULL ORDER BY api_id;"
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [r[0] for r in rows]


def fetch_points_for_api(conn, table: str, api_id: str, limit: int | None = None):
    sql = f"""
        SELECT 
            id,
            "MMSI"::text AS mmsi,
            "Ship type"   AS ship_type,
            "Length"      AS length,
            api_id,
            sensing_time_without_tz::text AS sensing_time,
            ST_X(geom) AS lon,
            ST_Y(geom) AS lat
        FROM {table}
        WHERE api_id = %s
        ORDER BY id
        { 'LIMIT %s' if limit is not None else ''}
    """
    with conn.cursor() as cur:
        if limit is not None:
            cur.execute(sql, (api_id, limit))
        else:
            cur.execute(sql, (api_id,))
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
    return [{k: v for k, v in zip(cols, r)} for r in rows]

BAND_SUFFIX = {
    "B02": "_B02_10m.jp2",
    "B03": "_B03_10m.jp2",
    "B04": "_B04_10m.jp2",
    "B08": "_B08_10m.jp2",
    "TCI": "_TCI_10m.jp2",
}

def find_band_members_in_safe_zip(zip_path: Path, want_bands: list[str]) -> dict[str, str]:

    want = set(want_bands)
    found: dict[str, str] = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        for b in want:
            suf = BAND_SUFFIX.get(b)
            if not suf:
                continue
            cand = next((n for n in names if n.endswith(suf) and "/IMG_DATA/" in n), None)
            if cand:
                found[b] = cand
    missing = [b for b in want if b not in found]
    if missing:
        logging.warning("Missing bands in %s: %s", zip_path.name, ",".join(missing))
    return found

def extract_members(zip_path: Path, members: dict[str, str], out_dir: Path) -> dict[str, Path]:

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for band, member in members.items():
            fname = Path(member).name
            dst = out_dir / fname
            if not dst.exists():
                with z.open(member) as src, open(dst, "wb") as f:
                    shutil.copyfileobj(src, f)
            paths[band] = dst
    return paths

def lonlat_to_rowcol(src: rasterio.DatasetReader, lon: float, lat: float) -> tuple[int, int]:

    if src.crs is None:
        raise ValueError("Source raster has no CRS; cannot geolocate.")
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = src.index(x, y)
    return int(row), int(col)

def make_centered_window(row: int, col: int, size_px: int) -> Window:
    half = size_px // 2
    return Window(col - half, row - half, size_px, size_px)

def write_multiband_geotiff(out_path: Path, ref_src: rasterio.DatasetReader, window: Window, arrays: list[np.ndarray],
                             dtype: str | None = None, meta_tags: dict | None = None):

    transform = ref_src.window_transform(window)
    height = int(window.height)
    width = int(window.width)
    count = len(arrays)
    dtype = dtype or arrays[0].dtype

    profile = ref_src.profile.copy()
    profile.update({
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": min(512, width),
        "blockysize": min(512, height),
        "nodata": 0,
    })

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(arr, i)
        if meta_tags:
            dst.update_tags(**{k: str(v) for k, v in meta_tags.items()})

def process_one_product(
    *,
    api_id: str,
    points: list[dict],
    s3_client,
    s3_bucket: str,
    s3_sentinel_prefix: str,
    s3_out_prefix: str,
    patch_size_px: int,
    work_dir: Path,
    bands: list[str],
    save_tci: bool = True,
    skip_existing: bool = True,
):

    safe_zip_key = s3_key_join(s3_sentinel_prefix, f"{api_id}.zip")
    local_zip = work_dir / f"{api_id}.zip"

    prod_dir = work_dir / api_id
    made = 0
    skipped = 0

    try:

        if not local_zip.exists():
            logging.info("Downloading %s from s3://%s/%s", api_id, s3_bucket, safe_zip_key)
            try:
                s3_client.download_file(s3_bucket, safe_zip_key, str(local_zip), Config=TRANSFER_CFG)
            except Exception as e:
                logging.error("Failed to download %s: %s", safe_zip_key, e)
                return 0, 0


        want = [b for b in bands if b != "TCI"] + (["TCI"] if save_tci else [])
        members = find_band_members_in_safe_zip(local_zip, want)
        if any(b not in members for b in [b for b in bands if b != "TCI"]):
            logging.error("Missing required 10m bands in %s; skipping.", api_id)
            return 0, 0

        extracted = extract_members(local_zip, members, prod_dir)


        with ExitStack() as stack:
            ref = stack.enter_context(rasterio.open(extracted[bands[0]]))  # e.g., B02 as reference
            band_srcs = {b: stack.enter_context(rasterio.open(extracted[b])) for b in bands if b != "TCI"}
            tci_src = None
            if save_tci and "TCI" in extracted:
                tci_src = stack.enter_context(rasterio.open(extracted["TCI"]))

            for rec in points:
                rec_id = rec["id"]
                mmsi = rec["mmsi"]
                ship_type = rec.get("ship_type", "Undefined")
                lon = float(rec["lon"])  # EPSG:4326
                lat = float(rec["lat"])  # EPSG:4326


                base = f"{_slug(ship_type)}_{_slug(mmsi)}_{rec_id}"
                mb_name = f"{base}.tif"      # multi-band B02,B03,B04,B08
                tci_name = f"{base}_TCI.tif"
                out_key_mb = s3_key_join(s3_out_prefix, api_id, mb_name)
                out_key_tci = s3_key_join(s3_out_prefix, api_id, tci_name)

                if skip_existing:
                    try:
                        s3_client.head_object(Bucket=s3_bucket, Key=out_key_mb)
                        if save_tci:
                            s3_client.head_object(Bucket=s3_bucket, Key=out_key_tci)
                        logging.debug("Exists on S3, skipping %s", base)
                        skipped += 1
                        continue
                    except Exception:
                        pass


                try:
                    row, col = lonlat_to_rowcol(ref, lon, lat)
                    win = make_centered_window(row, col, patch_size_px)
                except Exception as e:
                    logging.warning("Indexing error for rec %s (lon=%s,lat=%s): %s", rec_id, lon, lat, e)
                    continue


                arrays = [band_srcs[b].read(1, window=win, boundless=True, fill_value=0)
                          for b in bands if b != "TCI"]


                local_mb = prod_dir / mb_name
                meta_tags = {
                    "api_id": api_id,
                    "mmsi": mmsi,
                    "ship_type": ship_type,
                    "lon": lon,
                    "lat": lat,
                    "record_id": rec_id,
                    "sensing_time": rec.get("sensing_time", ""),
                    "bands": ",".join([b for b in bands if b != "TCI"]),
                }
                write_multiband_geotiff(local_mb, ref, win, arrays, meta_tags=meta_tags)
                del arrays  # free ASAP


                s3_client.upload_file(str(local_mb), s3_bucket, out_key_mb, Config=TRANSFER_CFG)


                if tci_src is not None:
                    tci_arr = tci_src.read(window=win, boundless=True, fill_value=0)
                    local_tci = prod_dir / tci_name
                    # tci_arr shape: (3, H, W)
                    write_multiband_geotiff(local_tci, tci_src, win,
                                            [tci_arr[0], tci_arr[1], tci_arr[2]],
                                            meta_tags=meta_tags)
                    del tci_arr
                    s3_client.upload_file(str(local_tci), s3_bucket, out_key_tci, Config=TRANSFER_CFG)
                    try:
                        local_tci.unlink()
                    except Exception:
                        pass


                try:
                    local_mb.unlink()
                except Exception:
                    pass

                made += 1

    finally:

        try:
            shutil.rmtree(prod_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            local_zip.unlink(missing_ok=True)
        except Exception:
            pass
        gc.collect()

    logging.info("%s \u2192 patches made=%d, skipped=%d", api_id, made, skipped)
    return made, skipped


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser(description="Cut Sentinel-2 patches for ship points per api_id and upload to S3.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--table", default="public.ship_predicted_positions_open_sea_test", help="Driver table name")
    p.add_argument("--patch-size-px", type=int, default=128, help="Patch size in pixels (e.g., 128)")
    p.add_argument("--bands", nargs="+", default=["B02", "B03", "B04", "B08"], help="10m bands to stack")
    p.add_argument("--save-tci", action="store_true", help="Also save TCI patches")
    p.add_argument("--limit", type=int, default=None, help="Per-api_id point limit for quick tests")
    p.add_argument("--workdir", default=None, help="Working directory (default: system temp)")
    p.add_argument("--s3-bucket", default=None, help="Override S3 bucket (else from config)")
    p.add_argument("--s3-sentinel-prefix", default=None, help="Override S3 prefix for SAFE zips (else from config.s3.sentinel2_prefix)")
    p.add_argument("--s3-out-prefix", default=None, help="Override S3 output prefix (else from config.s3.training_dataset_prefix or 'training-patches/')")
    p.add_argument("--no-skip-existing", action="store_true", help="Do not skip if outputs already exist on S3")
    p.add_argument("--log-level", default="INFO", help="Logging level")

    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_yaml(args.config)


    db_cfg = cfg.get("database", {})
    if not db_cfg:
        raise SystemExit("Missing 'database' section in config.yaml")


    s3_cfg = cfg.get("s3", {})
    bucket = args.s3_bucket or s3_cfg.get("bucket")
    if not bucket:
        raise SystemExit("Missing S3 bucket (either --s3-bucket or config.s3.bucket)")

    s2_prefix = args.s3_sentinel_prefix or s3_cfg.get("sentinel2_prefix", "sentinel2/")
    out_prefix = args.s3_out_prefix or s3_cfg.get("training_dataset_prefix", "training-patches/")

    s3_region = s3_cfg.get("region")
    ak = s3_cfg.get("access_key_id")
    sk = s3_cfg.get("secret_access_key")
    s3 = make_s3_client(s3_region, ak, sk)

    with open_pg(db_cfg) as conn:
        api_ids = fetch_api_ids(conn, args.table)
        logging.info("Found %d unique api_id(s) in %s", len(api_ids), args.table)

        work_dir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="s2patch_"))
        work_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Working dir: %s", work_dir)

        total_made = 0
        total_skipped = 0
        for api_id in api_ids:
            pts = fetch_points_for_api(conn, args.table, api_id, limit=args.limit)
            if not pts:
                logging.info("No points for %s; skipping.", api_id)
                continue

            made, skipped = process_one_product(
                api_id=api_id,
                points=pts,
                s3_client=s3,
                s3_bucket=bucket,
                s3_sentinel_prefix=s2_prefix,
                s3_out_prefix=out_prefix,
                patch_size_px=args.patch_size_px,
                work_dir=work_dir,
                bands=args.bands,
                save_tci=bool(args.save_tci),
                skip_existing=not args.no_skip_existing,
            )
            total_made += made
            total_skipped += skipped

            del pts
            gc.collect()

        logging.info("ALL DONE. New patches: %d, skipped (already on S3): %d", total_made, total_skipped)

        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()
