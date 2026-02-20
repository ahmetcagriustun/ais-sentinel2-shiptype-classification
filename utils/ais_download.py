import requests
from datetime import datetime
from utils.db_utils import get_date_list_from_db
from utils.s3_utils import s3_file_exists, upload_fileobj_to_s3
from utils.config_utils import load_config


def get_ais_url_and_zipname(date_str: str, base_url: str):
    """
    Build AIS download URL and filename based on the new layout:

    - 2025-01-01 and later:      https://aisdata.ais.dk/aisdk-YYYY-MM-DD.zip       (daily, no year folder)
    - 2024-03-01 .. 2024-12-31:  https://aisdata.ais.dk/2024/aisdk-YYYY-MM-DD.zip  (daily, year folder)
    - <= 2024-02-29 and earlier: https://aisdata.ais.dk/YYYY/aisdk-YYYY-MM.zip     (monthly, year folder)
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    base = base_url.rstrip("/")

    if dt >= datetime(2025, 1, 1):
        zipname = f"aisdk-{dt.strftime('%Y-%m-%d')}.zip"
        url = f"{base}/{zipname}"

    elif dt >= datetime(2024, 3, 1):
        zipname = f"aisdk-{dt.strftime('%Y-%m-%d')}.zip"
        url = f"{base}/2024/{zipname}"

    else:
        zipname = f"aisdk-{dt.strftime('%Y-%m')}.zip"
        url = f"{base}/{dt.strftime('%Y')}/{zipname}"

    return url, zipname


def download_url_to_s3_if_needed(url, s3_bucket, s3_prefix, s3_filename, s3_kwargs):

    if s3_file_exists(s3_bucket, s3_prefix, s3_filename, s3_kwargs):
        print(f"{s3_prefix}{s3_filename} already exists in S3, skipping.")
        return

    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=180)
    if resp.status_code != 200:
        print(f"Download failed ({resp.status_code}): {url}")
        return

    print(f"Uploading {s3_prefix}{s3_filename} to S3...")
    upload_fileobj_to_s3(resp.raw, s3_bucket, s3_prefix, s3_filename, s3_kwargs)
    print(f"Uploaded {s3_filename} to S3.")


def download_ais_zips_from_dates(
    db_config,
    s3_bucket,
    s3_prefix,
    s3_kwargs,
    date_table="ais_download_list",
    date_col="date",
    config_path="config.yaml"
):

    cfg = load_config(config_path)
    base_url = cfg.get("ais", {}).get("base_url", "https://aisdata.ais.dk")
    print(f"[ais.download] Using AIS base URL: {base_url}")

    date_list = get_date_list_from_db(db_config, date_table, date_col)
    print(f"{len(date_list)} unique dates loaded from database.")

    seen_zips = set()
    for date_str in date_list:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        url, zipname = get_ais_url_and_zipname(date_str, base_url)

        # Aylık paketleri aynı ay için tek kez indir
        is_monthly = dt < datetime(2024, 3, 1)
        if is_monthly:
            if zipname in seen_zips:
                continue
            seen_zips.add(zipname)

        try:
            download_url_to_s3_if_needed(url, s3_bucket, s3_prefix, zipname, s3_kwargs)
        except Exception as e:
            print(f"Failed to process {url}: {e}")
