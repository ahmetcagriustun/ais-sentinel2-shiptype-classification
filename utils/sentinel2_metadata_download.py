import time
import requests
import pandas as pd

from utils.config_utils import load_config
from utils.db_utils import create_pg_connection, ensure_table_exists, insert_dicts_to_table


def _post_with_retry(url: str, headers: dict, json_payload: dict, timeout: int = 120, max_tries: int = 6) -> requests.Response:
    """
    POST request with exponential backoff for transient HTTP errors.
    """
    for attempt in range(1, max_tries + 1):
        resp = requests.post(url, json=json_payload, headers=headers, timeout=timeout)

        if resp.status_code == 200:
            return resp

        # Retry on transient errors (rate-limit / gateway / temporary outage)
        if resp.status_code in (429, 500, 502, 503, 504):
            sleep_s = min(2 ** attempt, 60)
            print(f"[WARN] POST {url} -> {resp.status_code}. Retry {attempt}/{max_tries} in {sleep_s}s...")
            time.sleep(sleep_s)
            continue

        # Non-retryable errors
        raise RuntimeError(f"Request failed ({resp.status_code}): {resp.text}")

    raise RuntimeError(f"Request failed after retries ({resp.status_code}): {resp.text}")


def download_and_store_sentinel2_metadata(
    config_path: str = "config.yaml",
    start_date: str | None = None,
    end_date: str | None = None,
    bbox: list[float] | None = None,
    limit: int = 100,
    table_name: str = "public.sentinel_metadata_all",
) -> int:
    """
    Download Sentinel-2 L2A STAC metadata from Sentinel Hub Catalog and store a normalized subset into Postgres.

    Notes
    -----
    - Uses STAC /search pagination via links[].rel == 'next' and Sentinel Hub's POST-body cursor.
    - Stores only stable fields to avoid schema drift.
    """

    cfg = load_config(config_path)

    # --- Resolve params from config ---
    if bbox is None:
        bbox = cfg.get("project", {}).get("region_bbox")
    if start_date is None:
        start_date = cfg.get("project", {}).get("start_date")
    if end_date is None:
        end_date = cfg.get("project", {}).get("end_date")

    if not bbox or start_date is None or end_date is None:
        raise ValueError(
            f"Missing bbox/start_date/end_date. bbox={bbox}, start_date={start_date}, end_date={end_date}"
        )

    sh = cfg.get("sentinel_hub", {})
    auth_url = sh.get("auth_url")
    catalog_url = sh.get("catalog_url")
    client_id = (sh.get("client_id") or "").strip()
    client_secret = (sh.get("client_secret") or "").strip()

    if not all([auth_url, catalog_url, client_id, client_secret]):
        raise KeyError("Missing sentinel_hub config keys: auth_url, catalog_url, client_id, client_secret")

    # --- 1) Token ---
    auth_payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    r = requests.post(auth_url, data=auth_payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"SentinelHub Auth failed ({r.status_code}): {r.text}")

    access_token = r.json().get("access_token")
    if not access_token:
        raise RuntimeError("No access_token in auth response.")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # --- 2) STAC search (POST) + pagination using next.link body (Sentinel Hub style) ---
    query_payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start_date}/{end_date}",
        "bbox": bbox,
        "limit": int(limit),
    }

    features: list[dict] = []
    payload = query_payload.copy()

    while True:
        resp = _post_with_retry(catalog_url, headers=headers, json_payload=payload, timeout=120, max_tries=6)
        data = resp.json()

        features.extend(data.get("features", []) or [])

        # Sentinel Hub typically provides next page cursor as POST body in the 'next' link
        next_link = next((l for l in (data.get("links", []) or []) if l.get("rel") == "next"), None)
        if not next_link:
            break

        next_method = (next_link.get("method") or "POST").upper()
        next_body = next_link.get("body")

        if next_method == "POST" and isinstance(next_body, dict):
            payload = next_body
            # Safeguard: keep required search params if missing in next_body
            payload.setdefault("collections", query_payload["collections"])
            payload.setdefault("datetime", query_payload["datetime"])
            payload.setdefault("bbox", query_payload.get("bbox"))
            payload.setdefault("limit", query_payload.get("limit"))
        else:
            # If the API does not provide a POST body for next page, stop pagination safely
            break

    if not features:
        print("No Sentinel-2 metadata found for specified range.")
        return 0

    # --- 3) Normalize stable fields ---
    df = pd.json_normalize(features)

    out = pd.DataFrame({
        "api_id": df["id"] if "id" in df.columns else None,
        "datetime": df["properties.datetime"] if "properties.datetime" in df.columns else None,
        "cloud_cover": df["properties.eo:cloud_cover"] if "properties.eo:cloud_cover" in df.columns else None,
        "platform": df["properties.platform"] if "properties.platform" in df.columns else None,
        "constellation": df["properties.constellation"] if "properties.constellation" in df.columns else None,
        "proj_epsg": df["properties.proj:epsg"] if "properties.proj:epsg" in df.columns else None,
        "geom_geojson": df["geometry"].astype(str) if "geometry" in df.columns else None,
        "bbox": df["bbox"].astype(str) if "bbox" in df.columns else None,
    })

    out = out.dropna(subset=["api_id", "datetime"])

    # --- 4) Insert to Postgres ---
    conn = create_pg_connection(cfg["database"])

    schema_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        api_id        TEXT PRIMARY KEY,
        datetime      TIMESTAMPTZ,
        cloud_cover   DOUBLE PRECISION,
        platform      TEXT,
        constellation TEXT,
        proj_epsg     INTEGER,
        geom_geojson  TEXT,
        bbox          TEXT
    );
    """
    ensure_table_exists(conn, table_name, schema_sql)

    # Insert (naive). If duplicates happen, you'll want UPSERT later.
    insert_dicts_to_table(conn, table_name, out)
    conn.close()

    print(f"{len(out)} records written to {table_name}.")
    return len(out)