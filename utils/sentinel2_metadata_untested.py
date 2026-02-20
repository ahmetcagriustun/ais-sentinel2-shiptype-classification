import requests
import pandas as pd
from utils.config_utils import load_config
from utils.db_utils import create_pg_connection, insert_dicts_to_table

def download_and_store_sentinel2_metadata(
    config_path="config.yaml",
    start_date=None,
    end_date=None,
    bbox=None,
    limit=100
):

    config = load_config(config_path)
    
    if bbox is None:
        bbox = config["project"]["region_bbox"]
    if bbox is None:
        bbox = config["project"]["start_date"]
    if bbox is None:
        bbox = config["project"]["end_date"]
    
    # 1. Get API token
    auth_url = config["sentinel_hub"]["auth_url"]
    client_id = config["sentinel_hub"]["client_id"]
    client_secret = config["sentinel_hub"]["client_secret"]
    auth_payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(auth_url, data=auth_payload)
    if response.status_code != 200:
        raise RuntimeError(f"SentinelHub Auth failed: {response.text}")
    access_token = response.json()["access_token"]

    # 2. Query API with pagination
    catalog_url = config["sentinel_hub"]["catalog_url"]
    query_payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start_date}/{end_date}",
        "bbox": bbox,
        "limit": limit
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    features = []
    next_page = True
    payload = query_payload.copy()
    while next_page:
        resp = requests.post(catalog_url, json=payload, headers=headers)
        if resp.status_code != 200:
            print(f"API request failed: {resp.status_code} {resp.text}")
            break
        data = resp.json()
        features.extend(data.get("features", []))

        # Pagination: look for 'next' link
        next_link = next((l for l in data.get("links", []) if l.get("rel") == "next"), None)
        if next_link:
            payload["next"] = next_link["body"]["next"]
        else:
            next_page = False

    if not features:
        print("No Sentinel-2 metadata found for specified range.")
        return 0

    # 3. Convert to DataFrame and select required fields
    df = pd.json_normalize(features)
    # (Opsiyonel: burada gerekli alanlara dönüştürme veya filtreleme yapılabilir.)

    # 4. Insert to PostgreSQL
    conn = create_pg_connection(config["database"])
    insert_dicts_to_table(conn, "sentinel_metadata", df)
    conn.close()

    print(f"{len(df)} records written to sentinel_metadata table.")
    return len(df)
