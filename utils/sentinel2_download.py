import os
import time
import requests
import boto3
import yaml
import botocore
import io
import zipfile
import xml.etree.ElementTree as ET
import psycopg2

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_cds_token(username, password):
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    r = requests.post(url, data=data)
    r.raise_for_status()
    result = r.json()
    return result["access_token"], time.time()

def ensure_token_valid(token_time, max_age=600):
    # Returns True if the token is still valid (default: valid for 10 minutes)
    return (time.time() - token_time) < (max_age - 60)

def find_product_id(product_name, session, max_retries=5, retry_wait=10):

    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq '{product_name}.SAFE'"
    for attempt in range(max_retries):
        try:
            r = session.get(url)
            r.raise_for_status()
            js = r.json()
            if js.get("value"):
                return js["value"][0]["Id"]
            return None
        except requests.exceptions.HTTPError as e:
            status = getattr(r, 'status_code', None)
            if status and status >= 500:
                print(f"Server error {status} for {product_name} (Attempt {attempt+1}/{max_retries}). Retrying in {retry_wait}s...")
                time.sleep(retry_wait)
            else:
                print(f"Non-retryable HTTP error {status} for {product_name}: {e}")
                return None
        except Exception as e:
            print(f"Other error for {product_name}: {e}")
            return None
    print(f"Max retries exceeded for {product_name}, skipping.")
    return None

def download_product_zip(product_id, product_name, session, out_dir):
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    path = os.path.join(out_dir, f"{product_name}.zip")
    with session.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return path

def s3_file_exists(bucket, s3_prefix, filename, s3_kwargs):

    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    s3_key = s3_prefix + filename
    s3 = boto3.client("s3", **s3_kwargs)
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise

def upload_to_s3(path, bucket, s3_prefix, s3_kwargs):
    s3_filename = os.path.basename(path)
    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    s3_key = s3_prefix + s3_filename
    s3 = boto3.client("s3", **s3_kwargs)
    s3.upload_file(path, bucket, s3_key)
    print(f"Uploaded to S3: s3://{bucket}/{s3_key}")

def download_and_upload_products_by_name(
    product_name_list,
    config_path="config.yaml",
    out_dir="./downloads"
):

    config = load_config(config_path)
    cop = config["copernicus"]
    s3_conf = config["s3"]
    s3_prefix = s3_conf.get("sentinel2_prefix", "sentinel2/")
    bucket = s3_conf["bucket"]

    s3_kwargs = {}
    if s3_conf.get("region"): s3_kwargs["region_name"] = s3_conf["region"]
    if s3_conf.get("access_key_id") and s3_conf.get("secret_access_key"):
        s3_kwargs["aws_access_key_id"] = s3_conf["access_key_id"]
        s3_kwargs["aws_secret_access_key"] = s3_conf["secret_access_key"]

    os.makedirs(out_dir, exist_ok=True)
    access_token, token_time = get_cds_token(cop["username"], cop["password"])
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {access_token}"})

    for idx, prod_name in enumerate(product_name_list, 1):
        zip_filename = f"{prod_name}.zip"
        if s3_file_exists(bucket, s3_prefix, zip_filename, s3_kwargs):
            print(f"{zip_filename} already exists in S3. Skipping download.")
            continue

        if not ensure_token_valid(token_time):
            access_token, token_time = get_cds_token(cop["username"], cop["password"])
            session.headers.update({"Authorization": f"Bearer {access_token}"})
            print("Token renewed.")

        print(f"{idx}/{len(product_name_list)}: Searching for product {prod_name}...")
        prod_id = find_product_id(prod_name, session)
        if not prod_id:
            print(f"{prod_name} not found or server error, skipping.")
            continue

        print(f"Downloading: {prod_name}")
        try:
            zip_path = download_product_zip(prod_id, prod_name, session, out_dir)
            print(f"Downloaded to: {zip_path}")
            upload_to_s3(zip_path, bucket, s3_prefix, s3_kwargs)
            os.remove(zip_path)
            print(f"{prod_name} finished and local file deleted.\n")
        except Exception as e:
            print(f"Download/upload failed for {prod_name}: {e}")
def extract_sensing_time(xml_content):
    try:
        root = ET.fromstring(xml_content)
        sensing_time_element = root.find(".//SENSING_TIME")
        if sensing_time_element is not None:
            return sensing_time_element.text
    except Exception as e:
        print(f"Error parsing XML: {e}")
    return None

# --- PROCESS .ZIP FILES IN S3 ---
def process_zip_files_in_s3_streaming(s3_bucket, s3_prefix, s3_kwargs, conn, table_name):
    s3 = boto3.client("s3", **s3_kwargs)
    paginator = s3.get_paginator("list_objects_v2")
    inserted_count = 0
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".zip"):
                file_name = key.split("/")[-1].replace(".SAFE.zip", "")
                print(f"Processing: {key}")
                sensing_time = None
                try:
                    s3_response = s3.get_object(Bucket=s3_bucket, Key=key)
                    zip_content = s3_response["Body"].read()
                    with zipfile.ZipFile(io.BytesIO(zip_content), "r") as zip_ref:
                        xml_file_path = next((f for f in zip_ref.namelist() if "MTD_TL.xml" in f), None)
                        if xml_file_path:
                            with zip_ref.open(xml_file_path) as xml_file:
                                xml_content = xml_file.read()
                                sensing_time = extract_sensing_time(xml_content)
                        else:
                            print(f"MTD_TL.xml not found in {key}")
                except Exception as e:
                    print(f"Error processing {key}: {e}")

                # Insert this record immediately
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO sentinel_download_index (file_name, sensing_time) VALUES (%s, %s);",
                            (file_name, sensing_time)
                        )
                    conn.commit()
                    inserted_count += 1
                except Exception as e:
                    print(f"Error inserting {file_name}: {e}")
    print(f"Inserted {inserted_count} records to DB.")

def ensure_sentinel_download_index_table(conn, table_name: str = "public.sentinel_download_index"):
    """
    Create the 'sentinel_download_index' table if it does not exist.
    The table stores product file names and (optionally) parsed sensing_time.
    """
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        file_name     TEXT PRIMARY KEY,
        sensing_time  TIMESTAMPTZ NULL,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_sdi_sensing_time ON {table_name}(sensing_time);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def build_sentinel_download_index(
    config_path: str = "config.yaml",
    table_name: str = "public.sentinel_download_index",
    s3_prefix_override: str | None = None
):

    import psycopg2

    cfg = load_config(config_path)
    s3_conf = cfg.get("s3", {})
    db_conf = cfg.get("database", {})

    bucket = s3_conf["bucket"]
    s3_prefix = s3_prefix_override or s3_conf.get("sentinel2_prefix", "sentinel2/")
    s3_kwargs = {}
    if s3_conf.get("region"):
        s3_kwargs["region_name"] = s3_conf["region"]
    if s3_conf.get("access_key_id") and s3_conf.get("secret_access_key"):
        s3_kwargs["aws_access_key_id"] = s3_conf["access_key_id"]
        s3_kwargs["aws_secret_access_key"] = s3_conf["secret_access_key"]


    conn = psycopg2.connect(
        host=db_conf["host"],
        port=db_conf.get("port", 5432),
        dbname=db_conf["dbname"],
        user=db_conf["user"],
        password=db_conf["password"],
    )

    try:
        ensure_sentinel_download_index_table(conn, table_name=table_name)
        print(f"[s2.index] Ensured table: {table_name}")
        process_zip_files_in_s3_streaming(
            s3_bucket=bucket,
            s3_prefix=s3_prefix,
            s3_kwargs=s3_kwargs,
            conn=conn,
            table_name=table_name
        )
        print("[s2.index] Completed streaming ingest from S3.")
    finally:
        conn.close()
