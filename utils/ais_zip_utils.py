import zipfile
import os
from datetime import datetime

def extract_date_from_filename(fname):
    try:
        base = os.path.basename(fname).replace(".csv", "")
        date_part = base.split("-")[1:]  # örnek: ["2024", "08", "28"]
        if len(date_part) == 2:  # örnek: ["2024", "08"]
            return None  # bu bir günlük CSV değil
        date_str = "-".join(date_part)
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        return None

def process_zip_file(local_zip_path, needed_dates, db_conf, log_path):
    try:
        with zipfile.ZipFile(local_zip_path, "r") as z:
            for fname in z.namelist():
                if not fname.endswith(".csv"):
                    continue
                file_date = extract_date_from_filename(fname)
                if file_date and file_date in needed_dates:
                    try:
                        with z.open(fname) as csvf:
                            from utils.db_utils import bulk_copy_csv_to_ship_raw_data
                            bulk_copy_csv_to_ship_raw_data(db_conf, csvf)
                        print(f"{fname} yüklendi ({os.path.basename(local_zip_path)})")
                    except Exception as e:
                        log_error(log_path, f"Hata (CSV): {local_zip_path} -> {fname} -> {str(e)}")
        print(f"{local_zip_path} tamamlandı.")
    except Exception as e:
        log_error(log_path, f"Hata (ZIP): {local_zip_path} -> {str(e)}")

def log_error(log_path, msg):
    with open(log_path, "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{now}] {msg}\n")
