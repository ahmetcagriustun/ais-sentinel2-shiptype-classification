# AIS–Sentinel-2 Vessel Type Classification Pipeline

This repository contains the **end-to-end pipeline** used to build a ship-centered Sentinel-2 training dataset from AIS and Sentinel-2 data, and to train CNN models for vessel type classification.

The pipeline is orchestrated via a single CLI entry point:

```bash
python main.py <command> [options...]
```

and a shared configuration file:

```bash
config.yaml
```

---

## 1. Repository layout

High-level structure (relevant for running the pipeline):

```text
.
├── main.py                       # CLI entry point (AIS–Sentinel-2 pipeline)
├── config.yaml                   # Global configuration (DB, S3, project, training, cleaner)
├── sql/
│   ├── sentinel2_metadata_geom_and_water_area.sql
│   ├── sentinel2_metadata_select.sql
│   ├── create-sentinel_download_index_geom.sql
│   ├── create_ais_download_list.sql
│   ├── sql_create_point_geom.sql
│   ├── sql_create_sensing_time_without_tz.sql
│   ├── sql_create_index_sentinel.sql
│   ├── sql_create_index_ais1.sql
│   ├── sql_create_index_ais2.sql
│   ├── sql_create_index_ais3.sql
│   ├── sql_create_closest_timestamp_with_time_filter.sql
│   ├── sql_create_prediction_point_table.sql
│   └── sql_create_prediction_point_open_sea.sql
├── utils/
│   ├── ais_download.py                # Download AIS ZIPs to S3
│   ├── ais_parser.py                  # Parse AIS ZIPs on S3 → Postgres (ship_raw_data)
│   ├── ais_zip_utils.py               # Helpers for filtering AIS ZIP keys
│   ├── sentinel2_download.py          # Download Sentinel-2 SAFE archives → S3
│   ├── patches_sentinel2_from_db.py   # Build ship-centered Sentinel-2 patches
│   ├── clean_cloudy_s3.py             # Cloudy patch detection / cleaning on S3
│   ├── delete_clean_cloudy_s3.py      # (optional) Delete already-tagged cloudy patches
│   ├── rebucket_by_ship_type.py       # Re-bucket patches into ship-type folders
│   ├── closest_mmsi_timestamps_utils.py  # Build AIS–Sentinel temporal matches
│   ├── s3_utils.py                    # Generic S3 helpers
│   ├── db_utils.py                    # Generic Postgres helpers
│   ├── config_utils.py                # load_config()
│   └── ...
├── train_cnn3-4_resnet18.py
├── train_cnn3-4_resnet34.py
├── train_cnn3-5_resnet18.py
└── train_cnn3-6_resnet18.py           # Training scripts for different class setups
```

---

## 2. Prerequisites

- Python ≥ 3.9
- PostgreSQL + PostGIS (for AIS, Sentinel metadata, and derived tables)
- An S3-compatible bucket (e.g., AWS S3) for:
  - AIS ZIP archives
  - Sentinel-2 SAFE ZIPs
  - Ship-centered training patches
  - Cloud-cleaned patch sets and training outputs
- AIS data source:
  - The pipeline currently targets the `http://aisdata.ais.dk` layout (daily/monthly ZIPs).
- Sentinel-2 data source:
  - Copernicus Data Space / Sentinel-2 SAFE products (downloaded to S3).
- Sentinel-2 metadata source:
  - Sentinel Hub Catalog (STAC API) via OAuth2 client credentials (`sentinel_hub.*` in config).

Python dependencies (non-exhaustive):

- `boto3`, `psycopg2`, `pandas`, `numpy`, `rasterio`, `tifffile`, `torch`, `torchvision`, `PyYAML`, etc.

Use your preferred environment setup (`requirements.txt`, `conda`, etc.) to install them.

---

## 3. Configuration (`config.yaml`)

All credentials, S3 paths, and training hyperparameters are set in `config.yaml`.

A typical structure:

```yaml
sentinel_hub:
  auth_url: "https://services.sentinel-hub.com/oauth/token"
  catalog_url: "https://services.sentinel-hub.com/api/v1/catalog/search"
  client_id: "<sentinelhub-client-id>"
  client_secret: "<sentinelhub-client-secret>"

ais:
  base_url: "http://aisdata.ais.dk"

project:
  start_date: "2025-05-01T00:00:00Z"
  end_date:   "2025-05-01T23:59:59Z"
  region_bbox: [minLon, minLat, maxLon, maxLat]

database:
  host: "<postgres-host>"
  port: 5432
  dbname: "<dbname>"
  user: "<user>"
  password: "<password>"

copernicus:
  username: "<copernicus-username>"
  password: "<copernicus-password>"

s3:
  bucket: "<bucket-name>"
  region: "<aws-region>"

  # AIS ZIPs
  raw_data_prefix: "AIS/"

  # Sentinel-2 SAFE ZIPs
  sentinel2_prefix: "sentinel2/"

  # Training patches (raw → before rebucket)
  training_dataset_prefix: "training-patches-ship-predicted-positions/"

  # Re-bucketed training patches by ship type
  dest_training_by_type_prefix: "training-patches-ship-type-opensea/"

  # Cross-validation dataset prefix expected by training scripts
  cv_dataset_prefix: "training-patches-ship-type-opensea/"

  # Results & reports
  results_prefix: "results/"

  # Optional cleaner aliases
  images_prefix: "training-patches-ship-type-opensea/"
  cloudy_out_prefix: "cloudy_images/"

training:
  epochs: 30
  batch_size: 64
  learning_rate: 5e-4
  weight_decay: 1e-4
  num_workers: 4
  image_size: 128
  seed: 42
  early_stopping_patience: 7
  mixed_precision: true
  train_val_test_split: [0.8, 0.1, 0.1]

# Which bands are expected in training patches
bands_order: ["B02", "B03", "B04", "B08"]

quality:
  tci_brightness_threshold: 0.80
  max_cloud_ratio: 0.30
  vis_brightness_threshold: 0.75
  nir_threshold: 0.65
  ndvi_upper_for_cloud: 0.25
```

---

## 4. High-level pipeline

The typical **from scratch** workflow is:

1. Initialize DB schema (tables, indices, views)  
2. Select Sentinel-2 scenes  
3. Download selected Sentinel-2 SAFE archives to S3  
4. Build Sentinel-2 ZIP index (sensing time, file name, S3 location)  
5. Derive Sentinel-2 geometry table  
6. Build AIS download list based on Sentinel-2 coverage  
7. Download AIS ZIPs to S3  
8. Parse AIS ZIPs → `ship_raw_data` table in Postgres  
9. Build AIS–Sentinel temporal matches (`closest_mmsi_timestamps`)  
10. Compute prediction points (ship positions at sensing time)  
11. Filter prediction points to open-sea only  
12. Build Sentinel-2 image patches around prediction points  
13. Clean cloudy patches  
14. Re-bucket patches into class-wise folders  
15. Train CNN models using the generated dataset

All steps are operated via `main.py`.

---

## 5. End-to-end workflow (step-by-step)

This section provides a runnable, ordered command list that reproduces the full pipeline from metadata download to model training.

> **Note**
> - All commands assume you are in the repository root.
> - Most commands read defaults from `config.yaml` (DB credentials, S3 bucket/prefixes, project dates/bbox, etc.).
> - SQL scripts are located under `sql/`.

### 5.1. Sentinel-2 metadata → geometry & water area

1) Download Sentinel-2 metadata to Postgres:

```bash
python main.py --config config.yaml s2.metadata-download
```

2) Compute scene geometry and open-water area:

```bash
python main.py --config config.yaml db.exec-sql   --sql-file sql/sentinel2_metadata_geom_and_water_area.sql
```

### 5.2. Select Sentinel-2 products and download SAFE archives

3) Select scenes to use (creates a “selected” table):

```bash
python main.py --config config.yaml s2.select-file   --sql-file sql/sentinel2_metadata_select.sql
```

4) Download selected SAFE archives and upload to S3:

```bash
python main.py --config config.yaml s2.download   --table sentinel_metadata_selected   --column api_id   --out-dir ./downloads
```

### 5.3. Build Sentinel-2 download index (+ geometry)

5) Build download index from S3 SAFE ZIPs:

```bash
python main.py --config config.yaml s2.build-download-index   --table sentinel_download_index
```

6) Create/populate the geometry version of the index table:

```bash
python main.py --config config.yaml s2.build-download-index-geom   --sql-file sql/create-sentinel_download_index_geom.sql
```

### 5.4. AIS download → parse → point geometry

7) Create AIS download list:

```bash
python main.py --config config.yaml ais.build-list   --sql-file sql/create_ais_download_list.sql
```

8) Download AIS ZIPs to S3:

```bash
python main.py --config config.yaml ais.download   --table ais_download_list   --date-col date
```

9) Parse AIS ZIPs from S3 into Postgres:

```bash
python main.py --config config.yaml ais.parse-s3   --table ship_raw_data   --start-date 2024-01-01   --end-date 2025-01-01
```

10) Create PostGIS point geometry for AIS rows:

```bash
python main.py --config config.yaml db.exec-sql   --sql-file sql/sql_create_point_geom.sql
```

### 5.5. Sensing time, indexes, closest timestamps

11) Standardize Sentinel sensing time:

```bash
python main.py --config config.yaml db.exec-sql   --sql-file sql/sql_create_sensing_time_without_tz.sql
```

12) Create performance indexes:

```bash
python main.py --config config.yaml db.exec-sql --sql-file sql/sql_create_index_sentinel.sql
python main.py --config config.yaml db.exec-sql --sql-file sql/sql_create_index_ais1.sql
python main.py --config config.yaml db.exec-sql --sql-file sql/sql_create_index_ais2.sql
python main.py --config config.yaml db.exec-sql --sql-file sql/sql_create_index_ais3.sql
```

13) Create closest timestamp table (SQL definition):

```bash
python main.py --config config.yaml db.exec-sql   --sql-file sql/sql_create_closest_timestamp_with_time_filter.sql
```

> If your pipeline fills closest timestamps via Python, run:
```bash
python main.py --config config.yaml db.build-closest   --time-window-minutes 10   --truncate
```

### 5.6. Prediction points → open-sea filtering

14) Create prediction points:

```bash
python main.py --config config.yaml db.exec-sql   --sql-file sql/sql_create_prediction_point_table.sql
```

15) Filter prediction points to open-sea subset:

```bash
python main.py --config config.yaml db.exec-sql   --sql-file sql/sql_create_prediction_point_open_sea.sql
```

### 5.7. Patch generation → rebucket → cloud cleaning

16) Build Sentinel-2 patches:

```bash
python main.py --config config.yaml patches.build   --table public.ship_prediction_point_open_sea   --patch-size-px 128   --bands B02 B03 B04 B08   --save-tci
```

17) Re-bucket patches into ship-type folders:

```bash
python main.py --config config.yaml patches.rebucket
```

18) Remove cloudy patches (legacy direct script call):

```bash
python utils/clean_cloudy_s3.py
```

> If you prefer using the main CLI wrapper instead:
```bash
python main.py --config config.yaml patches.clean-cloudy   --bright-thresh 0.85   --max-cloud-ratio 0.20
```

### 5.8. Train CNN models

19) Train (example):

```bash
python train_cnn3-4_resnet18.py
```

Other configurations:

```bash
python train_cnn3-4_resnet34.py
python train_cnn3-5_resnet18.py
python train_cnn3-6_resnet18.py
```

---

## 6. Sentinel-2 workflow

Download metadata into a base table (example):

```bash
python main.py --config config.yaml s2.metadata-download   --start-date "2024-06-01T00:00:00Z"   --end-date "2024-06-02T00:00:00Z"   --bbox "minLon,minLat,maxLon,maxLat"   --limit 50   --table "public.sentinel_metadata_all"
```

- Downloads Sentinel-2 L2A STAC metadata from the Sentinel Hub Catalog API.
- Inserts results into `public.sentinel_metadata_all` (or a user-provided table).
- `--bbox` is optional; if omitted, uses `project.region_bbox` from config.

### 6.1. Select Sentinel-2 scenes

Use a SQL script (e.g. `sql/sentinel2_metadata_select.sql`) to select Sentinel-2 products over your region/time window into `sentinel_metadata_selected`:

```bash
python main.py s2.select-file   --sql-file sql/sentinel2_metadata_select.sql
```

This typically:

- Reads from your Sentinel-2 metadata base table,  
- Applies `project.region_bbox` and `project.start_date/end_date`,  
- Writes selected products to `public.sentinel_metadata_selected` (or similar).

### 6.2. Download Sentinel-2 SAFE archives to S3

Download all selected product names from Postgres and upload SAFE ZIPs to S3:

```bash
python main.py s2.download   --table sentinel_metadata_selected   --column api_id   --out-dir ./downloads
```

Behavior:

- Reads product names from `table.column` in Postgres.
- Calls `utils.sentinel2_download.download_and_upload_products_by_name`.
- Downloads each SAFE archive to `--out-dir`, then uploads to S3 under `s3.sentinel2_prefix`.

### 6.3. Build Sentinel-2 download index

Scan S3 for SAFE ZIPs and populate `sentinel_download_index`:

```bash
python main.py s2.build-download-index   --table sentinel_download_index
  # --prefix can override config.s3.sentinel2_prefix if needed
```

This will:

- List all `*.zip` under `s3://<bucket>/<sentinel2_prefix>`,
- Extract sensing time and product name (from `MTD*.xml`),
- Insert rows into `sentinel_download_index`.

### 6.4. Build Sentinel-2 geometry table

Create/populate `sentinel_download_index_geom` (e.g., adding geometries):

```bash
python main.py s2.build-download-index-geom   --sql-file sql/create-sentinel_download_index_geom.sql
```

This SQL script is responsible for:

- Creating `sentinel_download_index_geom`,
- Adding a geometry column,
- Possibly buffering or simplifying footprints.

---

## 7. AIS workflow

### 7.1. Build AIS download list

Create `ais_download_list` from `sentinel_download_index_geom` (time/space envelope around Sentinel-2 scenes):

```bash
python main.py ais.build-list   --sql-file sql/create_ais_download_list.sql
```

The script typically defines:

- Which days to download AIS data for,  
- Possibly restricted to scenes over open sea.

### 7.2. Download AIS ZIPs to S3

Use the download list to fetch AIS ZIPs and upload them to S3:

```bash
python main.py ais.download   --table ais_download_list   --date-col date
  # --prefix to override config.s3.raw_data_prefix if needed
```

This calls `utils.ais_download.download_ais_zips_from_dates` and:

- Fetches AIS ZIPs from `ais.base_url`,
- Stores them under `s3://<bucket>/<raw_data_prefix>/...`.

### 7.3. Parse AIS ZIPs from S3 into Postgres

Parse AIS CSVs inside ZIP archives directly from S3 into `ship_raw_data` (via `COPY`).

Examples:

```bash
python main.py ais.parse-s3   --table ship_raw_data   --start-date 2024-01-01   --end-date 2025-01-01

python main.py ais.parse-s3   --table ship_raw_data   --key-contains "aisdk-2023-12"
```

General behavior:

- Connects to S3 using `config.s3.raw_data_prefix`,
- Lists AIS ZIP keys and filters by:
  - Date range parsed from filenames (`--start-date`, `--end-date`),
  - And/or substring (`--key-contains`),
- Streams each ZIP and inserts AIS records into Postgres with `COPY`.

Useful options:

- `--prefix` – override `config.s3.raw_data_prefix`,
- `--start-date`, `--end-date` – filter by date encoded in ZIP filename,
- `--key-contains` – filter by substring in basename.

---

## 8. AIS–Sentinel temporal association

### 8.1. Create `closest_mmsi_timestamps` schema (SQL)

Use the dedicated SQL script to create the table and any needed indices:

```bash
python main.py db.exec-sql   --sql-file sql/sql_create_closest_timestamp_with_time_filter.sql
```

This script typically defines:

- `closest_mmsi_timestamps` structure,
- Relevant indices on MMSI, product id, and timestamps.

### 8.2. Populate `closest_mmsi_timestamps` with Python

Use the dedicated command to fill the table, based on:

- `ship_raw_data` (AIS),
- `sentinel_download_index_geom` (Sentinel-2 products and footprints).

Example:

```bash
python main.py db.build-closest --time-window-minutes 10
```

More options:

```bash
python main.py db.build-closest   --time-window-minutes 10   --truncate
```

Arguments:

- `--time-window-minutes` – half-width of the temporal window around Sentinel sensing time,
- `--no-create` – skip table creation (if already created),
- `--truncate` – clear existing rows before inserting,
- `--quiet` – reduce console logging.

---

## 9. Prediction points and open-sea filtering

Create prediction points at Sentinel sensing time (e.g., by linearly interpolating between nearest AIS samples) and filter to open-sea locations.

### 9.1. Create prediction point table

```bash
python main.py db.exec-sql   --sql-file sql/sql_create_prediction_point_table.sql
```

### 9.2. Filter to open-sea subset

```bash
python main.py db.exec-sql   --sql-file sql/sql_create_prediction_point_open_sea.sql
```

After these steps, you typically have tables like:

- `ship_prediction_point`  
- `ship_prediction_point_open_sea`

which are then used to drive patch extraction.

---

## 10. Patch generation (Sentinel-2 image chips)

Use the `patches.build` command to generate ship-centered Sentinel-2 patches on S3.

Example:

```bash
python main.py --config config.yaml patches.build   --table public.ship_prediction_point_open_sea   --patch-size-px 128   --bands B02 B03 B04 B08   --save-tci   --log-level INFO
```

Parameters:

- `--table` – source prediction point table (e.g., `public.ship_prediction_point_open_sea`),
- `--patch-size-px` – patch size in pixels (square patch),
- `--bands` – list of bands to extract, ordered as in `bands_order`,
- `--save-tci` – additionally save TCI images for quick visual inspection,
- `--log-level` – logging level for the underlying patch builder,
- `--extra` – forwarded verbatim to `utils.patches_sentinel2_from_db` (e.g., `--limit`, `--overwrite`).

The patches are uploaded to S3 under `s3.training_dataset_prefix`.

---

## 11. Cloud cleaning (patch filtering)

The recommended way is via `main.py`:

```bash
python main.py patches.clean-cloudy   --bright-thresh 0.85   --max-cloud-ratio 0.20
```

---

## 12. Re-bucketing patches by ship type

Once you have a cloud-cleaned patch set, you can re-arrange S3 keys so each class lives under its own prefix (useful for training).

Simple call:

```bash
python main.py patches.rebucket
```

---

## 13. Training CNN models

Training scripts read hyperparameters and S3 paths from `config.yaml`.

### 13.1. Example: 6-class ResNet-18

```bash
python train_cnn3-6_resnet18.py
```

This script typically:

1. Uses `cfg["s3"]["cv_dataset_prefix"]` to locate the re-bucketed dataset on S3.
2. Downloads the dataset locally under `results/cv_<timestamp>/data_cache/`.
3. Builds a label taxonomy (e.g., 6-class setup).
4. Performs stratified K-fold cross-validation (K inferred from configuration).
5. Trains a ResNet backbone and saves metrics/plots under `results/cv_<timestamp>/`.

Other training scripts (`train_cnn3-4_resnet18.py`, `train_cnn3-4_resnet34.py`, `train_cnn3-5_resnet18.py`) follow the same pattern but differ in:

- Label merging strategy (3/4/5/6 classes),
- Backbone architecture (ResNet-18 vs ResNet-34).

You can run them in the same way:

```bash
python train_cnn3-4_resnet18.py
python train_cnn3-4_resnet34.py
python train_cnn3-5_resnet18.py
```

---

## 14. Command reference

Quick reference for all `main.py` sub-commands:

```bash
# Global option
python main.py --config config.yaml <command> [...]

# Sentinel-2
python main.py s2.metadata-download [...]
python main.py s2.select-file --sql-file sql/sentinel2_metadata_select.sql
python main.py s2.download --table sentinel_metadata_selected --column api_id --out-dir ./downloads
python main.py s2.build-download-index --table sentinel_download_index
python main.py s2.build-download-index-geom --sql-file sql/create-sentinel_download_index_geom.sql

# AIS
python main.py ais.build-list --sql-file sql/create_ais_download_list.sql
python main.py ais.download --table ais_download_list --date-col date
python main.py ais.parse-s3 --table ship_raw_data [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--key-contains STR]

# AIS–Sentinel association & DB utilities
python main.py db.build-closest --time-window-minutes 10 [--truncate]
python main.py db.exec-sql --sql-file path/to/file.sql
python main.py db.exec-sql-batch --folder sql [--pattern PATTERN] [--single-transaction]

# Patches
python main.py patches.build --table public.ship_prediction_point_open_sea --patch-size-px 128 --bands B02 B03 B04 B08 [--save-tci]
python main.py patches.clean-cloudy --bright-thresh 0.85 --max-cloud-ratio 0.20
python main.py patches.rebucket
```

---

## Notes on external services and API stability

- AIS provider endpoints, file naming conventions, and availability may change over time.
- Copernicus Data Space and Sentinel Hub APIs (OAuth/token endpoints, Catalog/STAC responses, pagination format) may evolve.
- If requests start failing (401/403/5xx/400), review:
  - credentials in `config.yaml`
  - API endpoints (`sentinel_hub.auth_url`, `sentinel_hub.catalog_url`)
  - pagination logic and required request parameters
  - rate limits / service incidents
- This repository prioritizes reproducibility, but maintenance updates may be required to keep integrations working.
