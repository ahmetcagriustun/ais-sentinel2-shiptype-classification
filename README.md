# AIS–Sentinel-2 Integrated Pipeline for Ship Type Classification

This repository provides a reproducible and modular data-processing pipeline designed to integrate **Automatic Identification System (AIS)** records, **Sentinel-2 optical imagery**, and **OpenStreetMap (OSM)** water polygons for the purpose of **ship type classification and maritime pattern analysis**.  
The framework combines PostgreSQL/PostGIS-based spatial preprocessing, S3-oriented data management, and deep-learning–based model training (ResNet family architectures) for large-scale maritime analytics.

---

## 1. Project Overview

The workflow aims to generate a spatially and temporally consistent dataset by coupling vessel trajectories derived from AIS with the corresponding Sentinel-2 satellite scenes. The unified dataset enables supervised training of ship type classification models and supports open-science reproducibility.

The pipeline is divided into the following stages:

1. **Sentinel-2 Metadata Acquisition and Indexing**  
   - Metadata files are collected from Copernicus SciHub and stored in PostgreSQL.  
   - Each product’s spatial footprint is normalized to EPSG:4326 geometry (`geom_4326`).  

2. **OSM Water Polygon Processing**  
   - Global OSM water polygons are buffered (−1 km) to isolate open-sea areas.  
   - GIST indices are created to accelerate spatial intersections.

3. **AIS Download and Parsing**  
   - AIS daily ZIP archives are downloaded and parsed directly from AWS S3 storage.  
   - Records are filtered by region and time window (±10 min around Sentinel-2 sensing times).

4. **Ship Trajectory Interpolation and Prediction Points**  
   - Linear interpolation is applied between consecutive AIS messages to estimate vessel positions at Sentinel-2 overpass times.  
   - Resulting geometries are stored in `ship_predicted_positions` with derived `geom_4326`.

5. **Open-Sea Intersection and Patch Extraction**  
   - Predicted points are spatially intersected with buffered OSM water polygons to isolate open-sea vessels.  
   - Corresponding Sentinel-2 image patches (B02, B03, B04, B08 bands) are extracted, stored, and versioned on Amazon S3.

6. **Model Training**  
   - The `train_resnet_model.py` script supports K-fold cross-validation, class weighting, and optional class merging (e.g., *Sailing* + *Pleasure* → *Leisure*).  
   - Implemented architectures include ResNet-18, -34, and -50, trained using mixed-precision optimization with AdamW.

---

## 2. Repository Structure

```
├── main.py                       # CLI orchestrator for modular tasks
├── config.example.yaml            # Configuration template (DB, S3, local paths)
├── train_resnet_model.py          # ResNet training with K-fold and class merging
├── sql/
│   ├── sql_create_point_geom.sql
│   ├── sql_create_prediction_point_table.sql
│   ├── sql_create_closest_timestamp_with_time_filter.sql
│   ├── sql_create_index.sql
│   ├── sql_create_prediction_point_open_sea.sql
│   ├── create-sentinel_download_index_geom.sql
│   ├── sql_create_sensing_time_without_tz.sql
│   ├── create_ais_download_list.sql
│   └── sentinel2_metadata_select.sql
└── utils/
    ├── ais_parser.py
    ├── ais_zip_utils.py
    ├── ais_download.py
    ├── closest_mmsi_timestamps_utils.py
    ├── clean_cloudy_s3.py
    ├── db_utils.py
    ├── osm_utils_untested.py
    ├── patches_sentinel2_from_db.py
    ├── sentinel2_download.py
    ├── sentinel2_metadata_untested.py
    ├── s3_utils.py
    └── rebucket_by_ship_type.py
```

---

## 3. Database Architecture

A PostgreSQL + PostGIS backend ensures spatial consistency and efficient query performance.  
Principal tables include:

| Table | Description | Key Columns |
|:------|:-------------|:-------------|
| `sentinel_download_index` | Sentinel-2 metadata and product footprints | `file_name`, `sensing_time`, `geom_4326` |
| `ship_raw_data` | Raw AIS messages per day | `mmsi`, `BaseDateTime`, `geometry` |
| `ship_predicted_positions` | Interpolated vessel positions at Sentinel-2 sensing times | `api_id`, `mmsi`, `geom_4326`, `sensing_time_without_tz` |
| `open_sea_points` | Filtered predicted points within buffered OSM water polygons | `mmsi`, `geom_4326` |

All geometry columns are stored in **EPSG:4326** and indexed using **GIST**.

---

## 4. Configuration

The pipeline reads a YAML configuration file specifying:

```yaml
database:
  host: "<hostname>"
  port: 5432
  dbname: "<database>"
  user: "<user>"
  password: "<password>"

s3:
  bucket: "<bucket-name>"
  region: "eu-central-1"
  ais_prefix: "AIS/"
  sentinel_prefix: "Sentinel2/"
```

Rename `config.example.yaml` to `config.yaml` and edit according to your environment.

---

## 5. Running the Pipeline

### 5.1. Sentinel-2 Index Creation
```bash
python main.py s2.build-download-index --config config.yaml
```

### 5.2. AIS Download and Parsing
```bash
python main.py ais.download --config config.yaml
python main.py ais.parse-s3 --config config.yaml
```

### 5.3. Model Training
```bash
python train_resnet_model.py   --data_dir /path/to/training_patches   --results_dir ./results   --model resnet34   --epochs 20 --batch_size 64   --k_folds 5   --use_loss_weights   --merge "Sailing+Pleasure=Leisure"
```

---

## 6. Model Evaluation

During each fold, training and validation metrics are logged to console and stored in JSON summary files under `results/`.  
Key metrics include:

- Mean training and validation loss  
- Accuracy per fold  
- Class-weighted F1-scores (when applicable)  
- Fold-wise and aggregated confusion matrices  

All split indices are stored in reproducible JSON format for external replication.

---

## 7. Reproducibility and Open Science

This repository adheres to the principles of open and reproducible science:

- **Transparent data flow:** Each SQL file documents the exact transformations from raw to processed data.  
- **Versioned code:** Modular functions in `utils/` ensure deterministic behavior.  
- **Open data compatibility:** All scripts comply with Copernicus Open Access and IMO AIS data regulations.  
- **Hardware-agnostic design:** Fully operational on AWS SageMaker or local environments using PostgreSQL 17 + PostGIS 3.3.

---

## 8. Citation

If you use this repository or any part of its workflow in academic research, please cite:

> Üstün, A. Ç. (2025). *Image-Based Dynamic Ship Classification Using RNN and LSTM Models with AIS Data.* Doctoral course project, Ankara Technical University.

---

## 9. License

This repository is released under the **MIT License**, allowing academic and non-commercial reuse with attribution.

---

## 10. Contact

**Project Lead:**  
Ahmet Çağrı Üstün  
Geographic Information Systems Branch,  
Ankara Water and Sewerage Administration (ASKİ)  
📧 ahmetcagri.ustun [at] gmail.com
