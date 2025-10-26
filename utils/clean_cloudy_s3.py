# -*- coding: utf-8 -*-
"""
clean_cloudy_s3.py
==================

Purpose
-------
Identify and re-home cloud-contaminated Sentinel-2 training patches on Amazon S3
under a dedicated prefix, to prevent their participation in model training and
evaluation. This script favors quick heuristics on TCI (RGB) scenes when
available, and falls back to a multi-band (B02,B03,B04,B08) proxy when the TCI
is missing.

Key Features
------------
- Scans S3: s3://{bucket}/{images_prefix}/{Class}/*.tif
- Estimates cloud fraction from:
  (a) TCI luminance threshold (fast; default), or
  (b) Multi-band proxy using VIS & NIR intensity + NDVI upper bound (robust).
- Moves (or copies) “cloudy” patches to: s3://{bucket}/{cloudy_prefix}/{Class}/...
- Emits a CSV report and uploads it to s3://{bucket}/{results_prefix}/{run_id}/

Why this matters
----------------
Cloud-laden training samples bias feature distributions and degrade the
discriminative capacity of CNN classifiers for ship-type recognition. Automated
screening allows reproducible curation across large datasets and facilitates
transparent reporting in academic workflows.

CLI Examples
------------
# Original usage (kept for backward compatibility):
python clean_cloudy_s3.py --bright-thresh 0.85 --max-cloud-ratio 0.20

# Safe trial (no changes):
python clean_cloudy_s3.py --dry-run

# Copy instead of move (source remains):
python clean_cloudy_s3.py --copy-only

# Target a custom destination prefix:
python clean_cloudy_s3.py --out-prefix "processed/cloudy_v2/"

# For main.py pass-through compatibility:
python clean_cloudy_s3.py --config config.yaml --mode auto --include-tci --dry-run

Notes
-----
- Thresholds fall back to config.yaml: `quality.*` if CLI not provided.
- If you lack s3:DeleteObject permission, use `--copy-only` to avoid AccessDenied.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator, Tuple, Optional

import boto3
import botocore
import numpy as np

# Optional readers
try:
    import tifffile as tiff
    HAS_TIFF = True
except Exception:
    HAS_TIFF = False

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# Project util
from utils.config_utils import load_config

# ---------- Filename helpers ----------

TCI_RE = re.compile(r"_TCI\.tif$", flags=re.IGNORECASE)

def ensure_prefix(p: str) -> str:
    """Ensure trailing slash for S3 prefixes."""
    return p if p.endswith("/") else p + "/"

def s3_client(s3_cfg: dict):
    """Instantiate boto3.client('s3') with optional static keys and region."""
    kwargs = {}
    if s3_cfg.get("region"):
        kwargs["region_name"] = s3_cfg["region"]
    if s3_cfg.get("access_key_id") and s3_cfg.get("secret_access_key"):
        kwargs["aws_access_key_id"] = s3_cfg["access_key_id"]
        kwargs["aws_secret_access_key"] = s3_cfg["secret_access_key"]
    return boto3.client("s3", **kwargs)

def list_images(s3, bucket: str, prefix: str) -> Iterator[str]:
    """Yield *.tif keys under prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith(".tif"):
                yield k

def read_s3_object_bytes(s3, bucket: str, key: str) -> bytes:
    """Read whole S3 object into memory (suitable for small patches)."""
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()

def copy_object_safe(s3, bucket: str, src_key: str, dst_key: str):
    """Server-side copy with overwrite semantics; idempotent."""
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
        MetadataDirective="COPY",
    )

def rehome_object(s3, bucket: str, src_key: str, dst_key: str, *, copy_only: bool, delete: bool):
    """
    Re-home policy:
      - COPY-ONLY: copy to dst; keep src
      - DELETE: delete src (dangerous; rarely used)
      - MOVE (default): copy to dst, then best-effort delete src
    """
    if copy_only:
        copy_object_safe(s3, bucket, src_key, dst_key)
        return

    if delete:
        s3.delete_object(Bucket=bucket, Key=src_key)
        return

    # MOVE: copy then delete (best-effort delete)
    copy_object_safe(s3, bucket, src_key, dst_key)
    try:
        s3.delete_object(Bucket=bucket, Key=src_key)
    except Exception as e:
        print(f"[WARN] Delete failed for s3://{bucket}/{src_key}: {e}. "
              f"Copied to {dst_key}, source left in place.")

# ---------- Cloud estimators ----------

def estimate_cloud_ratio_from_tci(tci: np.ndarray, bright_thresh: float = 0.80) -> float:
    """
    Estimate cloud fraction from TCI luminance > threshold.

    Rationale
    ---------
    Perceptual luminance (Y) proxies clouds as high-reflectance, near-white pixels.

    Parameters
    ----------
    tci : np.ndarray
        H×W×3 uint16/uint8 array.
    bright_thresh : float
        Luminance threshold (0..1) after normalization by dtype max.

    Returns
    -------
    float
        Fraction of pixels labeled 'cloudy'.
    """
    tci = tci.astype(np.float32)
    if tci.max() > 1.0:
        # Normalize dynamically by metadata-implied range
        tci /= (255.0 if tci.dtype == np.uint8 else 65535.0)
    r, g, b = tci[..., 0], tci[..., 1], tci[..., 2]
    # ITU-R BT.709 luminance
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mask = luminance > bright_thresh
    return float(mask.mean())

def estimate_cloud_ratio_from_multiband(mb: np.ndarray,
                                        vis_thresh: float = 0.75,
                                        nir_thresh: float = 0.65,
                                        ndvi_upper: float = 0.25) -> float:
    """
    Multi-band proxy for clouds: bright in VIS & NIR, with low vegetation index.

    Heuristic
    ---------
    - VIS brightness: mean(B02,B03,B04)
    - NIR brightness: B08
    - NDVI upper bound: (B08 - B04) / (B08 + B04)

    Parameters are tuned for over-flagging avoidance on maritime scenes.

    Returns
    -------
    float
        Fraction of pixels passing the cloud mask.
    """
    C = mb.shape[2]
    if C < 4:  # fallback: only VIS brightness if NIR missing
        vis = mb.mean(axis=2)
        return float((vis > vis_thresh).mean())

    b02, b03, b04, b08 = mb[..., 0], mb[..., 1], mb[..., 2], mb[..., 3]
    vis = (b02 + b03 + b04) / 3.0
    nir = b08
    ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
    cloud_mask = (vis > vis_thresh) & (nir > nir_thresh) & (ndvi < ndvi_upper)
    return float(cloud_mask.mean())

# ---------- Readers ----------

def open_tci_from_s3(s3, bucket: str, key: str) -> Optional[np.ndarray]:
    """Return H×W×3 array from TCI GeoTIFF (uint16) if tifffile/PIL is available."""
    raw = read_s3_object_bytes(s3, bucket, key)
    if HAS_TIFF:
        with tiff.TiffFile(io.BytesIO(raw)) as tf:
            arr = tf.asarray()
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            return arr
    if HAS_PIL:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(im)
    raise RuntimeError("No reader available for TCI (install tifffile or Pillow).")

def open_multiband_from_s3(s3, bucket: str, key: str) -> np.ndarray:
    """Return H×W×C array from multi-band GeoTIFF using tifffile (required)."""
    if not HAS_TIFF:
        raise RuntimeError("tifffile is required for multi-band GeoTIFF reading.")
    raw = read_s3_object_bytes(s3, bucket, key)
    with tiff.TiffFile(io.BytesIO(raw)) as tf:
        arr = tf.asarray()
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] > 4:
            arr = arr[..., :4]
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 65535.0
        return arr

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Re-home cloud-contaminated S2 training patches on S3 with reproducible heuristics."
    )
    ap.add_argument("--config", default="config.yaml", type=str,
                    help="Path to YAML configuration.")
    # Thresholds (override config if provided)
    ap.add_argument("--bright-thresh", type=float, default=None,
                    help="TCI luminance threshold (0..1), default 0.80")
    ap.add_argument("--max-cloud-ratio", type=float, default=None,
                    help="Max allowed cloud ratio (0..1), default 0.30")
    ap.add_argument("--vis-thresh", type=float, default=None,
                    help="VIS brightness threshold for multiband, default 0.75")
    ap.add_argument("--nir-thresh", type=float, default=None,
                    help="NIR threshold for multiband, default 0.65")
    ap.add_argument("--ndvi-upper", type=float, default=None,
                    help="NDVI upper bound for clouds, default 0.25")

    # Destinations & actions
    # NOTE: --out-prefix is an alias to maintain compatibility with main.py passthrough.
    ap.add_argument("--cloudy-prefix", type=str, default=None,
                    help="S3 target prefix for cloudy samples (default: 'cloudy_images/' or config.s3.cloudy_out_prefix)")
    ap.add_argument("--out-prefix", type=str, default=None,
                    help="Alias of --cloudy-prefix for main.py integration.")
    ap.add_argument("--delete", action="store_true",
                    help="Delete cloudy samples instead of moving to cloudy-prefix (dangerous).")
    ap.add_argument("--copy-only", action="store_true",
                    help="Copy cloudy samples to target, keep source in place.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List planned actions without applying changes.")

    # Source selection
    ap.add_argument("--mode", type=str, default="auto", choices=["auto", "tci-only", "merged-only"],
                    help="Cloud estimator mode: prefer TCI (auto), force TCI, or force multiband.")
    ap.add_argument("--include-tci", action="store_true",
                    help="Also re-home the paired *_TCI.tif file when decision is based on merged (default: False).")

    args = ap.parse_args()

    # Load config
    cfg = load_config(args.config)
    s3_cfg = cfg.get("s3", {})
    qc_cfg = cfg.get("quality", {})  # optional thresholds in config

    # Resolve prefixes
    bucket = s3_cfg["bucket"]
    # Accept both 'images_prefix' (ship-type bucketed) or 'training_patches_prefix' (flat)
    images_prefix = s3_cfg.get("images_prefix") or s3_cfg.get("training_patches_prefix") or "training-patches/"
    images_prefix = ensure_prefix(images_prefix)
    results_prefix = ensure_prefix(s3_cfg.get("results_prefix", "results/"))

    cloudy_prefix = args.out_prefix or args.cloudy_prefix or s3_cfg.get("cloudy_out_prefix") or "cloudy_images/"
    cloudy_prefix = ensure_prefix(cloudy_prefix)

    # Threshold resolution (CLI > config > defaults)
    bright_thresh   = args.bright_thresh if args.bright_thresh is not None else float(qc_cfg.get("tci_brightness_threshold", 0.80))
    max_cloud_ratio = args.max_cloud_ratio if args.max_cloud_ratio is not None else float(qc_cfg.get("max_cloud_ratio", 0.30))
    vis_thresh      = args.vis_thresh if args.vis_thresh is not None else float(qc_cfg.get("vis_brightness_threshold", 0.75))
    nir_thresh      = args.nir_thresh if args.nir_thresh is not None else float(qc_cfg.get("nir_threshold", 0.65))
    ndvi_upper      = args.ndvi_upper if args.ndvi_upper is not None else float(qc_cfg.get("ndvi_upper_for_cloud", 0.25))

    s3 = s3_client(s3_cfg)

    print("=== CONFIG ===")
    print(json.dumps({
        "bucket": bucket,
        "images_prefix": images_prefix,
        "results_prefix": results_prefix,
        "cloudy_prefix": cloudy_prefix,
        "bright_thresh": bright_thresh,
        "max_cloud_ratio": max_cloud_ratio,
        "vis_thresh": vis_thresh,
        "nir_thresh": nir_thresh,
        "ndvi_upper": ndvi_upper,
        "mode": args.mode,
        "include_tci": bool(args.include_tci),
        "action": ("DELETE" if args.delete else ("COPY-ONLY" if args.copy_only else "MOVE")),
        "dry_run": args.dry_run
    }, indent=2))

    # Scan once and pair merged <-> TCI by basename
    keys = list(list_images(s3, bucket, images_prefix))
    merged = {}   # base_key -> merged_key
    tci    = {}   # base_key -> tci_key
    classes = set()

    for k in keys:
        rel = k[len(images_prefix):]  # "<Class>/<file>.tif" or "<file>.tif"
        parts = rel.split("/")
        cls = parts[0] if len(parts) > 1 else ""
        if cls:
            classes.add(cls)
        name = parts[-1]
        if TCI_RE.search(name):
            base = name.replace("_TCI.tif", ".tif").replace("_TCI.TIF", ".tif")
            base_key = f"{cls}/{base}" if cls else base
            tci[base_key] = k
        else:
            base_key = f"{cls}/{name}" if cls else name
            merged[base_key] = k

    total = 0
    examined = 0
    cloudy = 0

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"cloud_clean_report_{run_id}.csv"
    rows = []

    # Iterate over merged basenames (authoritative set)
    for base_key, merged_key in merged.items():
        total += 1

        # Resolve decision source
        has_tci = base_key in tci
        use_tci = (args.mode in ("auto", "tci-only")) and has_tci
        if args.mode == "tci-only" and not has_tci:
            # Strict tci-only: skip if TCI is absent
            continue
        if args.mode == "merged-only":
            use_tci = False

        # Read imagery & estimate cloud ratio
        try:
            if use_tci:
                arr = open_tci_from_s3(s3, bucket, tci[base_key])
                cloud_ratio = estimate_cloud_ratio_from_tci(arr, bright_thresh=bright_thresh)
                mode_used = "TCI"
            else:
                arr = open_multiband_from_s3(s3, bucket, merged_key)
                cloud_ratio = estimate_cloud_ratio_from_multiband(
                    arr, vis_thresh=vis_thresh, nir_thresh=nir_thresh, ndvi_upper=ndvi_upper
                )
                mode_used = "MULTIBAND"
        except Exception as e:
            print(f"[WARN] Read/estimate failed for {merged_key}: {e}")
            continue

        if cloud_ratio <= max_cloud_ratio:
            examined += 1
            continue

        # Mark as cloudy → plan re-home
        examined += 1
        cloudy += 1

        # Compute destinations
        # Preserve class subfolder (if any), e.g. <Class>/file.tif → cloudy_prefix/<Class>/file.tif
        dst_merged = merged_key.replace(images_prefix, cloudy_prefix, 1) if merged_key.startswith(images_prefix) else f"{cloudy_prefix}{merged_key}"
        dst_tci = None
        if has_tci and (use_tci or args.include_tci):
            tci_key = tci[base_key]
            dst_tci = tci_key.replace(images_prefix, cloudy_prefix, 1) if tci_key.startswith(images_prefix) else f"{cloudy_prefix}{tci_key}"

        # Apply (or simulate) actions
        action = "MOVE"
        if args.copy_only:
            action = "COPY-ONLY"
        if args.delete:
            action = "DELETE"

        if not args.dry_run:
            try:
                # merged file
                rehome_object(s3, bucket, merged_key, dst_merged, copy_only=args.copy_only, delete=args.delete)
                # paired TCI file (optional)
                if dst_tci:
                    rehome_object(s3, bucket, tci_key, dst_tci, copy_only=args.copy_only, delete=args.delete)
            except botocore.exceptions.ClientError as e:
                print(f"[ERROR] Re-home failed for {merged_key}: {e}")
                continue

        rows.append([
            base_key.split("/")[0] if "/" in base_key else "",
            images_prefix + base_key,
            tci.get(base_key, ""),
            mode_used,
            f"{cloud_ratio:.4f}",
            action,
            (dst_merged or ""),
            (dst_tci or "")
        ])

        if examined % 500 == 0:
            print(f" .. examined={examined} cloudy={cloudy} ({cloudy/max(1,examined):.1%})")

    # Write CSV and upload to S3 results/
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "merged_key", "tci_key", "mode", "cloud_ratio", "action", "dst_merged", "dst_tci"])
        w.writerows(rows)

    print("\n=== SUMMARY ===")
    print(f"Classes: {sorted(list(classes))}")
    print(f"Scanned (merged): {total}")
    print(f"Examined: {examined}")
    print(f"Cloudy (>{max_cloud_ratio:.0%}): {cloudy}  ->  {cloudy/max(1,examined):.1%}")

    s3_key = f"{results_prefix}{run_id}/cloud_clean_report.csv"
    s3.upload_file(str(csv_path), bucket, s3_key)
    print(f"Report uploaded: s3://{bucket}/{s3_key}")

    if args.dry_run:
        print("\n[DRY-RUN] No objects were moved/copied/deleted.")
    else:
        print("\nDone.")

if __name__ == "__main__":
    main()
