import argparse
import csv
import io
import json
import re
import time
from pathlib import Path

import boto3
import numpy as np

# Optional dependencies
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

# project util
from utils.config_utils import load_config

TCI_RE = re.compile(r"_TCI\.tif{1,2}$", re.IGNORECASE)

# -------------------- S3 helpers -------------------- #
def ensure_prefix(p: str) -> str:
    return p if p.endswith("/") else p + "/"

def s3_client(s3_cfg: dict):
    kw = {}
    if s3_cfg.get("region"): kw["region_name"] = s3_cfg["region"]
    if s3_cfg.get("access_key_id") and s3_cfg.get("secret_access_key"):
        kw["aws_access_key_id"] = s3_cfg["access_key_id"]
        kw["aws_secret_access_key"] = s3_cfg["secret_access_key"]
    return boto3.client("s3", **kw)

def list_images(s3, bucket: str, prefix: str):
    """Yield keys for .tif/.tiff under prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            low = key.lower()
            if low.endswith(".tif") or low.endswith(".tiff"):
                yield key

def copy_object_safe(s3, bucket: str, src_key: str, dst_key: str):
    # Gerekirse ACL eklenebilir: ACL='bucket-owner-full-control'
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
    )

def move_or_delete_pair(s3, bucket: str, src_key: str, dst_key: str,
                        delete: bool, copy_only: bool = False):
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


# -------------------- Image I/O -------------------- #
def read_tci_from_bytes(b: bytes) -> np.ndarray:
    """Read TCI bytes -> float32 [0,1], shape (H,W,3)."""
    if HAS_TIFF:
        arr = tiff.imread(io.BytesIO(b))
        if arr.ndim == 3 and arr.shape[0] in (3,4) and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1,2,0))  # (C,H,W)->(H,W,C)
    elif HAS_PIL:
        with Image.open(io.BytesIO(b)) as im:
            im = im.convert("RGB")
            arr = np.array(im)
    else:
        raise RuntimeError("tifffile or PIL is required to read TCI images.")
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0 if arr.max() <= 255 else arr.max()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] > 3:
        arr = arr[..., :3]
    return arr

def read_multiband_from_bytes(b: bytes) -> np.ndarray:
    if not HAS_TIFF:
        raise RuntimeError("tifffile is required to read multiband GeoTIFFs.")
    arr = tiff.imread(io.BytesIO(b))
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 16:
        arr = np.transpose(arr, (1,2,0))  # (C,H,W)->(H,W,C)
    arr = arr.astype(np.float32)
    # robust 2-98 percentile scaling
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        band = np.nan_to_num(arr[..., c], nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(band, [2, 98])
        if hi <= lo:
            mn, mx = band.min(), band.max()
            out[..., c] = 0.0 if mx <= mn else (band - mn) / (mx - mn + 1e-6)
        else:
            out[..., c] = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
    return out


# -------------------- Cloud heuristics -------------------- #
def estimate_cloud_ratio_from_tci(tci: np.ndarray, bright_thresh: float = 0.8) -> float:

    r, g, b = tci[...,0], tci[...,1], tci[...,2]
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    mask = luminance > bright_thresh
    return float(mask.mean())

def estimate_cloud_ratio_from_multiband(mb: np.ndarray,
                                        vis_thresh: float = 0.75,
                                        nir_thresh: float = 0.65,
                                        ndvi_upper: float = 0.25) -> float:
    C = mb.shape[2]
    if C < 4:  # fallback: only VIS brightness
        vis = mb.mean(axis=2)
        return float((vis > vis_thresh).mean())

    b02, b03, b04, b08 = mb[...,0], mb[...,1], mb[...,2], mb[...,3]
    vis = (b02 + b03 + b04) / 3.0
    nir = b08
    ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
    cloud_mask = (vis > vis_thresh) & (nir > nir_thresh) & (ndvi < ndvi_upper)
    return float(cloud_mask.mean())


# -------------------- Main -------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", type=str)

    # thresholds (override config)
    ap.add_argument("--bright-thresh", type=float, default=None, help="TCI brightness threshold (0-1), default 0.80")
    ap.add_argument("--max-cloud-ratio", type=float, default=None, help="Max allowed cloud ratio (0-1), default 0.30")
    ap.add_argument("--vis-thresh", type=float, default=None, help="VIS brightness threshold for multiband, default 0.75")
    ap.add_argument("--nir-thresh", type=float, default=None, help="NIR threshold for multiband, default 0.65")
    ap.add_argument("--ndvi-upper", type=float, default=None, help="NDVI upper bound for clouds, default 0.25")

    # destinations & actions
    ap.add_argument("--cloudy-prefix", type=str, default="cloudy_images/",
                    help="S3 target prefix for cloudy samples (default: 'cloudy_images/')")
    ap.add_argument("--delete", action="store_true",
                    help="Delete cloudy samples instead of moving to cloudy-prefix")
    ap.add_argument("--copy-only", action="store_true",
                    help="Copy to cloudy-prefix without deleting source (safe if no DeleteObject permission).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report; do not copy/move/delete")

    args = ap.parse_args()

    cfg = load_config(args.config)
    s3_cfg = cfg.get("s3", {})
    qc_cfg = cfg.get("quality", {})  # optional

    bucket = s3_cfg["bucket"]
    images_prefix = ensure_prefix(s3_cfg.get("images_prefix", "training-patches-ship-type2/"))
    results_prefix = ensure_prefix(s3_cfg.get("results_prefix", "results/"))
    cloudy_prefix = ensure_prefix(args.cloudy_prefix)

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
        "action": ("DELETE" if args.delete else ("COPY-ONLY" if args.copy_only else "MOVE")),
        "dry_run": args.dry_run
    }, indent=2))

    # Scan keys once
    keys = list(list_images(s3, bucket, images_prefix))
    merged = {}   # base_key -> merged_key
    tci    = {}   # base_key -> tci_key
    classes = set()

    for k in keys:
        rel = k[len(images_prefix):]  # "<Class>/<file>.tif"
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        cls = parts[0]
        classes.add(cls)
        name = parts[-1]
        if TCI_RE.search(name):
            base = name.replace("_TCI.tif", ".tif").replace("_TCI.TIF", ".tif")
            base_key = f"{cls}/{base}"
            tci[base_key] = k
        else:
            base_key = f"{cls}/{name}"
            merged[base_key] = k

    total = 0
    examined = 0
    cloudy = 0

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"cloud_clean_report_{run_id}.csv"

    rows = []
    print("\nScanning samples...")
    for base_key, merged_key in sorted(merged.items()):
        total += 1

        # Compute cloud ratio (prefer TCI)
        cloud_ratio = None
        mode = None
        if base_key in tci:
            resp = s3.get_object(Bucket=bucket, Key=tci[base_key])
            data = resp["Body"].read()
            arr = read_tci_from_bytes(data)
            cloud_ratio = estimate_cloud_ratio_from_tci(arr, bright_thresh=bright_thresh)
            mode = "TCI"
        else:
            resp = s3.get_object(Bucket=bucket, Key=merged_key)
            data = resp["Body"].read()
            mb = read_multiband_from_bytes(data)
            cloud_ratio = estimate_cloud_ratio_from_multiband(
                mb, vis_thresh=vis_thresh, nir_thresh=nir_thresh, ndvi_upper=ndvi_upper
            )
            mode = "MULTIBAND"

        examined += 1
        cls = base_key.split("/")[0]
        action = "KEEP"
        dst_merged = None
        dst_tci = None

        if cloud_ratio >= max_cloud_ratio:
            cloudy += 1
            if not args.dry_run:
                # destination under cloudy_prefix (preserve class folder)
                dst_merged = cloudy_prefix + base_key  # e.g., cloudy_images/Cargo/file.tif
                move_or_delete_pair(s3, bucket, merged_key, dst_merged,
                                    delete=args.delete, copy_only=args.copy_only)
                if base_key in tci:
                    tci_key = tci[base_key]
                    # keep class subpath
                    dst_tci = cloudy_prefix + tci_key[len(images_prefix):]  # e.g., cloudy_images/Cargo/file_TCI.tif
                    move_or_delete_pair(s3, bucket, tci_key, dst_tci,
                                        delete=args.delete, copy_only=args.copy_only)
                action = "DELETE" if args.delete else ("COPY-ONLY" if args.copy_only else "MOVE")
            else:
                action = "WOULD_" + ("DELETE" if args.delete else ("COPY-ONLY" if args.copy_only else "MOVE"))

        rows.append([
            cls, images_prefix + base_key, tci.get(base_key, ""),
            mode, f"{cloud_ratio:.4f}", action,
            (dst_merged or ""), (dst_tci or "")
        ])

        if examined % 500 == 0:
            print(f" .. examined={examined} cloudy={cloudy} ({cloudy/max(1,examined):.1%})")

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "merged_key", "tci_key", "mode", "cloud_ratio", "action", "dst_merged", "dst_tci"])
        w.writerows(rows)

    print("\n=== SUMMARY ===")
    print(f"Total samples (merged): {total}")
    print(f"Examined: {examined}")
    print(f"Cloudy (>{max_cloud_ratio:.0%}): {cloudy}  ->  {cloudy/max(1,examined):.1%}")

    # Upload report to S3 results/
    s3_key = f"{results_prefix}{run_id}/cloud_clean_report.csv"
    s3.upload_file(str(csv_path), bucket, s3_key)
    print(f"Report uploaded: s3://{bucket}/{s3_key}")

    if args.dry_run:
        print("\n[DRY-RUN] No objects were moved/copied/deleted.")
    else:
        print("\nDone.")

if __name__ == "__main__":
    main()