# utils/rebucket_auto.py
# -*- coding: utf-8 -*-
"""
Auto-rebucket S3 patches by Ship Type with zero runtime arguments.

Behavior
--------
- Reads S3 config and `training_dataset_prefix` from config.yaml using utils.config_utils.load_config.
- Scans every object under `training_dataset_prefix` (same bucket).
- Infers ship type from filename or path.
- Copies each object to: s3://<bucket>/training-patches-ship-type/<ShipType>/<basename>
- Does NOT delete or overwrite source. If destination already exists, it skips copying.
- No CLI flags. Running this module performs the operation.

Assumptions
-----------
config.yaml structure includes at least:
  s3:
    bucket: <your-bucket>
    region: <optional>
    training_dataset_prefix: <source/prefix/>

You may also optionally add:
  s3:
    dest_training_by_type_prefix: training-patches-ship-type/

Filename → ShipType inference
-----------------------------
1) If basename starts with '<ShipType>_' (letters/spaces/hyphens/underscores) -> use that token.
2) Else, first or second path segment under the source prefix that looks like a class name.
3) Otherwise 'unclassified'.
"""

import os
import re
import sys
from typing import Optional

import boto3
import botocore

from utils.config_utils import load_config


def ensure_trailing_slash(p: str) -> str:
    return p if p.endswith("/") else p + "/"


def is_plausible_shiptype(s: str) -> bool:
    if not s or s.isdigit():
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z _-]*$", s))


def extract_shiptype(src_prefix: str, key: str) -> str:
    base = os.path.basename(key)
    if "_" in base:
        first = base.split("_", 1)[0]
        if is_plausible_shiptype(first):
            return first

    sp = ensure_trailing_slash(src_prefix)
    tail = key[len(sp):] if key.startswith(sp) else key
    segs = [seg for seg in tail.split("/") if seg]
    for seg in segs[:2]:
        if is_plausible_shiptype(seg):
            return seg
    return "unclassified"


def head_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return False
        raise


def list_objects(s3, bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key and not key.endswith("/"):
                yield key


def copy_object(s3, bucket: str, src_key: str, dest_key: str) -> bool:
    if head_exists(s3, bucket, dest_key):
        print(f"[SKIP] exists: s3://{bucket}/{dest_key}")
        return False
    s3.copy_object(Bucket=bucket, Key=dest_key, CopySource={"Bucket": bucket, "Key": src_key})
    print(f"[COPY] s3://{bucket}/{src_key} -> s3://{bucket}/{dest_key}")
    return True


def main():
    cfg = load_config("config.yaml")
    s3_conf = cfg.get("s3", {})
    bucket = s3_conf["bucket"]
    region = s3_conf.get("region")
    src_prefix = s3_conf.get("training_dataset_prefix")
    if not src_prefix:
        print("[FATAL] 's3.training_dataset_prefix' is missing in config.yaml", file=sys.stderr)
        sys.exit(2)

    dest_prefix = s3_conf.get("dest_training_by_type_prefix", "training-patches-ship-type/")

    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")

    src_prefix = ensure_trailing_slash(src_prefix)
    dest_prefix = ensure_trailing_slash(dest_prefix)

    total = 0
    copied = 0

    print(f"[INFO] Bucket      : {bucket}")
    print(f"[INFO] Source      : s3://{bucket}/{src_prefix}")
    print(f"[INFO] Destination : s3://{bucket}/{dest_prefix}")
    print("[INFO] Start listing...")

    for key in list_objects(s3, bucket, src_prefix):
        total += 1
        ship = extract_shiptype(src_prefix, key)
        basename = os.path.basename(key)
        dest_key = f"{dest_prefix}{ship}/{basename}"
        try:
            if copy_object(s3, bucket, key, dest_key):
                copied += 1
        except botocore.exceptions.ClientError as e:
            print(f"[ERROR] copy failed {key} -> {dest_key}: {e}", file=sys.stderr)

    print(f"[DONE] scanned={total}, newly_copied={copied}")
    print("[NOTE] Sources are preserved; no deletions performed.")

if __name__ == "__main__":
    main()
