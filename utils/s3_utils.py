import boto3
import pandas as pd
from io import StringIO
import botocore
import tempfile
import os
import zipfile


def s3_csv_to_dataframe(s3_bucket, s3_key, encoding="utf-8", delimiter=","):
    """
    Downloads a CSV file from S3 and loads it into a Pandas DataFrame.

    Args:
        s3_bucket (str): S3 bucket name.
        s3_key (str): S3 key (path to CSV file).
        encoding (str): File encoding (default "utf-8").
        delimiter (str): Field separator (default "," for CSV).

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    s3 = boto3.client('s3')
    csv_obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    body = csv_obj['Body'].read().decode(encoding)
    df = pd.read_csv(StringIO(body), delimiter=delimiter)
    return df

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

def upload_fileobj_to_s3(fileobj, bucket, s3_prefix, filename, s3_kwargs):
    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    s3_key = s3_prefix + filename
    s3 = boto3.client("s3", **s3_kwargs)
    s3.upload_fileobj(fileobj, bucket, s3_key)

def list_s3_zip_files(bucket, prefix, s3_kwargs):
    import boto3
    s3 = boto3.client('s3', **s3_kwargs)
    paginator = s3.get_paginator('list_objects_v2')
    zips = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.zip'):
                zips.append(obj['Key'])
    return zips

def download_zip_from_s3(bucket, key, local_path, s3_kwargs):
    """Download a zip file from S3 to the given local path."""
    import boto3
    s3 = boto3.client('s3', **s3_kwargs)
    s3.download_file(bucket, key, local_path)

def list_zip_files_in_s3(bucket, prefix, s3_kwargs):
    """Return a list of .zip keys under the given S3 prefix."""
    s3 = boto3.client('s3', **s3_kwargs)
    paginator = s3.get_paginator('list_objects_v2')
    zip_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.zip'):
                zip_keys.append(key)
    return zip_keys
