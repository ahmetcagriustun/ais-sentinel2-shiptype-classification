import requests
import pandas as pd
from utils.db_utils import create_pg_connection, create_table, insert_dicts_to_table

def download_osm_data(
    bbox,
    osm_feature="way",
    tags=None,
    out="json",
    overpass_url="http://overpass-api.de/api/interpreter"
):
    """
    Downloads OSM data for a given bounding box and feature type using Overpass API.

    Args:
        bbox (list): [minLon, minLat, maxLon, maxLat]
        osm_feature (str): 'way', 'node', or 'relation'
        tags (dict or None): OSM tags to filter (e.g., {"highway": True, "waterway": True})
        out (str): Output format ('json' or 'xml')
        overpass_url (str): Overpass API endpoint

    Returns:
        pd.DataFrame: DataFrame of OSM elements and their attributes
    """
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"  # south,west,north,east!
    # OSM Overpass Query
    tag_filters = ""
    if tags:
        tag_filters = "".join([f'["{k}"{"=" + str(v) if v not in [True, None] else ""}]' for k, v in tags.items()])
    query = f"""
    [out:{out}][timeout:180];
    (
      {osm_feature}{tag_filters}({bbox_str});
    );
    out body;
    >;
    out skel qt;
    """
    print("Querying OSM Overpass API...")
    response = requests.post(overpass_url, data={"data": query})
    if response.status_code != 200:
        raise RuntimeError(f"OSM/Overpass API error: {response.text}")
    osm_json = response.json()
    # Elements -> DataFrame
    df = pd.json_normalize(osm_json.get("elements", []))
    print(f"Downloaded {df.shape[0]} OSM elements.")
    return df

def osm_df_to_postgres(
    df,
    table_name,
    config_path="config.yaml"
):
    """
    Saves OSM DataFrame to a PostgreSQL table. Table schema inferred from DataFrame columns.

    Args:
        df (pd.DataFrame): DataFrame to write.
        table_name (str): Target table name.
        config_path (str): Path to config.yaml.
    """
    columns = []
    for col, dtype in zip(df.columns, df.dtypes):
        if "int" in str(dtype):
            col_type = "INTEGER"
        elif "float" in str(dtype):
            col_type = "FLOAT"
        elif "datetime" in str(dtype):
            col_type = "TIMESTAMPTZ"
        elif "bool" in str(dtype):
            col_type = "BOOLEAN"
        else:
            col_type = "TEXT"
        columns.append(f'"{col}" {col_type}')
    schema_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n" + ",\n".join(columns) + "\n);"
    from utils.config_utils import load_config
    config = load_config(config_path)
    conn = create_pg_connection(config["database"])
    create_table(conn, table_name, schema_sql)
    insert_dicts_to_table(conn, table_name, df)
    conn.close()
    print(f"OSM data written to PostgreSQL table: {table_name}")

# ------------- KULLANIM ÖRNEĞİ (test için ayrı scriptte çağırabilirsin) -------------

if __name__ == "__main__":
    # Örnek: Sadece ana yollar (highway) çekmek için
    bbox = [27.0, 40.0, 27.5, 40.5]  # [minLon, minLat, maxLon, maxLat]
    tags = {"highway": True}
    df_osm = download_osm_data(bbox, osm_feature="way", tags=tags)
    osm_df_to_postgres(df_osm, table_name="osm_highways")
