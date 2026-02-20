import psycopg2
from typing import Mapping, Any

def batch_insert_closest_mmsi_timestamps(
    db_conf: Mapping[str, Any],
    verbose: bool = True,
    create_table_if_not_exists: bool = True,
    truncate_existing: bool = False,
    time_window_minutes: int = 10,
) -> None:

    conn = psycopg2.connect(**db_conf)
    try:
        # Ensure target table exists (and optionally truncate)
        with conn.cursor() as cur:
            if create_table_if_not_exists:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_name   = 'closest_mmsi_timestamps'
                    );
                    """
                )
                exists = cur.fetchone()[0]

                if not exists:
                    cur.execute(
                        """
                        CREATE TABLE closest_mmsi_timestamps (
                            "# Timestamp" timestamp without time zone,
                            "Type of mobile" varchar(255),
                            "MMSI" bigint,
                            "Latitude" double precision,
                            "Longitude" double precision,
                            "Navigational status" varchar(255),
                            "ROT" double precision,
                            "SOG" double precision,
                            "COG" double precision,
                            "Heading" double precision,
                            "IMO" varchar(255),
                            "Callsign" varchar(255),
                            "Name" varchar(255),
                            "Ship type" varchar(255),
                            "Cargo type" varchar(255),
                            "Width" double precision,
                            "Length" double precision,
                            "Type of position fixing device" varchar(255),
                            "Draught" double precision,
                            "Destination" varchar(255),
                            "ETA" varchar(255),
                            "Data source type" varchar(255),
                            "A" double precision,
                            "B" double precision,
                            "C" double precision,
                            "D" double precision,
                            id bigint,
                            geom geometry(POINT, 4326),
                            sensing_time_without_tz timestamp without time zone,
                            api_id varchar,
                            time_relation text,
                            time_diff numeric,
                            rank bigint
                        );
                        """
                    )
                    conn.commit()
                    if verbose:
                        print("[closest] Table closest_mmsi_timestamps created.")
                else:
                    if truncate_existing:
                        cur.execute("TRUNCATE TABLE closest_mmsi_timestamps;")
                        conn.commit()
                        if verbose:
                            print("[closest] Table closest_mmsi_timestamps truncated.")
                    elif verbose:
                        print("[closest] Table closest_mmsi_timestamps already exists, appending rows.")

        # Fetch api_id list
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT api_id
                FROM public.sentinel_download_index_geom
                WHERE api_id IS NOT NULL
                ORDER BY api_id;
                """
            )
            api_ids = [row[0] for row in cur.fetchall()]

        total = len(api_ids)
        if verbose:
            print(f"[closest] Found {total} api_id values in sentinel_download_index_geom.")

        # Half-width of time window around sensing_time_without_tz
        window = f"{int(time_window_minutes)} minutes"

        for idx, api_id in enumerate(api_ids, start=1):
            if verbose:
                print(f"[closest] [{idx}/{total}] Processing api_id={api_id} ...")

            sql = """
            INSERT INTO closest_mmsi_timestamps
            WITH target_image AS (
                SELECT * 
                FROM public.sentinel_download_index_geom
                WHERE api_id = %s
            ),
            closest_timestamps AS (
                SELECT 
                    sr.*,
                    sdi.sensing_time_without_tz,
                    sdi.api_id,
                    CASE 
                        WHEN sr."# Timestamp" <= sdi.sensing_time_without_tz THEN 'before'
                        WHEN sr."# Timestamp" > sdi.sensing_time_without_tz THEN 'after'
                    END AS time_relation,
                    ABS(EXTRACT(EPOCH FROM (sr."# Timestamp" - sdi.sensing_time_without_tz))) AS time_diff
                FROM 
                    ship_raw_data sr
                JOIN 
                    target_image sdi
                  ON sr."# Timestamp" BETWEEN sdi.sensing_time_without_tz - %s::interval
                                          AND sdi.sensing_time_without_tz + %s::interval
                 AND ST_Intersects(sr.geom, sdi.geom_4326)
            ),
            ranked_timestamps AS (
                SELECT 
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY "MMSI", time_relation
                        ORDER BY time_diff ASC
                    ) AS rank
                FROM closest_timestamps
            )
            SELECT *
            FROM ranked_timestamps
            WHERE rank = 1;
            """

            with conn.cursor() as cur:
                cur.execute(sql, (api_id, window, window))
                inserted = cur.rowcount

            conn.commit()

            if verbose:
                print(f"[closest] [{idx}/{total}] Done for api_id={api_id}, inserted={inserted} rows.")

        if verbose:
            print("[closest] All Sentinel-2 products processed. closest_mmsi_timestamps is ready.")
    finally:
        conn.close()
