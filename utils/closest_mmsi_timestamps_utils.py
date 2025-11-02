import psycopg2

def batch_insert_closest_mmsi_timestamps(db_conf, verbose=True, create_table_if_not_exists=True):
    conn = psycopg2.connect(**db_conf)
    cur = conn.cursor()

    if create_table_if_not_exists:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema='public'
                AND table_name='closest_mmsi_timestamps'
            );
        """)
        exists = cur.fetchone()[0]
        if not exists:
            cur.execute("""
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
                    geom geometry,
                    sensing_time_without_tz timestamp without time zone,
                    api_id varchar,
                    time_relation text,
                    time_diff numeric,
                    rank bigint
                );
            """)
            conn.commit()
            if verbose:
                print("Table closest_mmsi_timestamps created.")
        else:
            if verbose:
                print("Table closest_mmsi_timestamps already exists. Will append to it.")

    cur.execute("SELECT DISTINCT api_id FROM public.sentinel_download_index_geom;")
    api_ids = [row[0] for row in cur.fetchall()]

    for idx, api_id in enumerate(api_ids, 1):
        if verbose:
            print(f"[{idx}/{len(api_ids)}] Processing {api_id} ...")
        sql = '''
        INSERT INTO closest_mmsi_timestamps
        WITH target_image AS (
            SELECT * 
            FROM public.sentinel_download_index_geom
            WHERE api_id = %s
        )
        , closest_timestamps AS (
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
              ON sr."# Timestamp" BETWEEN sdi.sensing_time_without_tz - INTERVAL '10 minutes'
                                      AND sdi.sensing_time_without_tz + INTERVAL '10 minutes'
             AND ST_Intersects(sr.geom, ST_Transform(sdi.geom, 4326))
        )
        , ranked_timestamps AS (
            SELECT 
                *,  
                ROW_NUMBER() OVER (PARTITION BY "MMSI", time_relation ORDER BY time_diff ASC) AS rank
            FROM 
                closest_timestamps
        )
        SELECT 
            *  
        FROM 
            ranked_timestamps
        WHERE 
            rank = 1;
        '''
        cur.execute(sql, (api_id,))
        conn.commit()
        if verbose:
            print(f"    ✔️ Done for {api_id}")

    cur.close()
    conn.close()
    print("All Sentinel-2 products processed. Table closest_mmsi_timestamps is ready.")
