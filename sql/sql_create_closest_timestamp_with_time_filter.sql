SET work_mem = '512MB';
SET maintenance_work_mem = '2GB';
SET max_parallel_workers_per_gather = 4;
SET max_parallel_workers = 16;

CREATE TABLE closest_mmsi_timestamps AS
WITH closest_timestamps AS (
    SELECT 
        sr.*,  
        sdi.sensing_time_without_tz,
        sdi.api_id, 
        CASE 
            WHEN sr."# Timestamp" <= sdi.sensing_time_without_tz THEN 'before'
            WHEN sr."# Timestamp" >  sdi.sensing_time_without_tz THEN 'after'
        END AS time_relation,
        ABS(EXTRACT(EPOCH FROM (sr."# Timestamp" - sdi.sensing_time_without_tz))) AS time_diff
    FROM 
        ship_raw_data sr
    JOIN 
        sentinel_download_index_geom sdi
      ON sr."# Timestamp" BETWEEN sdi.sensing_time_without_tz - INTERVAL '10 minutes'
                              AND sdi.sensing_time_without_tz + INTERVAL '10 minutes'
     AND ST_Intersects(sr.geom, sdi.geom_4326)
),
ranked_timestamps AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY "MMSI", api_id, sensing_time_without_tz, time_relation
            ORDER BY time_diff ASC
        ) AS rank
    FROM closest_timestamps
)
SELECT 
    *
FROM ranked_timestamps
WHERE rank = 1
  AND time_relation IS NOT NULL;

