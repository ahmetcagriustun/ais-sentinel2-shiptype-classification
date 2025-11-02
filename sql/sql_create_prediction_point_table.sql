WITH pairs AS (
    SELECT
        b."MMSI"                  AS "MMSI",
        b."Ship type"             AS "Ship type",
        b."Length"                AS "Length",
        b.geom                    AS geom_before,
        a.geom                    AS geom_after,
        b."# Timestamp"           AS time_before,
        a."# Timestamp"           AS time_after,
        b.sensing_time_without_tz AS sensing_time_without_tz,
        b.api_id               AS api_id,
        b.id AS id,
        
        EXTRACT(EPOCH FROM (a."# Timestamp" - b."# Timestamp")) AS time_total,
        
        EXTRACT(EPOCH FROM (b.sensing_time_without_tz - b."# Timestamp")) AS time_past
        
    FROM closest_mmsi_timestamps b
    JOIN closest_mmsi_timestamps a
      ON b."MMSI" = a."MMSI"
     AND b.api_id = a.api_id
     AND b.sensing_time_without_tz = a.sensing_time_without_tz
     AND b.time_relation = 'before'
     AND a.time_relation = 'after'
),
calc AS (
    SELECT
        id,
        "MMSI",
        "Ship type",
        "Length",
        api_id,
        sensing_time_without_tz,
        
        CASE
            WHEN time_total > 0 THEN
                ST_LineInterpolatePoint(
                    ST_MakeLine(geom_before, geom_after),
                    time_past::double precision / time_total::double precision
                )
            ELSE
                ST_LineInterpolatePoint(
                    ST_MakeLine(geom_before, geom_after),
                    0.5
                )
        END AS geom
    FROM pairs
    WHERE time_total >= 0
)

SELECT
    id,
    "MMSI",
    "Ship type",
    "Length",
    api_id,
    sensing_time_without_tz,
    geom
INTO ship_predicted_positions
FROM calc
;

DELETE FROM ship_predicted_positions
 WHERE geom IS NULL
   OR ST_IsEmpty(geom);