ALTER TABLE public.sentinel_metadata_all
ADD COLUMN IF NOT EXISTS geom geometry(Polygon, 4326);

UPDATE public.sentinel_metadata_all
SET geom =
  ST_SetSRID(
    ST_MakeEnvelope(
      (regexp_match(bbox, '\[([^,]+),'))[1]::double precision,                     -- minLon
      (regexp_match(bbox, '\[[^,]+,\s*([^,]+),'))[1]::double precision,            -- minLat
      (regexp_match(bbox, '\[[^,]+,\s*[^,]+,\s*([^,]+),'))[1]::double precision,   -- maxLon
      (regexp_match(bbox, '\[[^,]+,\s*[^,]+,\s*[^,]+,\s*([^\]]+)\]'))[1]::double precision  -- maxLat
    ),
    4326
  )
WHERE geom IS NULL
  AND bbox IS NOT NULL
  AND bbox <> '';

ALTER TABLE public.sentinel_metadata_all
ADD COLUMN IF NOT EXISTS water_area DOUBLE PRECISION;

WITH intersection_areas AS (
    SELECT 
        smw.api_id,
        SUM(
            ST_Area(
                ST_Intersection(
                    ST_Transform(smw.geom, 3857),
                    wp.geom
                )
            )
        ) AS total_water_area
    FROM public.sentinel_metadata_all smw
    JOIN public.water_polygons_buffer wp
      ON ST_Intersects(ST_Transform(smw.geom, 3857), wp.geom)
    GROUP BY smw.api_id
)
UPDATE public.sentinel_metadata_all smw
SET water_area = ia.total_water_area
FROM intersection_areas ia
WHERE smw.api_id = ia.api_id;