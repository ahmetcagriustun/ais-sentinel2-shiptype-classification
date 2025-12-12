UPDATE public.ship_raw_data
SET geom = ST_SetSRID(ST_MakePoint("Longitude", "Latitude"), 4326)
WHERE "# Timestamp" >= '2023-10-01'::timestamp
  AND "# Timestamp" <  '2023-11-01'::timestamp
  AND geom IS NULL
  AND "Longitude" IS NOT NULL
  AND "Latitude"  IS NOT NULL
  AND "Longitude" BETWEEN -180 AND 180
  AND "Latitude"  BETWEEN -90  AND 90;


CREATE INDEX CONCURRENTLY idx_ship_raw_data_geom
ON public.ship_raw_data
USING GIST (geom);