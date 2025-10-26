-- Add columns only if they do not already exist
ALTER TABLE public.ship_raw_data
  ADD COLUMN IF NOT EXISTS id BIGSERIAL;

ALTER TABLE public.ship_raw_data
  ADD COLUMN IF NOT EXISTS geom geometry(POINT, 4326);

-- Populate geom only where it is currently NULL, and source coords are present
UPDATE public.ship_raw_data
SET geom = ST_SetSRID(ST_MakePoint("Longitude", "Latitude"), 4326)
WHERE geom IS NULL
  AND "Longitude" IS NOT NULL
  AND "Latitude"  IS NOT NULL;
