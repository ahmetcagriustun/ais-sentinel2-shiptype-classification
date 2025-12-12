CREATE INDEX IF NOT EXISTS idx_sentinel_geom
ON public.sentinel_download_index_geom
USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_sentinel_time
ON public.sentinel_download_index_geom (sensing_time_without_tz);

CREATE INDEX IF NOT EXISTS idx_sentinel_geom4326
ON public.sentinel_download_index_geom
USING GIST (geom_4326);

ANALYZE sentinel_download_index_geom;