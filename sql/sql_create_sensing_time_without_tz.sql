ALTER TABLE public.sentinel_download_index_geom
ADD COLUMN sensing_time_without_tz timestamp without time zone;

UPDATE public.sentinel_download_index_geom
SET sensing_time_without_tz = timezone('UTC', sensing_time);
