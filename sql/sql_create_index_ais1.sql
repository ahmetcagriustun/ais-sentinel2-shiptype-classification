CREATE INDEX idx_ship_raw_data_geom
ON public.ship_raw_data
USING GIST (geom);

ANALYZE ship_raw_data;