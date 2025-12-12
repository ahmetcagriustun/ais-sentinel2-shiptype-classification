CREATE INDEX idx_ship_raw_data_mmsi
ON public.ship_raw_data ("MMSI");

ANALYZE ship_raw_data;