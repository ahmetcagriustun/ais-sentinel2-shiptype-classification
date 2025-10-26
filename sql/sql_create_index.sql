-- Geometri için spatial index
CREATE INDEX idx_ship_raw_data_geom
ON public.ship_raw_data
USING GIST (geom);

-- MMSI için (büyük JOIN'ler ve filtreler için çok faydalı)
CREATE INDEX idx_ship_raw_data_mmsi
ON public.ship_raw_data ("MMSI");

-- Zaman damgası için
CREATE INDEX idx_ship_raw_data_timestamp
ON public.ship_raw_data ("# Timestamp");

--Sentinel_metadata geom_index
CREATE INDEX IF NOT EXISTS idx_sentinel_geom
ON public.sentinel_download_index_geom
USING GIST (geom);

--Sentinel_metadata timestamp_index
CREATE INDEX IF NOT EXISTS idx_sentinel_time
ON public.sentinel_download_index_geom (sensing_time_without_tz);

CREATE INDEX IF NOT EXISTS idx_sentinel_geom4326
ON public.sentinel_download_index_geom
USING GIST (geom_4326);
