DROP TABLE IF EXISTS public.ais_download_list;
CREATE TABLE public.ais_download_list AS
SELECT
    DATE(sensing_time) AS date,
    COUNT(*) AS count_per_day
FROM public.sentinel_download_index_geom
WHERE sensing_time IS NOT NULL              
GROUP BY DATE(sensing_time)
ORDER BY date;

-- İstersen sıkılaştır:
ALTER TABLE public.ais_download_list
    ALTER COLUMN date SET NOT NULL;
