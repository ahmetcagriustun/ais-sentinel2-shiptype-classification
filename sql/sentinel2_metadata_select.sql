CREATE TABLE public.sentinel_metadata_filtered AS SELECT *
FROM public.sentinel_metadata_all
WHERE cloud_cover < 5 AND water_area > 10000000;

CREATE TABLE public.sentinel_metadata_filtered_with_count AS
SELECT
    *,
    COUNT(*) OVER (
        PARTITION BY 
            EXTRACT(year FROM datetime), 
            EXTRACT(month FROM datetime), 
            EXTRACT(day FROM datetime)
    ) AS image_count
FROM public.sentinel_metadata_filtered;

CREATE TABLE public.sentinel_metadata_selected AS
SELECT * FROM public.sentinel_metadata_filtered_with_count
WHERE image_count > 0;




