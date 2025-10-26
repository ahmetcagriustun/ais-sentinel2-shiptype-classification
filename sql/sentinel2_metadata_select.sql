CREATE TABLE public.sentinel_metadata_filtered AS SELECT *
FROM public.sentinel_metadata_all
WHERE cloud_cover < 5 AND water_area > 10000000;

SELECT
    EXTRACT(year FROM datetime) AS year,
    EXTRACT(month FROM datetime) AS month,
    EXTRACT(day FROM datetime) AS day,
    COUNT(*) AS image_count
FROM public.sentinel_metadata_all
WHERE cloud_cover < 5 AND water_area > 10000000
GROUP BY year, month, day
ORDER BY image_count DESC;

SELECT
    *,
    COUNT(*) OVER (
        PARTITION BY 
            EXTRACT(year FROM datetime), 
            EXTRACT(month FROM datetime), 
            EXTRACT(day FROM datetime)
    ) AS image_count
FROM public.sentinel_metadata_filtered;


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
WHERE image_count > 9;

Select count(*) from public.sentinel_metadata_selected



