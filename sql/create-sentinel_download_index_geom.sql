CREATE TABLE sentinel_download_index_geom AS
SELECT
    LEFT(sdi.file_name, LENGTH(sdi.file_name) - 4) AS product_id,
    sdi.sensing_time,
    smw.*
FROM 
    public.sentinel_download_index sdi
JOIN 
    sentinel_metadata_all smw
ON 
    LEFT(sdi.file_name, LENGTH(sdi.file_name) - 4) = smw.api_id;

ALTER TABLE sentinel_download_index_geom ADD COLUMN geom_4326 geometry;
UPDATE sentinel_download_index_geom SET geom_4326 = ST_Transform(geom, 4326);