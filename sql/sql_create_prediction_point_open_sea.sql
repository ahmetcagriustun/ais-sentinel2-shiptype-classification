CREATE TABLE ship_predicted_positions_open_sea AS
SELECT spp.*
FROM ship_predicted_positions spp
WHERE EXISTS (
    SELECT 1
    FROM public.water_polygons_buffer wp
    WHERE ST_Intersects(
        spp.geom,
        ST_Transform(wp.geom, 4326)
    )
);
