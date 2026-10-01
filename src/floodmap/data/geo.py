"""Geospatial projections and metric area computations in Universal Transverse Mercator (UTM)."""


from pyproj import Transformer


def get_utm_epsg_for_bbox(bbox: tuple[float, float, float, float]) -> int:
    """Computes the appropriate UTM zone EPSG code for a bounding box in WGS84 coordinates.

    Parameters:
        bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    """
    center_lon = (bbox[0] + bbox[2]) / 2.0
    center_lat = (bbox[1] + bbox[3]) / 2.0

    # UTM Zone formula: 1 to 60
    zone = int((center_lon + 180.0) // 6.0) + 1
    # 326xx for Northern Hemisphere, 327xx for Southern Hemisphere
    base_epsg = 32600 if center_lat >= 0.0 else 32700
    return base_epsg + zone


def compute_projected_pixel_metrics(
    bbox: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> tuple[int, float, float, float]:
    """Computes exact metric pixel dimensions and area in the local UTM projected coordinate system.

    Parameters:
        bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84.
        shape: (height, width) raster dimensions in pixels.

    Returns:
        (utm_epsg, pixel_size_x_m, pixel_size_y_m, pixel_area_km2)
    """
    utm_epsg = get_utm_epsg_for_bbox(bbox)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)

    min_lon, min_lat, max_lon, max_lat = bbox
    # Transform opposite corners
    x_min, y_min = transformer.transform(min_lon, min_lat)
    x_max, y_max = transformer.transform(max_lon, max_lat)

    height, width = shape
    total_width_m = abs(x_max - x_min)
    total_height_m = abs(y_max - y_min)

    pixel_size_x_m = total_width_m / max(width, 1)
    pixel_size_y_m = total_height_m / max(height, 1)

    pixel_area_m2 = pixel_size_x_m * pixel_size_y_m
    pixel_area_km2 = pixel_area_m2 / 1.0e6

    return (utm_epsg, pixel_size_x_m, pixel_size_y_m, pixel_area_km2)
