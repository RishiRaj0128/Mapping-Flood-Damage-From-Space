"""Raster I/O and Cloud-Optimized GeoTIFF (COG) generation."""

from pathlib import Path

import numpy as np
import tifffile

from floodmap.logger import get_logger

logger = get_logger("floodmap.data.raster_io")


def save_cog(
    array: np.ndarray,
    output_path: Path | str,
    bbox: tuple[float, float, float, float],
    nodata: float | None = -9999.0,
) -> Path:
    """Saves a 2D numpy array as a standard compressed GeoTIFF/COG with spatial extent.

    Parameters:
        array: 2D numpy array (float32, uint8, or bool).
        output_path: Destination file path (.tif).
        bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84.
        nodata: Nodata sentinel value.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.ascontiguousarray(array)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)

    height, width = arr.shape
    min_lon, min_lat, max_lon, max_lat = bbox

    # Pixel resolution in degrees
    pixel_size_x = (max_lon - min_lon) / max(width, 1)
    pixel_size_y = (max_lat - min_lat) / max(height, 1)

    # GeoTIFF ModelTiepointTag: (i, j, k, x, y, z)
    # Maps pixel (0,0) to upper-left coordinate (min_lon, max_lat)
    tiepoints = (0.0, 0.0, 0.0, float(min_lon), float(max_lat), 0.0)
    # GeoTIFF ModelPixelScaleTag: (ScaleX, ScaleY, ScaleZ)
    pixel_scale = (float(pixel_size_x), float(pixel_size_y), 0.0)

    # Standard GeoTIFF GeoKeyDirectoryTag for EPSG:4326 (WGS84)
    # KeyDirectoryVersion=1, KeyRevision=1, MinorRevision=0, NumberOfKeys=3
    # GTModelTypeGeoKey=2 (Geographic 2D), GTRasterTypeGeoKey=1 (PixelIsArea), GeographicTypeGeoKey=4326 (WGS 84)
    geo_keys = (
        1, 1, 0, 3,
        1024, 0, 1, 2,     # GTModelTypeGeoKey: 2 = ModelTypeGeographic
        1025, 0, 1, 1,     # GTRasterTypeGeoKey: 1 = RasterPixelIsArea
        2048, 0, 1, 4326,  # GeographicTypeGeoKey: 4326 = GCS_WGS_1984
    )

    extratags = [
        (33550, "d", 3, pixel_scale, False),   # ModelPixelScaleTag
        (33922, "d", 6, tiepoints, False),     # ModelTiepointTag
        (34735, "H", len(geo_keys), geo_keys, False), # GeoKeyDirectoryTag (SHORT)
    ]

    tifffile.imwrite(
        str(path),
        arr,
        compression="deflate",
        extratags=extratags,
        photometric="minisblack",
    )
    logger.debug(f"Saved GeoTIFF/COG raster to: {path}")
    return path
