"""Copernicus GLO-30 DEM discovery and access module."""

from dataclasses import dataclass

from floodmap.cache import default_cache
from floodmap.data.stac_client import MultiCatalogSTACClient
from floodmap.logger import get_logger

logger = get_logger("floodmap.data.dem")

DEM_ATTRIBUTION = (
    "Produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and "
    "(c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS "
    "by the European Union and ESA; all rights reserved."
)


@dataclass
class DEMTileMetadata:
    """Metadata for a Copernicus GLO-30 elevation tile."""

    tile_id: str
    bbox: tuple[float, float, float, float]
    asset_url: str
    attribution: str = DEM_ATTRIBUTION


class CopernicusDEMClient:
    """Discovers and prepares Copernicus GLO-30 30-meter digital elevation model tiles."""

    def __init__(self, stac_client: MultiCatalogSTACClient | None = None):
        self.stac_client = stac_client or MultiCatalogSTACClient()

    def get_dem_tiles_for_bbox(
        self,
        bbox: tuple[float, float, float, float],
    ) -> list[DEMTileMetadata]:
        """Discovers GLO-30 DEM tiles covering the requested bounding box."""
        cache_key = {"bbox": bbox, "layer": "cop-dem-glo-30"}
        cached = default_cache.get_json("dem_tiles", cache_key)
        if cached and "tiles" in cached:
            logger.info(f"Loaded {len(cached['tiles'])} DEM tiles from cache.")
            return [DEMTileMetadata(**t) for t in cached["tiles"]]

        # Query Planetary Computer collection for Copernicus DEM
        items = self.stac_client.search_items(
            bbox=bbox,
            datetime_range="2010-01-01/2026-12-31",
            collections=["cop-dem-glo-30"],
            catalog_preference="pc",
            limit=20,
        )

        tiles: list[DEMTileMetadata] = []
        if items:
            for item in items:
                assets = item.get("assets", {})
                data_asset = assets.get("data", {})
                href = data_asset.get("href", "")
                tile_bbox = item.get("bbox", list(bbox))
                tiles.append(
                    DEMTileMetadata(
                        tile_id=item.get("id", "dem_tile"),
                        bbox=tuple(tile_bbox),  # type: ignore
                        asset_url=href,
                    )
                )
        else:
            # Fallback to direct AWS Copernicus DEM tile naming calculation
            # Copernicus GLO-30 tiles on AWS: e.g., Copernicus_DSM_COG_10_N27_00_E085_00_DEM.tif
            min_lon, min_lat, max_lon, max_lat = bbox
            lat_start = int(min_lat)
            lat_end = int(max_lat)
            lon_start = int(min_lon)
            lon_end = int(max_lon)

            for lat in range(lat_start, lat_end + 1):
                for lon in range(lon_start, lon_end + 1):
                    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
                    lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
                    tile_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
                    s3_url = f"https://copernicus-dem-30m.s3.amazonaws.com/{tile_name}/{tile_name}.tif"
                    tiles.append(
                        DEMTileMetadata(
                            tile_id=tile_name,
                            bbox=(float(lon), float(lat), float(lon + 1), float(lat + 1)),
                            asset_url=s3_url,
                        )
                    )

        # Cache the tiles
        tile_dicts = [t.__dict__ for t in tiles]
        default_cache.put_json("dem_tiles", cache_key, {"tiles": tile_dicts})
        logger.info(f"Discovered {len(tiles)} Copernicus GLO-30 DEM tiles.")
        return tiles
