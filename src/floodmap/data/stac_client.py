"""Unified STAC client supporting CDSE (primary) and Planetary Computer / Earth Search (fallbacks)."""

from typing import Any

import requests

from floodmap.cache import default_cache
from floodmap.config import settings
from floodmap.logger import get_logger

logger = get_logger("floodmap.data.stac")


class MultiCatalogSTACClient:
    """Client for discovering Sentinel-1, Sentinel-2 and DEM assets across public STAC APIs."""

    def __init__(
        self,
        cdse_url: str | None = None,
        pc_url: str | None = None,
        earth_search_url: str | None = None,
    ):
        self.cdse_url = cdse_url or settings.cdse_stac_url
        self.pc_url = pc_url or settings.planetary_computer_stac_url
        self.earth_search_url = earth_search_url or settings.earth_search_stac_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "FloodMap-Hackathon2026/0.1.0"})

    def search_items(
        self,
        bbox: tuple[float, float, float, float],
        datetime_range: str,
        collections: list[str],
        catalog_preference: str = "cdse",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Searches STAC catalog for items matching bbox and datetime range.

        Checks cache first. Falls back automatically if primary catalog fails or returns empty.
        """
        cache_params = {
            "bbox": bbox,
            "datetime": datetime_range,
            "collections": sorted(collections),
            "catalog": catalog_preference,
            "limit": limit,
        }

        cached = default_cache.get_json("stac_search", cache_params)
        if cached is not None and "features" in cached:
            logger.info(f"Returning {len(cached['features'])} items from STAC cache.")
            return cached["features"]

        # Order catalogs by preference
        catalogs = []
        if catalog_preference == "cdse":
            catalogs = [("CDSE", self.cdse_url), ("PlanetaryComputer", self.pc_url), ("EarthSearch", self.earth_search_url)]
        elif catalog_preference == "pc":
            catalogs = [("PlanetaryComputer", self.pc_url), ("EarthSearch", self.earth_search_url), ("CDSE", self.cdse_url)]
        else:
            catalogs = [("EarthSearch", self.earth_search_url), ("PlanetaryComputer", self.pc_url), ("CDSE", self.cdse_url)]

        features: list[dict[str, Any]] = []
        for cat_name, cat_url in catalogs:
            try:
                logger.info(f"Searching STAC via {cat_name} at {cat_url} for {collections}...")
                search_payload = {
                    "bbox": list(bbox),
                    "datetime": datetime_range,
                    "collections": collections,
                    "limit": limit,
                }
                resp = self.session.post(
                    f"{cat_url}/search",
                    json=search_payload,
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    cat_features = data.get("features", [])
                    if cat_features:
                        logger.info(f"Found {len(cat_features)} items in {cat_name}.")
                        features = cat_features
                        break
                else:
                    logger.warning(f"{cat_name} STAC returned status code {resp.status_code}.")
            except Exception as exc:
                logger.warning(f"Failed STAC search on {cat_name}: {exc}. Trying fallback...")

        # Cache the result
        default_cache.put_json("stac_search", cache_params, {"features": features})
        return features
