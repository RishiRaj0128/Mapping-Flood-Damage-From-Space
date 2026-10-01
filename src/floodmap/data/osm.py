"""Historical OpenStreetMap client with strict pre-event date enforcement."""

from datetime import datetime
from typing import Any

import requests

from floodmap.cache import default_cache
from floodmap.compliance.guards import validate_osm_timestamp
from floodmap.config import settings
from floodmap.logger import get_logger

logger = get_logger("floodmap.data.osm")

OSM_ATTRIBUTION = "(c) OpenStreetMap contributors."


def is_trishuli_bbox_match(
    bbox: tuple[float, float, float, float],
    tolerance: float = 0.15,
) -> bool:
    """Checks whether the requested bbox corresponds to the Trishuli case study AOI."""
    trishuli_bbox = settings.default_bbox
    return (
        abs(bbox[0] - trishuli_bbox[0]) <= tolerance
        and abs(bbox[1] - trishuli_bbox[1]) <= tolerance
        and abs(bbox[2] - trishuli_bbox[2]) <= tolerance
        and abs(bbox[3] - trishuli_bbox[3]) <= tolerance
    )


class OhsomeOSMClient:
    """Queries pre-event OpenStreetMap snapshots via the HeiGIT ohsome API.

    Guarantees strict compliance with the competition time cutoff (2026-07-27).
    """

    def __init__(self, api_url: str | None = None):
        self.api_url = api_url or settings.ohsome_api_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "FloodMap-Hackathon2026/0.1.0"})

    def query_elements_geometry(
        self,
        bbox: tuple[float, float, float, float],
        filter_expr: str,
        timestamp: str | datetime = "2026-07-27T00:00:00Z",
        properties: list[str] | None = None,
    ) -> dict[str, Any]:
        """Queries historical OSM geometries from ohsome with mandatory date validation.

        Parameters:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            filter_expr: ohsome filter expression, e.g. "building=* and geometry:polygon"
            timestamp: Historical timestamp (MUST be <= 2026-07-27)
            properties: Additional OSM tag keys to retain
        """
        # Hard compliance validation
        valid_dt = validate_osm_timestamp(timestamp)
        valid_time_str = valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        cache_params = {
            "bbox": bbox,
            "filter": filter_expr,
            "time": valid_time_str,
            "properties": properties or [],
        }

        cached = default_cache.get_json("osm_query", cache_params)
        if cached is not None:
            if "osm_source" not in cached:
                cached["osm_source"] = (
                    "cached_fixture:trishuli" if is_trishuli_bbox_match(bbox) else "cached_osm"
                )
            logger.info(f"Loaded OSM geometries for '{filter_expr}' from cache (source: {cached.get('osm_source')}).")
            return cached

        # ohsome /elements/geometry endpoint
        # bboxes parameter format: min_lon,min_lat,max_lon,max_lat
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        payload = {
            "bboxes": bbox_str,
            "time": valid_time_str,
            "filter": filter_expr,
        }
        if properties:
            payload["properties"] = ",".join(properties)

        url = f"{self.api_url}/elements/geometry"
        logger.info(
            f"Querying ohsome API at {valid_time_str} | Filter: {filter_expr} | BBox: {bbox_str}"
        )

        try:
            resp = self.session.post(url, data=payload, timeout=30)
            if resp.status_code == 200:
                geojson = resp.json()
                features_count = len(geojson.get("features", []))
                geojson["osm_source"] = "live_ohsome_api"
                logger.info(f"Retrieved {features_count} OSM elements from ohsome (source: live_ohsome_api).")
                default_cache.put_json("osm_query", cache_params, geojson)
                return geojson
            else:
                logger.warning(
                    f"ohsome API returned status {resp.status_code}: {resp.text[:200]}"
                )
        except Exception as exc:
            logger.warning(f"ohsome query failed: {exc}. Evaluating fallback options...")

        # Strict bbox gate: only serve Trishuli fixture if bbox matches Trishuli case study
        # Default Trishuli bbox: (85.15, 27.85, 85.45, 28.15)
        trishuli_bbox = settings.default_bbox
        is_trishuli_match = (
            abs(bbox[0] - trishuli_bbox[0]) <= 0.15
            and abs(bbox[1] - trishuli_bbox[1]) <= 0.15
            and abs(bbox[2] - trishuli_bbox[2]) <= 0.15
            and abs(bbox[3] - trishuli_bbox[3]) <= 0.15
        )

        sample_path = settings.output_dir / "samples" / "trishuli_pre_event_osm.json"
        if is_trishuli_match and sample_path.exists():
            try:
                import json
                with open(sample_path, encoding="utf-8") as f:
                    sample_fc = json.load(f)
                features = sample_fc.get("features", [])
                # Filter features matching requested layer
                if "building" in filter_expr:
                    filtered = [f for f in features if "building" in f.get("properties", {}) and f["properties"]["building"]]
                elif "bridge" in filter_expr:
                    filtered = [f for f in features if "bridge" in f.get("properties", {}) and f["properties"]["bridge"]]
                elif "highway" in filter_expr:
                    filtered = [f for f in features if "highway" in f.get("properties", {}) and f["properties"]["highway"]]
                elif "place" in filter_expr:
                    filtered = [f for f in features if "place" in f.get("properties", {}) and f["properties"]["place"]]
                elif "amenity" in filter_expr:
                    filtered = [f for f in features if "amenity" in f.get("properties", {}) and f["properties"]["amenity"]]
                else:
                    filtered = features

                logger.warning(
                    f"OSM source: cached fixture (Trishuli area match). "
                    f"Loaded {len(filtered)} features matching '{filter_expr}'."
                )
                return {
                    "type": "FeatureCollection",
                    "osm_source": "cached_fixture:trishuli",
                    "features": filtered,
                    "attribution": OSM_ATTRIBUTION,
                }
            except Exception as e:
                logger.warning(f"Failed to read sample fallback: {e}")
        elif not is_trishuli_match:
            logger.warning(
                f"OSM query failed for unseen bbox {bbox}. Strict compliance forbids "
                f"serving Trishuli fixture for an unmatching area. Returning empty OSM layer."
            )

        # Fallback empty FeatureCollection if API is unreachable and no valid fixture matches
        empty_fc: dict[str, Any] = {
            "type": "FeatureCollection",
            "osm_source": "empty_fallback",
            "features": [],
            "attribution": OSM_ATTRIBUTION,
        }
        return empty_fc

    def get_pre_event_buildings(
        self,
        bbox: tuple[float, float, float, float],
        timestamp: str | datetime = "2026-07-27T00:00:00Z",
    ) -> dict[str, Any]:
        """Fetches pre-event building footprints."""
        return self.query_elements_geometry(
            bbox=bbox,
            filter_expr="building=* and geometry:polygon",
            timestamp=timestamp,
            properties=["building", "name"],
        )

    def get_pre_event_roads(
        self,
        bbox: tuple[float, float, float, float],
        timestamp: str | datetime = "2026-07-27T00:00:00Z",
    ) -> dict[str, Any]:
        """Fetches pre-event road network."""
        return self.query_elements_geometry(
            bbox=bbox,
            filter_expr="highway in (motorway, trunk, primary, secondary, tertiary, unclassified, residential) and geometry:line",
            timestamp=timestamp,
            properties=["highway", "name", "bridge"],
        )

    def get_pre_event_bridges(
        self,
        bbox: tuple[float, float, float, float],
        timestamp: str | datetime = "2026-07-27T00:00:00Z",
    ) -> dict[str, Any]:
        """Fetches pre-event bridges."""
        return self.query_elements_geometry(
            bbox=bbox,
            filter_expr="bridge=yes and geometry:line",
            timestamp=timestamp,
            properties=["bridge", "highway", "name"],
        )

    def get_pre_event_settlements(
        self,
        bbox: tuple[float, float, float, float],
        timestamp: str | datetime = "2026-07-27T00:00:00Z",
    ) -> dict[str, Any]:
        """Fetches pre-event settlement points."""
        return self.query_elements_geometry(
            bbox=bbox,
            filter_expr="place in (city, town, village, hamlet) and geometry:point",
            timestamp=timestamp,
            properties=["place", "name", "population"],
        )

    def get_pre_event_hospitals(
        self,
        bbox: tuple[float, float, float, float],
        timestamp: str | datetime = "2026-07-27T00:00:00Z",
    ) -> dict[str, Any]:
        """Fetches pre-event hospitals and health centers."""
        return self.query_elements_geometry(
            bbox=bbox,
            filter_expr="amenity in (hospital, clinic) and (geometry:point or geometry:polygon)",
            timestamp=timestamp,
            properties=["amenity", "name"],
        )
