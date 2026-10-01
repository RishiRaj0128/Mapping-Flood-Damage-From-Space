"""Historical OpenStreetMap client with strict pre-event date enforcement and multi-source fallback.

Provides:
1. HeiGIT ohsome API (sub-tiling, custom User-Agent, exponential backoff retries).
2. Overpass API attic fallback ([date:"..."]) rotated across high-availability mirrors.
3. Pre-event cached fixtures strictly gated by AOI (Trishuli & Chamoli).
4. Mandatory pre-event cutoff enforcement: snapshot_dt = min(event_date - 1 day, 2026-07-27).
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any

import requests

from floodmap.cache import default_cache
from floodmap.compliance.guards import validate_osm_timestamp
from floodmap.config import settings
from floodmap.logger import get_logger

logger = get_logger("floodmap.data.osm")

OSM_ATTRIBUTION = "(c) OpenStreetMap contributors."

DEFAULT_USER_AGENT = (
    "FloodDamageSpaceHackathon2026/1.0 "
    "(https://github.com/RishiRaj0128/Mapping-Flood-Damage-From-Space; contact@floodmap.ai)"
)

OVERPASS_MIRRORS = [
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def is_trishuli_bbox_match(
    bbox: tuple[float, float, float, float],
    tolerance: float = 0.20,
) -> bool:
    """Checks whether requested bbox corresponds to Trishuli AOI."""
    trishuli_bbox = settings.default_bbox
    return (
        abs(bbox[0] - trishuli_bbox[0]) <= tolerance
        and abs(bbox[1] - trishuli_bbox[1]) <= tolerance
        and abs(bbox[2] - trishuli_bbox[2]) <= tolerance
        and abs(bbox[3] - trishuli_bbox[3]) <= tolerance
    )


def is_chamoli_bbox_match(
    bbox: tuple[float, float, float, float],
    tolerance: float = 0.20,
) -> bool:
    """Checks whether requested bbox corresponds to Chamoli AOI."""
    chamoli_bbox = (79.55, 30.35, 79.85, 30.65)
    return (
        abs(bbox[0] - chamoli_bbox[0]) <= tolerance
        and abs(bbox[1] - chamoli_bbox[1]) <= tolerance
        and abs(bbox[2] - chamoli_bbox[2]) <= tolerance
        and abs(bbox[3] - chamoli_bbox[3]) <= tolerance
    )


def get_pre_event_snapshot_datetime(
    event_date: str | datetime,
    cutoff_date: str = "2026-07-27T00:00:00Z",
) -> datetime:
    """Strictly enforces pre-event snapshot rule: min(event_date - 1 day, 2026-07-27)."""
    if isinstance(event_date, str):
        clean_str = event_date.replace("Z", "+00:00")
        if "T" not in clean_str:
            clean_str = f"{clean_str}T00:00:00+00:00"
        evt_dt = datetime.fromisoformat(clean_str)
    else:
        evt_dt = event_date

    one_day_before = evt_dt - timedelta(days=1)
    cutoff_dt = datetime.fromisoformat(cutoff_date.replace("Z", "+00:00"))
    snapshot_dt = min(one_day_before, cutoff_dt)
    return validate_osm_timestamp(snapshot_dt)


def parse_overpass_to_geojson(elements: list[dict], filter_expr: str) -> list[dict]:
    """Converts Overpass elements (nodes and ways with geometries) to GeoJSON features."""
    features = []
    is_building = "building" in filter_expr
    for elem in elements:
        tags = elem.get("tags", {})
        elem_id = elem.get("id")
        elem_type = elem.get("type")

        if elem_type == "node" and "lon" in elem and "lat" in elem:
            geom = {"type": "Point", "coordinates": [float(elem["lon"]), float(elem["lat"])]}
        elif elem_type == "way" and "geometry" in elem:
            pts = [[float(p["lon"]), float(p["lat"])] for p in elem["geometry"]]
            if len(pts) < 2:
                continue
            if is_building:
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                if len(pts) >= 4:
                    geom = {"type": "Polygon", "coordinates": [pts]}
                else:
                    continue
            else:
                geom = {"type": "LineString", "coordinates": pts}
        else:
            continue

        features.append({
            "type": "Feature",
            "id": f"{elem_type}/{elem_id}",
            "properties": tags,
            "geometry": geom,
        })
    return features


class OhsomeOSMClient:
    """Multi-source Historical OpenStreetMap client.

    Guarantees strict pre-event snapshot enforcement using ohsome with retries,
    Overpass attic rotation, and bbox-gated offline fixtures.
    """

    def __init__(self, api_url: str | None = None):
        self.api_url = api_url or settings.ohsome_api_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    def query_elements_geometry(
        self,
        bbox: tuple[float, float, float, float],
        filter_expr: str,
        timestamp: str | datetime = "2026-07-27T00:00:00Z",
        properties: list[str] | None = None,
    ) -> dict[str, Any]:
        """Queries historical OSM geometries from ohsome/Overpass with mandatory date validation.

        Parameters:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            filter_expr: Layer filter expression
            timestamp: Historical timestamp (MUST be <= 2026-07-27)
            properties: Additional OSM tag keys to retain
        """
        valid_dt = validate_osm_timestamp(timestamp)
        valid_time_str = valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        cache_params = {
            "bbox": bbox,
            "filter": filter_expr,
            "time": valid_time_str,
            "properties": properties or [],
        }

        # 1. Local Disk Cache Check
        cached = default_cache.get_json("osm_query", cache_params)
        if cached is not None and cached.get("features"):
            if "osm_source" not in cached:
                cached["osm_source"] = (
                    "cached_fixture:trishuli" if is_trishuli_bbox_match(bbox) else "cached_osm"
                )
            logger.info(
                f"Loaded {len(cached['features'])} OSM geometries for '{filter_expr}' "
                f"from cache (source: {cached.get('osm_source')})."
            )
            return cached

        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        logger.info(
            f"Querying historical OSM at {valid_time_str} | Filter: {filter_expr} | BBox: {bbox_str}"
        )

        # 2. Primary: HeiGIT ohsome API (with retries and exponential backoff)
        url = f"{self.api_url}/elements/geometry"
        payload = {
            "bboxes": bbox_str,
            "time": valid_time_str,
            "filter": filter_expr,
        }
        if properties:
            payload["properties"] = ",".join(properties)

        for attempt in range(2):
            try:
                resp = self.session.post(url, data=payload, timeout=25)
                if resp.status_code == 200:
                    geojson = resp.json()
                    feats = geojson.get("features", [])
                    if feats:
                        geojson["osm_source"] = "live_ohsome_api"
                        logger.info(f"Retrieved {len(feats)} OSM elements from live ohsome API.")
                        default_cache.put_json("osm_query", cache_params, geojson)
                        return geojson
                else:
                    logger.debug(f"ohsome attempt {attempt + 1} status {resp.status_code}")
            except Exception as exc:
                logger.debug(f"ohsome attempt {attempt + 1} failed: {exc}")
            time.sleep(1.0 * (2**attempt))

        # 3. Secondary: Overpass API Attic Query
        logger.info("Ohsome unavailable/rate-limited; attempting Overpass attic fallback query...")
        overpass_filter = ""
        if "building" in filter_expr:
            overpass_filter = (
                f'way["building"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});'
            )
        elif "highway" in filter_expr:
            overpass_filter = (
                f'way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential"]'
                f"({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});"
            )
        elif "bridge" in filter_expr:
            overpass_filter = f'way["bridge"="yes"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});'
        elif "place" in filter_expr:
            overpass_filter = (
                f'node["place"~"city|town|village|hamlet"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});'
            )
        elif "amenity" in filter_expr:
            overpass_filter = (
                f'node["amenity"~"hospital|clinic"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});'
                f'way["amenity"~"hospital|clinic"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});'
            )

        if overpass_filter:
            q_overpass = f"""
            [out:json][timeout:35][date:"{valid_time_str}"];
            (
              {overpass_filter}
            );
            out geom 10000;
            """
            for mirror in OVERPASS_MIRRORS:
                try:
                    resp = self.session.post(mirror, data={"data": q_overpass}, timeout=35)
                    if resp.status_code == 200:
                        elements = resp.json().get("elements", [])
                        parsed_feats = parse_overpass_to_geojson(elements, filter_expr)
                        if parsed_feats:
                            fc = {
                                "type": "FeatureCollection",
                                "osm_source": f"live_overpass_attic:{valid_time_str[:10]}",
                                "features": parsed_feats,
                                "attribution": OSM_ATTRIBUTION,
                            }
                            logger.info(
                                f"Retrieved {len(parsed_feats)} OSM elements from Overpass attic ({mirror})."
                            )
                            default_cache.put_json("osm_query", cache_params, fc)
                            return fc
                except Exception as e:
                    logger.debug(f"Overpass mirror {mirror} failed: {e}")
                time.sleep(1.0)

        # 4. Strict BBox-Gated Offline Fixtures
        # Trishuli fixture check
        if is_trishuli_bbox_match(bbox):
            fixture_file = settings.output_dir / "samples" / "trishuli_pre_event_osm.json"
            if fixture_file.exists():
                try:
                    with open(fixture_file, encoding="utf-8") as f:
                        fc = json.load(f)
                    all_feats = fc.get("features", [])
                    filtered = self._filter_features(all_feats, filter_expr)
                    logger.warning(
                        f"OSM source: cached fixture (Trishuli area match). "
                        f"Loaded {len(filtered)} features matching '{filter_expr}'."
                    )
                    out_fc = {
                        "type": "FeatureCollection",
                        "osm_source": "cached_fixture:trishuli",
                        "features": filtered,
                        "attribution": OSM_ATTRIBUTION,
                    }
                    default_cache.put_json("osm_query", cache_params, out_fc)
                    return out_fc
                except Exception as e:
                    logger.warning(f"Failed to load Trishuli fixture: {e}")

        # Chamoli fixture check
        if is_chamoli_bbox_match(bbox):
            fixture_file = settings.output_dir / "samples" / "chamoli_pre_event_osm.json"
            if fixture_file.exists():
                try:
                    with open(fixture_file, encoding="utf-8") as f:
                        fc = json.load(f)
                    all_feats = fc.get("features", [])
                    filtered = self._filter_features(all_feats, filter_expr)
                    logger.warning(
                        f"OSM source: cached fixture (Chamoli area match). "
                        f"Loaded {len(filtered)} features matching '{filter_expr}'."
                    )
                    out_fc = {
                        "type": "FeatureCollection",
                        "osm_source": "cached_fixture:chamoli",
                        "features": filtered,
                        "attribution": OSM_ATTRIBUTION,
                    }
                    default_cache.put_json("osm_query", cache_params, out_fc)
                    return out_fc
                except Exception as e:
                    logger.warning(f"Failed to load Chamoli fixture: {e}")

        # Fallback empty FeatureCollection if unseen area and remote APIs unreachable
        logger.warning(
            f"OSM query failed for unseen bbox {bbox}. Strict compliance forbids "
            f"serving mismatched fixtures. Returning empty OSM layer."
        )
        empty_fc: dict[str, Any] = {
            "type": "FeatureCollection",
            "osm_source": "empty_fallback",
            "features": [],
            "attribution": OSM_ATTRIBUTION,
        }
        return empty_fc

    def _filter_features(self, features: list[dict], filter_expr: str) -> list[dict]:
        """Filters features from cached multi-layer collections."""
        if "building" in filter_expr:
            return [f for f in features if "building" in f.get("properties", {}) and f["properties"]["building"]]
        elif "bridge" in filter_expr:
            return [f for f in features if "bridge" in f.get("properties", {}) and f["properties"]["bridge"]]
        elif "highway" in filter_expr:
            return [f for f in features if "highway" in f.get("properties", {}) and f["properties"]["highway"]]
        elif "place" in filter_expr:
            return [f for f in features if "place" in f.get("properties", {}) and f["properties"]["place"]]
        elif "amenity" in filter_expr:
            return [f for f in features if "amenity" in f.get("properties", {}) and f["properties"]["amenity"]]
        return features

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
