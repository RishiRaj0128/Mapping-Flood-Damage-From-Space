"""High-level Data Access and I/O orchestration."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from floodmap.compliance.guards import validate_input_dataset, validate_osm_timestamp
from floodmap.data.dem import CopernicusDEMClient, DEMTileMetadata
from floodmap.data.osm import OhsomeOSMClient
from floodmap.data.s1_selection import S1Pair, S1SceneMetadata, select_best_s1_pair
from floodmap.data.s2_selection import S2Pair, S2SceneMetadata, select_best_s2_pair
from floodmap.data.stac_client import MultiCatalogSTACClient
from floodmap.logger import get_logger

logger = get_logger("floodmap.data.io")


@dataclass
class DatasetBundle:
    """Complete bundle of raw satellite and geospatial layers for an AOI."""

    bbox: tuple[float, float, float, float]
    event_date: datetime
    s1_pair: S1Pair | None
    s2_pair: S2Pair | None
    dem_tiles: list[DEMTileMetadata]
    osm_buildings: dict[str, Any]
    osm_roads: dict[str, Any]
    osm_bridges: dict[str, Any]
    osm_settlements: dict[str, Any]
    osm_hospitals: dict[str, Any]
    is_cloud_compromised: bool
    osm_source: str = "unknown"


class DataLoader:
    """Orchestrates data discovery and ingestion for flood analysis."""

    def __init__(
        self,
        stac_client: MultiCatalogSTACClient | None = None,
        dem_client: CopernicusDEMClient | None = None,
        osm_client: OhsomeOSMClient | None = None,
    ):
        self.stac = stac_client or MultiCatalogSTACClient()
        self.dem = dem_client or CopernicusDEMClient(self.stac)
        self.osm = osm_client or OhsomeOSMClient()

    def load_dataset_bundle(
        self,
        bbox: tuple[float, float, float, float],
        event_date_str: str,
        search_window_days: int = 15,
        osm_snapshot_date: str = "2026-07-27T00:00:00Z",
    ) -> DatasetBundle:
        """Fetches all authorized input datasets for a given bounding box and event date.

        Guarantees that input datasets are strictly validated and OSM timestamp is compliant.
        """
        # Validate datasets and dates
        validate_input_dataset("sentinel-1")
        validate_input_dataset("sentinel-2")
        validate_input_dataset("copernicus-dem")
        validate_input_dataset("osm-pre-event")
        valid_osm_dt = validate_osm_timestamp(osm_snapshot_date)

        clean_date_str = event_date_str.replace("Z", "+00:00")
        if "T" not in clean_date_str:
            clean_date_str = f"{clean_date_str}T00:00:00+00:00"
        event_dt = datetime.fromisoformat(clean_date_str)

        start_dt = event_dt.fromtimestamp(
            event_dt.timestamp() - (search_window_days * 86400), tz=UTC
        )
        end_dt = event_dt.fromtimestamp(
            event_dt.timestamp() + (search_window_days * 86400), tz=UTC
        )
        dt_range = f"{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}"

        logger.info(f"Initiating dataset ingestion for bbox={bbox}, date_range={dt_range}")

        # 1. Search Sentinel-1
        s1_items = self.stac.search_items(
            bbox=bbox,
            datetime_range=dt_range,
            collections=["sentinel-1-grd", "sentinel-1-rtc", "SENTINEL-1"],
            catalog_preference="cdse",
        )
        s1_candidates = [S1SceneMetadata.from_stac_item(item) for item in s1_items]
        s1_pair = select_best_s1_pair(s1_candidates, event_dt, aoi_bbox=bbox)

        # 2. Search Sentinel-2
        s2_items = self.stac.search_items(
            bbox=bbox,
            datetime_range=dt_range,
            collections=["sentinel-2-l2a", "SENTINEL-2"],
            catalog_preference="cdse",
        )
        s2_candidates = [S2SceneMetadata.from_stac_item(item) for item in s2_items]
        s2_pair = select_best_s2_pair(s2_candidates, event_dt)

        # 3. Retrieve Copernicus DEM tiles
        dem_tiles = self.dem.get_dem_tiles_for_bbox(bbox)

        # 4. Ingest Historical OSM Layers (Hard-guarded <= 2026-07-27)
        osm_buildings = self.osm.get_pre_event_buildings(bbox, timestamp=valid_osm_dt)
        osm_roads = self.osm.get_pre_event_roads(bbox, timestamp=valid_osm_dt)
        osm_bridges = self.osm.get_pre_event_bridges(bbox, timestamp=valid_osm_dt)
        osm_settlements = self.osm.get_pre_event_settlements(bbox, timestamp=valid_osm_dt)
        osm_hospitals = self.osm.get_pre_event_hospitals(bbox, timestamp=valid_osm_dt)

        is_cloudy = s2_pair.is_cloud_compromised if s2_pair else True
        detected_osm_source = osm_buildings.get("osm_source") or osm_roads.get("osm_source") or "unknown"

        return DatasetBundle(
            bbox=bbox,
            event_date=event_dt,
            s1_pair=s1_pair,
            s2_pair=s2_pair,
            dem_tiles=dem_tiles,
            osm_buildings=osm_buildings,
            osm_roads=osm_roads,
            osm_bridges=osm_bridges,
            osm_settlements=osm_settlements,
            osm_hospitals=osm_hospitals,
            is_cloud_compromised=is_cloudy,
            osm_source=detected_osm_source,
        )
