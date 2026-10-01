"""Data ingestion, STAC search, orbit selection, DEM, and OSM snapshots."""

from floodmap.data.dem import CopernicusDEMClient, DEMTileMetadata
from floodmap.data.io import DataLoader, DatasetBundle
from floodmap.data.osm import OhsomeOSMClient
from floodmap.data.s1_selection import (
    S1Pair,
    S1SceneMetadata,
    select_best_s1_pair,
    validate_s1_pair,
)
from floodmap.data.s2_selection import (
    S2Pair,
    S2SceneMetadata,
    select_best_s2_pair,
)
from floodmap.data.stac_client import MultiCatalogSTACClient
from floodmap.data.tiling import ProgressReporter, TileWindow, compute_tiles_for_bbox

__all__ = [
    "MultiCatalogSTACClient",
    "S1SceneMetadata",
    "S1Pair",
    "validate_s1_pair",
    "select_best_s1_pair",
    "S2SceneMetadata",
    "S2Pair",
    "select_best_s2_pair",
    "CopernicusDEMClient",
    "DEMTileMetadata",
    "OhsomeOSMClient",
    "TileWindow",
    "compute_tiles_for_bbox",
    "ProgressReporter",
    "DataLoader",
    "DatasetBundle",
]
