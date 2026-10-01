"""Central configuration and parameters for the flood mapping pipeline."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Pipeline runtime settings and constants."""

    # Project metadata
    project_name: str = "Mapping Flood Damage from Space"
    version: str = "0.1.0"

    # Strict compliance cutoff for pre-event OSM
    max_osm_date: datetime = Field(
        default=datetime(2026, 7, 27, 23, 59, 59, tzinfo=UTC),
        description="Strict pre-event cutoff timestamp for OpenStreetMap queries.",
    )

    # Cache and output paths
    cache_dir: Path = Path(".cache/floodmap")
    output_dir: Path = Path("outputs")

    # STAC Endpoints
    cdse_stac_url: str = "https://catalogue.dataspace.copernicus.eu/stac"
    planetary_computer_stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    earth_search_stac_url: str = "https://earth-search.aws.element84.com/v1"

    # Credentials (optional for public/search; needed for full CDSE downloads)
    cdse_client_id: str | None = None
    cdse_client_secret: str | None = None
    planetary_computer_key: str | None = None

    # Ohsome API for historical OSM
    ohsome_api_url: str = "https://api.ohsome.org/v1"

    # Default Case Study: Trishuli - Bhote Koshi August 2026
    # Bounding Box: [min_lon, min_lat, max_lon, max_lat]
    default_bbox: tuple[float, float, float, float] = (85.15, 27.85, 85.45, 28.15)
    default_event_date: str = "2026-08-26"

    # Legal Attributions
    attributions: list[str] = [
        "Contains modified Copernicus Sentinel data 2026.",
        (
            "Produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and "
            "(c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS "
            "by the European Union and ESA; all rights reserved."
        ),
        "(c) OpenStreetMap contributors.",
        "Kuro Siwo training benchmark: Bountos et al., NeurIPS 2024.",
        "Sen1Floods11 benchmark: Bonafilia et al., CVPRW 2020.",
    ]


settings = Settings()
