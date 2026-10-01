"""Strict compliance guards for input datasets, OSM cutoff dates, and terminology."""

import re
from datetime import UTC, datetime

from floodmap.compliance.exceptions import (
    BannedDatasetError,
    ForbiddenTerminologyError,
    OSMDateViolationError,
)

# Strict Cutoff: 2026-07-27 23:59:59 UTC
OSM_HARD_CUTOFF = datetime(2026, 7, 27, 23, 59, 59, tzinfo=UTC)

# Allowed input datasets
ALLOWED_INPUT_DATASETS: set[str] = {
    "sentinel-1",
    "sentinel-2",
    "copernicus-dem",
    "copernicus-glo-30",
    "osm-pre-event",
    "kuro-siwo",
    "sen1floods11",
}

# Forbidden input datasets
BANNED_DATASETS: set[str] = {
    "worldpop",
    "ghsl",
    "jrc",
    "global_surface_water",
    "gsw",
    "worldcover",
    "esa_worldcover",
    "dynamic_world",
    "opera_dswx",
    "dswx",
    "glofas",
    "icimod",
    "cems",
    "emsr",
    "emsr927",
    "unosat",
}

# Forbidden absolute damage terminology
FORBIDDEN_WORDS = ["destroyed", "demolished", "flattened", "obliterated"]
ALLOWED_DAMAGE_TIERS = ["exposed", "likely hit", "unaffected", "uncertain"]


def validate_osm_timestamp(timestamp: str | datetime) -> datetime:
    """Validates that OpenStreetMap query timestamps are strictly on or before 2026-07-27.

    Raises OSMDateViolationError if the date is after the cutoff.
    """
    if isinstance(timestamp, str):
        # Support ISO 8601 strings e.g. '2026-07-27' or '2026-07-27T00:00:00Z'
        clean_ts = timestamp.replace("Z", "+00:00")
        if "T" not in clean_ts:
            clean_ts = f"{clean_ts}T23:59:59+00:00"
        dt = datetime.fromisoformat(clean_ts)
    else:
        dt = timestamp

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    if dt > OSM_HARD_CUTOFF:
        raise OSMDateViolationError(
            f"OSM timestamp {dt.isoformat()} violates strict competition cutoff "
            f"({OSM_HARD_CUTOFF.isoformat()}). Post-event OSM edits are strictly prohibited!"
        )
    return dt


def validate_input_dataset(dataset_name: str) -> None:
    """Validates that a dataset requested as pipeline input is officially authorized.

    Raises BannedDatasetError if an unauthorized or specifically banned dataset is passed.
    """
    normalized = dataset_name.strip().lower().replace(" ", "_").replace("-", "_")

    for banned in BANNED_DATASETS:
        if banned in normalized:
            raise BannedDatasetError(
                f"Dataset '{dataset_name}' is strictly prohibited as an input! "
                f"Matched banned pattern '{banned}'. CEMS/UNOSAT and external layers "
                f"may only be used in isolated evaluation."
            )

    is_allowed = any(
        allowed.replace("-", "_") in normalized for allowed in ALLOWED_INPUT_DATASETS
    )
    if not is_allowed:
        raise BannedDatasetError(
            f"Dataset '{dataset_name}' is not in the whitelist of allowed inputs: "
            f"{sorted(list(ALLOWED_INPUT_DATASETS))}"
        )


def validate_damage_terminology(text: str) -> None:
    """Validates that text output adheres to standard conservative terminology.

    Ensures terms like 'destroyed' are not used in place of 'exposed' or 'likely hit'.
    """
    lower = text.lower()
    for forbidden in FORBIDDEN_WORDS:
        pattern = rf"\b{re.escape(forbidden)}\b"
        if re.search(pattern, lower):
            raise ForbiddenTerminologyError(
                f"Found prohibited terminology '{forbidden}' in text: '{text}'. "
                f"Use 'exposed' or 'likely hit' with a confidence tier."
            )


def verify_required_attributions(document_text: str) -> bool:
    """Checks whether the mandatory attribution lines are present in the text."""
    lower_doc = document_text.lower()
    required_keywords = [
        "copernicus sentinel",
        "copernicus worlddem",
        "openstreetmap",
    ]
    return all(kw in lower_doc for kw in required_keywords)
