"""Tests for strict competition compliance rules."""

import pytest

from floodmap.compliance.exceptions import (
    BannedDatasetError,
    ForbiddenTerminologyError,
    OSMDateViolationError,
)
from floodmap.compliance.guards import (
    validate_damage_terminology,
    validate_input_dataset,
    validate_osm_timestamp,
    verify_required_attributions,
)
from floodmap.config import settings


def test_osm_cutoff_valid_dates():
    """Dates on or prior to 2026-07-27 must be accepted."""
    d1 = validate_osm_timestamp("2026-07-27T00:00:00Z")
    assert d1.year == 2026 and d1.month == 7 and d1.day == 27

    d2 = validate_osm_timestamp("2026-01-01")
    assert d2.year == 2026 and d2.month == 1


def test_osm_cutoff_violation_raises_error():
    """Any date after 2026-07-27 must raise OSMDateViolationError."""
    with pytest.raises(OSMDateViolationError):
        validate_osm_timestamp("2026-07-28T00:00:00Z")

    with pytest.raises(OSMDateViolationError):
        validate_osm_timestamp("2026-08-26T12:00:00Z")


def test_allowed_input_datasets():
    """Allowed datasets must pass validation without raising errors."""
    for ds in ["sentinel-1", "sentinel-2", "copernicus-dem", "osm-pre-event", "kuro-siwo"]:
        validate_input_dataset(ds)


def test_banned_datasets_raise_error():
    """Banned datasets (WorldPop, GHSL, CEMS, etc.) must raise BannedDatasetError."""
    banned_list = [
        "WorldPop",
        "GHSL",
        "JRC Global Surface Water",
        "ESA WorldCover",
        "Dynamic World",
        "OPERA DSWx",
        "GloFAS",
        "ICIMOD Flood Layer",
        "CEMS EMSR927",
        "UNOSAT Damage Map",
    ]
    for b in banned_list:
        with pytest.raises(BannedDatasetError):
            validate_input_dataset(b)


def test_damage_terminology():
    """Terminology validator enforces conservative language ('exposed'/'likely hit')."""
    # Allowed
    validate_damage_terminology("12 buildings exposed to flood waters.")
    validate_damage_terminology("3 road segments likely hit by debris flow.")

    # Forbidden
    with pytest.raises(ForbiddenTerminologyError):
        validate_damage_terminology("15 homes destroyed by the flash flood.")

    with pytest.raises(ForbiddenTerminologyError):
        validate_damage_terminology("The bridge was demolished.")


def test_attribution_verification():
    """Attribution verification ensures required copyright and benchmark notices."""
    doc = "\n".join(settings.attributions)
    assert verify_required_attributions(doc) is True

    # Missing parts must fail
    assert verify_required_attributions("Just a regular report.") is False
