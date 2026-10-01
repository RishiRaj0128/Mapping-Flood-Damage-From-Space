"""Shared pytest fixtures and test data."""


import pytest


@pytest.fixture
def sample_s1_stac_items():
    """Provides a list of synthetic Sentinel-1 STAC items with known orbits and dates."""
    return [
        {
            "id": "S1A_IW_GRDH_1SDV_20260815_orbit85_asc",
            "properties": {
                "datetime": "2026-08-15T00:30:00Z",
                "sat:relative_orbit": 85,
                "sat:orbit_state": "ascending",
                "sar:polarizations": ["VV", "VH"],
            },
            "assets": {"vv": {"href": "https://example.com/s1_20260815_vv.tif"}},
        },
        {
            "id": "S1A_IW_GRDH_1SDV_20260827_orbit85_asc",
            "properties": {
                "datetime": "2026-08-27T00:30:00Z",
                "sat:relative_orbit": 85,
                "sat:orbit_state": "ascending",
                "sar:polarizations": ["VV", "VH"],
            },
            "assets": {"vv": {"href": "https://example.com/s1_20260827_vv.tif"}},
        },
        {
            "id": "S1A_IW_GRDH_1SDV_20260828_orbit12_desc",
            "properties": {
                "datetime": "2026-08-28T12:00:00Z",
                "sat:relative_orbit": 12,
                "sat:orbit_state": "descending",
                "sar:polarizations": ["VV", "VH"],
            },
            "assets": {"vv": {"href": "https://example.com/s1_20260828_vv.tif"}},
        },
    ]


@pytest.fixture
def sample_s2_stac_items():
    """Provides a list of synthetic Sentinel-2 STAC items with varying cloud cover."""
    return [
        {
            "id": "S2A_MSIL2A_20260810_clear",
            "properties": {
                "datetime": "2026-08-10T04:45:00Z",
                "eo:cloud_cover": 5.2,
            },
            "assets": {
                "visual": {"href": "https://example.com/s2_20260810_tci.tif"},
                "scl": {"href": "https://example.com/s2_20260810_scl.tif"},
            },
        },
        {
            "id": "S2A_MSIL2A_20260828_cloudy",
            "properties": {
                "datetime": "2026-08-28T04:45:00Z",
                "eo:cloud_cover": 68.4,
            },
            "assets": {
                "visual": {"href": "https://example.com/s2_20260828_tci.tif"},
                "scl": {"href": "https://example.com/s2_20260828_scl.tif"},
            },
        },
    ]
