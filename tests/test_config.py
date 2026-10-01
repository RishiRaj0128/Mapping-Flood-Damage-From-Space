"""Tests for configuration settings and legal requirements."""

from floodmap.config import settings


def test_config_defaults():
    """Validates default settings and bounding box."""
    assert settings.project_name == "Mapping Flood Damage from Space"
    assert len(settings.default_bbox) == 4
    assert settings.default_event_date == "2026-08-26"
    assert settings.max_osm_date.year == 2026
    assert settings.max_osm_date.month == 7
    assert settings.max_osm_date.day == 27


def test_attributions_included():
    """Ensures all mandatory legal attribution lines are configured."""
    joined = " ".join(settings.attributions).lower()
    assert "copernicus sentinel" in joined
    assert "copernicus worlddem" in joined
    assert "openstreetmap" in joined
    assert "kuro siwo" in joined
