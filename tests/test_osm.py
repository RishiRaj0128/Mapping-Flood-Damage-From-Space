"""Tests for OpenStreetMap pre-event data retrieval and time enforcement."""

import pytest

from floodmap.compliance.exceptions import OSMDateViolationError
from floodmap.data.osm import OhsomeOSMClient


def test_osm_client_blocks_post_event_queries():
    """Client must reject queries with dates after 2026-07-27."""
    client = OhsomeOSMClient()
    bbox = (85.15, 27.85, 85.45, 28.15)

    with pytest.raises(OSMDateViolationError):
        client.get_pre_event_buildings(bbox, timestamp="2026-08-26T00:00:00Z")

    with pytest.raises(OSMDateViolationError):
        client.get_pre_event_roads(bbox, timestamp="2026-07-28T00:00:00Z")


def test_osm_client_allows_pre_event_queries(monkeypatch):
    """Client allows compliant pre-event queries."""
    client = OhsomeOSMClient()
    bbox = (85.15, 27.85, 85.45, 28.15)

    # Mock the session.post call
    class MockResponse:
        status_code = 200

        def json(self):
            return {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"building": "residential"}, "geometry": None}
                ],
            }

    monkeypatch.setattr(client.session, "post", lambda *args, **kwargs: MockResponse())

    res = client.get_pre_event_buildings(bbox, timestamp="2026-07-27T00:00:00Z")
    assert res["type"] == "FeatureCollection"
    assert len(res["features"]) == 1


def test_osm_fallback_is_bbox_gated(monkeypatch):
    """Fallback must only serve Trishuli fixture when bbox matches, and never for unseen areas."""
    client = OhsomeOSMClient()

    # Simulate network failure / 403 on ohsome API
    class FailingResponse:
        status_code = 403
        text = "Forbidden"

    monkeypatch.setattr(client.session, "post", lambda *args, **kwargs: FailingResponse())

    # 1. Trishuli BBox -> Should return Trishuli fixture with source flag
    trishuli_bbox = (85.15, 27.85, 85.45, 28.15)
    res_trishuli = client.get_pre_event_buildings(trishuli_bbox, timestamp="2026-07-27T00:00:00Z")
    assert res_trishuli["osm_source"] == "cached_fixture:trishuli"
    assert len(res_trishuli["features"]) > 0

    # 2. Chamoli BBox -> Must NEVER serve Trishuli fixture; returns empty_fallback
    chamoli_bbox = (79.40, 30.20, 79.80, 30.60)
    res_chamoli = client.get_pre_event_buildings(chamoli_bbox, timestamp="2026-07-27T00:00:00Z")
    assert res_chamoli["osm_source"] == "empty_fallback"
    assert len(res_chamoli["features"]) == 0

