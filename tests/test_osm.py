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
