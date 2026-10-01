"""Tests for Sentinel-1 pair selection and same-orbit validation."""

from datetime import UTC, datetime

import pytest

from floodmap.compliance.exceptions import CrossTrackPairError
from floodmap.data.s1_selection import (
    S1SceneMetadata,
    select_best_s1_pair,
    validate_s1_pair,
)


def test_s1_metadata_parsing(sample_s1_stac_items):
    """Verifies parsing of relative orbit, flight direction, and polarizations."""
    item = sample_s1_stac_items[0]
    meta = S1SceneMetadata.from_stac_item(item)

    assert meta.item_id == "S1A_IW_GRDH_1SDV_20260815_orbit85_asc"
    assert meta.relative_orbit == 85
    assert meta.direction == "ascending"
    assert "VV" in meta.polarizations


def test_s1_pair_validation_same_orbit():
    """Matching relative orbit and flight direction must validate cleanly."""
    pre = S1SceneMetadata(
        item_id="pre_orbit85",
        datetime=datetime(2026, 8, 15, tzinfo=UTC),
        relative_orbit=85,
        direction="ascending",
        polarizations=["VV", "VH"],
        assets={},
        raw_properties={},
    )
    post = S1SceneMetadata(
        item_id="post_orbit85",
        datetime=datetime(2026, 8, 27, tzinfo=UTC),
        relative_orbit=85,
        direction="ascending",
        polarizations=["VV", "VH"],
        assets={},
        raw_properties={},
    )

    validate_s1_pair(pre, post)


def test_s1_cross_track_orbit_mismatch_fails():
    """Different relative orbits must raise CrossTrackPairError."""
    pre = S1SceneMetadata(
        item_id="pre_orbit85",
        datetime=datetime(2026, 8, 15, tzinfo=UTC),
        relative_orbit=85,
        direction="ascending",
        polarizations=["VV", "VH"],
        assets={},
        raw_properties={},
    )
    post = S1SceneMetadata(
        item_id="post_orbit12",
        datetime=datetime(2026, 8, 28, tzinfo=UTC),
        relative_orbit=12,
        direction="ascending",
        polarizations=["VV", "VH"],
        assets={},
        raw_properties={},
    )

    with pytest.raises(CrossTrackPairError):
        validate_s1_pair(pre, post)


def test_s1_flight_direction_mismatch_fails():
    """Different flight directions (ascending vs descending) must raise CrossTrackPairError."""
    pre = S1SceneMetadata(
        item_id="pre_asc",
        datetime=datetime(2026, 8, 15, tzinfo=UTC),
        relative_orbit=85,
        direction="ascending",
        polarizations=["VV", "VH"],
        assets={},
        raw_properties={},
    )
    post = S1SceneMetadata(
        item_id="post_desc",
        datetime=datetime(2026, 8, 27, tzinfo=UTC),
        relative_orbit=85,
        direction="descending",
        polarizations=["VV", "VH"],
        assets={},
        raw_properties={},
    )

    with pytest.raises(CrossTrackPairError):
        validate_s1_pair(pre, post)


def test_select_best_s1_pair(sample_s1_stac_items):
    """select_best_s1_pair must identify the valid same-orbit pair."""
    candidates = [S1SceneMetadata.from_stac_item(it) for it in sample_s1_stac_items]
    event_dt = datetime(2026, 8, 26, tzinfo=UTC)

    pair = select_best_s1_pair(candidates, event_dt, preferred_repeat_days=12)

    assert pair is not None
    assert pair.relative_orbit == 85
    assert pair.direction == "ascending"
    assert pair.delta_days == 12.0


def test_select_best_s1_pair_strict_post_timing():
    """Scenes acquired prior to or on the event date cannot be used as post-event scenes."""
    event_dt = datetime(2026, 8, 26, 12, 0, 0, tzinfo=UTC)
    pre = S1SceneMetadata(
        item_id="pre_orbit85",
        datetime=datetime(2026, 8, 14, tzinfo=UTC),
        relative_orbit=85,
        direction="ascending",
        polarizations=["VV"],
        assets={},
        raw_properties={},
    )
    # Acquired morning of Aug 26, BEFORE the event at 12:00
    same_day_pre = S1SceneMetadata(
        item_id="same_day_morning",
        datetime=datetime(2026, 8, 26, 6, 0, 0, tzinfo=UTC),
        relative_orbit=85,
        direction="ascending",
        polarizations=["VV"],
        assets={},
        raw_properties={},
    )
    # No scene strictly after event_dt
    pair = select_best_s1_pair([pre, same_day_pre], event_dt)
    assert pair is None


def test_select_best_s1_pair_aoi_spatial_filtering():
    """Candidates outside or insufficiently covering the AOI must be filtered out."""
    event_dt = datetime(2026, 8, 26, tzinfo=UTC)
    aoi_bbox = (85.15, 27.85, 85.45, 28.15)

    # Disjoint scene in western Nepal (81.0, 29.0)
    disjoint_pre = S1SceneMetadata(
        item_id="disjoint_pre",
        datetime=datetime(2026, 8, 14, tzinfo=UTC),
        relative_orbit=85,
        direction="ascending",
        polarizations=["VV"],
        assets={},
        raw_properties={},
        bbox=(80.5, 28.5, 81.5, 29.5),
    )
    covering_post = S1SceneMetadata(
        item_id="covering_post",
        datetime=datetime(2026, 8, 27, tzinfo=UTC),
        relative_orbit=85,
        direction="ascending",
        polarizations=["VV"],
        assets={},
        raw_properties={},
        bbox=(85.0, 27.5, 86.0, 28.5),
    )

    pair = select_best_s1_pair([disjoint_pre, covering_post], event_dt, aoi_bbox=aoi_bbox)
    assert pair is None

