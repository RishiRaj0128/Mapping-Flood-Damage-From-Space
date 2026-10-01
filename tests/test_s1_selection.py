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
    assert pair.pre_scene.item_id == "S1A_IW_GRDH_1SDV_20260815_orbit85_asc"
    assert pair.post_scene.item_id == "S1A_IW_GRDH_1SDV_20260827_orbit85_asc"
    assert pair.delta_days == 12.0
