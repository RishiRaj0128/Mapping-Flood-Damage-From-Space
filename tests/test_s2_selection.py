"""Tests for Sentinel-2 cloud-aware pair selection."""

from datetime import UTC, datetime

from floodmap.data.s2_selection import (
    S2SceneMetadata,
    select_best_s2_pair,
)


def test_s2_metadata_parsing(sample_s2_stac_items):
    """Verifies parsing of cloud cover and SCL asset detection."""
    item = sample_s2_stac_items[0]
    meta = S2SceneMetadata.from_stac_item(item)

    assert meta.item_id == "S2A_MSIL2A_20260810_clear"
    assert meta.cloud_cover == 5.2
    assert meta.has_scl is True


def test_select_best_s2_pair_flags_cloudy(sample_s2_stac_items):
    """Pair selection flags cloud-compromised pairs when cloud threshold exceeded."""
    candidates = [S2SceneMetadata.from_stac_item(it) for it in sample_s2_stac_items]
    event_dt = datetime(2026, 8, 26, tzinfo=UTC)

    pair = select_best_s2_pair(candidates, event_dt, max_cloud_threshold=40.0)

    assert pair is not None
    assert pair.pre_scene.cloud_cover == 5.2
    assert pair.post_scene.cloud_cover == 68.4
    # 68.4% is > 40%, so it must be flagged for graceful degradation
    assert pair.is_cloud_compromised is True
