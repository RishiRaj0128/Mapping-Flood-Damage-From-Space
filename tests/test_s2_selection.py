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


def test_scl_cloud_fraction_calculation():
    """Computes exact cloud percentage from SCL raster array."""
    import numpy as np

    from floodmap.data.s2_selection import compute_aoi_cloud_fraction, compute_scl_cloud_mask

    # 10x10 array: 50 clear land (4=vegetation), 25 cloud high prob (9), 25 cloud shadow (3)
    scl = np.full((10, 10), 4, dtype=np.uint8)
    scl[:5, :] = 9  # 50 cloud pixels
    scl[5:7, :5] = 3  # 10 cloud shadow pixels

    cloud_mask = compute_scl_cloud_mask(scl)
    assert np.count_nonzero(cloud_mask) == 60

    cloud_pct = compute_aoi_cloud_fraction(scl)
    assert cloud_pct == 60.0


def test_s2_aoi_cloud_overrides_scene_cloud(sample_s2_stac_items):
    """If AOI cloud is low even though scene cloud is high, scene is not compromised."""
    import numpy as np

    candidates = [S2SceneMetadata.from_stac_item(it) for it in sample_s2_stac_items]
    event_dt = datetime(2026, 8, 26, tzinfo=UTC)

    # Post scene has 68.4% scene cloud, but AOI is clear (only 5% cloud)
    clear_aoi_scl = np.full((20, 20), 4, dtype=np.uint8)
    clear_aoi_scl[:1, :] = 9  # 20/400 = 5%
    candidates[1].set_aoi_cloud_from_scl(clear_aoi_scl)

    pair = select_best_s2_pair(candidates, event_dt, max_cloud_threshold=40.0)
    assert pair is not None
    assert pair.post_scene.aoi_cloud_cover == 5.0
    # Because AOI cloud (5%) is well below 40%, it should NOT be cloud compromised!
    assert pair.is_cloud_compromised is False

