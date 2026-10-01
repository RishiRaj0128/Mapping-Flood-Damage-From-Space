"""Tests for DEM slope, layover/shadow, and HAND terrain exclusion."""

import numpy as np

from floodmap.terrain.masks import (
    apply_terrain_exclusion,
    compute_hand,
    compute_layover_shadow_mask,
    compute_slope_degrees,
)


def test_compute_slope_degrees():
    """Flat terrain must have 0 deg slope; inclined ramp has predictable slope."""
    flat_dem = np.full((20, 20), 500.0, dtype=np.float32)
    flat_slope = compute_slope_degrees(flat_dem, cell_size_m=30.0)
    np.testing.assert_allclose(flat_slope, 0.0, atol=1e-3)

    # 45 degree slope: rise = run -> 30m rise per 30m pixel
    y, x = np.mgrid[0:20, 0:20]
    ramp_dem = (x * 30.0).astype(np.float32)
    ramp_slope = compute_slope_degrees(ramp_dem, cell_size_m=30.0)

    # Interior pixels should be 45 degrees
    np.testing.assert_allclose(ramp_slope[1:-1, 1:-1], 45.0, atol=1.0)


def test_compute_layover_shadow_mask():
    """Slopes exceeding look angle (35 deg) are flagged as layover/shadow."""
    # Steep 50 deg mountain slope
    y, x = np.mgrid[0:20, 0:20]
    steep_dem = (x * 40.0).astype(np.float32)  # ~53 deg slope
    mask = compute_layover_shadow_mask(steep_dem, incidence_angle_deg=35.0, cell_size_m=30.0)

    # Most pixels should be masked
    assert np.count_nonzero(mask) > 100


def test_compute_hand_valley():
    """HAND must be lowest along the drainage channel and increase on valley flanks."""
    h, w = 30, 30
    y, x = np.mgrid[0:h, 0:w]
    # Channel running down center (col 15)
    dist_to_channel = np.abs(x - 15)
    valley_dem = 500.0 + (dist_to_channel * 10.0) - (y * 2.0)

    hand = compute_hand(valley_dem, drainage_threshold=10)

    # Central channel should have lower HAND than valley walls
    channel_hand = np.mean(hand[:, 15])
    ridge_hand = np.mean(hand[:, 0])
    assert ridge_hand > channel_hand


def test_apply_terrain_exclusion():
    """Removes false positives on steep mountain slopes and high HAND ridges."""
    candidate_mask = np.ones((10, 10), dtype=bool)

    # Slope array: left half flat (3 deg), right half steep cliff (30 deg)
    slope = np.full((10, 10), 3.0, dtype=np.float32)
    slope[:, 5:] = 30.0

    # HAND array: top half valley floor (5m), bottom half mountain ridge (60m)
    hand = np.full((10, 10), 5.0, dtype=np.float32)
    hand[5:, :] = 60.0

    # Only top-left (rows 0:5, cols 0:5) has both slope <= 15 and hand <= 25
    filtered = apply_terrain_exclusion(
        candidate_mask, slope=slope, hand=hand, max_slope_deg=15.0, max_hand_m=25.0
    )

    assert np.count_nonzero(filtered) == 25
    assert np.all(filtered[0:5, 0:5])
    assert not np.any(filtered[5:, :])
    assert not np.any(filtered[:, 5:])
