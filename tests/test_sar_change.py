"""Tests for SAR log-ratio change detection and Otsu hysteresis thresholding."""

import numpy as np

from floodmap.sar.change import (
    apply_hysteresis_threshold,
    compute_adaptive_threshold_otsu,
    compute_log_ratio,
)


def test_compute_log_ratio():
    """Water inundation causes a negative dB difference."""
    pre_db = np.array([-10.0, -12.0, -8.0], dtype=np.float32)
    post_db = np.array([-16.0, -12.0, -4.0], dtype=np.float32)

    diff = compute_log_ratio(pre_db, post_db)
    # -16 - (-10) = -6 dB (strong flood signal)
    # -12 - (-12) = 0 dB (no change)
    # -4 - (-8) = +4 dB (brightening)
    np.testing.assert_allclose(diff, [-6.0, 0.0, 4.0], atol=1e-5)


def test_adaptive_otsu_threshold():
    """Otsu threshold identifies optimal separation between background and flood drops."""
    rng = np.random.default_rng(42)
    # Bimodal distribution: 80% background (slight variations around 0 dB) and 20% flood (-5 dB)
    bg_drops = rng.normal(loc=-0.5, scale=0.5, size=800)
    flood_drops = rng.normal(loc=-5.0, scale=0.8, size=200)
    diff = np.concatenate([bg_drops, flood_drops])

    core_thresh, relaxed_thresh = compute_adaptive_threshold_otsu(diff)

    # Core threshold should isolate the flood mode (between -3 and -6 dB)
    assert core_thresh <= -2.5
    assert relaxed_thresh > core_thresh
    assert relaxed_thresh <= -1.5


def test_hysteresis_threshold():
    """Hysteresis connects relaxed pixels to core seeds while discarding isolated noise."""
    # 10x10 diff grid with 0 dB background
    diff = np.zeros((10, 10), dtype=np.float32)

    # Connected flood feature at center (row 4-6, col 4-6)
    diff[4:7, 4:7] = -2.5  # Relaxed candidate
    diff[5, 5] = -4.5       # Core seed at center

    # Isolated speckle noise elsewhere (fails to reach core seed)
    diff[1, 1] = -2.5       # Relaxed without core seed
    # Isolated single pixel core seed
    diff[8, 8] = -4.5

    mask = apply_hysteresis_threshold(
        diff, core_threshold=-4.0, relaxed_threshold=-2.0, min_cluster_pixels=4
    )

    # Center cluster should be preserved (9 connected pixels)
    assert mask[5, 5]
    assert mask[4, 4]
    assert np.count_nonzero(mask[4:7, 4:7]) == 9

    # Isolated relaxed noise at (1,1) must be rejected
    assert not mask[1, 1]
    # Isolated single pixel core seed at (8,8) rejected by min_cluster_pixels=4
    assert not mask[8, 8]
