"""Tests for SAR speckle filtering and radiometric conversions."""

import numpy as np

from floodmap.sar.speckle import db_to_linear, enhanced_lee_filter, lee_filter, linear_to_db


def test_linear_to_db_and_db_to_linear():
    """Validates logarithmic conversion accuracy and invertibility."""
    linear = np.array([0.01, 0.1, 1.0, 10.0], dtype=np.float32)
    db = linear_to_db(linear)

    # 10*log10(1.0) = 0 dB, 10*log10(10.0) = 10 dB, 10*log10(0.1) = -10 dB
    np.testing.assert_allclose(db, [-20.0, -10.0, 0.0, 10.0], atol=1e-4)

    recovered = db_to_linear(db)
    np.testing.assert_allclose(recovered, linear, rtol=1e-4)


def test_lee_filter():
    """Lee filter must reduce variance in homogeneous areas while preserving edges."""
    rng = np.random.default_rng(42)
    # 50x50 image with a sharp edge: left side is 1.0, right side is 10.0
    img = np.ones((50, 50), dtype=np.float32)
    img[:, 25:] = 10.0

    # Add multiplicative speckle noise
    noise = rng.gamma(shape=1.0, scale=1.0, size=(50, 50))
    noisy = img * noise

    filtered = lee_filter(noisy, window_size=5, num_looks=1.0)
    # Verify backward compatible alias
    assert enhanced_lee_filter == lee_filter

    # In homogeneous left patch, filtered variance should be strictly less than noisy variance
    noisy_var = np.var(noisy[:, :20])
    filtered_var = np.var(filtered[:, :20])
    assert filtered_var < noisy_var

    # Sharp boundary difference between left and right should remain substantial
    left_mean = np.mean(filtered[:, :20])
    right_mean = np.mean(filtered[:, 30:])
    assert right_mean > (left_mean * 5.0)
