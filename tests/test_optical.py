"""Tests for optical MNDWI water index and debris flow detection."""

import numpy as np

from floodmap.optical.indices import (
    compute_mndwi,
    detect_debris_change,
    detect_optical_water,
)


def test_compute_mndwi_and_optical_water():
    """Water has high Green reflectance and strong SWIR absorption (MNDWI > 0)."""
    # Water pixel: Green=0.20, SWIR=0.02 -> (0.20-0.02)/(0.22) = +0.81
    # Dry soil: Green=0.10, SWIR=0.35 -> (0.10-0.35)/(0.45) = -0.55
    green = np.array([0.20, 0.10], dtype=np.float32)
    swir = np.array([0.02, 0.35], dtype=np.float32)

    mndwi = compute_mndwi(green, swir)
    assert mndwi[0] > 0.5
    assert mndwi[1] < 0.0

    water = detect_optical_water(mndwi, threshold=0.10)
    assert water[0] is True or water[0] == 1
    assert water[1] is False or water[1] == 0


def test_detect_debris_change():
    """Debris flow strips vegetation (NDVI drop) and deposits mineral sediment (SWIR rise)."""
    # Pixel 0: Debris hit (NDVI drops from 0.6 to 0.2 = -0.4, SWIR rises from 0.15 to 0.35 = +0.20)
    # Pixel 1: Unchanged vegetation (NDVI 0.6 -> 0.6, SWIR 0.15 -> 0.15)
    pre_ndvi = np.array([0.60, 0.60], dtype=np.float32)
    post_ndvi = np.array([0.20, 0.60], dtype=np.float32)
    pre_swir = np.array([0.15, 0.15], dtype=np.float32)
    post_swir = np.array([0.35, 0.15], dtype=np.float32)

    debris = detect_debris_change(
        pre_ndvi, post_ndvi, pre_swir, post_swir, ndvi_drop_thresh=0.15, swir_rise_thresh=0.05
    )

    assert debris[0] is True or debris[0] == 1
    assert debris[1] is False or debris[1] == 0


def test_debris_masked_under_cloud():
    """Cloud and shadow pixels in SCL mask must not be falsely flagged as debris."""
    pre_ndvi = np.array([0.60], dtype=np.float32)
    post_ndvi = np.array([0.10], dtype=np.float32)
    pre_swir = np.array([0.15], dtype=np.float32)
    post_swir = np.array([0.45], dtype=np.float32)

    # Cloud mask is True
    scl_cloud_mask = np.array([True])
    debris = detect_debris_change(
        pre_ndvi, post_ndvi, pre_swir, post_swir, scl_cloud_mask=scl_cloud_mask
    )
    assert debris[0] is False or debris[0] == 0
