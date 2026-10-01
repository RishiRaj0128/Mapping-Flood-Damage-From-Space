"""Optical water indices (MNDWI) and debris/sediment change detection."""

import numpy as np


def compute_mndwi(green: np.ndarray, swir: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Computes Modified Normalized Difference Water Index (MNDWI).

    MNDWI = (Green - SWIR) / (Green + SWIR)
    Enhances open water features while suppressing built-up, soil, and vegetation noise.
    """
    g = green.astype(np.float32)
    s = swir.astype(np.float32)
    denom = g + s
    denom = np.where(np.abs(denom) < eps, eps, denom)
    mndwi = (g - s) / denom
    return np.clip(mndwi, -1.0, 1.0)


def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Computes Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)
    """
    n = nir.astype(np.float32)
    r = red.astype(np.float32)
    denom = n + r
    denom = np.where(np.abs(denom) < eps, eps, denom)
    ndvi = (n - r) / denom
    return np.clip(ndvi, -1.0, 1.0)


def detect_optical_water(
    mndwi: np.ndarray,
    threshold: float = 0.10,
    scl_cloud_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Classifies water pixels from MNDWI, respecting the SCL cloud/shadow mask."""
    water = mndwi >= threshold
    if scl_cloud_mask is not None:
        water = water & (~scl_cloud_mask)
    return water


def detect_debris_change(
    pre_ndvi: np.ndarray,
    post_ndvi: np.ndarray,
    pre_swir: np.ndarray,
    post_swir: np.ndarray,
    scl_cloud_mask: np.ndarray | None = None,
    ndvi_drop_thresh: float = 0.15,
    swir_rise_thresh: float = 0.04,
) -> np.ndarray:
    """Detects mud and debris deposits characteristic of Himalayan flash floods.

    Debris flow signatures:
    1. Significant vegetation stripping / scour: (NDVI_post - NDVI_pre) < -ndvi_drop_thresh
    2. Fresh bare soil/mineral sediment deposit: (SWIR_post - SWIR_pre) > swir_rise_thresh
    3. Evaluated strictly outside cloud and cloud-shadow pixels.
    """
    ndvi_change = post_ndvi - pre_ndvi
    swir_change = post_swir - pre_swir

    is_debris = (ndvi_change <= -ndvi_drop_thresh) & (swir_change >= swir_rise_thresh)
    if scl_cloud_mask is not None:
        is_debris = is_debris & (~scl_cloud_mask)
    return is_debris
