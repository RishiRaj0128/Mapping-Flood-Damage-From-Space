"""Digital Elevation Model (DEM) analysis: Slope, Layover/Shadow, and HAND masks."""

import numpy as np
from scipy.ndimage import uniform_filter

from floodmap.logger import get_logger

logger = get_logger("floodmap.terrain.masks")


def compute_slope_degrees(dem: np.ndarray, cell_size_m: float = 30.0) -> np.ndarray:
    """Computes terrain slope in degrees from a digital elevation model.

    Parameters:
        dem: 2D numpy array containing elevation in meters.
        cell_size_m: Spatial resolution of each pixel in meters (Copernicus GLO-30 = 30m).
    """
    dem_clean = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    dy, dx = np.gradient(dem_clean, cell_size_m, cell_size_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)


def compute_layover_shadow_mask(
    dem: np.ndarray,
    incidence_angle_deg: float = 35.0,
    cell_size_m: float = 30.0,
) -> np.ndarray:
    """Generates a binary mask of radar layover and radar shadow zones.

    In mountainous areas:
    - Layover occurs when terrain slope toward radar exceeds the incidence angle.
    - Shadow occurs when terrain slope away from radar exceeds (90 - incidence angle).

    Returns:
        Boolean mask where True indicates pixels contaminated by layover or shadow.
    """
    slope = compute_slope_degrees(dem, cell_size_m=cell_size_m)
    # Layover: terrain slope towards sensor is steeper than look angle
    layover = slope >= incidence_angle_deg
    # Shadow: back slope steeper than grazing angle
    shadow = slope >= (90.0 - incidence_angle_deg)
    return layover | shadow


def compute_hand(
    dem: np.ndarray,
    drainage_threshold: int = 500,
) -> np.ndarray:
    """Computes Height Above Nearest Drainage (HAND) from DEM using pyflwdir.

    HAND measures the vertical elevation difference between a pixel and the nearest
    hydrologically connected stream/drainage channel. In flood analysis, it is the primary
    physical constraint preventing flood classification on mountain ridges.
    """
    try:
        import pyflwdir

        dem_clean = np.where(np.isfinite(dem), dem, np.nanmin(dem)).astype(np.float32)
        # Determine flow directions (D8 algorithm)
        flw = pyflwdir.from_dem(
            dem_clean,
            nodata=-9999.0,
        )
        # Upstream accumulation area to delineate stream network
        uparea = flw.upstream_area()
        streams = uparea >= drainage_threshold

        # Compute HAND relative to delineated streams
        hand = flw.hand(drain=streams, elevtn=dem_clean)
        return np.maximum(hand, 0.0)
    except Exception as exc:
        logger.warning(f"pyflwdir HAND computation fallback triggered ({exc}). Estimating relative relief.")
        # Robust local minimum relief estimation fallback
        dem_clean = np.where(np.isfinite(dem), dem, np.nanmin(dem))
        min_filter = uniform_filter(dem_clean, size=15, mode="reflect")
        return np.maximum(dem_clean - min_filter, 0.0)


def apply_terrain_exclusion(
    candidate_mask: np.ndarray,
    slope: np.ndarray,
    hand: np.ndarray,
    max_slope_deg: float = 15.0,
    max_hand_m: float = 25.0,
    layover_shadow_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Applies strict physical terrain constraints to eliminate mountain false-positives.

    Eliminates candidate flood pixels on:
    1. Slopes steeper than max_slope_deg (standing flood waters cannot persist on steep terrain).
    2. Vertical elevation far above the drainage channel (HAND > max_hand_m).
    3. Radar layover/shadow distortion zones.
    """
    valid_terrain = (slope <= max_slope_deg) & (hand <= max_hand_m)
    if layover_shadow_mask is not None:
        valid_terrain = valid_terrain & (~layover_shadow_mask)

    filtered = candidate_mask & valid_terrain
    suppressed_count = np.count_nonzero(candidate_mask) - np.count_nonzero(filtered)
    if suppressed_count > 0:
        logger.debug(f"Terrain exclusion eliminated {suppressed_count} false-positive pixels.")
    return filtered
