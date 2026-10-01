"""Digital Elevation Model (DEM) analysis: Slope, Layover/Shadow, and HAND masks."""

from typing import Any

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
    return_stats: bool = False,
    relax_steep_gorge: bool = True,
    gorge_hand_limit_m: float = 5.0,
    gorge_max_slope_deg: float = 25.0,
) -> tuple[np.ndarray, dict[str, int]] | np.ndarray:
    """Applies strict physical terrain constraints to eliminate mountain false-positives.

    Eliminates candidate flood pixels on:
    1. Slopes steeper than max_slope_deg (relaxed to gorge_max_slope_deg in immediate river gorges where HAND <= 5m).
    2. Vertical elevation far above the drainage channel (HAND > max_hand_m).
    3. Radar layover/shadow distortion zones.

    Returns:
        filtered_mask: Boolean mask of physically plausible flood pixels.
        stats (optional): Dictionary of pixel counts eliminated by each specific mask.
    """
    if relax_steep_gorge:
        allowed_slope = np.where(hand <= gorge_hand_limit_m, gorge_max_slope_deg, max_slope_deg)
    else:
        allowed_slope = max_slope_deg

    valid_slope = slope <= allowed_slope
    valid_hand = hand <= max_hand_m
    valid_terrain = valid_slope & valid_hand

    if layover_shadow_mask is not None:
        valid_terrain = valid_terrain & (~layover_shadow_mask)

    filtered = candidate_mask & valid_terrain

    # Compute individual exclusion metrics
    slope_removed = int(np.count_nonzero(candidate_mask & (~valid_slope)))
    hand_removed = int(np.count_nonzero(candidate_mask & (~valid_hand)))
    layover_removed = (
        int(np.count_nonzero(candidate_mask & layover_shadow_mask))
        if layover_shadow_mask is not None
        else 0
    )
    total_removed = int(np.count_nonzero(candidate_mask) - np.count_nonzero(filtered))

    stats = {
        "candidate_pixels": int(np.count_nonzero(candidate_mask)),
        "retained_pixels": int(np.count_nonzero(filtered)),
        "total_removed": total_removed,
        "removed_by_slope": slope_removed,
        "removed_by_hand": hand_removed,
        "removed_by_layover_shadow": layover_removed,
        "gorge_relaxed_pixels": (
            int(np.count_nonzero(candidate_mask & (hand <= gorge_hand_limit_m) & (slope > max_slope_deg) & (slope <= gorge_max_slope_deg)))
            if relax_steep_gorge
            else 0
        ),
    }

    logger.info(
        f"Terrain Exclusion Summary | Evaluated: {stats['candidate_pixels']} | "
        f"Removed by Slope (> {max_slope_deg} deg): {slope_removed} | "
        f"Removed by HAND (> {max_hand_m} m): {hand_removed} | "
        f"Removed by Layover/Shadow: {layover_removed} | "
        f"Retained: {stats['retained_pixels']}"
    )

    if return_stats:
        return filtered, stats
    return filtered


def compute_slope_hand_sensitivity(
    candidate_mask: np.ndarray,
    slope: np.ndarray,
    hand: np.ndarray,
    pixel_area_km2: float,
    layover_shadow_mask: np.ndarray | None = None,
    slope_thresholds: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0),
    hand_thresholds: tuple[float, ...] = (10.0, 25.0, 50.0),
    baseline_slope: float = 15.0,
    baseline_hand: float = 25.0,
) -> list[dict[str, Any]]:
    """Generates sensitivity analysis matrix (slope x HAND thresholds vs flooded area).

    Used for evaluation reports to demonstrate parameter robustness and physical impact.
    """
    # Compute baseline area
    baseline_filtered = apply_terrain_exclusion(
        candidate_mask,
        slope=slope,
        hand=hand,
        max_slope_deg=baseline_slope,
        max_hand_m=baseline_hand,
        layover_shadow_mask=layover_shadow_mask,
        relax_steep_gorge=True,
    )
    baseline_area = float(np.count_nonzero(baseline_filtered) * pixel_area_km2)

    results = []
    for s_thresh in slope_thresholds:
        for h_thresh in hand_thresholds:
            filtered = apply_terrain_exclusion(
                candidate_mask,
                slope=slope,
                hand=hand,
                max_slope_deg=s_thresh,
                max_hand_m=h_thresh,
                layover_shadow_mask=layover_shadow_mask,
                relax_steep_gorge=True,
            )
            retained = int(np.count_nonzero(filtered))
            area_km2 = float(retained * pixel_area_km2)
            pct_diff = (
                ((area_km2 - baseline_area) / baseline_area * 100.0)
                if baseline_area > 0
                else 0.0
            )

            results.append({
                "slope_deg": s_thresh,
                "hand_m": h_thresh,
                "retained_pixels": retained,
                "flooded_area_km2": round(area_km2, 4),
                "pct_change_vs_baseline": round(pct_diff, 2),
            })
    return results
