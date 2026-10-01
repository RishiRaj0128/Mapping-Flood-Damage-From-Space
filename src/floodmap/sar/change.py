"""SAR log-ratio change detection and adaptive hysteresis thresholding."""


import numpy as np
from scipy.ndimage import label


def compute_log_ratio(pre_db: np.ndarray, post_db: np.ndarray) -> np.ndarray:
    """Computes radiometric change in decibels: delta_dB = post_dB - pre_dB.

    In SAR imagery, standing water creates specular reflection away from the radar antenna,
    resulting in significant backscatter drop (delta_dB << 0, typically -2.0 to -6.0 dB or more).
    """
    return post_db - pre_db


def compute_adaptive_threshold_otsu(
    diff_array: np.ndarray,
    valid_mask: np.ndarray | None = None,
    nbins: int = 256,
    search_min_drop_db: float = -1.5,
) -> tuple[float, float]:
    """Computes data-driven adaptive thresholds on negative backscatter drop using Otsu's method.

    Returns:
        (core_threshold, relaxed_threshold): Thresholds for hysteresis segmentation.
        Values are negative (e.g. core=-3.8 dB, relaxed=-2.2 dB).
    """
    if valid_mask is None:
        valid_mask = np.isfinite(diff_array)
    else:
        valid_mask = valid_mask & np.isfinite(diff_array)

    vals = diff_array[valid_mask]
    # Filter to potential flood drops (negative changes)
    negative_drops = vals[vals < 0.0]

    if len(negative_drops) < 50:
        # Fallback conservative defaults if insufficient negative samples
        return (-3.5, -2.0)

    # Invert to positive drops for Otsu: drop_magnitude = -diff
    drops = -negative_drops
    hist, bin_edges = np.histogram(drops, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Otsu's inter-class variance maximization
    total_weight = np.sum(hist)
    if total_weight == 0:
        return (-3.5, -2.0)

    weight1 = np.cumsum(hist)
    weight2 = total_weight - weight1

    # Avoid division by zero
    valid_splits = (weight1 > 0) & (weight2 > 0)
    if not np.any(valid_splits):
        return (-3.5, -2.0)

    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1e-8)
    mean2 = (np.cumsum((hist * bin_centers)[::-1])[::-1]) / np.maximum(weight2, 1e-8)

    inter_class_variance = weight1 * weight2 * ((mean1 - mean2) ** 2)
    optimal_idx = np.argmax(inter_class_variance)
    optimal_drop = bin_centers[optimal_idx]

    # Ensure threshold meets physical plausibility (at least search_min_drop_db)
    core_drop = max(float(optimal_drop), abs(search_min_drop_db))
    relaxed_drop = core_drop * 0.65  # Relaxed neighborhood threshold

    core_threshold = -core_drop
    relaxed_threshold = -relaxed_drop

    return (float(core_threshold), float(relaxed_threshold))


def apply_hysteresis_threshold(
    diff_array: np.ndarray,
    core_threshold: float,
    relaxed_threshold: float,
    min_cluster_pixels: int = 5,
) -> np.ndarray:
    """Performs dual-threshold hysteresis segmentation on SAR log-ratio.

    Pixels with backscatter drop below core_threshold form seeds.
    Any contiguous pixels with drop below relaxed_threshold that connect
    to a seed are retained. Isolated speckle is eliminated.

    Parameters:
        diff_array: Log-ratio difference array (post - pre).
        core_threshold: Strict threshold for high-confidence core seeds (e.g. -3.8 dB).
        relaxed_threshold: Relaxed threshold for boundaries (e.g. -2.2 dB).
        min_cluster_pixels: Minimum contiguous cluster size to suppress isolated noise.
    """
    # Core seeds and relaxed candidates (remember: more negative = stronger flood signal)
    core_seeds = diff_array <= core_threshold
    relaxed_candidates = diff_array <= relaxed_threshold

    # Label connected components of relaxed candidates
    labeled_relaxed, num_features = label(relaxed_candidates)
    if num_features == 0:
        return np.zeros_like(diff_array, dtype=bool)

    # Find labels that contain at least one core seed
    labels_with_core = np.unique(labeled_relaxed[core_seeds])
    # Remove background label 0
    labels_with_core = labels_with_core[labels_with_core > 0]

    # Keep only relaxed clusters connected to core seeds
    retained_mask = np.isin(labeled_relaxed, labels_with_core)

    # Filter out tiny clusters
    if min_cluster_pixels > 1:
        retained_labeled, num_retained = label(retained_mask)
        counts = np.bincount(retained_labeled.ravel())
        large_labels = np.where(counts >= min_cluster_pixels)[0]
        large_labels = large_labels[large_labels > 0]
        retained_mask = np.isin(retained_labeled, large_labels)

    return retained_mask
