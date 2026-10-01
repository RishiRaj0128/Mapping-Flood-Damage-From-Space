"""SAR speckle filtering and radiometric conversions (dB)."""

import numpy as np
from scipy.ndimage import uniform_filter


def linear_to_db(linear_array: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Converts linear radar backscatter power/intensity to decibels (dB).

    Parameters:
        linear_array: Linear amplitude squared or intensity values (>= 0).
        eps: Small epsilon constant to prevent log(0).
    """
    clipped = np.maximum(linear_array, eps)
    return 10.0 * np.log10(clipped)


def db_to_linear(db_array: np.ndarray) -> np.ndarray:
    """Converts decibels (dB) back to linear radar backscatter intensity."""
    return 10.0 ** (db_array / 10.0)


def lee_filter(
    image: np.ndarray,
    window_size: int = 5,
    num_looks: float = 1.0,
    damping_factor: float = 1.0,
) -> np.ndarray:
    """Standard Lee speckle filter for SAR imagery in linear or decibel domain.

    Adapts filtering weights between pure averaging in homogeneous regions
    and edge preservation in heterogeneous/point target regions.

    Parameters:
        image: 2D numpy array (SAR backscatter).
        window_size: Filter window dimension (must be odd, default 5).
        num_looks: Equivalent number of looks (ENL) of the SAR product.
        damping_factor: Filter dampening parameter.
    """
    if window_size % 2 == 0:
        window_size += 1

    img = image.astype(np.float32)
    # Mask non-finite values
    valid_mask = np.isfinite(img)
    if not np.any(valid_mask):
        return image

    safe_img = np.where(valid_mask, img, 0.0)

    # Local mean and local mean of squares
    local_mean = uniform_filter(safe_img, size=window_size, mode="reflect")
    local_sq_mean = uniform_filter(safe_img**2, size=window_size, mode="reflect")

    # Local variance: Var(X) = E[X^2] - (E[X])^2
    local_var = np.maximum(local_sq_mean - (local_mean**2), 0.0)

    # Noise coefficient of variation
    cu = 1.0 / np.sqrt(max(num_looks, 0.1))
    # Local coefficient of variation
    ci = np.sqrt(local_var) / (np.maximum(np.abs(local_mean), 1e-6))

    # Weighting factor
    # W = 1 - (Cu / Ci)^2
    ratio = (cu / np.maximum(ci, 1e-6)) ** 2
    weights = np.clip(1.0 - ratio, 0.0, 1.0)

    # Filtered estimate: R_hat = mean + W * (I - mean)
    filtered = local_mean + weights * (safe_img - local_mean)
    return np.where(valid_mask, filtered, image)


# Backward-compatible alias
enhanced_lee_filter = lee_filter
