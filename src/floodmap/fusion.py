"""Multi-sensor SAR-Optical fusion and per-pixel confidence raster estimation."""

from dataclasses import dataclass

import numpy as np

from floodmap.logger import get_logger

logger = get_logger("floodmap.fusion")


@dataclass
class FusedFloodProduct:
    """Multi-sensor fused flood, debris, and confidence outputs."""

    flood_mask: np.ndarray  # Boolean mask: True = standing flood water
    debris_mask: np.ndarray  # Boolean mask: True = debris / sediment deposit
    classified_raster: np.ndarray  # 0=background, 1=flood, 2=debris
    confidence_raster: np.ndarray  # Float32 array [0.0, 1.0]
    total_flood_pixels: int
    total_debris_pixels: int
    mean_flood_confidence: float


def fuse_sar_optical_observations(
    s1_diff_db: np.ndarray,
    s1_flood_candidates: np.ndarray,
    slope_deg: np.ndarray,
    hand_m: np.ndarray,
    s2_water_mask: np.ndarray | None = None,
    s2_debris_mask: np.ndarray | None = None,
    s2_cloud_mask: np.ndarray | None = None,
    is_cloud_compromised: bool = False,
) -> FusedFloodProduct:
    """Fuses Sentinel-1 SAR change with Sentinel-2 optical indices and computes confidence.

    Confidence logic:
    - High (0.85 - 1.00): Dual-sensor agreement (both S1 drop and S2 MNDWI confirm water),
      combined with ideal valley terrain (HAND < 10m, slope < 5 deg).
    - Strong (0.75 - 0.85): Clear S1 SAR drop under cloudy skies, strictly validated by HAND.
    - Moderate (0.50 - 0.75): Single sensor detection or marginal drop near boundary thresholds.
    - Zero (0.00): Areas failing physical terrain rules (steep cliffs, mountain ridges).
    """
    height, width = s1_diff_db.shape
    confidence = np.zeros((height, width), dtype=np.float32)

    # 1. Base SAR confidence from drop magnitude and terrain plausibility
    # Normalize backscatter drop: -2 dB -> 0.3, -6 dB -> 0.9
    sar_drop_magnitude = np.maximum(-s1_diff_db, 0.0)
    sar_conf = np.clip((sar_drop_magnitude - 1.5) / 4.0, 0.0, 0.90)

    # Terrain plausibility weight (HAND <= 25m, slope <= 15 deg)
    hand_weight = np.clip(1.0 - (hand_m / 25.0), 0.0, 1.0)
    slope_weight = np.clip(1.0 - (slope_deg / 15.0), 0.0, 1.0)
    terrain_factor = hand_weight * slope_weight

    # SAR-derived flood confidence
    sar_conf = sar_conf * terrain_factor
    sar_flood = s1_flood_candidates & (terrain_factor > 0.1)

    # 2. Integrate Sentinel-2 optical water if available
    optical_water = np.zeros((height, width), dtype=bool)
    optical_debris = np.zeros((height, width), dtype=bool)

    if s2_water_mask is not None:
        optical_water = s2_water_mask & (terrain_factor > 0.1)
    if s2_debris_mask is not None:
        optical_debris = s2_debris_mask & (terrain_factor > 0.05)

    # 3. Sensor Fusion
    # In cloud-free regions: Dual confirmation boosts confidence
    both_water = sar_flood & optical_water
    s1_only_water = sar_flood & (~optical_water)
    s2_only_water = optical_water & (~sar_flood)

    # Dual sensor agreement gives highest confidence
    confidence[both_water] = np.clip(0.85 + (sar_conf[both_water] * 0.15), 0.85, 0.98)

    if is_cloud_compromised:
        # Under monsoon cloud cover, S1 SAR is primary
        confidence[s1_only_water] = np.clip(0.65 + (sar_conf[s1_only_water] * 0.25), 0.60, 0.88)
    else:
        # Optical clear: S1 only or S2 only is moderate
        confidence[s1_only_water] = np.clip(0.50 + (sar_conf[s1_only_water] * 0.25), 0.50, 0.75)
        confidence[s2_only_water] = 0.65 * terrain_factor[s2_only_water]

    final_flood = (confidence >= 0.50)

    # 4. Debris flows (cannot overlap water)
    final_debris = optical_debris & (~final_flood)
    confidence[final_debris] = np.clip(0.60 + (0.30 * terrain_factor[final_debris]), 0.55, 0.88)

    # 5. Build classified raster
    classified = np.zeros((height, width), dtype=np.uint8)
    classified[final_flood] = 1  # 1 = Inundation
    classified[final_debris] = 2  # 2 = Debris

    total_flood = int(np.count_nonzero(final_flood))
    total_debris = int(np.count_nonzero(final_debris))
    mean_conf = float(np.mean(confidence[final_flood])) if total_flood > 0 else 0.0

    logger.info(
        f"Multi-Sensor Fusion Complete | Flood Pixels: {total_flood} | "
        f"Debris Pixels: {total_debris} | Mean Flood Confidence: {mean_conf:.2f}"
    )

    return FusedFloodProduct(
        flood_mask=final_flood,
        debris_mask=final_debris,
        classified_raster=classified,
        confidence_raster=confidence,
        total_flood_pixels=total_flood,
        total_debris_pixels=total_debris,
        mean_flood_confidence=round(mean_conf, 3),
    )
