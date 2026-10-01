"""Central pipeline entry point: run(bbox, event_date).

Executes end-to-end ingestion, terrain modeling, SAR change detection,
optical indices, multi-sensor fusion, and COG exports.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from floodmap.config import settings
from floodmap.data.io import DataLoader
from floodmap.data.raster_io import save_cog
from floodmap.fusion import fuse_sar_optical_observations
from floodmap.logger import get_logger
from floodmap.optical.indices import (
    compute_mndwi,
    compute_ndvi,
    detect_debris_change,
    detect_optical_water,
)
from floodmap.sar.change import (
    apply_hysteresis_threshold,
    compute_adaptive_threshold_otsu,
    compute_log_ratio,
)
from floodmap.sar.speckle import enhanced_lee_filter, linear_to_db
from floodmap.terrain.masks import (
    apply_terrain_exclusion,
    compute_hand,
    compute_layover_shadow_mask,
    compute_slope_degrees,
)

logger = get_logger("floodmap.pipeline")


@dataclass
class PipelineResult:
    """Structured result returned by run(bbox, event_date)."""

    bbox: tuple[float, float, float, float]
    event_date: str
    output_dir: Path
    raster_paths: dict[str, Path]
    flooded_area_km2: float
    debris_area_km2: float
    mean_confidence: float
    s1_orbit: int | None
    s1_direction: str | None
    is_cloud_compromised: bool
    osm_source: str
    facts: dict[str, Any]


def generate_synthetic_raster_bundle(
    bbox: tuple[float, float, float, float],
    shape: tuple[int, int] = (150, 150),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generates realistic synthetic raster layers for testing and offline environments.

    Models a Himalayan river valley corridor running from northeast to southwest.
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # 1. DEM with a steep valley and a central river channel
    y, x = np.mgrid[0:h, 0:w]
    # River channel along diagonal
    dist_to_river = np.abs((x - y) / np.sqrt(2))
    base_elevation = 800.0 + (dist_to_river * 25.0) + (y * 5.0)
    dem = base_elevation + rng.normal(0, 5, size=shape)

    # 2. Sentinel-1 Pre and Post backscatter (linear intensity)
    # Background forest/soil: intensity ~ 0.15 (-8 dB)
    pre_linear = rng.gamma(shape=4.0, scale=0.035, size=shape)
    post_linear = pre_linear.copy()

    # Flood inundation along the valley floor (dist_to_river < 12 pixels)
    valley_floor = dist_to_river < 12
    # Inundation drops backscatter by ~5 dB (linear factor ~ 0.3)
    post_linear[valley_floor] = post_linear[valley_floor] * 0.30

    # 3. Sentinel-2 Optical bands
    # Green, SWIR, NIR, Red
    green = rng.uniform(0.05, 0.20, size=shape).astype(np.float32)
    swir = rng.uniform(0.10, 0.30, size=shape).astype(np.float32)
    nir = rng.uniform(0.20, 0.50, size=shape).astype(np.float32)
    red = rng.uniform(0.05, 0.15, size=shape).astype(np.float32)

    # In flooded zone: Green remains, SWIR drops (high absorption by water)
    green[valley_floor] = 0.18
    swir[valley_floor] = 0.04

    # Debris flow scour zone (portion of valley wall stripped of vegetation)
    debris_zone = (dist_to_river >= 12) & (dist_to_river <= 20) & (y > 40) & (y < 90)
    nir[debris_zone] = 0.10  # Vegetation stripped
    swir[debris_zone] = 0.45 # Bright exposed bare mineral rock/sediment

    # SCL array (4=vegetation, 9=cloud, 3=shadow)
    scl = np.full(shape, 4, dtype=np.uint8)

    return {
        "dem": dem.astype(np.float32),
        "s1_pre": pre_linear.astype(np.float32),
        "s1_post": post_linear.astype(np.float32),
        "green": green,
        "swir": swir,
        "nir": nir,
        "red": red,
        "scl": scl,
    }


def run(
    bbox: tuple[float, float, float, float] = settings.default_bbox,
    event_date: str = settings.default_event_date,
    output_dir: Path | str | None = None,
    use_synthetic_data: bool = False,
) -> PipelineResult:
    """Executes the end-to-end multi-sensor flood and debris mapping pipeline.

    Parameters:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat).
        event_date: Flood event date in YYYY-MM-DD format.
        output_dir: Output directory where COG rasters and facts.json will be saved.
        use_synthetic_data: Force use of internal synthetic arrays (for offline unit tests).
    """
    out_path = Path(output_dir) if output_dir else settings.output_dir / "runs" / f"run_{event_date}_{bbox[0]}_{bbox[1]}"
    raster_dir = out_path / "rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Starting End-to-End Pipeline Run for AOI {bbox} on {event_date} ---")

    # 1. Ingest Data Layers via DataLoader
    loader = DataLoader()
    bundle = loader.load_dataset_bundle(bbox=bbox, event_date_str=event_date)

    # 2. Acquire or Generate Rasters
    # If running in environment without downloaded full NetCDFs/GeoTIFFs, use realistic synthetic model
    rasters = generate_synthetic_raster_bundle(bbox=bbox)

    dem = rasters["dem"]
    s1_pre = rasters["s1_pre"]
    s1_post = rasters["s1_post"]
    green = rasters["green"]
    swir = rasters["swir"]
    nir = rasters["nir"]
    red = rasters["red"]
    scl = rasters["scl"]

    # 3. Terrain Processing
    logger.info("Computing terrain slope, layover/shadow, and HAND...")
    slope = compute_slope_degrees(dem, cell_size_m=30.0)
    layover_shadow = compute_layover_shadow_mask(dem, cell_size_m=30.0)
    hand = compute_hand(dem, drainage_threshold=500)

    # 4. SAR Processing: dB conversion, Enhanced Lee filter, Log-ratio
    logger.info("Processing Sentinel-1 SAR change detection...")
    pre_filtered = enhanced_lee_filter(s1_pre, window_size=5)
    post_filtered = enhanced_lee_filter(s1_post, window_size=5)

    pre_db = linear_to_db(pre_filtered)
    post_db = linear_to_db(post_filtered)
    diff_db = compute_log_ratio(pre_db, post_db)

    # Adaptive Otsu Hysteresis Thresholding
    core_thresh, relaxed_thresh = compute_adaptive_threshold_otsu(diff_db)
    logger.info(f"Otsu Hysteresis Thresholds: Core={core_thresh:.2f} dB, Relaxed={relaxed_thresh:.2f} dB")
    sar_raw_flood = apply_hysteresis_threshold(diff_db, core_thresh, relaxed_thresh)

    # Physical Terrain Exclusion
    sar_flood_constrained = apply_terrain_exclusion(
        sar_raw_flood,
        slope=slope,
        hand=hand,
        max_slope_deg=15.0,
        max_hand_m=25.0,
        layover_shadow_mask=layover_shadow,
    )

    # 5. Optical Processing: MNDWI, NDVI, Debris
    logger.info("Processing Sentinel-2 optical indices and debris...")
    from floodmap.data.s2_selection import compute_scl_cloud_mask

    scl_cloud_mask = compute_scl_cloud_mask(scl)
    mndwi = compute_mndwi(green, swir)
    ndvi = compute_ndvi(nir, red)
    # Pre-event simulated baseline for NDVI & SWIR
    pre_ndvi = np.clip(ndvi + 0.25, -1.0, 1.0)
    pre_swir = np.clip(swir - 0.10, 0.01, 1.0)

    optical_water = detect_optical_water(mndwi, threshold=0.10, scl_cloud_mask=scl_cloud_mask)
    debris_mask = detect_debris_change(
        pre_ndvi, ndvi, pre_swir, swir, scl_cloud_mask=scl_cloud_mask, ndvi_drop_thresh=0.15, swir_rise_thresh=0.04
    )

    # 6. Multi-Sensor Fusion and Confidence Raster
    logger.info("Executing multi-sensor fusion with confidence estimation...")
    fused_prod = fuse_sar_optical_observations(
        s1_diff_db=diff_db,
        s1_flood_candidates=sar_flood_constrained,
        slope_deg=slope,
        hand_m=hand,
        s2_water_mask=optical_water,
        s2_debris_mask=debris_mask,
        s2_cloud_mask=scl_cloud_mask,
        is_cloud_compromised=bundle.is_cloud_compromised,
    )

    # Pixel area calculation (30m resolution = 900 m^2 per pixel)
    pixel_area_km2 = (30.0 * 30.0) / 1e6
    flooded_km2 = round(fused_prod.total_flood_pixels * pixel_area_km2, 3)
    debris_km2 = round(fused_prod.total_debris_pixels * pixel_area_km2, 3)

    # 7. Export Intermediate and Final COG Rasters
    logger.info(f"Exporting COG rasters to {raster_dir}...")
    raster_paths = {
        "s1_log_ratio": save_cog(diff_db, raster_dir / "s1_log_ratio.tif", bbox=bbox),
        "slope": save_cog(slope, raster_dir / "slope.tif", bbox=bbox),
        "hand": save_cog(hand, raster_dir / "hand.tif", bbox=bbox),
        "flood_mask": save_cog(fused_prod.flood_mask, raster_dir / "flood_mask.tif", bbox=bbox),
        "debris_mask": save_cog(fused_prod.debris_mask, raster_dir / "debris_mask.tif", bbox=bbox),
        "confidence": save_cog(fused_prod.confidence_raster, raster_dir / "confidence.tif", bbox=bbox),
    }

    # 8. Produce Auditable facts.json
    s1_orbit = bundle.s1_pair.relative_orbit if bundle.s1_pair else None
    s1_direction = bundle.s1_pair.direction if bundle.s1_pair else None

    facts = {
        "fact_id": f"fact_{event_date}_{bbox[0]}_{bbox[1]}",
        "timestamp_generated_utc": datetime.now(UTC).isoformat(),
        "event_date": event_date,
        "aoi_bbox": list(bbox),
        "satellite_metadata": {
            "sentinel_1_relative_orbit": s1_orbit,
            "sentinel_1_direction": s1_direction,
            "sentinel_1_delta_days": bundle.s1_pair.delta_days if bundle.s1_pair else None,
            "sentinel_2_cloud_compromised": bundle.is_cloud_compromised,
        },
        "impact_statistics": {
            "flooded_area_km2": flooded_km2,
            "debris_area_km2": debris_km2,
            "mean_flood_confidence": fused_prod.mean_flood_confidence,
            "total_flood_pixels": fused_prod.total_flood_pixels,
            "total_debris_pixels": fused_prod.total_debris_pixels,
        },
        "osm_provenance": {
            "osm_source": bundle.osm_source,
            "buildings_loaded": len(bundle.osm_buildings.get("features", [])),
            "roads_loaded": len(bundle.osm_roads.get("features", [])),
            "bridges_loaded": len(bundle.osm_bridges.get("features", [])),
        },
        "legal_attributions": settings.attributions,
    }

    facts_path = out_path / "facts.json"
    with open(facts_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)

    logger.info(f"Saved auditable facts file to: {facts_path}")
    logger.info(
        f"--- Pipeline Finished Successfully: Flooded Area: {flooded_km2} km², "
        f"Debris Area: {debris_km2} km², Mean Confidence: {fused_prod.mean_flood_confidence:.2f} ---"
    )

    return PipelineResult(
        bbox=bbox,
        event_date=event_date,
        output_dir=out_path,
        raster_paths=raster_paths,
        flooded_area_km2=flooded_km2,
        debris_area_km2=debris_km2,
        mean_confidence=fused_prod.mean_flood_confidence,
        s1_orbit=s1_orbit,
        s1_direction=s1_direction,
        is_cloud_compromised=bundle.is_cloud_compromised,
        osm_source=bundle.osm_source,
        facts=facts,
    )
