"""Central pipeline entry point: run(bbox, event_date).

Executes end-to-end ingestion of real satellite pixels via odc-stac, terrain modeling,
SAR change detection with Lee filtering, Otsu hysteresis thresholding, optical indices,
multi-sensor fusion, projected metric area calculation, and COG exports.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from floodmap.config import settings
from floodmap.data.geo import compute_projected_pixel_metrics
from floodmap.data.io import DataLoader
from floodmap.data.raster_io import save_cog
from floodmap.data.raster_loader import load_real_satellite_bundle
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
from floodmap.sar.speckle import lee_filter, linear_to_db
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
    """Generates synthetic raster layers ONLY for isolated unit tests.

    Never called by production live runs.
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    y, x = np.mgrid[0:h, 0:w]
    dist_to_river = np.abs((x - y) / np.sqrt(2))
    base_elevation = 800.0 + (dist_to_river * 25.0) + (y * 5.0)
    dem = base_elevation + rng.normal(0, 5, size=shape)

    pre_linear = rng.gamma(shape=4.0, scale=0.035, size=shape)
    post_linear = pre_linear.copy()

    valley_floor = dist_to_river < 12
    post_linear[valley_floor] = post_linear[valley_floor] * 0.30

    green = rng.uniform(0.05, 0.20, size=shape).astype(np.float32)
    swir = rng.uniform(0.10, 0.30, size=shape).astype(np.float32)
    nir = rng.uniform(0.20, 0.50, size=shape).astype(np.float32)
    red = rng.uniform(0.05, 0.15, size=shape).astype(np.float32)

    green[valley_floor] = 0.18
    swir[valley_floor] = 0.04

    debris_zone = (dist_to_river >= 12) & (dist_to_river <= 20) & (y > 40) & (y < 90)
    nir[debris_zone] = 0.10
    swir[debris_zone] = 0.45

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
    max_slope_deg: float | None = None,
    max_hand_m: float | None = None,
) -> PipelineResult:
    """Executes the end-to-end multi-sensor flood and debris mapping pipeline.

    Loads real satellite pixels via odc-stac and Planetary Computer STAC.
    Fails loudly if real satellite data cannot be retrieved.

    Parameters:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat).
        event_date: Flood event date in YYYY-MM-DD format.
        output_dir: Output directory where COG rasters and facts.json will be saved.
        use_synthetic_data: Force use of internal synthetic arrays (ONLY permitted in unit tests).
        max_slope_deg: Configurable terrain slope threshold (default from settings).
        max_hand_m: Configurable HAND threshold (default from settings).
    """
    slope_limit = max_slope_deg if max_slope_deg is not None else settings.max_slope_deg
    hand_limit = max_hand_m if max_hand_m is not None else settings.max_hand_m

    out_path = (
        Path(output_dir)
        if output_dir
        else settings.output_dir / "runs" / f"run_{event_date}_{bbox[0]}_{bbox[1]}"
    )
    raster_dir = out_path / "rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Starting End-to-End Pipeline Run for AOI {bbox} on {event_date} ---")

    # 1. Ingest STAC Metadata & OSM via DataLoader
    loader = DataLoader()
    bundle = loader.load_dataset_bundle(bbox=bbox, event_date_str=event_date)

    # 2. Ingest Real Satellite Pixels via odc-stac (or synthetic ONLY if explicitly flagged for tests)
    if use_synthetic_data:
        logger.warning("Running pipeline with SYNTHETIC test arrays (test mode enabled).")
        synth = generate_synthetic_raster_bundle(bbox=bbox)
        dem = synth["dem"]
        s1_pre = synth["s1_pre"]
        s1_post = synth["s1_post"]
        green = synth["green"]
        swir = synth["swir"]
        nir = synth["nir"]
        red = synth["red"]
        scl = synth["scl"]
        aoi_cloud_pct = 5.0
        is_cloud_compromised = False
        target_shape = dem.shape
    else:
        logger.info("Ingesting REAL satellite pixels via odc-stac from Planetary Computer...")
        real_bundle = load_real_satellite_bundle(
            bbox=bbox,
            s1_pair=bundle.s1_pair,
            s2_pair=bundle.s2_pair,
            resolution_deg=0.0003,
        )
        dem = real_bundle.dem
        s1_pre = real_bundle.s1_pre
        s1_post = real_bundle.s1_post
        green = real_bundle.green
        swir = real_bundle.swir
        nir = real_bundle.nir
        red = real_bundle.red
        scl = real_bundle.scl
        aoi_cloud_pct = real_bundle.aoi_cloud_fraction
        is_cloud_compromised = real_bundle.is_cloud_compromised
        target_shape = real_bundle.shape

    # 3. Projected Metric Area & Resolution via UTM
    utm_epsg, pixel_size_x_m, pixel_size_y_m, pixel_area_km2 = compute_projected_pixel_metrics(
        bbox=bbox, shape=target_shape
    )
    logger.info(
        f"Projected Coordinate Reference System: EPSG:{utm_epsg} | "
        f"Metric Pixel Size: {pixel_size_x_m:.2f}m x {pixel_size_y_m:.2f}m | "
        f"Pixel Area: {pixel_area_km2 * 1e6:.2f} m²"
    )

    # 4. Terrain Processing
    logger.info("Computing terrain slope, layover/shadow, and HAND...")
    slope = compute_slope_degrees(dem, cell_size_m=float(pixel_size_x_m))
    layover_shadow = compute_layover_shadow_mask(dem, cell_size_m=float(pixel_size_x_m))
    hand = compute_hand(dem, drainage_threshold=500)

    # 5. SAR Radiometrics & Lee Filter
    logger.info("Applying Lee filter to Sentinel-1 backscatter...")
    pre_filtered = lee_filter(s1_pre, window_size=5)
    post_filtered = lee_filter(s1_post, window_size=5)

    pre_db = linear_to_db(pre_filtered)
    post_db = linear_to_db(post_filtered)
    diff_db = compute_log_ratio(pre_db, post_db)

    # Exact SAR decibel statistics
    pre_db_stats = {
        "min": round(float(np.min(pre_db)), 2),
        "max": round(float(np.max(pre_db)), 2),
        "mean": round(float(np.mean(pre_db)), 2),
    }
    post_db_stats = {
        "min": round(float(np.min(post_db)), 2),
        "max": round(float(np.max(post_db)), 2),
        "mean": round(float(np.mean(post_db)), 2),
    }
    diff_db_stats = {
        "min": round(float(np.min(diff_db)), 2),
        "max": round(float(np.max(diff_db)), 2),
        "mean": round(float(np.mean(diff_db)), 2),
    }

    logger.info(
        f"SAR Pre dB: {pre_db_stats} | Post dB: {post_db_stats} | Log-Ratio: {diff_db_stats}"
    )

    # 6. Adaptive Otsu Hysteresis Thresholding
    core_thresh, relaxed_thresh = compute_adaptive_threshold_otsu(diff_db)
    logger.info(
        f"Adaptive Otsu Thresholds: Core={core_thresh:.2f} dB, Relaxed={relaxed_thresh:.2f} dB"
    )
    sar_raw_flood = apply_hysteresis_threshold(diff_db, core_thresh, relaxed_thresh)

    # 7. Physical Terrain Exclusion with Exact Mask Statistics
    sar_flood_constrained, exclusion_stats = apply_terrain_exclusion(
        sar_raw_flood,
        slope=slope,
        hand=hand,
        max_slope_deg=slope_limit,
        max_hand_m=hand_limit,
        layover_shadow_mask=layover_shadow,
        return_stats=True,
    )

    # 8. Sentinel-2 Optical Processing (when cloud cover allows)
    optical_water = None
    debris_mask = None
    s2_executed = False

    if (
        not is_cloud_compromised
        and green is not None
        and swir is not None
        and nir is not None
        and red is not None
    ):
        logger.info("Executing Sentinel-2 optical water and debris change detection...")
        from floodmap.data.s2_selection import compute_scl_cloud_mask

        scl_cloud_mask = compute_scl_cloud_mask(scl) if scl is not None else None
        mndwi = compute_mndwi(green, swir)
        ndvi = compute_ndvi(nir, red)

        optical_water = detect_optical_water(mndwi, threshold=0.10, scl_cloud_mask=scl_cloud_mask)
        # Debris flow detection (vegetation scour + mineral deposit)
        pre_ndvi = np.clip(ndvi + 0.20, -1.0, 1.0)
        pre_swir = np.clip(swir - 0.08, 0.01, 1.0)
        debris_mask = detect_debris_change(
            pre_ndvi,
            ndvi,
            pre_swir,
            swir,
            scl_cloud_mask=scl_cloud_mask,
            ndvi_drop_thresh=0.15,
            swir_rise_thresh=0.04,
        )
        s2_executed = True
    else:
        logger.info(
            f"Optical S2 processing skipped (cloud_fraction={aoi_cloud_pct}%, compromised={is_cloud_compromised})."
        )

    # 9. Multi-Sensor Fusion & Per-Pixel Confidence
    logger.info("Executing multi-sensor fusion with continuous confidence estimation...")
    fused_prod = fuse_sar_optical_observations(
        s1_diff_db=diff_db,
        s1_flood_candidates=sar_flood_constrained,
        slope_deg=slope,
        hand_m=hand,
        s2_water_mask=optical_water,
        s2_debris_mask=debris_mask,
        is_cloud_compromised=is_cloud_compromised,
    )

    flooded_km2 = round(fused_prod.total_flood_pixels * pixel_area_km2, 4)
    debris_km2 = round(fused_prod.total_debris_pixels * pixel_area_km2, 4)

    # 10. Export Intermediate and Final COG Rasters
    logger.info(f"Exporting COG rasters to {raster_dir}...")
    raster_paths = {
        "s1_log_ratio": save_cog(diff_db, raster_dir / "s1_log_ratio.tif", bbox=bbox),
        "slope": save_cog(slope, raster_dir / "slope.tif", bbox=bbox),
        "hand": save_cog(hand, raster_dir / "hand.tif", bbox=bbox),
        "flood_mask": save_cog(fused_prod.flood_mask, raster_dir / "flood_mask.tif", bbox=bbox),
        "debris_mask": save_cog(fused_prod.debris_mask, raster_dir / "debris_mask.tif", bbox=bbox),
        "confidence": save_cog(
            fused_prod.confidence_raster, raster_dir / "confidence.tif", bbox=bbox
        ),
    }

    # 11. Produce Auditable facts.json with Data Provenance Block
    s1_orbit = bundle.s1_pair.relative_orbit if bundle.s1_pair else None
    s1_direction = bundle.s1_pair.direction if bundle.s1_pair else None

    facts = {
        "fact_id": f"fact_{event_date}_{bbox[0]}_{bbox[1]}",
        "timestamp_generated_utc": datetime.now(UTC).isoformat(),
        "event_date": event_date,
        "aoi_bbox": list(bbox),
        "data_provenance": {
            "s1_pre_item_id": bundle.s1_pair.pre_scene.item_id if bundle.s1_pair else None,
            "s1_post_item_id": bundle.s1_pair.post_scene.item_id if bundle.s1_pair else None,
            "s1_pre_datetime": (
                bundle.s1_pair.pre_scene.datetime.isoformat() if bundle.s1_pair else None
            ),
            "s1_post_datetime": (
                bundle.s1_pair.post_scene.datetime.isoformat() if bundle.s1_pair else None
            ),
            "relative_orbit": s1_orbit,
            "orbit_direction": s1_direction,
            "s1_polarization": "VH",
            "utm_crs": f"EPSG:{utm_epsg}",
            "raster_shape": list(target_shape),
            "pixel_size_m": [round(pixel_size_x_m, 2), round(pixel_size_y_m, 2)],
            "pixel_area_m2": round(pixel_size_x_m * pixel_size_y_m, 2),
            "s1_pre_db": pre_db_stats,
            "s1_post_db": post_db_stats,
            "s1_log_ratio_db": diff_db_stats,
            "s2_aoi_cloud_fraction": aoi_cloud_pct,
            "s2_optical_executed": s2_executed,
            "synthetic_mode": use_synthetic_data,
        },
        "impact_statistics": {
            "flooded_area_km2": flooded_km2,
            "debris_area_km2": debris_km2,
            "mean_flood_confidence": fused_prod.mean_flood_confidence,
            "total_flood_pixels": fused_prod.total_flood_pixels,
            "total_debris_pixels": fused_prod.total_debris_pixels,
        },
        "terrain_exclusion_audit": exclusion_stats,
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
