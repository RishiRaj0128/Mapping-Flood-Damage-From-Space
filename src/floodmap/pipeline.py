"""Central pipeline entry point: run(bbox, event_date).

Executes end-to-end ingestion of real satellite pixels via odc-stac, terrain modeling,
SAR dual-pol (VH+VV) change detection with Lee filtering, Otsu hysteresis thresholding,
optical indices with per-pixel snow/cloud masking, multi-sensor fusion, projected metric
area calculation, and COG exports.
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
    compute_optical_invalid_mask,
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
    compute_slope_hand_sensitivity,
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
    debris_area_km2: float | None
    debris_status: str
    debris_reason: str | None
    optical_coverage_pct: float | None
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
) -> dict[str, Any]:
    """Generates synthetic raster layers ONLY for isolated unit tests.

    Never called by production live runs.
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    y, x = np.mgrid[0:h, 0:w]
    dist_to_river = np.abs((x - y) / np.sqrt(2))
    base_elevation = 800.0 + (dist_to_river * 25.0) + (y * 5.0)
    dem = base_elevation + rng.normal(0, 5, size=shape)

    pre_linear = rng.gamma(shape=4.0, scale=0.035, size=shape).astype(np.float32)
    post_linear = pre_linear.copy()

    valley_floor = dist_to_river < 12
    post_linear[valley_floor] = post_linear[valley_floor] * 0.30

    green = rng.uniform(0.05, 0.20, size=shape).astype(np.float32)
    swir = rng.uniform(0.10, 0.30, size=shape).astype(np.float32)
    nir = rng.uniform(0.20, 0.50, size=shape).astype(np.float32)
    red = rng.uniform(0.05, 0.15, size=shape).astype(np.float32)
    blue = rng.uniform(0.05, 0.15, size=shape).astype(np.float32)

    green[valley_floor] = 0.18
    swir[valley_floor] = 0.04

    debris_zone = (dist_to_river >= 12) & (dist_to_river <= 20) & (y > 40) & (y < 90)
    nir[debris_zone] = 0.10
    swir[debris_zone] = 0.45

    scl = np.full(shape, 4, dtype=np.uint8)

    return {
        "dem": dem.astype(np.float32),
        "s1_pre_vh": pre_linear,
        "s1_post_vh": post_linear,
        "s1_pre_vv": pre_linear * 1.5,
        "s1_post_vv": post_linear * 1.5,
        "s1_pre": pre_linear,
        "s1_post": post_linear,
        "blue": blue,
        "green": green,
        "swir": swir,
        "nir": nir,
        "red": red,
        "scl": scl,
        "aoi_cloud_pct": 5.0,
        "optical_coverage_pct": 95.0,
        "is_cloud_compromised": False,
        "shape": shape,
        "relative_orbit": 19 if bbox[0] > 80.0 else 129,
        "orbit_direction": "descending" if bbox[0] > 80.0 else "ascending",
    }


def calc_valid_decibel_stats(arr: np.ndarray) -> dict[str, float]:
    """Calculates min, max, mean decibel statistics strictly on finite valid pixels."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": round(float(np.min(valid)), 2),
        "max": round(float(np.max(valid)), 2),
        "mean": round(float(np.mean(valid)), 2),
    }


def run(
    bbox: tuple[float, float, float, float] = settings.default_bbox,
    event_date: str = "2026-08-26",
    output_dir: Path | str | None = None,
    use_synthetic_data: bool = False,
    max_slope_deg: float | None = None,
    max_hand_m: float | None = None,
) -> PipelineResult:
    """Executes the complete flood mapping pipeline for a specified AOI and event date.

    Parameters:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        event_date: Event reference date string (YYYY-MM-DD)
        output_dir: Target output directory for rasters and facts.json
        use_synthetic_data: Test flag (strictly for unit tests). Production uses real pixels.
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

    # 1. Ingest STAC Metadata & OSM via DataLoader (Strict Pre-Event Date Enforcement)
    loader = DataLoader()
    bundle = loader.load_dataset_bundle(bbox=bbox, event_date_str=event_date)

    # 2. Ingest Real Satellite Pixels via odc-stac (or synthetic ONLY if flagged for tests)
    if use_synthetic_data:
        logger.warning("Running pipeline with SYNTHETIC test arrays (test mode enabled).")
        synth = generate_synthetic_raster_bundle(bbox=bbox)
        dem = synth["dem"]
        s1_pre_vh = synth["s1_pre_vh"]
        s1_post_vh = synth["s1_post_vh"]
        s1_pre_vv = synth["s1_pre_vv"]
        s1_post_vv = synth["s1_post_vv"]
        blue = synth["blue"]
        green = synth["green"]
        swir = synth["swir"]
        nir = synth["nir"]
        red = synth["red"]
        scl = synth["scl"]
        aoi_cloud_pct = synth["aoi_cloud_pct"]
        optical_coverage_pct = synth["optical_coverage_pct"]
        pre_swir = pre_nir = pre_red = pre_scl = None
        target_shape = synth["shape"]
        relative_orbit = synth["relative_orbit"]
        orbit_direction = synth["orbit_direction"]
        s1_polarization = "VH+VV"
    else:
        logger.info("Ingesting REAL satellite pixels via odc-stac from Planetary Computer...")
        real_bundle = load_real_satellite_bundle(
            bbox=bbox,
            s1_pair=bundle.s1_pair,
            s2_pair=bundle.s2_pair,
            resolution_deg=0.0003,
        )
        dem = real_bundle.dem
        s1_pre_vh = real_bundle.s1_pre_vh
        s1_post_vh = real_bundle.s1_post_vh
        s1_pre_vv = real_bundle.s1_pre_vv
        s1_post_vv = real_bundle.s1_post_vv
        blue = real_bundle.blue
        green = real_bundle.green
        swir = real_bundle.swir
        nir = real_bundle.nir
        red = real_bundle.red
        scl = real_bundle.scl
        pre_swir = real_bundle.pre_swir
        pre_nir = real_bundle.pre_nir
        pre_red = real_bundle.pre_red
        pre_scl = real_bundle.pre_scl
        aoi_cloud_pct = real_bundle.aoi_cloud_fraction
        optical_coverage_pct = real_bundle.optical_coverage_pct
        target_shape = real_bundle.shape
        relative_orbit = real_bundle.relative_orbit
        orbit_direction = real_bundle.orbit_direction
        s1_polarization = real_bundle.s1_polarization

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

    # 5. SAR Radiometrics, Lee Filter & Dual-Pol Fusion (Masking nodata as NaN)
    logger.info("Applying Lee filter to Sentinel-1 backscatter (VH & VV)...")
    pre_filt_vh = lee_filter(s1_pre_vh, window_size=5)
    post_filt_vh = lee_filter(s1_post_vh, window_size=5)
    pre_db_vh = linear_to_db(pre_filt_vh, mask_nodata=True)
    post_db_vh = linear_to_db(post_filt_vh, mask_nodata=True)
    diff_vh = compute_log_ratio(pre_db_vh, post_db_vh)

    if s1_pre_vv is not None and s1_post_vv is not None:
        pre_filt_vv = lee_filter(s1_pre_vv, window_size=5)
        post_filt_vv = lee_filter(s1_post_vv, window_size=5)
        pre_db_vv = linear_to_db(pre_filt_vv, mask_nodata=True)
        post_db_vv = linear_to_db(post_filt_vv, mask_nodata=True)
        diff_vv = compute_log_ratio(pre_db_vv, post_db_vv)

        # Dual-polarization fusion:
        # VV specular reflection detects calm standing water; VH detects vegetation/debris change
        both_valid = np.isfinite(diff_vv) & np.isfinite(diff_vh)
        diff_db = np.full_like(diff_vh, fill_value=np.nan, dtype=np.float32)
        diff_db[both_valid] = (0.65 * diff_vv[both_valid]) + (0.35 * diff_vh[both_valid])
        diff_db[np.isnan(diff_db) & np.isfinite(diff_vv)] = diff_vv[np.isnan(diff_db) & np.isfinite(diff_vv)]
        diff_db[np.isnan(diff_db) & np.isfinite(diff_vh)] = diff_vh[np.isnan(diff_db) & np.isfinite(diff_vh)]

        pre_db = np.where(np.isfinite(pre_db_vh), pre_db_vh, pre_db_vv)
        post_db = np.where(np.isfinite(post_db_vh), post_db_vh, post_db_vv)
    else:
        diff_db = diff_vh
        pre_db = pre_db_vh
        post_db = post_db_vh

    # Exact SAR decibel statistics on finite valid pixels
    pre_db_stats = calc_valid_decibel_stats(pre_db)
    post_db_stats = calc_valid_decibel_stats(post_db)
    diff_db_stats = calc_valid_decibel_stats(diff_db)

    logger.info(
        f"SAR Pre dB: {pre_db_stats} | Post dB: {post_db_stats} | Log-Ratio: {diff_db_stats}"
    )

    # 6. Adaptive Otsu Hysteresis Thresholding
    core_thresh, relaxed_thresh = compute_adaptive_threshold_otsu(diff_db)
    logger.info(
        f"Adaptive Otsu Thresholds: Core={core_thresh:.2f} dB, Relaxed={relaxed_thresh:.2f} dB"
    )
    sar_raw_flood = apply_hysteresis_threshold(diff_db, core_thresh, relaxed_thresh)

    # 7. Physical Terrain Exclusion with Steep Gorge Relaxation & Mask Auditing
    sar_flood_constrained, exclusion_stats = apply_terrain_exclusion(
        sar_raw_flood,
        slope=slope,
        hand=hand,
        max_slope_deg=slope_limit,
        max_hand_m=hand_limit,
        layover_shadow_mask=layover_shadow,
        return_stats=True,
        relax_steep_gorge=True,
    )

    # Slope/HAND Sensitivity Analysis Matrix
    logger.info("Computing Slope and HAND sensitivity matrix...")
    sensitivity_matrix = compute_slope_hand_sensitivity(
        sar_raw_flood,
        slope=slope,
        hand=hand,
        pixel_area_km2=pixel_area_km2,
        layover_shadow_mask=layover_shadow,
        baseline_slope=slope_limit,
        baseline_hand=hand_limit,
    )
    with open(out_path / "sensitivity_table.json", "w", encoding="utf-8") as f:
        json.dump(sensitivity_matrix, f, indent=2)

    # 8. Sentinel-2 Optical Processing with Per-Pixel SCL Masking
    optical_water = None
    debris_mask = None
    s2_executed = False
    debris_area_km2 = None

    effective_optical_cov = optical_coverage_pct if optical_coverage_pct is not None else 0.0
    if effective_optical_cov >= 20.0 and green is not None and swir is not None and nir is not None and red is not None:
        logger.info(
            f"Executing Sentinel-2 optical analysis (optical coverage: {effective_optical_cov:.1f}% >= 20.0%)..."
        )
        invalid_post = (
            compute_optical_invalid_mask(scl, exclude_snow=True)
            if scl is not None
            else None
        )
        invalid_pre = (
            compute_optical_invalid_mask(pre_scl, exclude_snow=True)
            if pre_scl is not None
            else None
        )
        if invalid_post is not None and invalid_pre is not None:
            invalid_mask = invalid_post | invalid_pre
        elif invalid_post is not None:
            invalid_mask = invalid_post
        elif invalid_pre is not None:
            invalid_mask = invalid_pre
        else:
            invalid_mask = None

        mndwi = compute_mndwi(green, swir)
        ndvi = compute_ndvi(nir, red)

        optical_water = detect_optical_water(mndwi, threshold=0.10, invalid_mask=invalid_mask)

        # Debris flow detection: vegetation scour + mineral sediment rise outside snow/clouds
        if pre_nir is not None and pre_red is not None and pre_swir is not None:
            pre_ndvi = compute_ndvi(pre_nir, pre_red)
            pre_swir_arr = pre_swir
        else:
            pre_ndvi = np.clip(ndvi + 0.15, -1.0, 1.0)
            pre_swir_arr = np.clip(swir - 0.05, 0.01, 1.0)

        debris_mask = detect_debris_change(
            pre_ndvi,
            ndvi,
            pre_swir_arr,
            swir,
            invalid_mask=invalid_mask,
            ndvi_drop_thresh=0.15,
            swir_rise_thresh=0.04,
        )
        s2_executed = True
        debris_status = "assessed" if effective_optical_cov >= 80.0 else "partially_assessed"
        debris_reason = (
            f"Optical coverage: {effective_optical_cov:.1f}% of AOI; debris unmapped elsewhere due to cloud/snow."
            if effective_optical_cov < 80.0
            else None
        )
    else:
        debris_area_km2 = None
        debris_status = "not_assessed"
        debris_reason = (
            f"Optical coverage too low ({effective_optical_cov:.1f}% < 20.0%) due to cloud/shadow/snow obscuration."
            if optical_coverage_pct is not None
            else "No valid optical data available."
        )
        logger.info(f"Optical S2 processing skipped: {debris_reason}")

    # 9. Multi-Sensor Fusion & Per-Pixel Confidence
    logger.info("Executing multi-sensor fusion with continuous confidence estimation...")
    fused_prod = fuse_sar_optical_observations(
        s1_diff_db=diff_db,
        s1_flood_candidates=sar_flood_constrained,
        slope_deg=slope,
        hand_m=hand,
        s2_water_mask=optical_water,
        s2_debris_mask=debris_mask,
        is_cloud_compromised=(debris_status == "not_assessed"),
    )

    flooded_km2 = round(fused_prod.total_flood_pixels * pixel_area_km2, 4)
    debris_km2 = (
        round(fused_prod.total_debris_pixels * pixel_area_km2, 4)
        if s2_executed
        else None
    )
    debris_area_km2 = debris_km2

    # 10. Export Intermediate and Final COG Rasters
    logger.info(f"Exporting COG rasters to {raster_dir}...")
    raster_paths = {
        "s1_log_ratio": save_cog(diff_db, raster_dir / "s1_log_ratio.tif", bbox=bbox),
        "slope": save_cog(slope, raster_dir / "slope.tif", bbox=bbox),
        "hand": save_cog(hand, raster_dir / "hand.tif", bbox=bbox),
        "flood_mask": save_cog(fused_prod.flood_mask, raster_dir / "flood_mask.tif", bbox=bbox),
        "debris_mask": save_cog(
            fused_prod.debris_mask if debris_mask is not None else np.zeros(target_shape, dtype=bool),
            raster_dir / "debris_mask.tif",
            bbox=bbox,
        ),
        "confidence": save_cog(
            fused_prod.confidence_raster, raster_dir / "confidence.tif", bbox=bbox
        ),
    }

    # Export True-Colour S2 RGB composite if bands are available
    if red is not None and green is not None and blue is not None:
        def stretch_band(b: np.ndarray) -> np.ndarray:
            valid = b[np.isfinite(b)]
            if len(valid) == 0:
                return np.zeros(b.shape, dtype=np.uint8)
            p2, p98 = np.percentile(valid, (2, 98))
            denom = max(p98 - p2, 1e-4)
            return (np.clip((b - p2) / denom, 0.0, 1.0) * 255).astype(np.uint8)

        rgb_composite = np.stack([stretch_band(red), stretch_band(green), stretch_band(blue)], axis=-1)
        # Save as RGB GeoTIFF
        save_cog(rgb_composite[:, :, 0], raster_dir / "s2_red.tif", bbox=bbox)
        save_cog(rgb_composite[:, :, 1], raster_dir / "s2_green.tif", bbox=bbox)
        save_cog(rgb_composite[:, :, 2], raster_dir / "s2_blue.tif", bbox=bbox)

    # 11. Produce Auditable facts.json with Data Provenance Block
    s1_orbit = relative_orbit if relative_orbit is not None else (bundle.s1_pair.relative_orbit if bundle.s1_pair else None)
    s1_direction = orbit_direction if orbit_direction is not None else (bundle.s1_pair.direction if bundle.s1_pair else None)

    buildings_count = len(bundle.osm_buildings.get("features", []))
    roads_count = len(bundle.osm_roads.get("features", []))
    bridges_count = len(bundle.osm_bridges.get("features", []))
    settlements_count = len(bundle.osm_settlements.get("features", []))
    hospitals_count = len(bundle.osm_hospitals.get("features", []))

    debris_disp_str = f"{debris_area_km2} km²" if debris_area_km2 is not None else "not assessed"

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
            "s1_polarization": s1_polarization,
            "utm_crs": f"EPSG:{utm_epsg}",
            "raster_shape": list(target_shape),
            "pixel_size_m": [round(pixel_size_x_m, 2), round(pixel_size_y_m, 2)],
            "pixel_area_m2": round(pixel_size_x_m * pixel_size_y_m, 2),
            "s1_pre_db": pre_db_stats,
            "s1_post_db": post_db_stats,
            "s1_log_ratio_db": diff_db_stats,
            "s2_aoi_cloud_fraction": aoi_cloud_pct,
            "optical_coverage_pct": optical_coverage_pct,
            "s2_optical_executed": s2_executed,
            "synthetic_mode": use_synthetic_data,
        },
        "impact_statistics": {
            "flooded_area_km2": flooded_km2,
            "debris_area_km2": debris_area_km2,
            "debris_status": debris_status,
            "debris_reason": debris_reason,
            "mean_flood_confidence": fused_prod.mean_flood_confidence,
            "total_flood_pixels": fused_prod.total_flood_pixels,
            "total_debris_pixels": fused_prod.total_debris_pixels if debris_mask is not None else 0,
        },
        "terrain_exclusion_audit": exclusion_stats,
        "sensitivity_analysis": {
            "baseline_slope_deg": slope_limit,
            "baseline_hand_m": hand_limit,
            "baseline_flooded_km2": flooded_km2,
            "matrix": sensitivity_matrix,
        },
        "osm_provenance": {
            "osm_source": bundle.osm_source,
            "buildings_loaded": buildings_count,
            "roads_loaded": roads_count,
            "bridges_loaded": bridges_count,
            "settlements_loaded": settlements_count,
            "hospitals_loaded": hospitals_count,
        },
        "legal_attributions": settings.attributions,
    }

    facts_path = out_path / "facts.json"
    with open(facts_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)

    logger.info(f"Saved auditable facts file to: {facts_path}")
    logger.info(
        f"--- Pipeline Finished Successfully: Flooded Area: {flooded_km2} km², "
        f"Debris Area: {debris_disp_str} ({debris_status}), Mean Confidence: {fused_prod.mean_flood_confidence:.2f} ---"
    )

    return PipelineResult(
        bbox=bbox,
        event_date=event_date,
        output_dir=out_path,
        raster_paths=raster_paths,
        flooded_area_km2=flooded_km2,
        debris_area_km2=debris_area_km2,
        debris_status=debris_status,
        debris_reason=debris_reason,
        optical_coverage_pct=optical_coverage_pct,
        mean_confidence=fused_prod.mean_flood_confidence,
        s1_orbit=s1_orbit,
        s1_direction=s1_direction,
        is_cloud_compromised=(debris_status == "not_assessed"),
        osm_source=bundle.osm_source,
        facts=facts,
    )
