"""Real satellite raster ingestion using odc-stac and Planetary Computer."""

from dataclasses import dataclass

import numpy as np
import odc.stac
import planetary_computer
import pystac_client

from floodmap.config import settings
from floodmap.data.s1_selection import S1Pair
from floodmap.data.s2_selection import S2Pair, compute_aoi_cloud_fraction
from floodmap.logger import get_logger

logger = get_logger("floodmap.data.raster_loader")


@dataclass
class LoadedRasterBundle:
    """Complete bundle of real satellite raster arrays clipped to AOI."""

    dem: np.ndarray
    s1_pre_vh: np.ndarray  # Linear intensity (NaN for nodata)
    s1_post_vh: np.ndarray  # Linear intensity (NaN for nodata)
    s1_pre_vv: np.ndarray | None  # Linear intensity (NaN for nodata)
    s1_post_vv: np.ndarray | None  # Linear intensity (NaN for nodata)
    s1_pre: np.ndarray  # Primary or fused linear backscatter
    s1_post: np.ndarray
    blue: np.ndarray | None
    green: np.ndarray | None
    swir: np.ndarray | None
    nir: np.ndarray | None
    red: np.ndarray | None
    scl: np.ndarray | None
    pre_green: np.ndarray | None = None
    pre_swir: np.ndarray | None = None
    pre_nir: np.ndarray | None = None
    pre_red: np.ndarray | None = None
    pre_scl: np.ndarray | None = None
    aoi_cloud_fraction: float | None = None
    optical_coverage_pct: float | None = None
    is_cloud_compromised: bool = True
    shape: tuple[int, int] = (0, 0)
    s1_polarization: str = "VH"
    relative_orbit: int | None = None
    orbit_direction: str | None = None


def clean_linear_sar(arr: np.ndarray) -> np.ndarray:
    """Masks zeros, negative numbers, and non-finite values as NaN BEFORE dB conversion."""
    a = arr.astype(np.float32)
    return np.where(np.isfinite(a) & (a > 0.0), a, np.nan)


def load_real_satellite_bundle(
    bbox: tuple[float, float, float, float],
    s1_pair: S1Pair | None,
    s2_pair: S2Pair | None,
    resolution_deg: float = 0.0003,  # ~30 meters at equator
) -> LoadedRasterBundle:
    """Loads real Sentinel-1 RTC (VH & VV), Sentinel-2 L2A, and Copernicus DEM pixels via odc-stac.

    Fails loudly with RuntimeError if real satellite pixels cannot be loaded.
    Masks nodata as NaN before decibel conversion.
    """
    if s1_pair is None:
        raise RuntimeError("Cannot load real satellite data: No valid Sentinel-1 pair provided!")

    logger.info(
        f"Loading real satellite pixels via Planetary Computer STAC for AOI {bbox} | "
        f"S1 Pre: {s1_pair.pre_scene.item_id} | S1 Post: {s1_pair.post_scene.item_id}"
    )

    try:
        catalog = pystac_client.Client.open(
            settings.planetary_computer_stac_url,
            modifier=planetary_computer.sign_inplace,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to Planetary Computer STAC: {exc}") from exc

    # 1. Load Sentinel-1 RTC Dual-Pol (VH and VV)
    try:
        s1_search = catalog.search(
            collections=["sentinel-1-rtc"],
            ids=[s1_pair.pre_scene.item_id, s1_pair.post_scene.item_id],
        )
        s1_items = list(s1_search.items())
        if len(s1_items) < 2:
            pre_date_str = s1_pair.pre_scene.datetime.strftime("%Y-%m-%d")
            post_date_str = s1_pair.post_scene.datetime.strftime("%Y-%m-%d")
            s1_search = catalog.search(
                collections=["sentinel-1-rtc"],
                bbox=list(bbox),
                datetime=f"{pre_date_str}/{post_date_str}",
            )
            s1_items = list(s1_search.items())

        if not s1_items:
            raise RuntimeError(
                f"No Sentinel-1 RTC STAC items found on Planetary Computer for IDs: "
                f"{s1_pair.pre_scene.item_id}, {s1_pair.post_scene.item_id}"
            )

        # Extract relative orbit and direction directly from product metadata
        item_props = s1_items[0].properties
        relative_orbit = item_props.get("sat:relative_orbit") or s1_pair.pre_scene.relative_orbit
        orbit_direction = item_props.get("sat:orbit_state") or s1_pair.pre_scene.flight_direction

        bands_to_load = ["vh", "vv"]
        ds_s1 = odc.stac.load(
            s1_items,
            bands=bands_to_load,
            bbox=list(bbox),
            resolution=resolution_deg,
            crs="EPSG:4326",
            chunks={},
        )
        logger.info(f"Loaded Sentinel-1 RTC xarray: {ds_s1.dims} | Relative Orbit: {relative_orbit}")

        s1_times = ds_s1.time.values
        if len(s1_times) < 2:
            raise RuntimeError(
                f"Expected at least 2 time slices for Sentinel-1 pair, but received {len(s1_times)}"
            )

        s1_pre_vh = clean_linear_sar(ds_s1["vh"].values[0])
        s1_post_vh = clean_linear_sar(ds_s1["vh"].values[-1])

        if "vv" in ds_s1:
            s1_pre_vv = clean_linear_sar(ds_s1["vv"].values[0])
            s1_post_vv = clean_linear_sar(ds_s1["vv"].values[-1])
        else:
            s1_pre_vv = s1_post_vv = None

    except Exception as exc:
        raise RuntimeError(f"Failed to load real Sentinel-1 RTC pixels via odc-stac: {exc}") from exc

    target_shape = s1_pre_vh.shape

    # 2. Load Copernicus GLO-30 DEM
    try:
        dem_search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=list(bbox),
        )
        dem_items = list(dem_search.items())
        if not dem_items:
            raise RuntimeError(f"No Copernicus GLO-30 DEM items found for AOI {bbox}")

        ds_dem = odc.stac.load(
            dem_items,
            bands=["data"],
            like=ds_s1.odc.geobox,
            chunks={},
        )
        dem_array = ds_dem["data"].values[0].astype(np.float32)
        dem_array = np.where(np.isfinite(dem_array), dem_array, np.nanmin(dem_array))
        logger.info(
            f"Loaded Copernicus DEM: min={float(np.min(dem_array)):.1f}m, "
            f"max={float(np.max(dem_array)):.1f}m, mean={float(np.mean(dem_array)):.1f}m"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load Copernicus GLO-30 DEM pixels via odc-stac: {exc}") from exc

    # 3. Load Sentinel-2 Optical Bands & SCL (Pre & Post)
    blue = green = swir = nir = red = scl = None
    pre_green = pre_swir = pre_nir = pre_red = pre_scl = None
    aoi_cloud_pct = None
    optical_coverage_pct = None
    is_cloud_compromised = True

    if s2_pair is not None:
        try:
            s2_ids = [s2_pair.post_scene.item_id]
            if s2_pair.pre_scene and s2_pair.pre_scene.item_id != s2_pair.post_scene.item_id:
                s2_ids = [s2_pair.pre_scene.item_id, s2_pair.post_scene.item_id]

            s2_search = catalog.search(
                collections=["sentinel-2-l2a"],
                ids=s2_ids,
            )
            s2_items = list(s2_search.items())
            if s2_items:
                ds_s2 = odc.stac.load(
                    s2_items,
                    bands=["B02", "B03", "B04", "B08", "B11", "SCL"],
                    like=ds_s1.odc.geobox,
                    chunks={},
                )
                # If both pre and post loaded
                if len(ds_s2.time.values) >= 2:
                    pre_green = (ds_s2["B03"].values[0] / 10000.0).astype(np.float32)
                    pre_swir = (ds_s2["B11"].values[0] / 10000.0).astype(np.float32)
                    pre_nir = (ds_s2["B08"].values[0] / 10000.0).astype(np.float32)
                    pre_red = (ds_s2["B04"].values[0] / 10000.0).astype(np.float32)
                    pre_scl = ds_s2["SCL"].values[0].astype(np.uint8)

                blue = (ds_s2["B02"].values[-1] / 10000.0).astype(np.float32)
                green = (ds_s2["B03"].values[-1] / 10000.0).astype(np.float32)
                red = (ds_s2["B04"].values[-1] / 10000.0).astype(np.float32)
                nir = (ds_s2["B08"].values[-1] / 10000.0).astype(np.float32)
                swir = (ds_s2["B11"].values[-1] / 10000.0).astype(np.float32)
                scl = ds_s2["SCL"].values[-1].astype(np.uint8)

                # Invalid classes: 0 (nodata), 1 (saturated), 3 (cloud shadow),
                # 8 (med prob cloud), 9 (high prob cloud), 10 (cirrus), 11 (snow/ice)
                invalid_post = np.isin(scl, (0, 1, 3, 8, 9, 10, 11))
                if pre_scl is not None:
                    invalid_pre = np.isin(pre_scl, (0, 1, 3, 8, 9, 10, 11))
                    combined_invalid = invalid_post | invalid_pre
                else:
                    combined_invalid = invalid_post

                usable_pixels = int(np.count_nonzero(~combined_invalid))
                optical_coverage_pct = round(float((usable_pixels / max(scl.size, 1)) * 100.0), 2)
                aoi_cloud_pct = compute_aoi_cloud_fraction(scl)

                if optical_coverage_pct >= 20.0:
                    logger.info(
                        f"Sentinel-2 optical coverage usable: {optical_coverage_pct:.1f}% of AOI is clear "
                        f"(>= 20.0%). Optical water & debris detection activated with per-pixel SCL masking."
                    )
                    is_cloud_compromised = False
                else:
                    logger.warning(
                        f"Sentinel-2 usable optical coverage too low ({optical_coverage_pct:.1f}% < 20.0%) "
                        f"due to cloud/snow obscuration. Debris assessment skipped."
                    )
                    is_cloud_compromised = True
        except Exception as exc:
            logger.warning(f"Could not load real Sentinel-2 optical bands: {exc}. Proceeding S1-only.")

    return LoadedRasterBundle(
        dem=dem_array,
        s1_pre_vh=s1_pre_vh,
        s1_post_vh=s1_post_vh,
        s1_pre_vv=s1_pre_vv,
        s1_post_vv=s1_post_vv,
        s1_pre=s1_pre_vh,
        s1_post=s1_post_vh,
        blue=blue,
        green=green,
        swir=swir,
        nir=nir,
        red=red,
        scl=scl,
        pre_green=pre_green,
        pre_swir=pre_swir,
        pre_nir=pre_nir,
        pre_red=pre_red,
        pre_scl=pre_scl,
        aoi_cloud_fraction=aoi_cloud_pct,
        optical_coverage_pct=optical_coverage_pct,
        is_cloud_compromised=is_cloud_compromised,
        shape=target_shape,
        s1_polarization="VH+VV" if s1_pre_vv is not None else "VH",
        relative_orbit=int(relative_orbit) if relative_orbit is not None else None,
        orbit_direction=str(orbit_direction) if orbit_direction is not None else None,
    )
