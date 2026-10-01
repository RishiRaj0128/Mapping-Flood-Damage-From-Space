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
    s1_pre: np.ndarray  # Linear intensity
    s1_post: np.ndarray  # Linear intensity
    green: np.ndarray | None
    swir: np.ndarray | None
    nir: np.ndarray | None
    red: np.ndarray | None
    scl: np.ndarray | None
    aoi_cloud_fraction: float | None
    is_cloud_compromised: bool
    shape: tuple[int, int]
    s1_polarization: str


def load_real_satellite_bundle(
    bbox: tuple[float, float, float, float],
    s1_pair: S1Pair | None,
    s2_pair: S2Pair | None,
    resolution_deg: float = 0.0003,  # ~30 meters at equator
) -> LoadedRasterBundle:
    """Loads real Sentinel-1 RTC, Sentinel-2 L2A, and Copernicus DEM pixels via odc-stac.

    Fails loudly with RuntimeError if real satellite pixels cannot be loaded.
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

    # 1. Load Sentinel-1 RTC Pre and Post
    try:
        s1_search = catalog.search(
            collections=["sentinel-1-rtc"],
            ids=[s1_pair.pre_scene.item_id, s1_pair.post_scene.item_id],
            bbox=list(bbox),
        )
        s1_items = list(s1_search.items())
        if len(s1_items) < 2:
            # Fallback search by relative orbit and acquisition dates if specific IDs differ across catalogs
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

        # Prefer VH polarization for flood detection, fallback to VV
        pol = "vh" if "vh" in [p.lower() for p in s1_pair.pre_scene.polarizations] else "vv"
        ds_s1 = odc.stac.load(
            s1_items,
            bands=[pol],
            bbox=list(bbox),
            resolution=resolution_deg,
            crs="EPSG:4326",
            chunks={},
        )
        logger.info(f"Loaded Sentinel-1 RTC xarray: {ds_s1.dims}")

        # Separate pre and post by time
        s1_times = ds_s1.time.values
        if len(s1_times) < 2:
            raise RuntimeError(
                f"Expected at least 2 time slices for Sentinel-1 pair, but received {len(s1_times)}"
            )

        s1_array = ds_s1[pol].values
        # s1_times is sorted ascending: index 0 is pre, index -1 is post
        s1_pre = s1_array[0].astype(np.float32)
        s1_post = s1_array[-1].astype(np.float32)

        # Replace NaNs with local median or epsilon
        s1_pre = np.where(np.isfinite(s1_pre) & (s1_pre > 0), s1_pre, 1e-4)
        s1_post = np.where(np.isfinite(s1_post) & (s1_post > 0), s1_post, 1e-4)

    except Exception as exc:
        raise RuntimeError(f"Failed to load real Sentinel-1 RTC pixels via odc-stac: {exc}") from exc

    target_shape = s1_pre.shape

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

    # 3. Load Sentinel-2 Optical Bands & SCL
    green = swir = nir = red = scl = None
    aoi_cloud_pct = None
    is_cloud_compromised = True

    if s2_pair is not None:
        try:
            s2_search = catalog.search(
                collections=["sentinel-2-l2a"],
                ids=[s2_pair.post_scene.item_id],
                bbox=list(bbox),
            )
            s2_items = list(s2_search.items())
            if s2_items:
                ds_s2 = odc.stac.load(
                    s2_items,
                    bands=["B03", "B11", "B08", "B04", "SCL"],
                    like=ds_s1.odc.geobox,
                    chunks={},
                )
                green = (ds_s2["B03"].values[0] / 10000.0).astype(np.float32)
                swir = (ds_s2["B11"].values[0] / 10000.0).astype(np.float32)
                nir = (ds_s2["B08"].values[0] / 10000.0).astype(np.float32)
                red = (ds_s2["B04"].values[0] / 10000.0).astype(np.float32)
                scl = ds_s2["SCL"].values[0].astype(np.uint8)

                aoi_cloud_pct = compute_aoi_cloud_fraction(scl)
                if aoi_cloud_pct > 40.0:
                    logger.warning(
                        f"S2 skipped / degraded: AOI cloud fraction ({aoi_cloud_pct:.1f}%) "
                        f"exceeds threshold (40.0%). S1 SAR will be used as primary."
                    )
                    is_cloud_compromised = True
                else:
                    logger.info(
                        f"S2 optical loaded successfully: AOI cloud fraction is {aoi_cloud_pct:.1f}% "
                        f"(<= 40.0%). Optical water & debris detection activated."
                    )
                    is_cloud_compromised = False
        except Exception as exc:
            logger.warning(f"Could not load real Sentinel-2 optical bands: {exc}. Proceeding S1-only.")

    return LoadedRasterBundle(
        dem=dem_array,
        s1_pre=s1_pre,
        s1_post=s1_post,
        green=green,
        swir=swir,
        nir=nir,
        red=red,
        scl=scl,
        aoi_cloud_fraction=aoi_cloud_pct,
        is_cloud_compromised=is_cloud_compromised,
        shape=target_shape,
        s1_polarization=pol.upper(),
    )
