"""Sentinel-2 optical pair selection with cloud awareness and SCL inspection."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from floodmap.logger import get_logger

logger = get_logger("floodmap.data.s2")

# Sentinel-2 Scene Classification Layer (SCL) Cloud & Shadow Class IDs
# 3: Cloud shadow, 8: Cloud medium prob, 9: Cloud high prob, 10: Thin cirrus
SCL_CLOUD_SHADOW_CLASSES = (3, 8, 9, 10)


def compute_scl_cloud_mask(scl_array: np.ndarray) -> np.ndarray:
    """Computes binary cloud/shadow mask from Sentinel-2 SCL band.

    Returns boolean array where True indicates obscured by cloud, shadow, or cirrus.
    """
    return np.isin(scl_array, SCL_CLOUD_SHADOW_CLASSES)


def compute_aoi_cloud_fraction(scl_array: np.ndarray) -> float:
    """Computes exact percentage (0-100%) of AOI pixels obscured by clouds and shadows."""
    valid_mask = scl_array != 0  # 0 is NO_DATA
    if not np.any(valid_mask):
        return 100.0
    cloud_mask = compute_scl_cloud_mask(scl_array) & valid_mask
    cloud_pct = (np.count_nonzero(cloud_mask) / np.count_nonzero(valid_mask)) * 100.0
    return float(cloud_pct)


@dataclass
class S2SceneMetadata:
    """Metadata for an individual Sentinel-2 L2A optical scene."""

    item_id: str
    datetime: datetime
    cloud_cover: float  # Percentage (0 - 100) from scene metadata
    has_scl: bool  # Scene Classification Layer available
    assets: dict[str, str]
    raw_properties: dict[str, Any]
    aoi_cloud_cover: float | None = None  # Exact AOI cloud fraction from SCL band
    bbox: tuple[float, float, float, float] | None = None
    geometry: dict[str, Any] | None = None

    @classmethod
    def from_stac_item(cls, item: dict[str, Any]) -> "S2SceneMetadata":
        props = item.get("properties", {})
        item_id = item.get("id", "")

        dt_str = props.get("datetime") or item.get("datetime")
        if dt_str:
            clean_dt = dt_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_dt)
        else:
            dt = datetime.now(UTC)

        cloud_cover = float(props.get("eo:cloud_cover") or props.get("cloud_cover") or 100.0)

        assets = {
            k: v.get("href", "") for k, v in item.get("assets", {}).items() if isinstance(v, dict)
        }
        has_scl = "scl" in assets or "SCL" in assets or any("scl" in k.lower() for k in assets)

        item_bbox = tuple(item["bbox"]) if item.get("bbox") and len(item["bbox"]) == 4 else None
        item_geom = item.get("geometry")

        return cls(
            item_id=item_id,
            datetime=dt,
            cloud_cover=cloud_cover,
            has_scl=has_scl,
            assets=assets,
            raw_properties=props,
            bbox=item_bbox,
            geometry=item_geom,
        )

    def set_aoi_cloud_from_scl(self, scl_array: np.ndarray) -> float:
        """Sets the exact AOI cloud fraction by inspecting the SCL raster array."""
        self.aoi_cloud_cover = compute_aoi_cloud_fraction(scl_array)
        return self.aoi_cloud_cover


@dataclass
class S2Pair:
    """Selected pre-event and post-event Sentinel-2 optical pair."""

    pre_scene: S2SceneMetadata
    post_scene: S2SceneMetadata
    is_cloud_compromised: bool
    aoi_cloud_cover_post: float | None = None


def select_best_s2_pair(
    candidates: list[S2SceneMetadata],
    event_date: datetime,
    max_cloud_threshold: float = 40.0,
) -> S2Pair | None:
    """Selects the clearest optical pre and post pair relative to the event date.

    If cloud cover exceeds threshold for post-event scene, marks the pair as cloud compromised
    so the pipeline gracefully falls back to S1-dominant processing.
    """
    if not candidates:
        logger.warning("No Sentinel-2 candidate scenes found.")
        return None

    posts = [c for c in candidates if c.datetime >= event_date]
    pres = [c for c in candidates if c.datetime < event_date]

    if not posts or not pres:
        logger.warning("Missing pre or post Sentinel-2 scenes.")
        return None

    # Sort candidates by combined score: low cloud cover (preferring AOI-specific) + temporal closeness
    def score_scene(s: S2SceneMetadata) -> float:
        days_diff = abs((s.datetime - event_date).total_seconds()) / 86400.0
        effective_cloud = s.aoi_cloud_cover if s.aoi_cloud_cover is not None else s.cloud_cover
        return (effective_cloud * 1.5) + (days_diff * 0.5)

    pres.sort(key=score_scene)
    posts.sort(key=score_scene)

    best_pre = pres[0]
    best_post = posts[0]

    pre_cloud = best_pre.aoi_cloud_cover if best_pre.aoi_cloud_cover is not None else best_pre.cloud_cover
    post_cloud = best_post.aoi_cloud_cover if best_post.aoi_cloud_cover is not None else best_post.cloud_cover

    is_cloudy = post_cloud > max_cloud_threshold or pre_cloud > max_cloud_threshold

    cloud_type_str = "AOI cloud" if best_post.aoi_cloud_cover is not None else "scene cloud"
    if is_cloudy:
        logger.warning(
            f"Sentinel-2 post scene has high {cloud_type_str} ({post_cloud:.1f}%). "
            f"Flagging for graceful S1-dominant fallback."
        )
    else:
        logger.info(
            f"Selected Sentinel-2 optical pair | "
            f"Pre: {best_pre.datetime.strftime('%Y-%m-%d')} ({pre_cloud:.1f}% cloud) | "
            f"Post: {best_post.datetime.strftime('%Y-%m-%d')} ({post_cloud:.1f}% cloud)"
        )

    return S2Pair(
        pre_scene=best_pre,
        post_scene=best_post,
        is_cloud_compromised=is_cloudy,
        aoi_cloud_cover_post=best_post.aoi_cloud_cover,
    )
