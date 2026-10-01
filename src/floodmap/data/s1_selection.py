"""Sentinel-1 SAR scene selection and same-orbit pair enforcement."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from floodmap.compliance.exceptions import CrossTrackPairError
from floodmap.logger import get_logger

logger = get_logger("floodmap.data.s1")


@dataclass
class S1SceneMetadata:
    """Metadata for an individual Sentinel-1 GRD/SLC scene."""

    item_id: str
    datetime: datetime
    relative_orbit: int
    direction: str  # 'ascending' or 'descending'
    polarizations: list[str]
    assets: dict[str, str]  # asset key -> href
    raw_properties: dict[str, Any]
    bbox: tuple[float, float, float, float] | None = None
    geometry: dict[str, Any] | None = None

    @classmethod
    def from_stac_item(cls, item: dict[str, Any]) -> "S1SceneMetadata":
        """Parses a STAC item dictionary into S1SceneMetadata."""
        props = item.get("properties", {})
        item_id = item.get("id", "")

        dt_str = props.get("datetime") or item.get("datetime")
        if dt_str:
            clean_dt = dt_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_dt)
        else:
            dt = datetime.now(UTC)

        # Standard STAC SAT & SAR extensions
        rel_orbit = (
            props.get("sat:relative_orbit")
            or props.get("s1:relative_orbit_number")
            or props.get("relative_orbit")
            or 0
        )
        direction = (
            props.get("sat:orbit_state")
            or props.get("s1:orbit_direction")
            or props.get("orbit_state")
            or "unknown"
        ).lower()

        pols = props.get("sar:polarizations") or props.get("polarization") or ["VV", "VH"]

        assets = {
            k: v.get("href", "") for k, v in item.get("assets", {}).items() if isinstance(v, dict)
        }

        item_bbox = tuple(item["bbox"]) if item.get("bbox") and len(item["bbox"]) == 4 else None
        item_geom = item.get("geometry")

        return cls(
            item_id=item_id,
            datetime=dt,
            relative_orbit=int(rel_orbit),
            direction=direction,
            polarizations=pols,
            assets=assets,
            raw_properties=props,
            bbox=item_bbox,
            geometry=item_geom,
        )


@dataclass
class S1Pair:
    """A verified, same-track Sentinel-1 pre-event and post-event pair."""

    pre_scene: S1SceneMetadata
    post_scene: S1SceneMetadata
    delta_days: float

    @property
    def relative_orbit(self) -> int:
        return self.pre_scene.relative_orbit

    @property
    def direction(self) -> str:
        return self.pre_scene.direction


def validate_s1_pair(pre_scene: S1SceneMetadata, post_scene: S1SceneMetadata) -> None:
    """Enforces strict same-orbit and same flight direction invariants.

    Raises CrossTrackPairError if orbits or flight directions do not match.
    """
    if pre_scene.relative_orbit != post_scene.relative_orbit:
        raise CrossTrackPairError(
            f"Sentinel-1 pair rejected: Incompatible relative orbits! "
            f"Pre-event orbit={pre_scene.relative_orbit} ({pre_scene.item_id}) vs "
            f"Post-event orbit={post_scene.relative_orbit} ({post_scene.item_id}). "
            f"Cross-track change detection is strictly invalid in complex mountain terrain."
        )

    if pre_scene.direction != post_scene.direction:
        raise CrossTrackPairError(
            f"Sentinel-1 pair rejected: Incompatible flight directions! "
            f"Pre-event direction={pre_scene.direction} vs "
            f"Post-event direction={post_scene.direction}. "
            f"Radar look geometry differs between ascending and descending passes."
        )

    if post_scene.datetime <= pre_scene.datetime:
        raise ValueError(
            f"Post-scene date ({post_scene.datetime.isoformat()}) must be after "
            f"pre-scene date ({pre_scene.datetime.isoformat()})."
        )


def check_scene_covers_aoi(
    scene: S1SceneMetadata,
    aoi_bbox: tuple[float, float, float, float] | None,
    min_overlap_ratio: float = 0.5,
) -> bool:
    """Verifies that a Sentinel-1 scene sufficiently covers the requested AOI bounding box."""
    if aoi_bbox is None:
        return True
    if scene.bbox is None and scene.geometry is None:
        return True

    from shapely.geometry import box, shape

    aoi_poly = box(*aoi_bbox)
    if scene.geometry:
        scene_poly = shape(scene.geometry)
    elif scene.bbox:
        scene_poly = box(*scene.bbox)
    else:
        return True

    if not scene_poly.intersects(aoi_poly):
        return False

    intersection_area = scene_poly.intersection(aoi_poly).area
    coverage_ratio = intersection_area / aoi_poly.area
    return coverage_ratio >= min_overlap_ratio


def select_best_s1_pair(
    candidates: list[S1SceneMetadata],
    event_date: datetime,
    preferred_repeat_days: int = 12,
    aoi_bbox: tuple[float, float, float, float] | None = None,
    min_aoi_overlap: float = 0.5,
) -> S1Pair | None:
    """Finds the optimal same-track pre/post pair around the event date.

    Prioritizes:
    1. Post-event scene strictly after event_date (post.datetime > event_date).
    2. Full spatial overlap of the AOI bounding box for both pre and post frames.
    3. Pre-event scene on the EXACT same relative orbit and direction,
       closest to 12 days prior (or multiples of 12 / 6 days).
    """
    if not candidates:
        logger.warning("No Sentinel-1 candidates available for pair selection.")
        return None

    # Separate into pre and post candidates (Strict inequality: post must be AFTER the event)
    posts = [
        c for c in candidates
        if c.datetime > event_date and check_scene_covers_aoi(c, aoi_bbox, min_aoi_overlap)
    ]
    pres = [
        c for c in candidates
        if c.datetime < event_date and check_scene_covers_aoi(c, aoi_bbox, min_aoi_overlap)
    ]

    if not posts:
        logger.warning(
            f"No valid post-event Sentinel-1 scenes found after event date: {event_date.isoformat()} "
            f"covering AOI {aoi_bbox}."
        )
        return None
    if not pres:
        logger.warning(
            f"No valid pre-event Sentinel-1 scenes found prior to event date: {event_date.isoformat()} "
            f"covering AOI {aoi_bbox}."
        )
        return None

    # Sort post scenes ascending by time from event date
    posts.sort(key=lambda s: abs((s.datetime - event_date).total_seconds()))

    best_pair: S1Pair | None = None
    best_score = float("inf")

    for post in posts:
        # Filter pre scenes matching exact relative orbit and direction
        matching_pres = [
            p
            for p in pres
            if p.relative_orbit == post.relative_orbit and p.direction == post.direction
        ]

        if not matching_pres:
            continue

        for pre in matching_pres:
            delta_days = (post.datetime - pre.datetime).total_seconds() / 86400.0
            # Score deviation from preferred repeat (e.g. 12 days)
            score = abs(delta_days - preferred_repeat_days)

            if score < best_score:
                best_score = score
                best_pair = S1Pair(
                    pre_scene=pre,
                    post_scene=post,
                    delta_days=round(delta_days, 2),
                )

    if best_pair:
        logger.info(
            f"Selected S1 Pair | Orbit: {best_pair.relative_orbit} ({best_pair.direction}) | "
            f"Pre: {best_pair.pre_scene.datetime.strftime('%Y-%m-%d')} | "
            f"Post: {best_pair.post_scene.datetime.strftime('%Y-%m-%d')} | "
            f"Delta: {best_pair.delta_days} days"
        )
        validate_s1_pair(best_pair.pre_scene, best_pair.post_scene)
        return best_pair

    logger.warning("No compatible same-track Sentinel-1 pair could be constructed.")
    return None
