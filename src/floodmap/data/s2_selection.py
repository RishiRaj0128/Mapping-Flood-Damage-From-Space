"""Sentinel-2 optical pair selection with cloud awareness and SCL inspection."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from floodmap.logger import get_logger

logger = get_logger("floodmap.data.s2")


@dataclass
class S2SceneMetadata:
    """Metadata for an individual Sentinel-2 L2A optical scene."""

    item_id: str
    datetime: datetime
    cloud_cover: float  # Percentage (0 - 100)
    has_scl: bool  # Scene Classification Layer available
    assets: dict[str, str]
    raw_properties: dict[str, Any]

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

        return cls(
            item_id=item_id,
            datetime=dt,
            cloud_cover=cloud_cover,
            has_scl=has_scl,
            assets=assets,
            raw_properties=props,
        )


@dataclass
class S2Pair:
    """Selected pre-event and post-event Sentinel-2 optical pair."""

    pre_scene: S2SceneMetadata
    post_scene: S2SceneMetadata
    is_cloud_compromised: bool


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

    # Sort candidates by combined score: low cloud cover + temporal closeness to event
    def score_scene(s: S2SceneMetadata) -> float:
        days_diff = abs((s.datetime - event_date).total_seconds()) / 86400.0
        # Weight cloud cover heavily, but prefer scenes within 30 days
        return (s.cloud_cover * 1.5) + (days_diff * 0.5)

    pres.sort(key=score_scene)
    posts.sort(key=score_scene)

    best_pre = pres[0]
    best_post = posts[0]

    is_cloudy = (
        best_post.cloud_cover > max_cloud_threshold or best_pre.cloud_cover > max_cloud_threshold
    )

    if is_cloudy:
        logger.warning(
            f"Sentinel-2 post scene has high cloud cover ({best_post.cloud_cover:.1f}%). "
            f"Flagging for graceful S1-dominant fallback."
        )
    else:
        logger.info(
            f"Selected Sentinel-2 optical pair | "
            f"Pre: {best_pre.datetime.strftime('%Y-%m-%d')} ({best_pre.cloud_cover:.1f}% cloud) | "
            f"Post: {best_post.datetime.strftime('%Y-%m-%d')} ({best_post.cloud_cover:.1f}% cloud)"
        )

    return S2Pair(
        pre_scene=best_pre,
        post_scene=best_post,
        is_cloud_compromised=is_cloudy,
    )
