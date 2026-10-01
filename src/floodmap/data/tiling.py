"""Spatial partitioning, windowed reads, and progress reporting."""

import math
from collections.abc import Callable
from dataclasses import dataclass

from floodmap.logger import get_logger

logger = get_logger("floodmap.data.tiling")


@dataclass
class TileWindow:
    """Represents a spatial tile with coordinate bounds and grid indices."""

    col_idx: int
    row_idx: int
    total_cols: int
    total_rows: int
    bbox: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    buffered_bbox: tuple[float, float, float, float]
    overlap_deg: float

    @property
    def tile_id(self) -> str:
        return f"tile_c{self.col_idx:02d}_r{self.row_idx:02d}"


class ProgressReporter:
    """Simple progress tracker with logging and callback support."""

    def __init__(
        self,
        total_steps: int,
        task_name: str = "Processing",
        callback: Callable[[int, int, str], None] | None = None,
    ):
        self.total_steps = max(total_steps, 1)
        self.current_step = 0
        self.task_name = task_name
        self.callback = callback

    def step(self, description: str = "") -> None:
        """Advances progress by one step and invokes callbacks."""
        self.current_step += 1
        pct = (self.current_step / self.total_steps) * 100.0
        msg = f"[{self.task_name}] Step {self.current_step}/{self.total_steps} ({pct:.1f}%) - {description}"
        logger.info(msg)
        if self.callback:
            self.callback(self.current_step, self.total_steps, description)


def compute_tiles_for_bbox(
    bbox: tuple[float, float, float, float],
    max_tile_size_deg: float = 0.25,  # ~27km at equator
    overlap_deg: float = 0.02,  # ~2.2km overlap buffer
) -> list[TileWindow]:
    """Divides an arbitrary bounding box into overlapping tiles.

    Parameters:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        max_tile_size_deg: Maximum span of an individual tile in degrees
        overlap_deg: Margin added around each tile to avoid edge-effect artifacts
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    width = max_lon - min_lon
    height = max_lat - min_lat

    num_cols = max(1, math.ceil(width / max_tile_size_deg))
    num_rows = max(1, math.ceil(height / max_tile_size_deg))

    col_width = width / num_cols
    row_height = height / num_rows

    tiles: list[TileWindow] = []

    for r in range(num_rows):
        for c in range(num_cols):
            t_min_lon = min_lon + c * col_width
            t_max_lon = min_lon + (c + 1) * col_width
            t_min_lat = min_lat + r * row_height
            t_max_lat = min_lat + (r + 1) * row_height

            buffered_bbox = (
                t_min_lon - overlap_deg,
                t_min_lat - overlap_deg,
                t_max_lon + overlap_deg,
                t_max_lat + overlap_deg,
            )

            tiles.append(
                TileWindow(
                    col_idx=c,
                    row_idx=r,
                    total_cols=num_cols,
                    total_rows=num_rows,
                    bbox=(t_min_lon, t_min_lat, t_max_lon, t_max_lat),
                    buffered_bbox=buffered_bbox,
                    overlap_deg=overlap_deg,
                )
            )

    logger.info(f"Subdivided AOI {bbox} into {len(tiles)} tiles ({num_cols}x{num_rows}).")
    return tiles
