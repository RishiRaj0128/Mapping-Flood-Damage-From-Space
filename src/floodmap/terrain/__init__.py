"""Terrain and elevation constraints (slope, HAND, layover/shadow)."""

from floodmap.terrain.masks import (
    apply_terrain_exclusion,
    compute_hand,
    compute_layover_shadow_mask,
    compute_slope_degrees,
)

__all__ = [
    "compute_slope_degrees",
    "compute_layover_shadow_mask",
    "compute_hand",
    "apply_terrain_exclusion",
]
