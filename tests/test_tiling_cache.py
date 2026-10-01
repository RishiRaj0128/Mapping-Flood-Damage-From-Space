"""Tests for spatial tiling and disk caching."""

from pathlib import Path

from floodmap.cache import DiskCache
from floodmap.data.tiling import ProgressReporter, compute_tiles_for_bbox


def test_compute_tiles_for_bbox():
    """Tiling must subdivide a large bounding box into overlapping tiles."""
    # 0.5 deg x 0.5 deg area with 0.25 deg tiles -> 2x2 = 4 tiles
    bbox = (85.0, 27.0, 85.5, 27.5)
    tiles = compute_tiles_for_bbox(bbox, max_tile_size_deg=0.25, overlap_deg=0.01)

    assert len(tiles) == 4
    for tile in tiles:
        assert tile.total_cols == 2
        assert tile.total_rows == 2
        # Buffered bbox must be wider than base bbox
        assert tile.buffered_bbox[0] < tile.bbox[0]
        assert tile.buffered_bbox[2] > tile.bbox[2]


def test_disk_cache_roundtrip(tmp_path: Path):
    """DiskCache must store, retrieve, and isolate cache keys."""
    cache = DiskCache(cache_dir=tmp_path)
    namespace = "test_space"
    params = {"bbox": [85.0, 27.0, 85.5, 27.5], "mode": "fast"}
    payload = {"data": [1, 2, 3], "status": "ok"}

    assert cache.get_json(namespace, params) is None

    saved_path = cache.put_json(namespace, params, payload)
    assert saved_path.exists()

    retrieved = cache.get_json(namespace, params)
    assert retrieved == payload


def test_progress_reporter():
    """ProgressReporter tracks current step and fires callbacks."""
    steps_recorded = []

    def callback(curr, total, desc):
        steps_recorded.append((curr, total, desc))

    reporter = ProgressReporter(total_steps=3, task_name="TestTask", callback=callback)
    reporter.step("Step 1")
    reporter.step("Step 2")
    reporter.step("Step 3")

    assert reporter.current_step == 3
    assert len(steps_recorded) == 3
    assert steps_recorded[0] == (1, 3, "Step 1")
    assert steps_recorded[-1] == (3, 3, "Step 3")
