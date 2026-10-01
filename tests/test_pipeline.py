"""End-to-end integration tests for pipeline run(bbox, event_date)."""

import json
from pathlib import Path

from floodmap.pipeline import PipelineResult, run


def test_end_to_end_pipeline_trishuli(tmp_path: Path):
    """Executes the full pipeline on the Trishuli case study."""
    trishuli_bbox = (85.15, 27.85, 85.45, 28.15)
    event_date = "2026-08-26"
    out_dir = tmp_path / "trishuli_run"

    res = run(
        bbox=trishuli_bbox,
        event_date=event_date,
        output_dir=out_dir,
        use_synthetic_data=True,
    )

    assert isinstance(res, PipelineResult)
    assert res.bbox == trishuli_bbox
    assert res.event_date == event_date
    assert res.flooded_area_km2 > 0.0
    assert 0.0 <= res.mean_confidence <= 1.0

    # Verify all 6 COG rasters were saved
    expected_cogs = ["s1_log_ratio", "slope", "hand", "flood_mask", "debris_mask", "confidence"]
    for name in expected_cogs:
        assert name in res.raster_paths
        assert res.raster_paths[name].exists()
        assert res.raster_paths[name].stat().st_size > 0

    # Verify facts.json structure and data_provenance block
    facts_file = out_dir / "facts.json"
    assert facts_file.exists()
    with open(facts_file, encoding="utf-8") as f:
        facts_data = json.load(f)

    assert "fact_id" in facts_data
    assert "data_provenance" in facts_data
    prov = facts_data["data_provenance"]
    assert "utm_crs" in prov
    assert "raster_shape" in prov
    assert "pixel_size_m" in prov
    assert "pixel_area_m2" in prov
    assert "s1_pre_db" in prov
    assert "s1_post_db" in prov
    assert "s1_log_ratio_db" in prov
    assert "terrain_exclusion_audit" in facts_data
    assert len(facts_data["legal_attributions"]) >= 3


def test_end_to_end_pipeline_chamoli(tmp_path: Path):
    """Executes the pipeline on Chamoli 2021 disaster area to verify multi-event generalizability."""
    # Chamoli, Uttarakhand: 7 Feb 2021 rock/ice avalanche and debris flood
    chamoli_bbox = (79.55, 30.35, 79.85, 30.65)
    event_date = "2021-02-07"
    out_dir = tmp_path / "chamoli_run"

    res = run(
        bbox=chamoli_bbox,
        event_date=event_date,
        output_dir=out_dir,
        use_synthetic_data=True,
    )

    assert isinstance(res, PipelineResult)
    assert res.bbox == chamoli_bbox
    assert res.event_date == event_date
    # Must NEVER serve Trishuli cached fixture for Chamoli
    assert res.osm_source != "cached_fixture:trishuli"

    # COGs must be successfully generated
    assert (out_dir / "rasters" / "flood_mask.tif").exists()
    assert (out_dir / "rasters" / "confidence.tif").exists()
    assert (out_dir / "facts.json").exists()

    with open(out_dir / "facts.json", encoding="utf-8") as f:
        chamoli_facts = json.load(f)

    # Chamoli is in UTM Zone 44N (EPSG:32644) whereas Trishuli is Zone 45N (EPSG:32645)
    assert chamoli_facts["data_provenance"]["utm_crs"] == "EPSG:32644"
