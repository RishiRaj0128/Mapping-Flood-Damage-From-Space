"""Tests for UTM projection selection and metric pixel area computations."""

from floodmap.data.geo import compute_projected_pixel_metrics, get_utm_epsg_for_bbox


def test_utm_epsg_determination():
    """Calculates correct UTM zone EPSG for Nepal (Zone 45N) and Uttarakhand (Zone 44N)."""
    # Nepal (Trishuli, lon ~85.3, lat ~28.0) -> Zone 45N -> EPSG:32645
    trishuli_bbox = (85.15, 27.85, 85.45, 28.15)
    epsg_nepal = get_utm_epsg_for_bbox(trishuli_bbox)
    assert epsg_nepal == 32645

    # Uttarakhand (Chamoli, lon ~79.7, lat ~30.5) -> Zone 44N -> EPSG:32644
    chamoli_bbox = (79.55, 30.35, 79.85, 30.65)
    epsg_chamoli = get_utm_epsg_for_bbox(chamoli_bbox)
    assert epsg_chamoli == 32644


def test_compute_projected_pixel_metrics():
    """Computes exact metric pixel dimensions and area in projected UTM coordinate system."""
    trishuli_bbox = (85.15, 27.85, 85.45, 28.15)
    # Approx 0.3 deg span (~30 km) with 1000x1000 raster -> ~30m resolution
    shape = (1000, 1000)
    utm_epsg, px_x, px_y, area_km2 = compute_projected_pixel_metrics(trishuli_bbox, shape)

    assert utm_epsg == 32645
    # Pixel resolution should be close to 30 meters
    assert 25.0 <= px_x <= 35.0
    assert 25.0 <= px_y <= 38.0
    # Single pixel area in km2: ~30m * 30m = 900 m2 = 0.0009 km2
    assert 0.0006 <= area_km2 <= 0.0013
