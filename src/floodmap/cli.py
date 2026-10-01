"""Command line interface for the flood mapping pipeline."""

import argparse

from floodmap.config import settings
from floodmap.data.io import DataLoader
from floodmap.logger import get_logger

logger = get_logger("floodmap.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mapping Flood Damage from Space - Multimodal Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run Trishuli case study demonstration")
    demo_parser.add_argument(
        "--bbox",
        type=str,
        default=f"{settings.default_bbox[0]},{settings.default_bbox[1]},{settings.default_bbox[2]},{settings.default_bbox[3]}",
        help="Bounding box (min_lon,min_lat,max_lon,max_lat)",
    )
    demo_parser.add_argument(
        "--date",
        type=str,
        default=settings.default_event_date,
        help="Event date (YYYY-MM-DD)",
    )

    # Ingest / Data discovery command
    ingest_parser = subparsers.add_parser("ingest", help="Discover and ingest data for an AOI")
    ingest_parser.add_argument(
        "--bbox",
        type=str,
        required=True,
        help="Bounding box min_lon,min_lat,max_lon,max_lat",
    )
    ingest_parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Event date YYYY-MM-DD",
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run end-to-end flood mapping pipeline")
    run_parser.add_argument(
        "--bbox",
        type=str,
        required=True,
        help="Bounding box min_lon,min_lat,max_lon,max_lat",
    )
    run_parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Event date YYYY-MM-DD",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory",
    )

    return parser


def parse_bbox_str(bbox_str: str):
    parts = [float(x.strip()) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("BBox must have 4 coordinates: min_lon,min_lat,max_lon,max_lat")
    return (parts[0], parts[1], parts[2], parts[3])


def main():
    parser = build_parser()
    args = parser.parse_args()
    bbox = parse_bbox_str(args.bbox)

    if args.command in ("demo", "run"):
        from floodmap.pipeline import run
        output_dir = getattr(args, "output_dir", None)
        result = run(bbox=bbox, event_date=args.date, output_dir=output_dir)
        print("\n================ PIPELINE RESULT ================")
        print(f"AOI: {result.bbox}")
        print(f"Event Date: {result.event_date}")
        print(f"Flooded Area: {result.flooded_area_km2} km²")
        print(f"Debris Flow Area: {result.debris_area_km2} km²")
        print(f"Mean Flood Confidence: {result.mean_confidence:.2f}")
        print(f"OSM Source: {result.osm_source}")
        print(f"COGs Generated in: {result.output_dir / 'rasters'}")
        print(f"Facts File: {result.output_dir / 'facts.json'}")
        print("=================================================\n")

    elif args.command == "ingest":
        logger.info(f"Executing ingest for bbox={bbox} on date={args.date}")
        loader = DataLoader()
        bundle = loader.load_dataset_bundle(bbox=bbox, event_date_str=args.date)

        logger.info("=== Dataset Ingestion Summary ===")
        logger.info(f"S1 Pair: {bundle.s1_pair.pre_scene.item_id if bundle.s1_pair else 'None'}")
        logger.info(f"S2 Pair: {bundle.s2_pair.pre_scene.item_id if bundle.s2_pair else 'None'}")
        logger.info(f"Cloud Compromised: {bundle.is_cloud_compromised}")
        logger.info(f"Copernicus DEM Tiles: {len(bundle.dem_tiles)}")
        logger.info(
            f"OSM Buildings: {len(bundle.osm_buildings.get('features', []))}, "
            f"Roads: {len(bundle.osm_roads.get('features', []))}, "
            f"Bridges: {len(bundle.osm_bridges.get('features', []))}"
        )
        logger.info("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
