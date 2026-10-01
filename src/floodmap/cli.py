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

    return parser


def parse_bbox_str(bbox_str: str):
    parts = [float(x.strip()) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("BBox must have 4 coordinates: min_lon,min_lat,max_lon,max_lat")
    return (parts[0], parts[1], parts[2], parts[3])


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command in ("demo", "ingest"):
        bbox = parse_bbox_str(args.bbox)
        logger.info(f"Executing {args.command} for bbox={bbox} on date={args.date}")
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
