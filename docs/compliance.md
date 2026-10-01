# Compliance & Provenance Framework

## 1. Allowed vs. Prohibited Inputs

### Allowed Inputs
- **Sentinel-1 GRD / SLC**: CDSE, Planetary Computer, Earth Search.
- **Sentinel-2 L2A**: CDSE, Planetary Computer, Earth Search.
- **Copernicus DEM (GLO-30)**: AWS Open Data (`s3://copernicus-dem-30m`) and Planetary Computer (`cop-dem-glo-30`).
- **OpenStreetMap History**: Sourced via `ohsome` API strictly pinned to `timestamp <= 2026-07-27`.
- **Training Datasets**: Kuro Siwo (Bountos et al., NeurIPS 2024), Sen1Floods11 (Bonafilia et al., 2020).

### Explicitly Prohibited as Runtime Inputs
- WorldPop
- Global Human Settlement Layer (GHSL)
- JRC Global Surface Water (GSW)
- ESA WorldCover
- Dynamic World
- OPERA DSWx
- GloFAS / Flood hydrographs
- ICIMOD flood layers
- Copernicus Emergency Management Service (CEMS / EMSR) or UNOSAT products (Reserved for isolated evaluation only)
- Post-event OpenStreetMap updates (timestamp > 2026-07-27)

## 2. Attribution Requirements
All outputs, maps, sitreps, and documentation must display the following legal notices:
- "Contains modified Copernicus Sentinel data [year]."
- "Produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."
- "(c) OpenStreetMap contributors."
- Citations: Bountos et al. 2024 (Kuro Siwo), Bonafilia et al. 2020 (Sen1Floods11).
