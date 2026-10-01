# Architecture Overview: Mapping Flood Damage from Space

## System Pipeline

```mermaid
graph TD
    A[Inputs: Bounding Box + Event Date] --> B[Compliance Guards]
    B --> C{Data Ingestion Layer}
    C -->|STAC Search & Filter| D1[Sentinel-1 Pre/Post Pair]
    C -->|STAC Cloud Aware| D2[Sentinel-2 Pre/Post Pair]
    C -->|AWS Open Data / STAC| D3[Copernicus GLO-30 DEM]
    C -->|ohsome API Guarded <= 2026-07-27| D4[Pre-Event OSM Snapshot]
    D1 --> E[SAR Processing & Change Detection]
    D2 --> F[Optical Water/Debris Indices]
    D3 --> G[Terrain, Slope & HAND Constraints]
    E & F & G --> H[Fused Flood & Debris Mask with Confidence]
    H & D4 --> I[Damage Assessment: Buildings, Roads, Bridges]
    I & D4 --> J[Cut-Off Network Connectivity Analysis]
    H & J --> K[Bilingual Sitrep EN/NE & Interactive Dashboard]
```

## Hard Invariants
1. **Input Restrictions**: Only Sentinel-1, Sentinel-2, Copernicus DEM (GLO-30), OSM (<= 2026-07-27), and designated datasets (Kuro Siwo, Sen1Floods11).
2. **Evaluation Isolation**: Reference validation (EMSR927) is strictly isolated within `evaluation/` and never imported by runtime packages.
3. **Same-Orbit Rule**: Sentinel-1 pre/post pairs must share the exact same relative orbit and flight direction. Cross-track pairs are hard-rejected.
4. **Terminology**: Only "exposed" or "likely hit" classifications with explicit confidence tiers. Never "destroyed".
5. **Auditable Numbers**: Every statistic in sitrep and outputs derives directly from structured `facts.json`.
