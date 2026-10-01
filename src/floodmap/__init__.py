"""Mapping Flood Damage from Space.

Multimodal satellite pipeline for mountain flood and debris mapping,
damage assessment, and cut-off settlement analysis.
"""

from floodmap.pipeline import PipelineResult, run

__version__ = "0.1.0"
__all__ = ["run", "PipelineResult"]
