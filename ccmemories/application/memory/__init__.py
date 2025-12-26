"""Memory management module for CCMemoriesMCP Server."""

from .add_phases import (
    AddPipeline,
    AddResult,
    DuplicateDetectionPhase,
    Phase1Result,
    Phase2Result,
    Phase3Result,
    ReplacementInfo,
    SmartReplacementPhase,
    StoragePhase,
)
from .fusion import FusionEngine, RanxFusionEngine, create_fusion_engine
from .manager import MemoryManager
from .protocols import SearchEngine
from .search_strategies import SearchContext, SearchResult, SearchPipeline

__all__ = [
    "AddResult",
    "MemoryManager",
    "ReplacementInfo",
    "SearchEngine",
    "FusionEngine",
    "RanxFusionEngine",
    "create_fusion_engine",
    "SearchContext",
    "SearchResult",
    "SearchPipeline",
    "AddPipeline",
    "DuplicateDetectionPhase",
    "SmartReplacementPhase",
    "StoragePhase",
    "Phase1Result",
    "Phase2Result",
    "Phase3Result",
]
