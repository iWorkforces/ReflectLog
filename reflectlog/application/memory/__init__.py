'''Memory management module for ReflectLogMCP Server.'''

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
from .search_strategies import SearchContext, SearchPipeline, SearchResult

__all__ = [
    'AddPipeline',
    'AddResult',
    'DuplicateDetectionPhase',
    'FusionEngine',
    'MemoryManager',
    'Phase1Result',
    'Phase2Result',
    'Phase3Result',
    'RanxFusionEngine',
    'ReplacementInfo',
    'SearchContext',
    'SearchEngine',
    'SearchPipeline',
    'SearchResult',
    'SmartReplacementPhase',
    'StoragePhase',
    'create_fusion_engine',
]
