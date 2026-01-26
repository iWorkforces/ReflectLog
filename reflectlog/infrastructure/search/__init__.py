"""Search engine implementations for ReflectLogMCP.

This subpackage contains implementations of search engine backends.
Each backend implements the ISearchBackend protocol from core.search.

Note: The actual engine implementations are in the parent infrastructure/
directory. This module re-exports them for backward compatibility.
"""

# Re-export from parent module for backward compatibility
from reflectlog.infrastructure.tantivy_engine import TantivyConfig, TantivyEngine
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine

__all__ = [
    "USearchEngine",
    "USearchConfig",
    "TantivyEngine",
    "TantivyConfig",
]
