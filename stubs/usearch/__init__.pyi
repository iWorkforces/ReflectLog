"""Type stubs for usearch library."""

# Re-export index module for direct imports
from usearch.index import Index as Index
from usearch.index import Match as Match
from usearch.index import Matches as Matches

__all__ = ["Index", "Match", "Matches"]
