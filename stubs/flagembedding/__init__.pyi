"""Type stubs for FlagEmbedding library.

This provides type hints for the FlagReranker class used in cross-encoder
reranking. Only APIs used by ReflectLog are stubbed.

Reference: https://github.com/FlagOpen/FlagEmbedding
"""

from typing import Protocol

class FlagRerankerProtocol(Protocol):
    """Protocol for FlagReranker model.

    FlagReranker provides cross-encoder reranking using BGE models.
    It jointly encodes query-document pairs for relevance scoring.

    Example:
        >>> reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        >>> scores = reranker.compute_score([("query", "doc1"), ("query", "doc2")])
    """

    def compute_score(
        self,
        sentence_pairs: list[tuple[str, str]],
        *,
        batch_size: int = 256,
        max_length: int = 512,
        normalize: bool = True,
    ) -> float | list[float]:
        """Compute relevance scores for query-document pairs.

        Args:
            sentence_pairs: List of (query, document) tuples to score.
            batch_size: Batch size for inference (default: 256).
            max_length: Maximum token length for pairs (default: 512).
            normalize: Apply sigmoid to normalize scores to 0-1 range (default: True).

        Returns:
            Single float score if one pair, or list of scores for multiple pairs.
            Scores are normalized to 0-1 range when normalize=True.
        """
        ...
