'''Reranking protocols for ReflectLogMCP.

This module defines protocols for relevance scoring and reranking operations.
Rerankers improve search result quality by re-scoring results using
additional signals like LLM relevance or cross-encoder similarity.
'''

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IRankingResult(Protocol):
    '''Result from a reranking operation.

    This class represents a reranked document with its relevance score
    and any additional metadata from the reranking process.
    '''

    content: str
    score: float
    metadata: dict[str, Any]


@runtime_checkable
class IReranker(Protocol):
    '''Protocol for relevance scoring implementations.

    This protocol defines the interface for reranking components that
    score documents by relevance to a query. Implementations can use
    local models, remote APIs, or hybrid approaches.

    Attributes:
        name: Reranker identifier for configuration.
    '''

    @property
    def name(self) -> str:
        '''Reranker identifier for configuration.'''
        ...

    async def rerank(
        self,
        query: str,
        documents: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[IRankingResult]:
        '''Score documents by relevance to query.

        Args:
            query: The search query.
            documents: List of documents to score.
            context: Optional additional context (e.g., timestamps).

        Returns:
            List of reranked results with scores.
        '''
        ...

    async def rerank_single(
        self,
        query: str,
        document: str,
        context: dict[str, Any] | None = None,
    ) -> IRankingResult:
        '''Score a single document by relevance.

        Args:
            query: The search query.
            document: Document to score.
            context: Optional additional context.

        Returns:
            Reranked result with score.
        '''
        ...

    async def close(self) -> None:
        '''Release resources.'''
        ...


@runtime_checkable
class IRerankerProvider(Protocol):
    '''Protocol for reranker provider implementations.

    This protocol defines the interface for providers that handle
    communication with reranking backends (LLM APIs, local models, etc.).

    Attributes:
        name: Provider identifier.
    '''

    @property
    def name(self) -> str:
        '''Provider identifier.'''
        ...

    async def score_batch(
        self,
        query: str,
        documents: list[str],
        prompt: str,
    ) -> list[float]:
        '''Score multiple documents efficiently.

        Args:
            query: The search query.
            documents: Documents to score.
            prompt: Scoring prompt to use.

        Returns:
            List of scores in same order as documents.
        '''
        ...

    async def score_single(
        self,
        query: str,
        document: str,
        prompt: str,
    ) -> float:
        '''Score a single document.

        Args:
            query: The search query.
            document: Document to score.
            prompt: Scoring prompt to use.

        Returns:
            Relevance score.
        '''
        ...


@runtime_checkable
class IRerankerConfig(Protocol):
    '''Protocol for reranker configuration.

    This protocol defines the configuration interface that rerankers
    use to configure themselves.
    '''

    @property
    def reranker_engine(self) -> str:
        '''Reranker engine type.'''
        ...

    @property
    def llm_model(self) -> str:
        '''LLM model name.'''
        ...

    @property
    def llm_api_base_url(self) -> str:
        '''API base URL.'''
        ...

    @property
    def cross_encoder_model(self) -> str:
        '''Cross-encoder model name.'''
        ...

    @property
    def cross_encoder_device(self) -> str:
        '''Cross-encoder device: cpu, cuda, mps.'''
        ...
