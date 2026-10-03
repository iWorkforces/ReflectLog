"""Immutable, privacy-safe search failure snapshots."""

from dataclasses import dataclass

from reflectlog.core.enums import SearchComponent


@dataclass(frozen=True)
class ComponentFailureSnapshot:
    """Cumulative failures and the latest failure's non-sensitive metadata."""

    count: int = 0
    last_failure_at: str | None = None
    exception_type: str | None = None


@dataclass(frozen=True)
class SearchFailureSnapshot:
    """Detached failure state for all four search components."""

    semantic: ComponentFailureSnapshot = ComponentFailureSnapshot()
    tantivy: ComponentFailureSnapshot = ComponentFailureSnapshot()
    cross_encoder: ComponentFailureSnapshot = ComponentFailureSnapshot()
    openrouter_reranker: ComponentFailureSnapshot = ComponentFailureSnapshot()

    def to_dict(self) -> dict[str, dict[str, int | str | None]]:
        """Return fresh dictionaries keyed by serialized component names."""
        return {
            component.value: {
                "count": snapshot.count,
                "last_failure_at": snapshot.last_failure_at,
                "exception_type": snapshot.exception_type,
            }
            for component, snapshot in (
                (SearchComponent.SEMANTIC, self.semantic),
                (SearchComponent.TANTIVY, self.tantivy),
                (SearchComponent.CROSS_ENCODER, self.cross_encoder),
                (SearchComponent.OPENROUTER_RERANKER, self.openrouter_reranker),
            )
        }
