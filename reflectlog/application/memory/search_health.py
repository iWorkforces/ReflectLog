"""Search failure accounting independent of manager and storage locks."""

from datetime import UTC, datetime
import threading

from reflectlog.core.enums import SearchComponent
from reflectlog.core.search_health import (
    ComponentFailureSnapshot,
    SearchFailureSnapshot,
)


class SearchFailureRecorder:
    """Accumulate failures under a private lock without retaining exceptions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._components = {
            component: ComponentFailureSnapshot() for component in SearchComponent
        }

    def record(self, component: SearchComponent, error: BaseException) -> None:
        """Count a failure, storing only its type name and an aware UTC time."""
        exception_type = type(error).__name__
        with self._lock:
            previous = self._components[component]
            self._components[component] = ComponentFailureSnapshot(
                count=previous.count + 1,
                last_failure_at=datetime.now(UTC).isoformat(),
                exception_type=exception_type,
            )

    def snapshot(self) -> SearchFailureSnapshot:
        """Copy the current immutable component states under the private lock."""
        with self._lock:
            return SearchFailureSnapshot(
                semantic=self._components[SearchComponent.SEMANTIC],
                tantivy=self._components[SearchComponent.TANTIVY],
                cross_encoder=self._components[SearchComponent.CROSS_ENCODER],
                openrouter_reranker=self._components[
                    SearchComponent.OPENROUTER_RERANKER
                ],
            )
