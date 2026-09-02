"""Workspace storage coordination protocols and value types."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class LeaseMode(StrEnum):
    """Portalocker lease mode for a workspace root."""

    SHARED = "shared"
    EXCLUSIVE = "exclusive"


@dataclass(frozen=True)
class WorkspaceStoragePaths:
    """Stable sidecar paths for one workspace."""

    workspace_id: str
    root: str
    lock_path: str
    generation_path: str


@runtime_checkable
class IStorageLease(Protocol):
    """Held workspace lease that must be released exactly once."""

    @property
    def workspace_id(self) -> str: ...

    @property
    def mode(self) -> LeaseMode: ...

    def release(self) -> None: ...

    def __enter__(self) -> IStorageLease: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None: ...


@runtime_checkable
class IStorageCoordinator(Protocol):
    """Coordinates exclusive/shared workspace access across processes."""

    @property
    def timeout(self) -> float: ...

    def paths_for(self, workspace_id: str) -> WorkspaceStoragePaths: ...

    def acquire(
        self,
        workspace_id: str,
        mode: LeaseMode = LeaseMode.EXCLUSIVE,
        *,
        timeout: float | None = None,
    ) -> AbstractContextManager[IStorageLease]: ...

    def read_generation(self, workspace_id: str) -> int: ...

    def publish_generation(self, workspace_id: str, generation: int) -> None: ...


def exclusive_lease(
    coordinator: IStorageCoordinator,
    workspace_id: str,
    *,
    timeout: float | None = None,
) -> AbstractContextManager[IStorageLease]:
    """Acquire an exclusive workspace lease."""
    return coordinator.acquire(
        workspace_id, LeaseMode.EXCLUSIVE, timeout=timeout
    )


def shared_lease(
    coordinator: IStorageCoordinator,
    workspace_id: str,
    *,
    timeout: float | None = None,
) -> AbstractContextManager[IStorageLease]:
    """Acquire a shared workspace lease."""
    return coordinator.acquire(workspace_id, LeaseMode.SHARED, timeout=timeout)


def iter_lease_modes() -> Iterator[LeaseMode]:
    """Yield every supported lease mode."""
    yield from LeaseMode
