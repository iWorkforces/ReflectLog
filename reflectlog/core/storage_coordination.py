"""Workspace storage coordination protocols and value types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from reflectlog.core.exceptions import LeaseUpgradeError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager


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

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None: ...


@runtime_checkable
class IStorageCoordinator(Protocol):
    """Coordinates exclusive/shared workspace access across processes.

    An EXCLUSIVE owner may nest EXCLUSIVE and SHARED leases. A SHARED-only
    owner may nest SHARED, but SHARED-to-EXCLUSIVE raises LeaseUpgradeError
    immediately and leaves the held lease untouched. Other threads and
    processes wait up to the acquisition timeout as before.
    """

    @property
    def timeout(self) -> float: ...

    def paths_for(self, workspace_id: str) -> WorkspaceStoragePaths: ...

    def acquire(
        self,
        workspace_id: str,
        mode: LeaseMode = LeaseMode.EXCLUSIVE,
        *,
        timeout: float | None = None,
    ) -> AbstractContextManager[IStorageLease]:
        """Acquire a lease; reject a SHARED-only owner's EXCLUSIVE request.

        LeaseUpgradeError is immediate, without changing the held lease.
        Legal nesting reuses ownership; other owners wait up to the timeout.
        """
        ...

    def read_generation(self, workspace_id: str) -> int: ...

    def publish_generation(self, workspace_id: str, generation: int) -> None: ...

    def is_held(self, workspace_id: str, mode: LeaseMode | None = None) -> bool:
        """Report this thread's ownership, not permission to upgrade.

        None accepts either mode; SHARED also accepts EXCLUSIVE ownership.
        Only EXCLUSIVE ownership permits nesting an EXCLUSIVE request.
        """
        ...


def reject_shared_upgrade(coordinator: IStorageCoordinator, workspace_id: str) -> None:
    """Reject an EXCLUSIVE request by a thread holding only a SHARED lease."""
    if coordinator.is_held(workspace_id) and not coordinator.is_held(
        workspace_id, LeaseMode.EXCLUSIVE
    ):
        raise LeaseUpgradeError(
            f"SHARED-to-EXCLUSIVE lease upgrades are forbidden for {workspace_id}"
        )


def exclusive_lease(
    coordinator: IStorageCoordinator,
    workspace_id: str,
    *,
    timeout: float | None = None,
) -> AbstractContextManager[IStorageLease]:
    """Acquire an exclusive workspace lease."""
    return coordinator.acquire(workspace_id, LeaseMode.EXCLUSIVE, timeout=timeout)


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
