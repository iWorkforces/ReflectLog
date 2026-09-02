"""Portalocker workspace storage coordinator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
import threading

import portalocker
from portalocker.exceptions import AlreadyLocked, LockException

from reflectlog.core.exceptions import GenerationError, LeaseTimeoutError
from reflectlog.core.storage_coordination import (
    IStorageLease,
    LeaseMode,
    WorkspaceStoragePaths,
)
from reflectlog.utility.security import validate_workspace_id

PRODUCTION_LEASE_TIMEOUT = 30.0
LOCK_NAME = ".reflectlog.writer.lock"
GENERATION_NAME = ".reflectlog.storage-generation"


@dataclass
class _PortalockerLease:
    workspace_id: str
    mode: LeaseMode
    _on_release: Callable[[], None] | None
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        callback = self._on_release
        self._on_release = None
        if callback is not None:
            callback()

    def __enter__(self) -> IStorageLease:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        self.release()


class PortalockerStorageCoordinator:
    """Per-workspace Portalocker leases and generation sidecars."""

    def __init__(
        self,
        indexes_root: str,
        *,
        timeout: float = PRODUCTION_LEASE_TIMEOUT,
    ) -> None:
        if timeout <= 0:
            raise LeaseTimeoutError("Coordinator timeout must be positive")
        self._indexes_root = os.path.abspath(indexes_root)
        self._timeout = timeout
        self._local = threading.RLock()
        self._lease_depth: dict[tuple[str, LeaseMode], int] = {}
        self._os_locks: dict[tuple[str, LeaseMode], portalocker.Lock] = {}

    @property
    def timeout(self) -> float:
        return self._timeout

    def paths_for(self, workspace_id: str) -> WorkspaceStoragePaths:
        safe_id = validate_workspace_id(workspace_id).lower()
        root = os.path.join(self._indexes_root, safe_id)
        return WorkspaceStoragePaths(
            workspace_id=safe_id,
            root=root,
            lock_path=os.path.join(root, LOCK_NAME),
            generation_path=os.path.join(root, GENERATION_NAME),
        )

    def acquire(
        self,
        workspace_id: str,
        mode: LeaseMode = LeaseMode.EXCLUSIVE,
        *,
        timeout: float | None = None,
    ) -> _PortalockerLease:
        paths = self.paths_for(workspace_id)
        wait = self._timeout if timeout is None else timeout
        if wait <= 0:
            raise LeaseTimeoutError(
                f"Workspace lease timeout is not positive for {paths.workspace_id}"
            )
        os.makedirs(paths.root, exist_ok=True)
        if not os.path.exists(paths.lock_path):
            with open(paths.lock_path, "a", encoding="utf-8") as handle:
                _ = handle.write("")

        key = (paths.workspace_id, mode)
        with self._local:
            depth = self._lease_depth.get(key, 0)
            if depth > 0:
                self._lease_depth[key] = depth + 1
                return _PortalockerLease(
                    workspace_id=paths.workspace_id,
                    mode=mode,
                    _on_release=self._release_callback(key),
                )

        flags = (
            portalocker.LOCK_EX | portalocker.LOCK_NB
            if mode is LeaseMode.EXCLUSIVE
            else portalocker.LOCK_SH | portalocker.LOCK_NB
        )
        lock = portalocker.Lock(paths.lock_path, timeout=wait, flags=flags)
        try:
            _ = lock.acquire()
        except AlreadyLocked as exc:
            raise LeaseTimeoutError(
                f"Timed out acquiring {mode} lease for {paths.workspace_id}"
            ) from exc
        except LockException as exc:
            raise LeaseTimeoutError(
                f"Failed to acquire {mode} lease for {paths.workspace_id}"
            ) from exc

        with self._local:
            self._lease_depth[key] = 1
            self._os_locks[key] = lock
        return _PortalockerLease(
            workspace_id=paths.workspace_id,
            mode=mode,
            _on_release=self._release_callback(key),
        )

    def _release_callback(self, key: tuple[str, LeaseMode]) -> Callable[[], None]:
        def _on_release() -> None:
            with self._local:
                depth = self._lease_depth.get(key, 0)
                if depth <= 1:
                    _ = self._lease_depth.pop(key, None)
                    lock = self._os_locks.pop(key, None)
                else:
                    self._lease_depth[key] = depth - 1
                    lock = None
            if lock is not None:
                lock.release()

        return _on_release

    def is_held(
        self, workspace_id: str, mode: LeaseMode | None = None
    ) -> bool:
        """Return True when this process already holds a lease for the workspace."""
        safe_id = validate_workspace_id(workspace_id).lower()
        with self._local:
            if mode is None:
                return any(
                    key[0] == safe_id and depth > 0
                    for key, depth in self._lease_depth.items()
                )
            return self._lease_depth.get((safe_id, mode), 0) > 0

    def read_generation(self, workspace_id: str) -> int:
        path = self.paths_for(workspace_id).generation_path
        if not os.path.exists(path):
            return 0
        try:
            with open(path, encoding="utf-8") as handle:
                raw = handle.read()
        except OSError as exc:
            raise GenerationError(
                "Storage generation sidecar is unreadable"
            ) from exc
        text = raw.strip()
        if text == "":
            raise GenerationError("Storage generation sidecar is empty")
        try:
            generation = int(text)
        except ValueError as exc:
            raise GenerationError(
                "Storage generation sidecar is corrupt"
            ) from exc
        if generation < 0:
            raise GenerationError("Storage generation sidecar is corrupt")
        return generation

    def publish_generation(self, workspace_id: str, generation: int) -> None:
        if generation < 0:
            raise GenerationError("Storage generation must be non-negative")
        paths = self.paths_for(workspace_id)
        os.makedirs(paths.root, exist_ok=True)
        temp_path = f"{paths.generation_path}.{os.getpid()}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                _ = handle.write(f"{generation}\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, paths.generation_path)
            _fsync_directory(paths.root)
        except OSError as exc:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise GenerationError("Failed to publish storage generation") from exc


def _fsync_directory(directory: str) -> None:
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        _ = os.fsync(dir_fd)
    except OSError:
        return
    finally:
        os.close(dir_fd)
