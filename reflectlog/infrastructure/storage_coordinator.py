"""Portalocker workspace storage coordinator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import os
import threading
import time

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


def _empty_holder_map() -> dict[int, int]:
    return {}


@dataclass
class _WorkspaceLeaseState:
    mutex: threading.Lock
    cond: threading.Condition
    os_lock: portalocker.Lock | None = None
    exclusive_depth: int = 0
    exclusive_owner: int | None = None
    shared_holders: dict[int, int] = field(default_factory=_empty_holder_map)

    def shared_depth(self) -> int:
        return sum(self.shared_holders.values())

    def add_shared(self, owner: int) -> None:
        self.shared_holders[owner] = self.shared_holders.get(owner, 0) + 1

    def drop_shared(self, owner: int) -> None:
        remaining = self.shared_holders.get(owner, 0) - 1
        if remaining <= 0:
            _ = self.shared_holders.pop(owner, None)
        else:
            self.shared_holders[owner] = remaining


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
        self._states: dict[str, _WorkspaceLeaseState] = {}

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

    def _state_for(self, workspace_id: str) -> _WorkspaceLeaseState:
        with self._local:
            state = self._states.get(workspace_id)
            if state is None:
                mutex = threading.Lock()
                state = _WorkspaceLeaseState(
                    mutex=mutex, cond=threading.Condition(mutex)
                )
                self._states[workspace_id] = state
            return state

    def _take_os_lock(
        self,
        paths: WorkspaceStoragePaths,
        mode: LeaseMode,
        wait: float,
        state: _WorkspaceLeaseState,
    ) -> None:
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
        state.os_lock = lock

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

        state = self._state_for(paths.workspace_id)
        deadline = time.monotonic() + wait
        exclusive = mode is LeaseMode.EXCLUSIVE
        owner = threading.get_ident()
        with state.cond:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LeaseTimeoutError(
                        f"Timed out acquiring {mode} lease for {paths.workspace_id}"
                    )
                if exclusive:
                    if state.exclusive_depth > 0 and state.exclusive_owner == owner:
                        state.exclusive_depth += 1
                        break
                    if state.exclusive_depth > 0:
                        _ = state.cond.wait(timeout=remaining)
                        continue
                    other_shared = sum(
                        count
                        for thread_id, count in state.shared_holders.items()
                        if thread_id != owner
                    )
                    if other_shared > 0:
                        _ = state.cond.wait(timeout=remaining)
                        continue
                    if state.os_lock is None:
                        self._take_os_lock(paths, LeaseMode.EXCLUSIVE, remaining, state)
                    state.exclusive_depth = 1
                    state.exclusive_owner = owner
                    break
                if state.exclusive_depth > 0:
                    if state.exclusive_owner == owner:
                        state.add_shared(owner)
                        break
                    _ = state.cond.wait(timeout=remaining)
                    continue
                if state.shared_depth() > 0 and state.os_lock is not None:
                    state.add_shared(owner)
                    break
                self._take_os_lock(paths, LeaseMode.SHARED, remaining, state)
                state.add_shared(owner)
                break

        return _PortalockerLease(
            workspace_id=paths.workspace_id,
            mode=mode,
            _on_release=self._release_callback(paths.workspace_id, mode, owner),
        )

    def _release_callback(
        self, workspace_id: str, mode: LeaseMode, owner: int
    ) -> Callable[[], None]:
        def _on_release() -> None:
            state = self._state_for(workspace_id)
            with state.cond:
                if mode is LeaseMode.EXCLUSIVE:
                    state.exclusive_depth = max(0, state.exclusive_depth - 1)
                    if state.exclusive_depth == 0:
                        state.exclusive_owner = None
                else:
                    state.drop_shared(owner)
                lock = None
                if state.exclusive_depth == 0 and state.shared_depth() == 0:
                    lock = state.os_lock
                    state.os_lock = None
                try:
                    if lock is not None:
                        lock.release()
                finally:
                    state.cond.notify_all()

        return _on_release

    def is_held(self, workspace_id: str, mode: LeaseMode | None = None) -> bool:
        """Return True when the calling thread already holds a lease."""
        safe_id = validate_workspace_id(workspace_id).lower()
        owner = threading.get_ident()
        state = self._state_for(safe_id)
        with state.cond:
            owns_exclusive = (
                state.exclusive_depth > 0 and state.exclusive_owner == owner
            )
            owns_shared = owner in state.shared_holders
            if mode is None:
                return owns_exclusive or owns_shared
            if mode is LeaseMode.EXCLUSIVE:
                return owns_exclusive
            return owns_shared or owns_exclusive

    def read_generation(self, workspace_id: str) -> int:
        path = self.paths_for(workspace_id).generation_path
        if not os.path.exists(path):
            return 0
        try:
            with open(path, encoding="utf-8") as handle:
                raw = handle.read()
        except OSError as exc:
            raise GenerationError("Storage generation sidecar is unreadable") from exc
        text = raw.strip()
        if text == "":
            raise GenerationError("Storage generation sidecar is empty")
        try:
            generation = int(text)
        except ValueError as exc:
            raise GenerationError("Storage generation sidecar is corrupt") from exc
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
