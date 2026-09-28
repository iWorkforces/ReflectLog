from annotationlib import Format
import asyncio
import inspect
import threading
from unittest.mock import MagicMock

import anyio
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.memory.workspace_registry import WorkspaceRegistry
from reflectlog.application.utils.security import SecretString
from reflectlog.core.exceptions import ConfigurationError


def fake_manager(config: Config) -> MagicMock:
    manager = MagicMock(spec=MemoryManager)
    manager.config = config
    manager.closed = False
    manager.close.side_effect = lambda: setattr(manager, "closed", True)
    return manager


def was_closed(manager: MemoryManager) -> bool:
    return vars(manager)["closed"] is True


@pytest.fixture
def config() -> Config:
    return Config(workspace_id="", openrouter_api_key=SecretString("sk-test"))


@pytest.mark.unit
async def test_isolation_and_canonical_identity(config: Config) -> None:
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    assert created == []
    async with registry.acquire("ALPHA") as alpha:
        async with registry.acquire("alpha") as same:
            async with registry.acquire("beta") as beta:
                assert alpha is same
                assert alpha is not beta
                assert alpha.config.workspace_id == "alpha"
                assert beta.config.workspace_id == "beta"
                assert config.workspace_id == ""
    assert len(created) == 2
    await registry.close()


@pytest.mark.unit
@pytest.mark.parametrize("workspace_id", [".", "..", "abc..def", "../other", "a/b", ""])
async def test_invalid_workspace_id_is_rejected(
    config: Config, workspace_id: str
) -> None:
    registry = WorkspaceRegistry(config, fake_manager)
    with pytest.raises(ConfigurationError, match="WORKSPACE_ID"):
        async with registry.acquire(workspace_id):
            pytest.fail("Invalid workspace reached manager construction")


@pytest.mark.unit
async def test_acquire_requires_explicit_workspace(config: Config) -> None:
    registry = WorkspaceRegistry(config, fake_manager)
    parameters = inspect.signature(
        registry.acquire, annotation_format=Format.STRING
    ).parameters
    assert parameters["workspace_id"].default is inspect.Parameter.empty


@pytest.mark.unit
async def test_configured_workspace_is_not_selected(config: Config) -> None:
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(
        Config(workspace_id="Default", openrouter_api_key=config.openrouter_api_key),
        factory,
    )
    assert created == []
    async with registry.acquire("Selected") as manager:
        assert manager.config.workspace_id == "selected"
    assert len(created) == 1
    await registry.close()


@pytest.mark.unit
async def test_single_flight_construction(config: Config) -> None:
    created: list[MagicMock] = []
    release = anyio.Event()
    entered = anyio.Event()

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    seen: list[MemoryManager] = []

    async def worker() -> None:
        async with registry.acquire("Shared") as manager:
            seen.append(manager)
            entered.set()
            await release.wait()

    async with anyio.create_task_group() as group:
        for _ in range(12):
            group.start_soon(worker)
        await entered.wait()
        release.set()
    assert len(created) == 1
    assert all(manager is created[0] for manager in seen)
    assert len(seen) == 12
    await registry.close()


@pytest.mark.unit
async def test_sliding_ttl_and_active_protection(config: Config) -> None:
    now = [0.0]
    registry = WorkspaceRegistry(config, fake_manager, clock=lambda: now[0])
    async with registry.acquire("one") as first:
        now[0] = 899
        await registry.prune()
        assert not was_closed(first)
    now[0] = 1700
    async with registry.acquire("ONE") as reused:
        assert reused is first
    now[0] = 2600
    await registry.prune()
    assert was_closed(first)
    async with registry.acquire("one") as replacement:
        assert replacement is not first
    await registry.close()


@pytest.mark.unit
async def test_idle_cap_never_evicts_active(config: Config) -> None:
    now = [0.0]
    registry = WorkspaceRegistry(config, fake_manager, clock=lambda: now[0], max_idle=2)
    async with registry.acquire("pinned") as pinned:
        idle: list[MemoryManager] = []
        for workspace in ("one", "two", "three"):
            now[0] += 1
            async with registry.acquire(workspace) as manager:
                idle.append(manager)
        assert was_closed(idle[0])
        assert not was_closed(idle[1])
        assert not was_closed(idle[2])
        assert not was_closed(pinned)
    await registry.close()
    assert was_closed(pinned)


@pytest.mark.unit
async def test_default_idle_cap_is_eight(config: Config) -> None:
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    for index in range(9):
        async with registry.acquire(f"workspace-{index}"):
            pass
    assert was_closed(created[0])
    assert all(not was_closed(manager) for manager in created[1:])
    await registry.close()


@pytest.mark.unit
async def test_close_drains_and_rejects_new_acquisitions(config: Config) -> None:
    registry = WorkspaceRegistry(config, fake_manager)
    acquired = anyio.Event()
    release = anyio.Event()
    completed = anyio.Event()
    held: list[MemoryManager] = []

    async def user() -> None:
        async with registry.acquire("live") as manager:
            held.append(manager)
            acquired.set()
            await release.wait()

    async def closer() -> None:
        await registry.close()
        completed.set()

    async with anyio.create_task_group() as group:
        group.start_soon(user)
        await acquired.wait()
        group.start_soon(closer)
        await anyio.lowlevel.checkpoint()
        assert not completed.is_set()
        assert not was_closed(held[0])
        release.set()
    assert completed.is_set()
    assert was_closed(held[0])
    with pytest.raises(RuntimeError, match="closed"):
        async with registry.acquire("live"):
            pytest.fail("Closed registry returned a manager")
    await registry.close()


@pytest.mark.unit
async def test_cancelled_construction_waits_for_native_worker(config: Config) -> None:
    started = threading.Event()
    finish = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        started.set()
        finish.wait(timeout=5)
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    completed = anyio.Event()

    async def user() -> None:
        async with registry.acquire("slow"):
            completed.set()

    async with anyio.create_task_group() as group:
        group.start_soon(user)
        await anyio.to_thread.run_sync(started.wait)
        group.cancel_scope.cancel()
        finish.set()
    assert len(created) == 1
    assert created[0].closed is False
    await registry.close()
    assert created[0].closed


@pytest.mark.unit
async def test_direct_cancel_during_construction_keeps_completed_manager_owned(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        started.set()
        assert finish.wait(timeout=5)
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)

    async def user() -> None:
        async with registry.acquire("slow"):
            pytest.fail("Cancelled acquisition returned a manager")

    user_task = asyncio.create_task(user())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        user_task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await user_task
        assert len(created) == 1
        assert registry._entries["slow"].manager is created[0]
        assert registry._entries["slow"].active == 0
        await registry.close()
        assert was_closed(created[0])
        assert created[0].close.call_count == 1
    finally:
        finish.set()


@pytest.mark.unit
async def test_direct_cancel_during_failed_construction_propagates_cancellation(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()

    def factory(_: Config) -> MemoryManager:
        started.set()
        assert finish.wait(timeout=5)
        raise OSError("constructor failed")

    registry = WorkspaceRegistry(config, factory)

    async def user() -> None:
        async with registry.acquire("broken"):
            pytest.fail("Failed acquisition returned a manager")

    user_task = asyncio.create_task(user())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        user_task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await user_task
        assert registry._entries == {}
        await registry.close()
    finally:
        finish.set()


@pytest.mark.unit
async def test_reaper_expires_idle_without_another_acquisition(config: Config) -> None:
    now = [0.0]
    expired = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        manager.close.side_effect = expired.set
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory, clock=lambda: now[0])
    async with anyio.create_task_group() as group:
        group.start_soon(registry.run_reaper, 0)
        async with registry.acquire("idle"):
            now[0] = 901
            await anyio.lowlevel.checkpoint()
            assert not expired.is_set()
        now[0] = 1801
        await anyio.to_thread.run_sync(expired.wait)
        assert expired.is_set()
        await registry.close()
    assert created[0].close.call_count == 1


@pytest.mark.unit
async def test_failed_eviction_quarantines_workspace_only(config: Config) -> None:
    now = [0.0]
    attempts: list[str] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        if concrete.workspace_id == "broken":

            def close_once() -> None:
                attempts.append("broken")
                if len(attempts) == 1:
                    raise OSError("storage unavailable")
                manager.closed = True

            manager.close.side_effect = close_once
        return manager

    registry = WorkspaceRegistry(config, factory, clock=lambda: now[0])
    async with registry.acquire("broken") as broken:
        pass
    now[0] = 901
    await registry.prune()
    assert attempts == ["broken"]
    with pytest.raises(RuntimeError, match="pending close"):
        async with registry.acquire("BROKEN"):
            pytest.fail("Quarantined manager was reused")
    async with registry.acquire("healthy") as healthy:
        assert healthy.config.workspace_id == "healthy"
    assert not was_closed(broken)
    await registry.close()
    assert attempts == ["broken", "broken"]
    assert was_closed(broken)
    assert was_closed(healthy)


@pytest.mark.unit
async def test_direct_cancel_during_eviction_waits_for_close(config: Config) -> None:
    started = threading.Event()
    finish = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def persist() -> None:
            started.set()
            assert finish.wait(timeout=5)
            manager.closed = True

        manager.close.side_effect = persist
        created.append(manager)
        return manager

    now = [0.0]
    registry = WorkspaceRegistry(config, factory, clock=lambda: now[0])
    async with registry.acquire("idle") as manager:
        pass
    now[0] = 901
    prune_task = asyncio.create_task(registry.prune())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        prune_task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await prune_task
        assert was_closed(manager)
        await registry.close()
        assert created[0].close.call_count == 1
    finally:
        finish.set()


@pytest.mark.unit
async def test_direct_cancel_during_failed_eviction_quarantines_for_retry(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    attempts = [0]

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def persist() -> None:
            attempts[0] += 1
            if attempts[0] == 1:
                started.set()
                assert finish.wait(timeout=5)
                raise OSError("storage unavailable")
            manager.closed = True

        manager.close.side_effect = persist
        return manager

    now = [0.0]
    registry = WorkspaceRegistry(config, factory, clock=lambda: now[0])
    async with registry.acquire("broken") as manager:
        pass
    now[0] = 901
    prune_task = asyncio.create_task(registry.prune())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        prune_task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await prune_task
        with pytest.raises(RuntimeError, match="pending close"):
            async with registry.acquire("broken"):
                pytest.fail("Failed manager was reused")
        assert not was_closed(manager)
        await registry.close()
        assert was_closed(manager)
        assert attempts == [2]
    finally:
        finish.set()


@pytest.mark.unit
async def test_shutdown_retries_failed_closes_without_losing_managers(
    config: Config,
) -> None:
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        if concrete.workspace_id == "broken":
            attempts = [0]

            def close_once() -> None:
                attempts[0] += 1
                if attempts[0] == 1:
                    raise OSError("storage unavailable")
                manager.closed = True

            manager.close.side_effect = close_once
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    async with registry.acquire("broken"):
        pass
    async with registry.acquire("healthy"):
        pass
    with pytest.raises(ExceptionGroup, match="could not be closed"):
        await registry.close()
    assert not was_closed(created[0])
    assert was_closed(created[1])
    await registry.close()
    assert was_closed(created[0])
    assert created[0].close.call_count == 2
    assert created[1].close.call_count == 1


@pytest.mark.unit
async def test_concurrent_closers_report_persist_failure_and_can_retry(
    config: Config,
) -> None:
    acquired = anyio.Event()
    release = anyio.Event()
    attempts = [0]

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def close_once() -> None:
            attempts[0] += 1
            if attempts[0] == 1:
                raise OSError("storage unavailable")
            manager.closed = True

        manager.close.side_effect = close_once
        return manager

    registry = WorkspaceRegistry(config, factory)
    held: list[MemoryManager] = []

    async def user() -> None:
        async with registry.acquire("broken") as manager:
            held.append(manager)
            acquired.set()
            await release.wait()

    failures: list[ExceptionGroup] = []

    async def closer() -> None:
        try:
            await registry.close()
        except ExceptionGroup as error:
            failures.append(error)

    async with anyio.create_task_group() as group:
        group.start_soon(user)
        await acquired.wait()
        group.start_soon(closer)
        while not registry._closing:
            await anyio.lowlevel.checkpoint()
        group.start_soon(closer)
        await anyio.lowlevel.checkpoint()
        release.set()

    assert len(failures) == 2
    assert all("storage unavailable" in str(error.exceptions[0]) for error in failures)
    assert not was_closed(held[0])
    await registry.close()
    assert was_closed(held[0])
    assert attempts == [2]


@pytest.mark.unit
async def test_close_queued_behind_failed_persist_shares_attempt(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    second_entered = anyio.Event()
    attempts = [0]

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def persist() -> None:
            attempts[0] += 1
            if attempts[0] == 1:
                started.set()
                assert finish.wait(timeout=5)
                raise OSError("storage unavailable")
            manager.closed = True

        manager.close.side_effect = persist
        return manager

    registry = WorkspaceRegistry(config, factory)
    async with registry.acquire("broken") as manager:
        pass

    async def closer(entered: anyio.Event | None = None) -> ExceptionGroup | None:
        if entered is not None:
            entered.set()
        try:
            await registry.close()
        except ExceptionGroup as error:
            return error
        return None

    first = asyncio.create_task(closer())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        second = asyncio.create_task(closer(second_entered))
        await second_entered.wait()
        await anyio.lowlevel.checkpoint()
        finish.set()
        first_failure = await first
        second_failure = await second
        assert isinstance(first_failure, ExceptionGroup)
        assert second_failure is first_failure
        assert attempts == [1]
        assert not was_closed(manager)
        await registry.close()
        assert was_closed(manager)
        assert attempts == [2]
    finally:
        finish.set()


@pytest.mark.unit
async def test_cancelled_shutdown_waits_for_manager_persistence(config: Config) -> None:
    started = threading.Event()
    finish = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def persist() -> None:
            started.set()
            assert finish.wait(timeout=5)
            manager.closed = True

        manager.close.side_effect = persist
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    async with registry.acquire("live") as manager:
        pass

    async with anyio.create_task_group() as group:
        group.start_soon(registry.close)
        assert await anyio.to_thread.run_sync(started.wait, 5)
        group.cancel_scope.cancel()
        finish.set()

    assert was_closed(manager)
    await registry.close()
    assert created[0].close.call_count == 1


@pytest.mark.unit
async def test_direct_cancel_during_shutdown_waits_for_persistence(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def persist() -> None:
            started.set()
            assert finish.wait(timeout=5)
            manager.closed = True

        manager.close.side_effect = persist
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    async with registry.acquire("live") as manager:
        pass

    close_task = asyncio.create_task(registry.close())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        close_task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        assert was_closed(manager)
        with pytest.raises(RuntimeError, match="closed"):
            async with registry.acquire("live"):
                pytest.fail("Closing registry returned a manager")
        await registry.close()
        assert created[0].close.call_count == 1
    finally:
        finish.set()


@pytest.mark.unit
async def test_direct_cancel_during_failed_shutdown_reports_to_other_closer(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    acquired = anyio.Event()
    release = anyio.Event()
    second_waiting = anyio.Event()
    attempts = [0]

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def persist() -> None:
            attempts[0] += 1
            if attempts[0] == 1:
                started.set()
                assert finish.wait(timeout=5)
                raise OSError("storage unavailable")
            manager.closed = True

        manager.close.side_effect = persist
        return manager

    registry = WorkspaceRegistry(config, factory)
    held: list[MemoryManager] = []

    async def user() -> None:
        async with registry.acquire("broken") as manager:
            held.append(manager)
            acquired.set()
            await release.wait()

    user_task = asyncio.create_task(user())
    await acquired.wait()

    async def second_closer() -> None:
        second_waiting.set()
        await registry.close()

    first = asyncio.create_task(registry.close())
    try:
        while not registry._closing:
            await anyio.lowlevel.checkpoint()
        second = asyncio.create_task(second_closer())
        await second_waiting.wait()
        await anyio.lowlevel.checkpoint()
        release.set()
        assert await anyio.to_thread.run_sync(started.wait, 5)
        first.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(ExceptionGroup, match="could not be closed") as failure:
            await second
        assert "storage unavailable" in str(failure.value.exceptions[0])
        await user_task
        assert not was_closed(held[0])
        await registry.close()
        assert was_closed(held[0])
        assert attempts == [2]
    finally:
        release.set()
        finish.set()


@pytest.mark.unit
async def test_direct_cancel_of_pinned_user_drains_closing_registry(
    config: Config,
) -> None:
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    acquired = anyio.Event()
    held: list[MemoryManager] = []

    async def user() -> None:
        async with registry.acquire("live") as manager:
            held.append(manager)
            acquired.set()
            await anyio.sleep_forever()

    user_task = asyncio.create_task(user())
    try:
        await acquired.wait()
        close_task = asyncio.create_task(registry.close())
        while not registry._closing:
            await anyio.lowlevel.checkpoint()
        with pytest.raises(RuntimeError, match="closed"):
            async with registry.acquire("new"):
                pytest.fail("Closing registry accepted a new manager")
        user_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await user_task
        await close_task
        assert registry._entries == {}
        assert was_closed(held[0])
        await registry.close()
        assert created[0].close.call_count == 1
    finally:
        user_task.cancel()
