'''Unit tests for reflectlog.application.utils.config_reload module.'''

from collections.abc import Generator
import signal
import threading
from unittest.mock import MagicMock, patch

import pytest

from reflectlog.application.utils.config_reload import (
    ConfigReloadManager,
    get_reload_manager,
    setup_signal_handler,
)


class TestConfigReloadManagerInit:
    '''Tests for ConfigReloadManager.__init__.'''

    def test_init_stores_config_provider(self) -> None:
        '''Test that config_provider callable is stored.'''
        provider = MagicMock()
        manager = ConfigReloadManager(config_provider=provider)
        assert manager._config_provider is provider

    def test_init_config_is_none(self) -> None:
        '''Test that config starts as None (lazy loading).'''
        provider = MagicMock()
        manager = ConfigReloadManager(config_provider=provider)
        assert manager._config is None

    def test_init_reload_count_is_zero(self) -> None:
        '''Test that reload count starts at zero.'''
        provider = MagicMock()
        manager = ConfigReloadManager(config_provider=provider)
        assert manager._reload_count == 0

    def test_init_creates_rlock(self) -> None:
        '''Test that an RLock is created for thread safety.'''
        provider = MagicMock()
        manager = ConfigReloadManager(config_provider=provider)
        assert isinstance(manager._lock, type(threading.RLock()))


class TestConfigReloadManagerGetConfig:
    '''Tests for ConfigReloadManager.get_config.'''

    def test_get_config_calls_provider_on_first_access(self) -> None:
        '''Test that config_provider is called on first get_config.'''
        mock_config = MagicMock()
        provider = MagicMock(return_value=mock_config)
        manager = ConfigReloadManager(config_provider=provider)

        result = manager.get_config()

        provider.assert_called_once()
        assert result is mock_config

    def test_get_config_returns_cached_on_second_call(self) -> None:
        '''Test that second call returns cached config without calling provider again.'''
        mock_config = MagicMock()
        provider = MagicMock(return_value=mock_config)
        manager = ConfigReloadManager(config_provider=provider)

        first = manager.get_config()
        second = manager.get_config()

        provider.assert_called_once()
        assert first is second
        assert first is mock_config

    def test_get_config_logs_on_load(self) -> None:
        '''Test that loading config logs an info message.'''
        mock_config = MagicMock()
        provider = MagicMock(return_value=mock_config)
        manager = ConfigReloadManager(config_provider=provider)

        with patch("reflectlog.application.utils.config_reload.logger") as mock_logger:
            _ = manager.get_config()
            mock_logger.info.assert_called_once_with(
                "Configuration loaded",
                extra={"reload_count": 0},
            )

    def test_get_config_does_not_log_on_cached_access(self) -> None:
        '''Test that cached access does not log again.'''
        mock_config = MagicMock()
        provider = MagicMock(return_value=mock_config)
        manager = ConfigReloadManager(config_provider=provider)

        # First call loads and logs
        _ = manager.get_config()

        with patch("reflectlog.application.utils.config_reload.logger") as mock_logger:
            _ = manager.get_config()
            mock_logger.info.assert_not_called()


class TestConfigReloadManagerReloadConfig:
    '''Tests for ConfigReloadManager.reload_config.'''

    def test_reload_invalidates_cache_and_reloads(self) -> None:
        '''Test that reload clears cached config and provides fresh one.'''
        config_v1 = MagicMock(name="config_v1")
        config_v2 = MagicMock(name="config_v2")
        provider = MagicMock(side_effect=[config_v1, config_v2])
        manager = ConfigReloadManager(config_provider=provider)

        first = manager.get_config()
        reloaded = manager.reload_config()

        assert first is config_v1
        assert reloaded is config_v2
        assert provider.call_count == 2

    def test_reload_increments_reload_count(self) -> None:
        '''Test that each reload increments the count.'''
        provider = MagicMock()
        manager = ConfigReloadManager(config_provider=provider)

        assert manager.get_reload_count() == 0

        _ = manager.reload_config()
        assert manager.get_reload_count() == 1

        _ = manager.reload_config()
        assert manager.get_reload_count() == 2

    def test_reload_logs_invalidation(self) -> None:
        '''Test that reload logs the invalidation message.'''
        provider = MagicMock()
        manager = ConfigReloadManager(config_provider=provider)

        with patch("reflectlog.application.utils.config_reload.logger") as mock_logger:
            _ = manager.reload_config()

            # Should log invalidation then load
            calls = mock_logger.info.call_args_list
            assert len(calls) == 2
            assert (
                calls[0].args[0]
                == "Configuration invalidated, will reload on next access"
            )
            assert calls[0].kwargs["extra"] == {"reload_count": 1}
            assert calls[1].args[0] == "Configuration loaded"
            assert calls[1].kwargs["extra"] == {"reload_count": 1}

    def test_reload_returns_new_config(self) -> None:
        '''Test that reload returns the newly loaded config.'''
        config_new = MagicMock(name="config_new")
        provider = MagicMock(side_effect=[MagicMock(), config_new])
        manager = ConfigReloadManager(config_provider=provider)

        _ = manager.get_config()  # initial load
        result = manager.reload_config()

        assert result is config_new


class TestConfigReloadManagerGetReloadCount:
    '''Tests for ConfigReloadManager.get_reload_count.'''

    def test_initial_count_is_zero(self) -> None:
        '''Test that reload count starts at zero.'''
        provider = MagicMock()
        manager = ConfigReloadManager(config_provider=provider)
        assert manager.get_reload_count() == 0

    def test_count_tracks_multiple_reloads(self) -> None:
        '''Test that count increases with each reload.'''
        provider = MagicMock()
        manager = ConfigReloadManager(config_provider=provider)

        for i in range(5):
            _ = manager.reload_config()
            assert manager.get_reload_count() == i + 1

    def test_count_not_incremented_by_get_config(self) -> None:
        '''Test that get_config does not increment reload count.'''
        provider = MagicMock()
        manager = ConfigReloadManager(config_provider=provider)

        _ = manager.get_config()
        _ = manager.get_config()
        _ = manager.get_config()

        assert manager.get_reload_count() == 0


class TestConfigReloadManagerThreadSafety:
    '''Tests for thread safety of ConfigReloadManager.'''

    def test_concurrent_get_config_calls_provider_once(self) -> None:
        '''Test that concurrent get_config calls only invoke provider once.'''
        call_count = 0
        barrier = threading.Barrier(5)
        mock_config = MagicMock()

        def slow_provider():
            nonlocal call_count
            call_count += 1
            return mock_config

        manager = ConfigReloadManager(config_provider=slow_provider)
        results: list[object] = []
        errors: list[Exception] = []

        def worker():
            try:
                _ = barrier.wait(timeout=2)
                result = manager.get_config()
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert len(results) == 5
        # All results should be the same config
        assert all(r is mock_config for r in results)
        # Provider should only be called once (lazy init)
        assert call_count == 1

    def test_concurrent_reload_increments_correctly(self) -> None:
        '''Test that concurrent reloads maintain correct count.'''
        provider = MagicMock()
        manager = ConfigReloadManager(config_provider=provider)
        barrier = threading.Barrier(10)
        errors: list[Exception] = []

        def worker():
            try:
                _ = barrier.wait(timeout=2)
                _ = manager.reload_config()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert manager.get_reload_count() == 10


class TestSetupSignalHandler:
    '''Tests for setup_signal_handler function.'''

    @pytest.fixture(autouse=True)
    def _reset_global_state(self) -> Generator[None]:
        '''Reset global singleton state before each test.'''
        import reflectlog.application.utils.config_reload as module

        with module._global_lock:
            module._global_reload_manager = None
        yield
        with module._global_lock:
            module._global_reload_manager = None

    @patch("reflectlog.application.utils.config_reload.signal.signal")
    def test_registers_sighup_handler(self, mock_signal: MagicMock) -> None:
        '''Test that SIGHUP handler is registered with signal module.'''
        provider = MagicMock()

        _ = setup_signal_handler(provider)

        mock_signal.assert_called_once()
        args = mock_signal.call_args
        assert args[0][0] == signal.SIGHUP

    @patch("reflectlog.application.utils.config_reload.signal.signal")
    def test_returns_reload_manager(self, mock_signal: MagicMock) -> None:
        '''Test that setup returns a ConfigReloadManager instance.'''
        provider = MagicMock()

        result = setup_signal_handler(provider)

        assert isinstance(result, ConfigReloadManager)

    @patch("reflectlog.application.utils.config_reload.signal.signal")
    def test_singleton_returns_same_instance(self, mock_signal: MagicMock) -> None:
        '''Test that second call returns same manager (singleton).'''
        provider1 = MagicMock()
        provider2 = MagicMock()

        first = setup_signal_handler(provider1)
        second = setup_signal_handler(provider2)

        assert first is second
        # signal.signal should only be called once (singleton)
        mock_signal.assert_called_once()

    @patch("reflectlog.application.utils.config_reload.signal.signal")
    def test_logs_handler_registration(self, mock_signal: MagicMock) -> None:
        '''Test that handler registration is logged.'''
        provider = MagicMock()

        with patch("reflectlog.application.utils.config_reload.logger") as mock_logger:
            _ = setup_signal_handler(provider)
            mock_logger.info.assert_called_with(
                "SIGHUP handler registered for configuration reload",
            )

    @patch("reflectlog.application.utils.config_reload.signal.signal")
    def test_sighup_handler_triggers_reload(self, mock_signal: MagicMock) -> None:
        '''Test that the registered SIGHUP handler triggers config reload.'''
        mock_config = MagicMock()
        provider = MagicMock(return_value=mock_config)

        _ = setup_signal_handler(provider)

        # Extract the handler function that was registered
        handler = mock_signal.call_args[0][1]

        # Simulate SIGHUP signal
        handler(signal.SIGHUP, None)

        # reload_config calls get_config which calls provider
        assert provider.call_count >= 1

    @patch("reflectlog.application.utils.config_reload.signal.signal")
    def test_sighup_handler_logs_signal_receipt(self, mock_signal: MagicMock) -> None:
        '''Test that SIGHUP handler logs signal receipt.'''
        provider = MagicMock()

        _ = setup_signal_handler(provider)
        handler = mock_signal.call_args[0][1]

        with patch("reflectlog.application.utils.config_reload.logger") as mock_logger:
            handler(signal.SIGHUP, None)

            mock_logger.info.assert_any_call(
                "SIGHUP received, reloading configuration",
                extra={"signal": "SIGHUP", "signum": signal.SIGHUP},
            )

    @patch("reflectlog.application.utils.config_reload.signal.signal")
    def test_sighup_handler_increments_reload_count(
        self, mock_signal: MagicMock
    ) -> None:
        '''Test that SIGHUP handler increments reload count.'''
        provider = MagicMock()

        manager = setup_signal_handler(provider)
        handler = mock_signal.call_args[0][1]

        assert manager.get_reload_count() == 0

        handler(signal.SIGHUP, None)
        assert manager.get_reload_count() == 1

        handler(signal.SIGHUP, None)
        assert manager.get_reload_count() == 2


class TestGetReloadManager:
    '''Tests for get_reload_manager function.'''

    @pytest.fixture(autouse=True)
    def _reset_global_state(self) -> Generator[None]:
        '''Reset global singleton state before each test.'''
        import reflectlog.application.utils.config_reload as module

        with module._global_lock:
            module._global_reload_manager = None
        yield
        with module._global_lock:
            module._global_reload_manager = None

    def test_returns_none_before_setup(self) -> None:
        '''Test that get_reload_manager returns None before setup.'''
        result = get_reload_manager()
        assert result is None

    @patch("reflectlog.application.utils.config_reload.signal.signal")
    def test_returns_manager_after_setup(self, mock_signal: MagicMock) -> None:
        '''Test that get_reload_manager returns manager after setup.'''
        provider = MagicMock()
        expected = setup_signal_handler(provider)

        result = get_reload_manager()

        assert result is expected
        assert isinstance(result, ConfigReloadManager)

    @patch("reflectlog.application.utils.config_reload.signal.signal")
    def test_returns_same_instance(self, mock_signal: MagicMock) -> None:
        '''Test that get_reload_manager always returns same instance.'''
        provider = MagicMock()
        _ = setup_signal_handler(provider)

        first = get_reload_manager()
        second = get_reload_manager()

        assert first is second
