"""Runtime configuration reload support.

Provides SIGHUP signal handling for reloading configuration without
restarting the server process. This is useful for applying configuration
changes at runtime without service interruption.

Example:
    # Send SIGHUP to trigger reload
    kill -HUP <pid>

    # Server will reinitialize configuration from environment
    # and log the reload action
"""

from collections.abc import Callable
import logging
import signal
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflectlog.application.config import Config

logger = logging.getLogger(__name__)


class ConfigReloadManager:
    """Manages runtime configuration reloading via SIGHUP.

    Thread-safe manager that invalidates cached configuration
    when SIGHUP signal is received, forcing re-initialization
    from environment variables on next access.
    """

    def __init__(self, config_provider: Callable[[], Config]) -> None:
        """Initialize reload manager.

        Args:
            config_provider: Function that returns Config instance.
                Called after reload to obtain fresh configuration.
        """
        self._config_provider = config_provider
        self._config: Config | None = None
        self._lock = threading.RLock()
        self._reload_count = 0

    def get_config(self) -> Config:
        """Get current configuration, loading if not initialized.

        Thread-safe: Uses double-checked locking pattern.

        Returns:
            Current Config instance.
        """
        with self._lock:
            if self._config is None:
                self._config = self._config_provider()
                logger.info(
                    "Configuration loaded",
                    extra={"reload_count": self._reload_count},
                )
        return self._config

    def reload_config(self) -> Config:
        """Force configuration reload.

        Invalidates cached configuration and forces re-initialization
        on next access. Thread-safe.

        Returns:
            New Config instance after reload.
        """
        with self._lock:
            self._config = None
            self._reload_count += 1
            logger.info(
                "Configuration invalidated, will reload on next access",
                extra={"reload_count": self._reload_count},
            )

        return self.get_config()

    def get_reload_count(self) -> int:
        with self._lock:
            return self._reload_count


_global_reload_manager: ConfigReloadManager | None = None
_global_lock = threading.Lock()


def setup_signal_handler(config_provider: Callable[[], Config]) -> ConfigReloadManager:
    """Setup SIGHUP signal handler for configuration reload.

    Registers signal handler that invalidates cached configuration
    when SIGHUP is received. Thread-safe singleton pattern.

    Args:
        config_provider: Function that returns Config instance.

    Returns:
        ConfigReloadManager instance.
    """
    global _global_reload_manager, _global_lock

    with _global_lock:
        if _global_reload_manager is not None:
            return _global_reload_manager

        def handle_sighup(signum: int, frame: object) -> None:
            """Handle SIGHUP signal.

            Invalidates cached configuration and logs the reload action.
            """
            logger.info(
                "SIGHUP received, reloading configuration",
                extra={"signal": "SIGHUP", "signum": signum},
            )
            manager = _global_reload_manager
            if manager:
                manager.reload_config()

        signal.signal(signal.SIGHUP, handle_sighup)
        _global_reload_manager = ConfigReloadManager(config_provider)

        logger.info("SIGHUP handler registered for configuration reload")

        return _global_reload_manager


def get_reload_manager() -> ConfigReloadManager | None:
    """Get global reload manager instance.

    Returns:
        ConfigReloadManager if setup_signal_handler has been called, None otherwise.
    """
    global _global_reload_manager, _global_lock

    with _global_lock:
        return _global_reload_manager
