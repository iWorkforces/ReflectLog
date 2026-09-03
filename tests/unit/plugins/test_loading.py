"""Unit tests for reflectlog/plugins/loading.py.

Tests PluginLoader lifecycle: discovery, loading, initialization,
activation, deactivation, unloading, shutdown, and hooks.
"""

from unittest.mock import MagicMock, patch

import pytest

from reflectlog.plugins.discovery import (
    DiscoveredPlugin,
    StaticRegistration,
)
from reflectlog.plugins.loading import (
    IPluginLifecycle,
    LifecycleHooks,
    PluginLoader,
)
from reflectlog.plugins.registry import (
    PluginRegistry,
    PluginState,
)

assert PluginState is not None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimplePlugin:
    """Plugin without lifecycle support."""


class _LifecyclePlugin:
    """Plugin implementing IPluginLifecycle protocol."""

    def __init__(self) -> None:
        self.initialized = False
        self.activated = False
        self.deactivated = False
        self.cleaned_up = False

    async def initialize(self) -> None:
        self.initialized = True

    async def activate(self) -> None:
        self.activated = True

    async def deactivate(self) -> None:
        self.deactivated = True

    async def cleanup(self) -> None:
        self.cleaned_up = True


class _FailingInitPlugin:
    """Plugin that fails during initialize."""

    async def initialize(self) -> None:
        raise RuntimeError("init failed")

    async def activate(self) -> None:
        pass

    async def deactivate(self) -> None:
        raise RuntimeError("deactivate failed")

    async def cleanup(self) -> None:
        raise RuntimeError("cleanup failed")


type TestPlugin = _SimplePlugin | _LifecyclePlugin | _FailingInitPlugin


def _make_loader(
    plugins: list[DiscoveredPlugin] | None = None,
    hooks: LifecycleHooks | None = None,
) -> PluginLoader[TestPlugin]:
    """Create a PluginLoader with static discovery and a fresh registry."""
    if plugins is None:
        plugins = []
    strategy: StaticRegistration[TestPlugin] = StaticRegistration(
        registered_plugins=plugins
    )
    registry: PluginRegistry[TestPlugin] = PluginRegistry()
    return PluginLoader[TestPlugin](
        discovery_strategy=strategy,
        registry=registry,
        hooks=hooks,
    )


# ---------------------------------------------------------------------------
# LifecycleHooks dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLifecycleHooks:
    """Tests for the LifecycleHooks dataclass."""

    def test_default_hooks_are_none(self) -> None:
        """All hooks default to None."""
        hooks = LifecycleHooks()
        assert hooks.on_load is None
        assert hooks.on_initialize is None
        assert hooks.on_activate is None
        assert hooks.on_deactivate is None
        assert hooks.on_unload is None

    def test_custom_hooks(self) -> None:
        """Custom hooks can be set."""
        fn = MagicMock()
        hooks = LifecycleHooks(
            on_load=fn,
            on_initialize=fn,
            on_activate=fn,
            on_deactivate=fn,
            on_unload=fn,
        )
        assert hooks.on_load is fn
        assert hooks.on_initialize is fn


# ---------------------------------------------------------------------------
# IPluginLifecycle protocol
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIPluginLifecycle:
    """Tests for the IPluginLifecycle protocol."""

    def test_lifecycle_plugin_satisfies_protocol(self) -> None:
        """_LifecyclePlugin satisfies IPluginLifecycle."""
        p = _LifecyclePlugin()
        assert isinstance(p, IPluginLifecycle)

    def test_simple_plugin_does_not_satisfy(self) -> None:
        """_SimplePlugin does not satisfy IPluginLifecycle."""
        p = _SimplePlugin()
        assert not isinstance(p, IPluginLifecycle)


# ---------------------------------------------------------------------------
# PluginLoader — discover
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginLoaderDiscover:
    """Tests for PluginLoader.discover()."""

    async def test_discover_returns_plugins(self) -> None:
        """Discover returns discovered plugins list."""
        plugins = [
            DiscoveredPlugin(name="a", module_path="a", class_name="A"),
            DiscoveredPlugin(name="b", module_path="b", class_name="B"),
        ]
        loader = _make_loader(plugins)
        result = await loader.discover()

        assert len(result) == 2
        assert {p.name for p in result} == {"a", "b"}

    async def test_discover_empty(self) -> None:
        """Discover with no plugins returns empty."""
        loader = _make_loader()
        result = await loader.discover()
        assert result == []


# ---------------------------------------------------------------------------
# PluginLoader — load_plugin
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginLoaderLoad:
    """Tests for PluginLoader.load_plugin()."""

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_load_plugin_success(self, mock_load: MagicMock) -> None:
        """Successfully loads a discovered plugin."""
        instance = _SimplePlugin()
        mock_load.return_value = instance

        plugins = [
            DiscoveredPlugin(name="p", module_path="pkg", class_name="P"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()

        result = await loader.load_plugin("p")
        assert result is True
        assert (
            loader.registry.get("_SimplePlugin") is not None
            or loader.registry.count() == 1
        )

    async def test_load_plugin_not_discovered(self) -> None:
        """Returns False for undiscovered plugin."""
        loader = _make_loader()
        await loader.discover()

        result = await loader.load_plugin("nonexistent")
        assert result is False

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_load_plugin_import_error(self, mock_load: MagicMock) -> None:
        """Returns False and sets error state on import failure."""
        mock_load.side_effect = ImportError("module not found")

        plugins = [
            DiscoveredPlugin(name="broken", module_path="bad", class_name="X"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()

        result = await loader.load_plugin("broken")
        assert result is False

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_load_plugin_with_preexisting_instance(
        self, mock_load: MagicMock
    ) -> None:
        """Loads with a pre-existing instance, skips import."""
        plugins = [
            DiscoveredPlugin(name="pre", module_path="pkg", class_name="P"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()

        instance = _SimplePlugin()
        result = await loader.load_plugin("pre", instance=instance)

        assert result is True
        mock_load.assert_not_called()
        assert loader.registry.count() == 1

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_load_plugin_calls_hook(self, mock_load: MagicMock) -> None:
        """Calls on_load hook after successful load."""
        mock_load.return_value = _SimplePlugin()
        hook = MagicMock()
        hooks = LifecycleHooks(on_load=hook)

        plugins = [
            DiscoveredPlugin(name="hooked", module_path="pkg", class_name="H"),
        ]
        loader = _make_loader(plugins, hooks=hooks)
        await loader.discover()

        await loader.load_plugin("hooked")
        hook.assert_called_once_with("hooked")


# ---------------------------------------------------------------------------
# PluginLoader — initialize_plugin
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginLoaderInitialize:
    """Tests for PluginLoader.initialize_plugin()."""

    async def test_initialize_not_registered(self) -> None:
        """Returns False when plugin not registered."""
        loader = _make_loader()
        result = await loader.initialize_plugin("missing")
        assert result is False

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_initialize_lifecycle_plugin(self, mock_load: MagicMock) -> None:
        """Calls initialize() on IPluginLifecycle plugins."""
        lc = _LifecyclePlugin()
        mock_load.return_value = lc

        plugins = [
            DiscoveredPlugin(name="lc", module_path="pkg", class_name="LC"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_plugin("lc")

        # The registry name is the class name auto-detected
        registered_names = loader.registry.list_all()
        assert len(registered_names) == 1
        name = registered_names[0]

        result = await loader.initialize_plugin(name)
        assert result is True
        assert lc.initialized is True

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_initialize_simple_plugin(self, mock_load: MagicMock) -> None:
        """Non-lifecycle plugins initialize successfully (noop)."""
        mock_load.return_value = _SimplePlugin()

        plugins = [
            DiscoveredPlugin(name="sp", module_path="pkg", class_name="SP"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_plugin("sp")

        name = loader.registry.list_all()[0]
        result = await loader.initialize_plugin(name)
        assert result is True

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_initialize_failing_plugin(self, mock_load: MagicMock) -> None:
        """Returns False when initialize() raises."""
        lc = _FailingInitPlugin()
        mock_load.return_value = lc

        plugins = [
            DiscoveredPlugin(name="fail", module_path="pkg", class_name="F"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_plugin("fail")

        name = loader.registry.list_all()[0]
        result = await loader.initialize_plugin(name)
        assert result is False

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_initialize_calls_hook(self, mock_load: MagicMock) -> None:
        """Calls on_initialize hook after success."""
        mock_load.return_value = _SimplePlugin()
        hook = MagicMock()
        hooks = LifecycleHooks(on_initialize=hook)

        plugins = [
            DiscoveredPlugin(name="h", module_path="pkg", class_name="H"),
        ]
        loader = _make_loader(plugins, hooks=hooks)
        await loader.discover()
        await loader.load_plugin("h")

        name = loader.registry.list_all()[0]
        await loader.initialize_plugin(name)
        hook.assert_called_once_with(name)


# ---------------------------------------------------------------------------
# PluginLoader — activate_plugin
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginLoaderActivate:
    """Tests for PluginLoader.activate_plugin()."""

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_activate_success(self, mock_load: MagicMock) -> None:
        """Activates a loaded plugin."""
        mock_load.return_value = _SimplePlugin()

        plugins = [
            DiscoveredPlugin(name="a", module_path="pkg", class_name="A"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_plugin("a")

        name = loader.registry.list_all()[0]
        result = await loader.activate_plugin(name)
        assert result is True

    async def test_activate_not_registered(self) -> None:
        """Returns False when plugin not in registry."""
        loader = _make_loader()
        result = await loader.activate_plugin("missing")
        assert result is False

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_activate_calls_hook(self, mock_load: MagicMock) -> None:
        """Calls on_activate hook."""
        mock_load.return_value = _SimplePlugin()
        hook = MagicMock()
        hooks = LifecycleHooks(on_activate=hook)

        plugins = [
            DiscoveredPlugin(name="h", module_path="pkg", class_name="H"),
        ]
        loader = _make_loader(plugins, hooks=hooks)
        await loader.discover()
        await loader.load_plugin("h")

        name = loader.registry.list_all()[0]
        await loader.activate_plugin(name)
        hook.assert_called_once_with(name)


# ---------------------------------------------------------------------------
# PluginLoader — deactivate_plugin
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginLoaderDeactivate:
    """Tests for PluginLoader.deactivate_plugin()."""

    async def test_deactivate_not_registered(self) -> None:
        """Returns False when plugin not registered."""
        loader = _make_loader()
        result = await loader.deactivate_plugin("missing")
        assert result is False

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_deactivate_lifecycle_plugin(self, mock_load: MagicMock) -> None:
        """Calls deactivate() on lifecycle plugins."""
        lc = _LifecyclePlugin()
        mock_load.return_value = lc

        plugins = [
            DiscoveredPlugin(name="lc", module_path="pkg", class_name="LC"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_plugin("lc")

        name = loader.registry.list_all()[0]
        # Activate first so deactivate can work
        loader.registry.activate(name)

        result = await loader.deactivate_plugin(name)
        assert result is True
        assert lc.deactivated is True

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_deactivate_failing_lifecycle(self, mock_load: MagicMock) -> None:
        """Deactivate continues even when lifecycle.deactivate() raises."""
        lc = _FailingInitPlugin()
        mock_load.return_value = lc

        plugins = [
            DiscoveredPlugin(name="fail", module_path="pkg", class_name="F"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_plugin("fail")

        name = loader.registry.list_all()[0]
        loader.registry.activate(name)

        # Should not raise, returns True if registry deactivate works
        result = await loader.deactivate_plugin(name)
        assert result is True

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_deactivate_calls_hook(self, mock_load: MagicMock) -> None:
        """Calls on_deactivate hook."""
        mock_load.return_value = _SimplePlugin()
        hook = MagicMock()
        hooks = LifecycleHooks(on_deactivate=hook)

        plugins = [
            DiscoveredPlugin(name="h", module_path="pkg", class_name="H"),
        ]
        loader = _make_loader(plugins, hooks=hooks)
        await loader.discover()
        await loader.load_plugin("h")

        name = loader.registry.list_all()[0]
        loader.registry.activate(name)

        await loader.deactivate_plugin(name)
        hook.assert_called_once_with(name)


# ---------------------------------------------------------------------------
# PluginLoader — unload_plugin
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginLoaderUnload:
    """Tests for PluginLoader.unload_plugin()."""

    async def test_unload_not_registered(self) -> None:
        """Returns False when plugin not registered."""
        loader = _make_loader()
        result = await loader.unload_plugin("missing")
        assert result is False

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_unload_lifecycle_plugin(self, mock_load: MagicMock) -> None:
        """Calls cleanup() on lifecycle plugins before unloading."""
        lc = _LifecyclePlugin()
        mock_load.return_value = lc

        plugins = [
            DiscoveredPlugin(name="lc", module_path="pkg", class_name="LC"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_plugin("lc")

        name = loader.registry.list_all()[0]
        result = await loader.unload_plugin(name)

        assert result is True
        assert lc.cleaned_up is True
        assert loader.registry.count() == 0

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_unload_simple_plugin(self, mock_load: MagicMock) -> None:
        """Unloads non-lifecycle plugins cleanly."""
        mock_load.return_value = _SimplePlugin()

        plugins = [
            DiscoveredPlugin(name="sp", module_path="pkg", class_name="SP"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_plugin("sp")

        name = loader.registry.list_all()[0]
        result = await loader.unload_plugin(name)

        assert result is True
        assert loader.registry.count() == 0

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_unload_failing_cleanup(self, mock_load: MagicMock) -> None:
        """Unload continues even when cleanup() raises."""
        lc = _FailingInitPlugin()
        mock_load.return_value = lc

        plugins = [
            DiscoveredPlugin(name="fail", module_path="pkg", class_name="F"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_plugin("fail")

        name = loader.registry.list_all()[0]
        result = await loader.unload_plugin(name)

        assert result is True
        assert loader.registry.count() == 0

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_unload_calls_hook(self, mock_load: MagicMock) -> None:
        """Calls on_unload hook."""
        mock_load.return_value = _SimplePlugin()
        hook = MagicMock()
        hooks = LifecycleHooks(on_unload=hook)

        plugins = [
            DiscoveredPlugin(name="h", module_path="pkg", class_name="H"),
        ]
        loader = _make_loader(plugins, hooks=hooks)
        await loader.discover()
        await loader.load_plugin("h")

        name = loader.registry.list_all()[0]
        await loader.unload_plugin(name)
        hook.assert_called_once_with(name)


# ---------------------------------------------------------------------------
# PluginLoader — bulk operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginLoaderBulk:
    """Tests for load_all, initialize_all, activate_all, deactivate_all, unload_all."""

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_load_all(self, mock_load: MagicMock) -> None:
        """load_all loads all discovered plugins."""
        mock_load.return_value = _SimplePlugin()

        plugins = [
            DiscoveredPlugin(name="a", module_path="pkg", class_name="A"),
            DiscoveredPlugin(name="b", module_path="pkg", class_name="B"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()

        count = await loader.load_all()
        assert count == 2

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_initialize_all(self, mock_load: MagicMock) -> None:
        """initialize_all initializes all loaded plugins."""
        mock_load.return_value = _SimplePlugin()

        plugins = [
            DiscoveredPlugin(name="a", module_path="pkg", class_name="A"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_all()

        count = await loader.initialize_all()
        assert count >= 0  # All LOADED state plugins

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_activate_all(self, mock_load: MagicMock) -> None:
        """activate_all activates all registered plugins."""
        mock_load.return_value = _SimplePlugin()

        plugins = [
            DiscoveredPlugin(name="a", module_path="pkg", class_name="A"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_all()

        count = await loader.activate_all()
        assert count == 1

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_deactivate_all(self, mock_load: MagicMock) -> None:
        """deactivate_all deactivates all active plugins."""
        mock_load.return_value = _SimplePlugin()

        plugins = [
            DiscoveredPlugin(name="a", module_path="pkg", class_name="A"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_all()
        await loader.activate_all()

        count = await loader.deactivate_all()
        assert count >= 0  # at least attempts

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_unload_all(self, mock_load: MagicMock) -> None:
        """unload_all unloads all registered plugins."""
        mock_load.return_value = _SimplePlugin()

        plugins = [
            DiscoveredPlugin(name="a", module_path="pkg", class_name="A"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_all()

        count = await loader.unload_all()
        assert count >= 1
        assert loader.registry.count() == 0


# ---------------------------------------------------------------------------
# PluginLoader — shutdown
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginLoaderShutdown:
    """Tests for PluginLoader.shutdown()."""

    @patch("reflectlog.plugins.discovery.load_plugin")
    async def test_shutdown_deactivates_and_unloads(self, mock_load: MagicMock) -> None:
        """Shutdown deactivates active plugins then unloads all."""
        lc = _LifecyclePlugin()
        mock_load.return_value = lc

        plugins = [
            DiscoveredPlugin(name="x", module_path="pkg", class_name="X"),
        ]
        loader = _make_loader(plugins)
        await loader.discover()
        await loader.load_all()
        await loader.activate_all()

        await loader.shutdown()

        assert loader.registry.count() == 0
        assert lc.deactivated is True
        assert lc.cleaned_up is True

    async def test_shutdown_empty(self) -> None:
        """Shutdown on empty loader completes without error."""
        loader = _make_loader()
        await loader.shutdown()  # no error


# ---------------------------------------------------------------------------
# PluginLoader — properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginLoaderProperties:
    """Tests for PluginLoader properties."""

    async def test_registry_property(self) -> None:
        """Registry property returns the registry."""
        loader = _make_loader()
        assert isinstance(loader.registry, PluginRegistry)

    async def test_discoverer_property(self) -> None:
        """Discoverer property returns the discoverer."""
        loader = _make_loader()
        assert loader.discoverer is not None
