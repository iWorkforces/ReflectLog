"""Unit tests for reflectlog/plugins/discovery.py.

Tests plugin discovery via entry points, directory scan, static registration,
composite discovery, load_plugin function, and PluginDiscoverer orchestrator.
"""

import importlib
import importlib.metadata
import pkgutil
import types
from unittest.mock import MagicMock, patch
from typing import Any, cast

import pytest

from reflectlog.plugins.discovery import (
    CompositeDiscovery,
    DirectoryScanDiscovery,
    DiscoveredPlugin,
    EntryPointDiscovery,
    PluginDiscoverer,
    PluginDiscoveryStrategy,
    StaticRegistration,
    load_plugin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _BasePlugin:
    """Base plugin class for directory scan tests."""


class _ConcretePlugin(_BasePlugin):
    """Concrete plugin subclass."""


class _AnotherPlugin(_BasePlugin):
    """Another concrete plugin subclass."""


def _make_entry_point(
    name: str = "test_plugin",
    value: str = "my_pkg.plugins:MyPlugin",
    group: str = "reflectlog.plugins",
) -> MagicMock:
    """Create a mock entry point."""
    ep = MagicMock()
    ep.name = name
    ep.value = value
    ep.group = group
    ep.__str__ = lambda self: f"{name} = {value}"
    return ep


# ---------------------------------------------------------------------------
# DiscoveredPlugin dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDiscoveredPlugin:
    """Tests for the DiscoveredPlugin dataclass."""

    def test_required_fields(self) -> None:
        """DiscoveredPlugin stores name, module_path, class_name."""
        dp = DiscoveredPlugin(
            name="foo",
            module_path="foo.bar",
            class_name="FooPlugin",
        )
        assert dp.name == "foo"
        assert dp.module_path == "foo.bar"
        assert dp.class_name == "FooPlugin"

    def test_default_version(self) -> None:
        """Default version is 0.0.0."""
        dp = DiscoveredPlugin(name="x", module_path="x", class_name="X")
        assert dp.version == "0.0.0"

    def test_default_entry_point_is_none(self) -> None:
        """Default entry_point is None."""
        dp = DiscoveredPlugin(name="x", module_path="x", class_name="X")
        assert dp.entry_point is None

    def test_custom_version_and_entry_point(self) -> None:
        """Custom version and entry_point are stored."""
        dp = DiscoveredPlugin(
            name="x",
            module_path="x",
            class_name="X",
            version="1.0.0",
            entry_point="x = x:X",
        )
        assert dp.version == "1.0.0"
        assert dp.entry_point == "x = x:X"


# ---------------------------------------------------------------------------
# PluginDiscoveryStrategy base class
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginDiscoveryStrategy:
    """Tests for the base PluginDiscoveryStrategy."""

    async def test_discover_raises_not_implemented(self) -> None:
        """Base discover() raises NotImplementedError."""
        strategy: PluginDiscoveryStrategy[object] = PluginDiscoveryStrategy()
        with pytest.raises(NotImplementedError):
            await strategy.discover()


# ---------------------------------------------------------------------------
# EntryPointDiscovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntryPointDiscovery:
    """Tests for entry point discovery."""

    @patch("reflectlog.plugins.discovery.importlib.metadata.entry_points")
    async def test_discover_with_colon_in_value(self, mock_eps: MagicMock) -> None:
        """Entry point with 'module:Class' is parsed correctly."""
        ep = _make_entry_point("myplugin", "my_pkg.mod:MyClass")
        mock_eps.return_value = [ep]

        strategy = EntryPointDiscovery(group="reflectlog.plugins", plugin_type=object)
        result = await strategy.discover()

        assert len(result) == 1
        assert result[0].name == "myplugin"
        assert result[0].module_path == "my_pkg.mod"
        assert result[0].class_name == "MyClass"
        assert result[0].entry_point is not None

    @patch("reflectlog.plugins.discovery.importlib.metadata.entry_points")
    async def test_discover_without_colon_in_value(self, mock_eps: MagicMock) -> None:
        """Entry point without ':' treats value as module path, empty class."""
        ep = _make_entry_point("myplugin", "my_pkg.mod")
        mock_eps.return_value = [ep]

        strategy = EntryPointDiscovery(group="reflectlog.plugins", plugin_type=object)
        result = await strategy.discover()

        assert len(result) == 1
        assert result[0].module_path == "my_pkg.mod"
        assert result[0].class_name == ""

    @patch("reflectlog.plugins.discovery.importlib.metadata.entry_points")
    async def test_discover_multiple_entry_points(self, mock_eps: MagicMock) -> None:
        """Multiple entry points are all discovered."""
        eps = [
            _make_entry_point("p1", "mod1:C1"),
            _make_entry_point("p2", "mod2:C2"),
            _make_entry_point("p3", "mod3:C3"),
        ]
        mock_eps.return_value = eps

        strategy = EntryPointDiscovery(group="reflectlog.plugins", plugin_type=object)
        result = await strategy.discover()

        assert len(result) == 3
        names = {p.name for p in result}
        assert names == {"p1", "p2", "p3"}

    @patch("reflectlog.plugins.discovery.importlib.metadata.entry_points")
    async def test_discover_empty(self, mock_eps: MagicMock) -> None:
        """No entry points returns empty list."""
        mock_eps.return_value = []

        strategy = EntryPointDiscovery(group="reflectlog.plugins", plugin_type=object)
        result = await strategy.discover()

        assert result == []

    @patch("reflectlog.plugins.discovery.importlib.metadata.entry_points")
    async def test_discover_with_type_error_and_select(
        self, mock_eps: MagicMock
    ) -> None:
        """Falls back to .select() on TypeError (Python 3.9 compat)."""
        mock_eps.side_effect = [TypeError("no group arg")]

        # Second call (without group=...) returns object with select
        selectable = MagicMock()
        ep = _make_entry_point("fallback", "fb_mod:FbClass")
        selectable.select.return_value = [ep]
        mock_eps.side_effect = [TypeError("no group arg"), selectable]
        # Patch so the second call returns the selectable object
        call_count = 0
        original_side_effect = mock_eps.side_effect

        def side_effect_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("no group arg")
            return selectable

        mock_eps.side_effect = side_effect_fn

        strategy = EntryPointDiscovery(group="reflectlog.plugins", plugin_type=object)
        result = await strategy.discover()

        assert len(result) == 1
        assert result[0].name == "fallback"

    @patch("reflectlog.plugins.discovery.importlib.metadata.entry_points")
    async def test_discover_with_type_error_no_select(
        self, mock_eps: MagicMock
    ) -> None:
        """Falls back to iteration + filter when no .select() available."""
        ep_match = _make_entry_point("match", "m:M", group="reflectlog.plugins")
        ep_other = _make_entry_point("other", "o:O", group="other.group")

        call_count = 0

        def side_effect_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("no group arg")
            # Return list-like without select
            return [ep_match, ep_other]

        mock_eps.side_effect = side_effect_fn

        strategy = EntryPointDiscovery(group="reflectlog.plugins", plugin_type=object)
        result = await strategy.discover()

        assert len(result) == 1
        assert result[0].name == "match"

    @patch("reflectlog.plugins.discovery.importlib.metadata.entry_points")
    async def test_discover_rsplit_on_multiple_colons(
        self, mock_eps: MagicMock
    ) -> None:
        """rsplit(':', 1) handles value with multiple colons correctly."""
        ep = _make_entry_point("nested", "my.pkg:sub:Cls")
        mock_eps.return_value = [ep]

        strategy = EntryPointDiscovery(group="reflectlog.plugins", plugin_type=object)
        result = await strategy.discover()

        assert result[0].module_path == "my.pkg:sub"
        assert result[0].class_name == "Cls"


# ---------------------------------------------------------------------------
# DirectoryScanDiscovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDirectoryScanDiscovery:
    """Tests for directory scan discovery."""

    async def test_discover_finds_subclasses(self) -> None:
        """Discovers classes that are subclasses of the base class."""
        pkg_mod = types.ModuleType("test_pkg")
        pkg_mod.__file__ = "/fake/test_pkg/__init__.py"

        plugin_mod = MagicMock()
        plugin_mod.ConcretePlugin = _ConcretePlugin
        plugin_mod.AnotherPlugin = _AnotherPlugin
        plugin_mod._BasePlugin = _BasePlugin
        setattr(plugin_mod, "__dir__", lambda self: ["ConcretePlugin", "AnotherPlugin", "_BasePlugin"])

        _orig = importlib.import_module

        def import_side_effect(name: str):
            if name == "test_pkg":
                return pkg_mod
            if name == "test_pkg.plugin_foo":
                return plugin_mod
            return _orig(name)

        with patch("reflectlog.plugins.discovery.importlib.import_module", side_effect=import_side_effect):
            with patch("pkgutil.iter_modules", return_value=[(None, "test_pkg.plugin_foo", False)]):
                strategy = DirectoryScanDiscovery(
                    package_names=["test_pkg"],
                    plugin_base_class=_BasePlugin,
                )
                result = await strategy.discover()

        names = {p.class_name for p in result}
        assert "ConcretePlugin" in names
        assert "AnotherPlugin" in names
        assert "_BasePlugin" not in names

    @patch("pkgutil.iter_modules")
    @patch("reflectlog.plugins.discovery.importlib.import_module")
    async def test_discover_skips_packages(
        self, mock_import: MagicMock, mock_iter: MagicMock
    ) -> None:
        """Skips sub-packages (ispkg=True)."""
        pkg_mod = types.ModuleType("test_pkg")
        pkg_mod.__file__ = "/fake/test_pkg/__init__.py"
        mock_import.return_value = pkg_mod
        mock_iter.return_value = [
            (None, "test_pkg.subpkg", True),  # ispkg=True
        ]

        strategy = DirectoryScanDiscovery(
            package_names=["test_pkg"],
            plugin_base_class=_BasePlugin,
        )
        result = await strategy.discover()

        assert result == []

    @patch("reflectlog.plugins.discovery.importlib.import_module")
    async def test_discover_skips_namespace_packages(
        self, mock_import: MagicMock
    ) -> None:
        """Skips namespace packages (__file__ is None)."""
        pkg_mod = types.ModuleType("ns_pkg")
        pkg_mod.__file__ = None
        mock_import.return_value = pkg_mod

        strategy = DirectoryScanDiscovery(
            package_names=["ns_pkg"],
            plugin_base_class=_BasePlugin,
        )
        result = await strategy.discover()

        assert result == []

    @patch("reflectlog.plugins.discovery.importlib.import_module")
    async def test_discover_handles_import_error(self, mock_import: MagicMock) -> None:
        """Import errors are silently skipped."""
        mock_import.side_effect = ImportError("not installed")

        strategy = DirectoryScanDiscovery(
            package_names=["missing_pkg"],
            plugin_base_class=_BasePlugin,
        )
        result = await strategy.discover()

        assert result == []

    async def test_discover_multiple_packages(self) -> None:
        """Scans all listed packages."""
        pkg1 = types.ModuleType("pkg1")
        pkg1.__file__ = "/fake/pkg1/__init__.py"
        pkg2 = types.ModuleType("pkg2")
        pkg2.__file__ = "/fake/pkg2/__init__.py"

        Sub1 = type("Sub1", (_BasePlugin,), {})
        Sub2 = type("Sub2", (_BasePlugin,), {})

        mod1 = MagicMock()
        mod1.Sub1 = Sub1
        setattr(mod1, "__dir__", lambda self: ["Sub1"])

        mod2 = MagicMock()
        mod2.Sub2 = Sub2
        setattr(mod2, "__dir__", lambda self: ["Sub2"])

        _orig = importlib.import_module

        def import_side_effect(name: str):
            mapping = {
                "pkg1": pkg1,
                "pkg2": pkg2,
                "pkg1.plugin_a": mod1,
                "pkg2.plugin_b": mod2,
            }
            if name in mapping:
                return mapping[name]
            return _orig(name)

        def iter_side_effect(paths, prefix=""):
            if "pkg1" in prefix:
                return [(None, "pkg1.plugin_a", False)]
            return [(None, "pkg2.plugin_b", False)]

        with patch("reflectlog.plugins.discovery.importlib.import_module", side_effect=import_side_effect):
            with patch("pkgutil.iter_modules", side_effect=iter_side_effect):
                strategy = DirectoryScanDiscovery(
                    package_names=["pkg1", "pkg2"],
                    plugin_base_class=_BasePlugin,
                )
                result = await strategy.discover()

        class_names = {p.class_name for p in result}
        assert "Sub1" in class_names
        assert "Sub2" in class_names
    async def test_discover_naming_convention(self) -> None:
        """Verifies naming convention for discovered plugins."""
        pkg_mod = types.ModuleType("mypkg")
        pkg_mod.__file__ = "/fake/mypkg/__init__.py"

        MyImpl = type("MyImpl", (_BasePlugin,), {})
        plugin_mod = MagicMock()
        plugin_mod.MyImpl = MyImpl
        setattr(plugin_mod, "__dir__", lambda self: ["MyImpl"])

        _orig = importlib.import_module

        def import_side_effect(name: str):
            if name == "mypkg":
                return pkg_mod
            if name == "mypkg.plugin_x":
                return plugin_mod
            return _orig(name)

        with patch("reflectlog.plugins.discovery.importlib.import_module", side_effect=import_side_effect):
            with patch("pkgutil.iter_modules", return_value=[(None, "mypkg.plugin_x", False)]):
                strategy = DirectoryScanDiscovery(
                    package_names=["mypkg"],
                    plugin_base_class=_BasePlugin,
                )
                result = await strategy.discover()

        assert len(result) >= 1
        assert result[0].name == "mypkg_MyImpl"
        assert result[0].module_path == "mypkg.plugin_x"
        assert result[0].class_name == "MyImpl"

# ---------------------------------------------------------------------------
# StaticRegistration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStaticRegistration:
    """Tests for static plugin registration."""

    async def test_returns_copy_of_registered(self) -> None:
        """Returns a copy, not the original list."""
        plugins = [
            DiscoveredPlugin(name="a", module_path="a", class_name="A"),
        ]
        strategy = StaticRegistration(registered_plugins=plugins)
        result = await strategy.discover()

        assert result == plugins
        assert result is not plugins

    async def test_empty_registration(self) -> None:
        """Empty list returns empty."""
        strategy = StaticRegistration(registered_plugins=[])
        result = await strategy.discover()
        assert result == []

    async def test_multiple_plugins(self) -> None:
        """All statically registered plugins are returned."""
        plugins = [
            DiscoveredPlugin(name="a", module_path="a", class_name="A"),
            DiscoveredPlugin(name="b", module_path="b", class_name="B"),
            DiscoveredPlugin(name="c", module_path="c", class_name="C"),
        ]
        strategy = StaticRegistration(registered_plugins=plugins)
        result = await strategy.discover()

        assert len(result) == 3
        assert {p.name for p in result} == {"a", "b", "c"}

    async def test_mutation_after_discover_does_not_affect_original(self) -> None:
        """Mutating result doesn't affect internal state."""
        plugins = [
            DiscoveredPlugin(name="x", module_path="x", class_name="X"),
        ]
        strategy = StaticRegistration(registered_plugins=plugins)
        result = await strategy.discover()
        result.append(DiscoveredPlugin(name="y", module_path="y", class_name="Y"))

        second = await strategy.discover()
        assert len(second) == 1


# ---------------------------------------------------------------------------
# CompositeDiscovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompositeDiscovery:
    """Tests for composite discovery combining multiple strategies."""

    async def test_combines_results(self) -> None:
        """Results from all strategies are combined."""
        s1 = StaticRegistration(
            registered_plugins=[
                DiscoveredPlugin(name="a", module_path="a", class_name="A"),
            ]
        )
        s2 = StaticRegistration(
            registered_plugins=[
                DiscoveredPlugin(name="b", module_path="b", class_name="B"),
            ]
        )

        composite = CompositeDiscovery(strategies=cast(list[PluginDiscoveryStrategy[Any]], [s1, s2]))
        result = await composite.discover()

        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"a", "b"}

    async def test_deduplicates_by_name(self) -> None:
        """Duplicate names are deduplicated, first wins."""
        s1 = StaticRegistration(
            registered_plugins=[
                DiscoveredPlugin(name="dup", module_path="first", class_name="First"),
            ]
        )
        s2 = StaticRegistration(
            registered_plugins=[
                DiscoveredPlugin(name="dup", module_path="second", class_name="Second"),
            ]
        )

        composite = CompositeDiscovery(strategies=cast(list[PluginDiscoveryStrategy[Any]], [s1, s2]))
        result = await composite.discover()

        assert len(result) == 1
        assert result[0].module_path == "first"

    async def test_empty_strategies(self) -> None:
        """No strategies returns empty."""
        composite = CompositeDiscovery(strategies=[])
        result = await composite.discover()
        assert result == []

    async def test_all_empty_strategies(self) -> None:
        """All strategies returning empty gives empty."""
        s1 = StaticRegistration(registered_plugins=[])
        s2 = StaticRegistration(registered_plugins=[])

        composite = CompositeDiscovery(strategies=cast(list[PluginDiscoveryStrategy[Any]], [s1, s2]))
        result = await composite.discover()
        assert result == []


# ---------------------------------------------------------------------------
# load_plugin (module-level function)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadPlugin:
    """Tests for the load_plugin function."""

    @patch("reflectlog.plugins.discovery.importlib.import_module")
    async def test_load_with_class_name(self, mock_import: MagicMock) -> None:
        """Loads and instantiates the specified class."""
        mock_cls = MagicMock(return_value="instance")
        mock_module = MagicMock()
        mock_module.MyPlugin = mock_cls
        mock_import.return_value = mock_module

        dp = DiscoveredPlugin(name="test", module_path="pkg.mod", class_name="MyPlugin")
        result = await load_plugin(dp)

        mock_import.assert_called_once_with("pkg.mod")
        mock_cls.assert_called_once()
        assert result == "instance"

    @patch("reflectlog.plugins.discovery.importlib.import_module")
    async def test_load_without_class_name(self, mock_import: MagicMock) -> None:
        """Returns the module itself when no class_name."""
        mock_module = MagicMock()
        mock_import.return_value = mock_module

        dp = DiscoveredPlugin(name="test", module_path="pkg.mod", class_name="")
        result = await load_plugin(dp)

        assert result is mock_module

    @patch("reflectlog.plugins.discovery.importlib.import_module")
    async def test_load_raises_on_missing_module(self, mock_import: MagicMock) -> None:
        """Raises ImportError when module doesn't exist."""
        mock_import.side_effect = ImportError("no such module")

        dp = DiscoveredPlugin(name="test", module_path="bad.mod", class_name="X")
        with pytest.raises(ImportError, match="no such module"):
            await load_plugin(dp)

    @patch("reflectlog.plugins.discovery.importlib.import_module")
    async def test_load_raises_on_missing_class(self, mock_import: MagicMock) -> None:
        """Raises AttributeError when class doesn't exist in module."""
        mock_module = MagicMock(spec=[])
        mock_import.return_value = mock_module

        dp = DiscoveredPlugin(name="test", module_path="pkg.mod", class_name="Missing")
        with pytest.raises(AttributeError):
            await load_plugin(dp)


# ---------------------------------------------------------------------------
# PluginDiscoverer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginDiscoverer:
    """Tests for the PluginDiscoverer orchestrator class."""

    async def test_discover_plugins(self) -> None:
        """Discovers plugins via strategy."""
        plugins = [
            DiscoveredPlugin(name="a", module_path="a", class_name="A"),
            DiscoveredPlugin(name="b", module_path="b", class_name="B"),
        ]
        strategy = StaticRegistration(registered_plugins=plugins)
        discoverer: PluginDiscoverer[object] = PluginDiscoverer(strategy)

        result = await discoverer.discover_plugins()

        assert len(result) == 2
        assert discoverer.discovered_plugins == result

    async def test_discovered_plugins_returns_copy(self) -> None:
        """discovered_plugins property returns a copy."""
        strategy = StaticRegistration(
            registered_plugins=[
                DiscoveredPlugin(name="x", module_path="x", class_name="X"),
            ]
        )
        discoverer: PluginDiscoverer[object] = PluginDiscoverer(strategy)
        await discoverer.discover_plugins()

        copy1 = discoverer.discovered_plugins
        copy2 = discoverer.discovered_plugins
        assert copy1 == copy2
        assert copy1 is not copy2

    @patch("reflectlog.plugins.discovery.importlib.import_module")
    async def test_load_plugin_by_name(self, mock_import: MagicMock) -> None:
        """Loads a specific plugin by name."""
        mock_cls = MagicMock(return_value=_ConcretePlugin())
        mock_module = MagicMock()
        mock_module.ConcretePlugin = mock_cls
        mock_import.return_value = mock_module

        plugins = [
            DiscoveredPlugin(
                name="concrete", module_path="pkg", class_name="ConcretePlugin"
            ),
        ]
        strategy = StaticRegistration(registered_plugins=plugins)
        discoverer: PluginDiscoverer[_BasePlugin] = PluginDiscoverer(strategy)
        await discoverer.discover_plugins()

        instance = await discoverer.load_plugin("concrete")
        assert instance is not None

    async def test_load_plugin_not_found(self) -> None:
        """Returns None when plugin name not discovered."""
        strategy = StaticRegistration(registered_plugins=[])
        discoverer: PluginDiscoverer[object] = PluginDiscoverer(strategy)
        await discoverer.discover_plugins()

        result = await discoverer.load_plugin("nonexistent")
        assert result is None

    @patch("reflectlog.plugins.discovery.importlib.import_module")
    async def test_load_plugin_cached(self, mock_import: MagicMock) -> None:
        """Second load of same plugin returns cached instance."""
        obj = _ConcretePlugin()
        mock_cls = MagicMock(return_value=obj)
        mock_module = MagicMock()
        mock_module.C = mock_cls
        mock_import.return_value = mock_module

        plugins = [
            DiscoveredPlugin(name="c", module_path="pkg", class_name="C"),
        ]
        strategy = StaticRegistration(registered_plugins=plugins)
        discoverer: PluginDiscoverer[_BasePlugin] = PluginDiscoverer(strategy)
        await discoverer.discover_plugins()

        first = await discoverer.load_plugin("c")
        second = await discoverer.load_plugin("c")

        assert first is second
        mock_cls.assert_called_once()

    @patch("reflectlog.plugins.discovery.importlib.import_module")
    async def test_load_all_plugins(self, mock_import: MagicMock) -> None:
        """Loads all discovered plugins."""
        mock_module = MagicMock()
        mock_module.A = MagicMock(return_value="inst_a")
        mock_module.B = MagicMock(return_value="inst_b")
        mock_import.return_value = mock_module

        plugins = [
            DiscoveredPlugin(name="a", module_path="pkg", class_name="A"),
            DiscoveredPlugin(name="b", module_path="pkg", class_name="B"),
        ]
        strategy = StaticRegistration(registered_plugins=plugins)
        discoverer: PluginDiscoverer[object] = PluginDiscoverer(strategy)
        await discoverer.discover_plugins()

        loaded = await discoverer.load_all_plugins()
        assert len(loaded) == 2

    @patch("reflectlog.plugins.discovery.importlib.import_module")
    async def test_loaded_plugins_property(self, mock_import: MagicMock) -> None:
        """loaded_plugins returns dict of name -> instance."""
        mock_module = MagicMock()
        mock_module.X = MagicMock(return_value="x_inst")
        mock_import.return_value = mock_module

        plugins = [
            DiscoveredPlugin(name="x", module_path="pkg", class_name="X"),
        ]
        strategy = StaticRegistration(registered_plugins=plugins)
        discoverer: PluginDiscoverer[object] = PluginDiscoverer(strategy)
        await discoverer.discover_plugins()
        await discoverer.load_plugin("x")

        loaded = discoverer.loaded_plugins
        assert "x" in loaded
        assert loaded is not discoverer.loaded_plugins  # copy
