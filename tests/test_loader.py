"""
Tests for the plugin loader (py4D_browser.plugins).

Fake plugin modules are registered in ``sys.modules`` under the
``py4d_browser_plugin`` namespace and ``pkgutil.iter_modules`` is
monkeypatched, so these tests never see (or depend on) the real sample
plugins.
"""

import importlib.machinery
import pkgutil
import sys
import types

import pytest

from py4D_browser.plugin_api import PluginAPI1


def make_plugin(name, **attrs):
    """
    Build a plugin class that records the ``parent``/``api`` kwargs it
    received, plus any class attributes passed in (``api_version``,
    ``full_access``, ``post_init``, ...).
    """

    def __init__(self, **kwargs):
        # `parent` is only present for full_access plugins
        self.parent = kwargs.get("parent")
        self.api = kwargs["api"]

    namespace = {
        "plugin_id": f"test.{name}",
        "__init__": __init__,
    }
    namespace.update(attrs)
    return type(f"Fake{name}Plugin", (), namespace)


@pytest.fixture
def load_fake_plugins(viewer, monkeypatch):
    """
    Load the given (name, plugin_class) pairs through the real loader,
    with the discovery mechanism pointed at the fakes only.
    """

    def _load(specs):
        for name, cls in specs:
            module = types.ModuleType(f"py4d_browser_plugin.{name}")
            setattr(module, cls.__name__, cls)
            monkeypatch.setitem(sys.modules, f"py4d_browser_plugin.{name}", module)

        infos = [
            pkgutil.ModuleInfo(importlib.machinery.FileFinder, name, False)
            for name, _ in specs
        ]
        monkeypatch.setattr(pkgutil, "iter_modules", lambda *a, **k: iter(infos))

        viewer.load_plugins()
        return viewer.loaded_plugins

    return _load


########## Parent / api dispatch ##########


def test_compatible_plugin_receives_api_object(viewer, load_fake_plugins):
    plugin_cls = make_plugin("compat", api_version=(1, 6))

    loaded = load_fake_plugins([("compat", plugin_cls)])

    assert len(loaded) == 1
    entry = loaded[0]
    assert entry["id"] == plugin_cls.plugin_id
    assert type(entry["api"]) is PluginAPI1
    # standard plugins get the API object as `api` and NO `parent`
    assert entry["parent"] is None
    assert entry["api"] is not viewer
    assert entry["plugin"].parent is None
    assert entry["plugin"].api is entry["api"]


def test_missing_api_version_is_skipped(viewer, load_fake_plugins, capsys):
    plugin_cls = make_plugin("noversion")

    loaded = load_fake_plugins([("noversion", plugin_cls)])

    assert loaded == []
    out = capsys.readouterr().out
    assert "Skipping plugin" in out
    assert "test.noversion" in out
    assert "api_version" in out


def test_full_access_plugin_receives_both(viewer, load_fake_plugins):
    plugin_cls = make_plugin("fullaccess", api_version=(1, 6), full_access=True)

    loaded = load_fake_plugins([("fullaccess", plugin_cls)])

    entry = loaded[0]
    assert entry["parent"] is viewer
    assert type(entry["api"]) is PluginAPI1
    assert entry["api"] is not viewer
    assert entry["plugin"].parent is viewer
    assert entry["plugin"].api is entry["api"]


########## Version incompatibility does not abort loading ##########


@pytest.mark.parametrize("api_version", [(1, 9), (2, 0), (99, 0)])
def test_ahead_plugin_is_skipped(viewer, load_fake_plugins, capsys, api_version):
    plugin_cls = make_plugin("ahead", api_version=api_version)

    loaded = load_fake_plugins([("ahead", plugin_cls)])

    assert loaded == []
    out = capsys.readouterr().out
    assert "Skipping plugin" in out
    assert "test.ahead" in out


def test_mixed_plugins_only_compatible_one_loads(viewer, load_fake_plugins, capsys):
    good = make_plugin("good", api_version=(1, 6))
    ahead = make_plugin("ahead", api_version=(2, 0))

    loaded = load_fake_plugins([("good", good), ("ahead", ahead)])

    assert [e["id"] for e in loaded] == ["test.good"]
    out = capsys.readouterr().out
    assert "Skipping plugin" in out
    assert "test.ahead" in out


########## post_init ##########


def test_post_init_receives_same_objects(viewer, load_fake_plugins):
    def post_init(self, **kwargs):
        self.post_init_parent = kwargs["parent"]
        self.post_init_api = kwargs["api"]

    plugin_cls = make_plugin(
        "postinit", api_version=(1, 6), full_access=True, post_init=post_init
    )

    loaded = load_fake_plugins([("postinit", plugin_cls)])

    plugin = loaded[0]["plugin"]
    assert plugin.post_init_parent is plugin.parent is viewer
    assert plugin.post_init_api is plugin.api


########## Dev-only plugins ##########


def test_dev_only_plugin_skipped_silently_when_disabled(
    viewer, load_fake_plugins, capsys
):
    viewer.settings.setValue("gui/dev_plugins", "0")
    plugin_cls = make_plugin("devonly", api_version=(1, 6), dev_only=True)

    loaded = load_fake_plugins([("devonly", plugin_cls)])

    assert loaded == []
    out = capsys.readouterr().out
    # Skipped silently: no "Loading plugin" and no "Skipping plugin" line.
    assert "test.devonly" not in out
    assert "Loading plugin" not in out
    assert "Skipping plugin" not in out


def test_dev_only_plugin_loads_when_enabled(viewer, load_fake_plugins):
    viewer.settings.setValue("gui/dev_plugins", "1")
    plugin_cls = make_plugin("devonly", api_version=(1, 6), dev_only=True)

    try:
        loaded = load_fake_plugins([("devonly", plugin_cls)])
        assert [e["id"] for e in loaded] == ["test.devonly"]
    finally:
        # restore the default so the session sees dev plugins off again
        viewer.settings.setValue("gui/dev_plugins", "0")


def test_normal_plugin_loads_regardless_of_dev_setting(viewer, load_fake_plugins):
    viewer.settings.setValue("gui/dev_plugins", "0")
    plugin_cls = make_plugin("normal", api_version=(1, 6))

    loaded = load_fake_plugins([("normal", plugin_cls)])

    assert [e["id"] for e in loaded] == ["test.normal"]


########## Real sample plugins ##########


def test_real_sample_plugins_load(viewer):
    # Reload with the real discovery mechanism (no fakes patched in),
    # in case the fake-plugin tests above ran first.
    viewer.load_plugins()

    ids = {entry["id"] for entry in viewer.loaded_plugins}
    assert {
        "py4DGUI.internal.calibration",
        "py4DGUI.internal.logging",
        "py4DGUI.internal.metadata",
        "py4DGUI.internal.copy_to_tab",
    } <= ids

    for entry in viewer.loaded_plugins:
        assert type(entry["api"]) is PluginAPI1
        # all sample plugins use the standard (non-full-access) mode,
        # so they are passed the API object only (no `parent`)
        assert entry["parent"] is None
