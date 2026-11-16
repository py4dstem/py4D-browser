import pkgutil
import importlib
import inspect
import traceback

from PyQt5.QtWidgets import QMenu, QAction

from py4D_browser.plugin_api import (
    PluginAPIVersionError,
    resolve_api_class,
)
from py4D_browser.utils import strtobool

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py4D_browser import DataViewer

__all__ = ["load_plugins", "unload_plugins"]


def load_plugins(self: "DataViewer"):
    """
    The py4D_browser plugin mechanics are inspired by Nion Swift:
    https://nionswift.readthedocs.io/en/stable/api/plugins.html

    Plugins should create a module in the py4d_browser_plugin namespace
    and should define a class with the `plugin_id` attribute and an
    `api_version` attribute declaring the plugin API version it was
    written against (see py4D_browser.plugin_api).

    On loading the class is initialized using
        ExamplePlugin(api=..., plugin_menu=..., plugin_action=...)
    with all arguments passed as kwargs:

    * `api` is an instance of the versioned plugin API matching the
      plugin's declared `api_version`.
    * `parent` (the raw DataViewer) is passed only when the plugin sets
      `full_access = True`.

    A plugin that does not declare `api_version`, or that requires an API
    version this browser does not provide, is logged and skipped; loading
    the remaining plugins continues.

    A plugin that sets `dev_only = True` is for development/testing use and
    is skipped **silently** (no log output) unless the `gui/dev_plugins`
    setting in the config file is truthy. This lets development plugins be
    committed and tracked in-repo without exposing them to end users, who
    must opt in by setting `gui/dev_plugins = true`.
    """

    import py4d_browser_plugin

    self.loaded_plugins = []  # we need to hold on to these objects to keep them alive

    # Dev-only plugins are hidden from end users unless they opt in via the
    # config file. Default is off.
    dev_plugins_enabled = strtobool(self.settings.value("gui/dev_plugins", "0"))

    for module_info in pkgutil.iter_modules(getattr(py4d_browser_plugin, "__path__")):

        try:
            module = importlib.import_module(
                py4d_browser_plugin.__name__ + "." + module_info.name
            )
        except Exception as e:
            print(
                f"Attempting to import plugin {module_info.name} raised exception:\n{e}"
            )
            print(traceback.print_exc())
            continue

        for name, member in inspect.getmembers(module, inspect.isclass):
            plugin_id = getattr(member, "plugin_id", None)

            if plugin_id:
                # Dev-only plugins are skipped silently (no log) unless the
                # user has opted in via the `gui/dev_plugins` setting.
                if bool(getattr(member, "dev_only", False)) and not dev_plugins_enabled:
                    continue

                print(f"Loading plugin: {plugin_id} \tfrom: {name}")
                try:
                    # Resolve the versioned API this plugin was written
                    # against. Declaring `api_version` is required; a plugin
                    # that omits it (or requires a version this browser does
                    # not provide) is skipped with a log message.
                    api_version = getattr(member, "api_version", None)
                    if api_version is None:
                        raise PluginAPIVersionError(
                            "does not declare an `api_version`. Declare "
                            "`api_version = (1, 6)` (or the version this "
                            "plugin was built against) to load it."
                        )
                    api = resolve_api_class(api_version)(self)

                    # `api` is always passed. The raw DataViewer as `parent`
                    # is passed only for full_access plugins (an unversioned
                    # escape hatch with no compatibility guarantees).
                    parent_obj = (
                        self if bool(getattr(member, "full_access", False)) else None
                    )

                    plugin_menu = (
                        QMenu(getattr(member, "display_name", "DEFAULT_NAME"))
                        if getattr(member, "uses_plugin_menu", False)
                        else None
                    )
                    if plugin_menu:
                        self.processing_menu.addMenu(plugin_menu)

                    plugin_action = (
                        QAction(getattr(member, "display_name", "DEFAULT_NAME"))
                        if getattr(member, "uses_single_action", False)
                        else None
                    )
                    if plugin_action:
                        self.processing_menu.addAction(plugin_action)

                    init_kwargs = dict(
                        api=api,
                        plugin_menu=plugin_menu,
                        plugin_action=plugin_action,
                    )
                    if parent_obj is not None:
                        init_kwargs["parent"] = parent_obj

                    self.loaded_plugins.append(
                        {
                            "plugin": member(**init_kwargs),
                            "parent": parent_obj,
                            "api": api,
                            "menu": plugin_menu,
                            "action": plugin_action,
                            "id": plugin_id,
                        }
                    )
                    if parent_obj is not None:
                        print(
                            f"  Note: {plugin_id!r} loaded in full-access mode, "
                            f"bypassing the stable API and capable of arbitrary badness"
                        )
                except PluginAPIVersionError as exc:
                    print(f"Skipping plugin {plugin_id!r} ({name}): {exc}")
                except Exception as exc:
                    print(f"Failed to load plugin.\n{exc}")
                    print(traceback.print_exc())

    # run post-initialization so that plugins can see all other loaded plugins
    for plugin_dict in self.loaded_plugins:
        plugin = plugin_dict["plugin"]
        if hasattr(plugin, "post_init"):
            kwargs = dict(api=plugin_dict["api"])
            if plugin_dict["parent"] is not None:
                kwargs["parent"] = plugin_dict["parent"]
            plugin.post_init(**kwargs)


def unload_plugins(self: "DataViewer"):
    for plugin in self.loaded_plugins:
        try:
            plugin["plugin"].close()
        except Exception as e:
            print(f"Error {e} while unloading plugin {plugin['id']}")


class py4DBrowserPlugin:

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "my.plugin.identifier"

    # The plugin API version this plugin was written against (see
    # py4D_browser.plugin_api). REQUIRED: a plugin that does not declare it
    # is skipped with a log message. The loader passes a compatible API
    # object as `api`; a plugin requiring a version the browser does not
    # provide is skipped.
    api_version = (1, 6)

    # Set to True to also receive the raw DataViewer as `parent` (bypassing
    # the versioned API; no compatibility guarantees). The API object is
    # always passed as `api` regardless.
    full_access = False

    # Set to True to mark this plugin as for development/testing use only.
    # Such plugins are skipped silently (no log output) unless the user opts
    # in by setting `gui/dev_plugins = true` in the config file. This lets
    # development plugins be committed and tracked without exposing them to
    # end users.
    dev_only = False

    ######## optional flags ########
    display_name = "Example Plugin"

    # Plugins may add a top-level menu on their own, or can opt to have
    # a submenu located under Plugins>[display_name], which is created before
    # initialization and its QMenu object passed as `plugin_menu`
    uses_plugin_menu = False

    # If the plugin only needs a single action button, the browser can opt
    # to have that menu item created automatically under Plugins>[Display Name]
    # and its QAction object passed as `plugin_action`
    uses_single_action = False

    def __init__(self, api, **kwargs):
        # `api` is the versioned plugin API object (see `api_version`).
        # `parent` (the raw DataViewer) is present only when `full_access`
        # is set.
        self.api = api

    def post_init(self, api, **kwargs):
        # This is called after *all* plugins are loaded and __init__'ed
        # to enabled to plugins to discover and hook into one another.
        # ADDED IN v1.5.0 (currently called with the same `api` object, and
        # `parent` for full_access plugins, passed to __init__)
        pass

    def close(self):
        pass  # perform any shutdown activities

