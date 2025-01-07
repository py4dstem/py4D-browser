import pkgutil
import importlib
import inspect

from PyQt5.QtWidgets import QMenu

__all__ = ["load_plugins", "unload_plugins"]


def load_plugins(self):
    """
    The py4D_browser plugin mechanics are inspired by Nion Swift:
    https://nionswift.readthedocs.io/en/stable/api/plugins.html

    Plugins should create a module in the py4d_browser_plugin namespace
    and should define a class with the `plugin_id` attribute

    On loading the class is initialized using
        ExamplePlugin(parent=self)
    with additional arguments potentially passed as kwargs


    """

    import py4d_browser_plugin

    self.loaded_plugins = []

    for module_info in pkgutil.iter_modules(getattr(py4d_browser_plugin, "__path__")):

        module = importlib.import_module(
            py4d_browser_plugin.__name__ + "." + module_info.name
        )

        for name, member in inspect.getmembers(module, inspect.isclass):
            plugin_id = getattr(member, "plugin_id", None)

            if plugin_id:
                print(f"Loading plugin: {plugin_id}")
                plugin_menu = (
                    QMenu(getattr(member, "display_name", "DEFAULT_NAME"))
                    if getattr(member, "uses_plugin_menu", False)
                    else None
                )
                if plugin_menu:
                    self.processing_menu.addMenu(plugin_menu)
                self.loaded_plugins.append(member(parent=self, plugin_menu=plugin_menu))


def unload_plugins(self):
    # NOTE: This is currently not actually called!
    for plugin in self.loaded_plugins:
        plugin.close()


class ExamplePlugin:

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "my.plugin.identifier"

    # optional flags

    # Plugins may add a top-level menu on their own, or can opt to have
    # a submenu located under Plugins>[display_name], which is created before
    # initialization and its QMenu object passed as `plugin_menu`
    uses_plugin_menu = False
    display_name = "Example Plugin"

    def __init__(self, parent, *args, **kwargs):
        self.parent = parent

    def close(self):
        pass  # perform any shutdown activities
