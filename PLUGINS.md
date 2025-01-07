# Creating a Plugin

The py4D_browser plugin mechanics are inspired by Nion Swift:
https://nionswift.readthedocs.io/en/stable/api/plugins.html

Plugins should create a module in the `py4d_browser_plugin` namespace and should define a class with the `plugin_id` attribute

```python
class ExamplePlugin:

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "my.plugin.identifier"

    # optional flags

    # Plugins may add a top-level menu on their own, or can opt to have 
    # a submenu located under Plugins>[display_name], which is created before
    # initialization and its QMenu object passed as `plugin_menu`
    uses_plugin_menu = False 
    display_name = "Example Plugin"

    def __init__(self, parent, argv, *args, **kwargs):
        self.parent = parent

        if "--do-stuff" in argv:
            pass

    def close(self):
        pass  # perform any shutdown activities
                    

```

On loading the class is initialized using
```python
ExamplePlugin(parent=self, argv=argv)
```
where `self` is the `DataViewer` instance (the main window object) and argv is the list of command line arguments passed on launch. 

The current implementation of the plugin interface is thus extremely simple: the plugin object gets a reference to the main window, and can in theory do whatever artitrarily stupid things it wants with it, and there are no guarantees on compatibility between different versions of the browser and plugins. Swift solves this using the API Broker, which interposes all actions taken by the plugin. While we may adopt such an interface in version 2.0, for now we simply have the following design guidelines that should ensure compatibility:

* If the plugin adds menu items, it should only add items to its own menu (not to ones already existing in the GUI). The plugin is permitted to add a menu to the top bar on its own, or (preferably) can set the `uses_plugin_menu` attribute which will initialize a menu under Plugins>MyPluginDisplayName which gets passed to the initializer as `plugin_menu`
* The plugin should *never* render an image to the views directly. To display images, plugins should always call `set_virtual_image` or `set_diffraction_image` using raw, unscaled data. If the plugin needs to produce a customized display, it cannot do that in the existing views and must create its own window. 