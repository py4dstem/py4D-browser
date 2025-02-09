# py4DGUI Plugins

Over time, we have substantially pared down the functionality available in the browser, removing things such as pre-processing, file conversion, and data analysis. This has allowed the browser code to become much cleaner, and focused primarily on its core functionality of visualizing 4D-STEM data. In doing so, we have made the browser more robust and maintainable. 
With the introduction of plugins in version 1.3.0, we hope to enable easy expansibility of the capabilities of the browser without complicating the core implementation.

## Known Plugins
We hope to maintain a list of existing plugins here. If you produce a browser plugin, feel free to message `sezelt` or create a PR to be added to this list.

### Pre-packaged plugins
Parts of what used to be "core" functionality are now implemented using the plugin interface to separate them from the core browser code. These are packaged with py4DGUI and always available:
* `Calibration`: Allows for the calibration of the scale bars using known physical distances. **Note:** This plugin is currently considered "badly behaved" because of the way it accesses the detector ROI objects directly. An abstract interface for this behavior will be created in the future, but for now this plugin should not be considered an "example" to follow.
* `tcBF`: Allows for the computation of tilt-corrected brightfield images. This also accesses detector ROIs directly and should be considered "badly behaved".

### External plugins
* [EMPAD2 Raw File Reader](https://github.com/sezelt/empad2): This also previously was present in the core browser code and would add an additional menu if the external package was installed. This adds the ability to import the "concatenated" raw binary data from the TFS EMPAD-G2 detector. This plugin is considered conforming to the guidelines.  

# Creating a Plugin

The py4D_browser plugin mechanics are inspired by [Nion Swift](https://nionswift.readthedocs.io/en/stable/api/plugins.html), particularly how plugins are installed, discovered, and loaded. 

Plugins should create a module in the `py4d_browser_plugin` namespace and should define a class with the `plugin_id` attribute

```python
class ExamplePlugin:

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "my.plugin.identifier"

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

    def __init__(self, parent, **kwargs):
        self.parent = parent

    def close(self):
        pass  # perform any shutdown activities                   

```

On loading the class is initialized using
```python
ExamplePlugin(parent=self, [...])
```
where `self` is the `DataViewer` instance (the main window object). All arguments will always be passed as keywords, including any additional arguments that are provided as a result of setting various optional flags. Plugins are loaded as the last step after constructing the `DataViewer`, before its `show()` method is called.  

The current implementation of the plugin interface is thus extremely simple: the plugin object gets a reference to the main window, and can in theory do whatever artitrarily stupid things it wants with it, and there are no guarantees on compatibility between different versions of the browser and plugins. Swift solves this using the API Broker, which interposes all actions taken by the plugin. While we may adopt such an interface in version 2.0, for now we simply have the following design guidelines that should ensure compatibility:

* If the plugin adds menu items, it should only add items to its own menu (not to ones already existing in the GUI). The plugin is permitted to add a menu to the top bar on its own, or (preferably) can set the `uses_plugin_menu` attribute which will initialize a menu under Plugins>MyPluginDisplayName which gets passed to the initializer as `plugin_menu`.
* If the plugin adds a single menu item, it can have the browser create and insert that action item automatically by setting `uses_single_action`. The `QAction` object will be passed in as `plugin_action`. 
* The plugin should *never* render an image to the views directly. To display images, plugins should always call `set_virtual_image` or `set_diffraction_image` using raw, unscaled data. If the plugin needs to produce a customized display, it cannot do that in the existing views and must create its own window. 
* The plugin should not retain references to any objects in the `DataViewer`, as that may prevent objects from being freed at the right times. For example, do not do something like `self.current_datacube = self.parent.datacube`, as until this reference is cleared the browser could not free memory after closing a dataset and opening a new one. 
* The plugin is allowed to read/write from the QSettings of the GUI, but should only do so in a top-level section with the same name as `plugin_id`, i.e. `value = self.parent.settings(self.plugin_id + "/my_setting", default_value)`.

## Accessing the detectors

With version 1.3.0, there is a new API for accessing the ROI selections made using the detectors on the two views. Plugins should only interact with the detectors via this API, as the implementation details of the ROI objects themselves are considered internal and subject to change. Calling `get_diffraction_detector` or `get_virtual_image_detector` yields a `DetectorInfo` object containing the properties of the current detector and the information (either a slice or a mask array) needed to produce the selection it represents.

## Namespace packages

Namespace packages are a way to split a package across multiple sources, which can be provided by different distributions. This allows the py4DGUI to import this special namespace and have all plugins, regardless of their source, appear under that import. Details can be found in [PEP 420](https://peps.python.org/pep-0420/).

In order to create a plugin, create a directory called `py4d_browser_plugin` under your `src` directory, and then create a directory for your plugin within that folder. _Do not place an `__init__.py` file in the `py4d_browser_plugin` folder, or the import mechanism will be broken for all plugins._