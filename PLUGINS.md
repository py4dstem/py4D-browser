# py4DGUI Plugins

Over time, we have substantially pared down the functionality available in the browser, removing things such as pre-processing, file conversion, and data analysis. This has allowed the browser code to become much cleaner, and focused primarily on its core functionality of visualizing 4D-STEM data. In doing so, we have made the browser more robust and maintainable. 
With the introduction of plugins in version 1.3.0, we hope to enable easy expansibility of the capabilities of the browser without complicating the core implementation.

## Known Plugins
We hope to maintain a list of existing plugins here. If you produce a browser plugin, feel free to message `sezelt` or create a PR to be added to this list.

### Pre-packaged plugins
Parts of what used to be "core" functionality are now implemented using the plugin interface to separate them from the core browser code. These are packaged with py4DGUI and always available:
* `Calibration`: Allows for the calibration of the scale bars using known physical distances.
* `tcBF`: Allows for the computation of tilt-corrected brightfield images. Uses the versioned plugin API (v1.6) and the detector getters, and displays its reconstruction in its own virtual-image tab (see below) so it does not overwrite the built-in virtual image.
* `Copy Image to Tab`: Copies the raw array currently on display — the virtual image (the built-in image, or another plugin tab's image) or the diffraction pattern — into a new virtual-image tab, so it can be kept alongside the live display without being overwritten. The new tab mirrors the source view's **scaling and display range** (not its colormap), so the copy looks like what you were viewing (e.g. a copy of a square-root-scaled diffraction pattern is itself square-root-scaled).
* `Open Raw...`: Loads an arbitrary raw binary file as a datacube. Prompts for the numpy dtype, endianness, in-file offset, per-pattern stride (pitch), and the full 4D shape; validates the settings fit the file size; supports memmap and bin-on-load. Named presets ship with defaults in the plugin and can be extended via the INI file.

### External plugins
* [EMPAD2 Raw File Reader](https://github.com/sezelt/empad2): This also previously was present in the core browser code and would add an additional menu if the external package was installed. This adds the ability to import the "concatenated" raw binary data from the TFS EMPAD-G2 detector. This plugin is considered conforming to the guidelines.  

# Creating a Plugin

The py4D_browser plugin mechanics are inspired by [Nion Swift](https://nionswift.readthedocs.io/en/stable/api/plugins.html), particularly how plugins are installed, discovered, and loaded. 

Plugins should create a module in the `py4d_browser_plugin` namespace and should define a class with the `plugin_id` attribute

```python
class ExamplePlugin:

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "my.plugin.identifier"

    # the version of the plugin API this plugin was built against.
    # REQUIRED: a plugin that does not declare this is skipped with a log
    # message. The browser passes it the matching API object (see below),
    # or skips the plugin if this version is not supported.
    # (Declare the version you actually build against; the browser loads
    # any 1.x minor at or below the one it provides and skips newer ones.)
    api_version = (1, 6)

    ######## optional flags ########
    display_name = "Example Plugin"

    # Plugins are always passed the versioned plugin API object as `api`
    # and should only use that surface. Setting this to True additionally
    # passes the raw DataViewer window as `parent` (with the API object
    # still available as `api`) for plugins that need to go beyond the
    # API. This bypasses all compatibility guarantees.
    full_access = False

    # Mark this plugin as for development/testing use only. Such plugins
    # are skipped silently (no log output) unless the user opts in by
    # setting `gui/dev_plugins = true` in the config file. This lets
    # development plugins be committed and tracked in-repo without
    # exposing them to end users.
    dev_only = False

    # Plugins may add a top-level menu on their own, or can opt to have
    # a submenu located under Plugins>[display_name], which is created before
    # initialization and its QMenu object passed as `plugin_menu`
    uses_plugin_menu = False

    # If the plugin only needs a single action button, the browser can opt
    # to have that menu item created automatically under Plugins>[Display Name]
    # and its QAction object passed as `plugin_action`
    uses_single_action = False

    def __init__(self, api, **kwargs):
        self.api = api

    def close(self):
        pass  # perform any shutdown activities                   

```

On loading the class is initialized using
```python
ExamplePlugin(api=api, [...])
```
where `api` is always the versioned plugin API object described in [The Plugin API](#the-plugin-api-v16) (see below). If the plugin sets `full_access = True`, it additionally receives the `DataViewer` instance (the main window object) as `parent`. All arguments will always be passed as keywords, including any additional arguments that are provided as a result of setting various optional flags. Plugins are loaded as the last step after constructing the `DataViewer`, before its `show()` method is called.  

## Versioning and compatibility

Plugins declare which version of the plugin API they were built against using the class-level `api_version` attribute, a `(major, minor)` tuple. The browser resolves this against its table of supported API versions:

* If the plugin's **major** version is supported and its minor version is at most the latest minor for that major, the plugin is passed the matching API class. Within a major version the API is strictly additive: new features may appear, but existing ones never change behavior.
* If the major version is unknown, or the plugin requests a minor version newer than the browser provides, the plugin is **skipped**: a `PluginAPIVersionError` naming the plugin and both versions is logged, and loading of all other plugins continues. The fix is to upgrade `py4D_browser` (or, for plugin authors, to build the plugin against an API version their users have).
* Multiple major API versions coexist in the same browser, so a plugin built against 1.x continues to work after a 2.x API is introduced, as long as the browser still supports major 1.
* `api_version` is **required**: a plugin that does not declare it is **skipped** with a log message telling it to declare one (e.g. `api_version = (1, 6)`). There is no legacy fallback.
* `full_access = True` is an unversioned escape hatch: the plugin receives the raw `DataViewer` as `parent` in addition to the API object as `api`, and any breakage caused by the browser changing its internals is the plugin's responsibility.
* **Targeting a specific API version:** by convention, the API minor version always matches the `py4D_browser` package minor version of the release that introduced it. So a plugin can require a given API simply by pinning the browser minor in its dependency — e.g. `py4d_browser>=1.6` guarantees API v1.6 (and its features, like virtual-image tabs) is available. Not every browser minor adds new API (some are bugfix-only releases), so there may be browser minors with no API change — but a browser's minor is always at least as high as any API minor it ships, so pinning the browser minor reliably lower-bounds the API a plugin gets.

### Development-only plugins

A plugin can set `dev_only = True` to mark itself as for development/testing use. Such plugins are **skipped silently** (no log message, no menu item) unless the user opts in by setting `gui/dev_plugins = true` in the browser's config file (`GUI_config.ini`, under the `[gui]` section). By default the setting is off, so development plugins can be committed and tracked in-repo without being exposed to end users. When enabled, a `dev_only` plugin loads exactly like any other plugin.

The following design guidelines should still be followed:

* If the plugin adds menu items, it should only add items to its own menu (not to ones already existing in the GUI). The plugin is permitted to add a menu to the top bar on its own, or (preferably) can set the `uses_plugin_menu` attribute which will initialize a menu under Plugins>MyPluginDisplayName which gets passed to the initializer as `plugin_menu`.
* If the plugin adds a single menu item, it can have the browser create and insert that action item automatically by setting `uses_single_action`. The `QAction` object will be passed in as `plugin_action`. 
* The plugin should *never* render an image to the views directly. To display images, plugins should always call `set_virtual_image` or `set_diffraction_image` using raw, unscaled data. If the plugin needs to produce a customized display, it cannot do that in the existing views and must create its own window. **Exception (v1.6):** an image-producing plugin should *prefer* to display its output in its own **virtual-image tab** (created with `create_virtual_image_tab`, see below) rather than overwriting the built-in virtual image with `set_virtual_image`. This keeps the browser's own pipeline visible and lets several image-producing plugins show output at the same time. Use `set_virtual_image` only when the plugin genuinely wants to drive the built-in pane.
* The plugin should not retain references to any objects in the `DataViewer`, as that may prevent objects from being freed at the right times. For example, do not do something like `self.current_datacube = self.api.datacube`, as until this reference is cleared the browser could not free memory after closing a dataset and opening a new one. 
* Likewise, keep references to virtual-image tabs only as long as you need them — i.e. while the tab is open and you intend to drive it. A **closed** tab is not freed by the browser (its raw image array remains reachable through `tab.image`), so holding a reference to a tab after `close_virtual_image_tab` pins that image in memory until the process exits. In particular, do not accumulate a list of every tab you have ever created and keep it for the life of the session.
* The plugin is allowed to read/write from the QSettings of the GUI, but should only do so in a top-level section with the same name as `plugin_id`, i.e. `value = self.api.settings.value(self.plugin_id + "/my_setting", default_value)`.

## The Plugin API (v1.6)

The v1.6 API object is a broker between the plugin and the browser. Plugins should use only this surface; the browser is free to change its internals as long as the API object keeps working. The full v1.6 surface:

* **`api_version`** — `(1, 6)`, for introspection.
* **`datacube`** — read-only access to the currently loaded `DataCube` (or `None`). Mutating the cube object (e.g. its calibration) is fine; to replace the whole cube, use `set_datacube` below.
* **`set_datacube(datacube, refresh=True)`** — replace the currently loaded datacube (e.g. a cube a plugin computed, or read from a file format the browser doesn't natively open). With `refresh` (the default) the normal post-load machinery runs: both views are reset and `signal_datacube_changed` is emitted. Pass `refresh=False` to swap in the cube without redrawing or notifying listeners. In either case the previous cube is released (a garbage-collection pass is run) so its memory is reclaimed.
* **Signals** — `signal_diffraction_data_changed`, `signal_virtual_image_data_changed`, `signal_datacube_changed`, and (new in v1.6) `signal_current_virtual_image_changed`. These are the live signals from the main window, so `connect`/`disconnect` through the API object behaves exactly as if they were accessed directly.
* **Pane setters** — `set_virtual_image(vimg, reset, pixel_size, pixel_units)`, `set_diffraction_image(dp, reset, pixel_size, pixel_units)`, `set_result_image(vimg, reset, pixel_size, pixel_units, title)`. Always pass raw, unscaled data.
* **`set_scalebar(view, pixel_size, units)`** — update the scale bar of one pane, where `view` is one of `"diffraction"`, `"real_space"`, or `"result"`. This is the supported way for plugins to change a scale bar; do not touch the scale bar objects directly.
* **Result-pane registration (new in v1.6)** — `register_result_callback(title, cleanup, callback_diffraction_pattern_changed=None, callback_virtual_image_changed=None, callback_datacube_changed=None)` makes a plugin drive the result pane instead of the built-in FFT/EWPC. It is a **singleton**: registering a new callback replaces the previous one and runs the previous `cleanup` first, so pass a `cleanup` that restores the view (e.g. removes any ROIs you added). The `callback_*` functions are invoked on the matching signal (a no-arg callable).
* **View ROIs (new in v1.6)** — `add_diffraction_roi(item)`, `remove_diffraction_roi(item)`, `add_real_space_roi(item)`, `remove_real_space_roi(item)` attach and detach a pre-built pyqtgraph item (an ROI such as `pg.ROI`/`pg.LineROI`/`pg.PolygonROI`, or a line/text/path) to the built-in diffraction view or the built-in real-space (virtual image) view. Each returns the item. The plugin owns the item's lifetime: `remove_*` detaches it without deleting it, so it can be re-added. This is the supported way to place an ROI in a built-in view — do not reach into the viewer's `*_widget.getView()` objects directly (and do not retain references to those widgets).
* **Image-shape accessors (new in v1.6)** — `diffraction_image_shape` and `real_space_image_shape` return the shape of the raw (unscaled) image currently on display in the built-in diffraction / real-space views, or `None` if no image is loaded. These expose the *shape* (not the full array) so a plugin can size ROIs against the displayed view without retaining a reference to a large array.
* **Detector getters** — `get_diffraction_detector()` and `get_virtual_image_detector()`, each returning a `DetectorInfo` (see below).
* **Virtual-image tabs (new in v1.6)** — `create_virtual_image_tab(title)`, `close_virtual_image_tab(tab)`, `virtual_image_tabs`, `current_virtual_image`, `current_diffraction_image`, `current_virtual_image_settings`, and `current_diffraction_image_settings`. See [Virtual-image tabs](#virtual-image-tabs-v16) below.
* **3D volume tabs (new in v1.6)** — `create_volume_tab(title, select=False)` adds a tab that renders a 3D volume (OpenGL, with user-adjustable color/alpha transfer functions). Volume tabs are tracked in `virtual_image_tabs` and closed with the same `close_virtual_image_tab` as image tabs; when a volume tab is visible, `current_virtual_image` is a 3D array. See [3D volume tabs](#3d-volume-tabs-v16) below.
* **Qt plumbing** — `qtapp` (the `QApplication`), `qt_window` (a `QWidget` to use as a dialog parent — **not** the main window itself), `status_bar` (the status bar), and `settings` (the QSettings).

### Virtual-image tabs (v1.6)

The virtual-image pane is now a tabbed pane. Tab index 0 is the **default tab** — the browser's own virtual-image pipeline (driven by `set_virtual_image`). It is **not closable**, and it is **not** included in the `virtual_image_tabs` list. Every other tab is a plugin-created **virtual-image tab**, an independent display a plugin drives to show its own image without overwriting the built-in one.

* **`create_virtual_image_tab(title, select=False)`** — add a new tab labeled `title` and return a `VirtualImageTab` object for the plugin to drive. By default the new tab is **not** selected (the built-in virtual image stays visible until the user clicks it); pass `select=True` to immediately switch the pane to the new tab. Reuse a tab by keeping a reference to the returned object (or finding it in `virtual_image_tabs` by `title`) and calling `set_image` on it again, rather than creating a new tab each time.
* **`virtual_image_tabs`** — a copy of the list of open plugin tabs (the default tab is not in it). Read-only; close a tab with `close_virtual_image_tab(tab)` or `tab.close()`.
* **`current_virtual_image`** — the raw array currently on display in the virtual-image pane: the built-in virtual image when the default tab is visible, else the visible plugin tab's raw array (or `None`).
* **`current_diffraction_image`** — the raw array currently on display in the diffraction pane (or `None` if no image is loaded).
* **`current_virtual_image_settings`** — the display settings of the virtual-image view currently on screen, as a `VirtualImageSettings` record with fields `scaling`, `autorange`, `percentiles`, and `levels`: the built-in pane's when the default tab is visible, else the visible image tab's. Returns `None` when the visible view has no 2D display settings (an empty built-in pane or image tab, or a volume tab). Use this to read a view's current scaling and range (e.g. to mirror them onto a new tab); each tab's own `scaling` / `autorange` / `autorange_percentiles` / `levels` / `set_levels` control its display independently.
* **`current_diffraction_image_settings`** — the display settings of the built-in **diffraction** (reciprocal-space) pane, as a `VirtualImageSettings` record with the same fields (`scaling`, `autorange`, `percentiles`, `levels`). Returns `None` when the diffraction pane is empty (no image loaded). Use this to read the DP pane's current scaling and range (e.g. to mirror them onto a tab copied from the DP pane).
* **`close_virtual_image_tab(tab)`** / **`tab.close()`** — close a plugin tab. This detaches the tab's ROIs and annotations from its view, removes the tab from the pane, and frees its widget. It is idempotent, so it is safe to call more than once. Note that ROIs/annotations are *detached*, not destroyed — a plugin that wants to reuse an item can do so.

A `VirtualImageTab` object (returned by `create_virtual_image_tab`) exposes:

* **`set_image(image, reset=True, pixel_size=None, pixel_units=None)`** — set the raw image shown in the tab (a real 2D array, or a complex 2D array rendered as an RGB image). The tab applies **its own** scaling (Linear/Log/Square Root) and autoscale percentiles — a new tab starts by inheriting the built-in pane's current settings, then keeps its own and is independent of the built-in pane and of every other tab (see `scaling` / `autorange` / `autorange_percentiles` below). With `reset` (the default), the display levels are re-derived from the **scaled** data at the tab's percentiles (so outliers don't wash out the display); pass `reset=False` to re-derive only while `autorange` is on, otherwise keeping the tab's current levels. If `pixel_size`/`pixel_units` are given, the tab's own scale bar is updated too (unit strings are displayed in Unicode, e.g. `"A"` → `"Å"`).
* **`set_scalebar(pixel_size=None, units=None)`** — update the tab's scale bar independently of its image.
* **`add_roi(roi)`** / **`add_annotation(annotation)`** — attach a pre-built pyqtgraph ROI (e.g. `pg.ROI`, `pg.LineROI`, `pg.PolygonROI`) or any pyqtgraph `QGraphicsItem` (line, text, path, …) to the tab's view; both return the item. Tracked and detached when the tab is closed.
* **`colormap`** (get/set) — the tab's active colormap. A new tab is seeded with the browser's **default virtual-image colormap** (the `gui/realspace_colormap` setting, the same default the built-in pane starts with), and each tab keeps its own colormap from then on. Set it to a `pyqtgraph.ColorMap` object or a colormap name (resolved through the browser's colormap lookup) to change the tab independently of the other panes.
* **Per-tab display settings (new in v1.6)** — each tab carries its **own** scaling and display range, shown in a fixed strip at the **bottom of the tab view** (a Scaling selector, an Autorange latch, and low/high autoscale-percentile spinboxes). A new tab starts by inheriting the built-in pane's current settings, then diverges as it is changed, so tabs never share display state. The strip is collapsible and **hidden by default** (so a fresh tab's view owns the full height): a toggle button in the tab's bottom-left corner — hugging the pane frame while collapsed, the strip's left edge while shown — or `controls_visible` below, brings the whole strip in and out to reclaim the vertical space. Exposed through the API as:
  * **`scaling`** (get/set) — the tab's scaling mode: `"Linear"`, `"Log"`, or `"Square Root"`. Setting it re-renders and re-derives the range.
  * **`autorange`** (get/set) — whether the range auto re-derives from the (scaled) data (the Autorange latch). When off, the range is frozen at the last re-derive or `set_levels`.
  * **`autorange_percentiles`** (get/set) — the autoscale percentiles `(low, high)` in 0–100 used when `autorange` is on.
  * **`levels`** (get) — the current `(low, high)` display level range, or `None` before an image has been set.
  * **`set_levels(low, high)`** — set the display level range explicitly and re-render (freezes the range even while `autorange` is on).
  * **`controls_visible`** (get/set) — whether the bottom display-control strip is shown. Defaults to `False` (hidden, so the view owns the full height). Set it to `True` to show the whole strip; set it back to `False` to hide it, leaving only the corner toggle button. Each tab remembers its own state.
* **`title`** (get/set), **`image`** (the last array passed to `set_image`), **`scale_bar`**, **`widget`** (the tab's backing container `QWidget`: the image view plus the fixed bottom strip), **`image_view`** (the underlying `pyqtgraph.ImageView` that renders the image — use this, not `widget`, for pyqtgraph view operations), **`rois`**, **`annotations`**, **`closed`**.
* **`signal_data_changed`** — emitted (with the tab as its argument) after every `set_image`.

**How the result pane tracks tabs.** The result pane (the built-in FFT, or a plugin's registered `callback_virtual_image_changed`) responds to updates on the **currently visible** tab via `signal_current_virtual_image_changed`, which fires when the visible tab's image changes or when the user switches tabs. `signal_virtual_image_data_changed` still fires on updates to the **default** tab, but the result machinery is driven by the new signal. Note that the result driver is a **singleton** — only one result handler (the built-in, or one plugin) is active at a time — so tabs can *coexist* in the pane, but they do not each get their own simultaneous result display.

**Example** (auto tcBF-style):

```python
tab = self.api.create_virtual_image_tab("tcBF")   # create once, reuse after
tab.set_image(
    reconstruction,
    reset=True,
    pixel_size=self.api.datacube.calibration.get_R_pixel_size(),
    pixel_units=self.api.datacube.calibration.get_R_pixel_units(),
)
# ... later, to add a marker:
line = pg.InfiniteLine(pos=12.0, pen=pg.mkPen("r"))
tab.add_annotation(line)
```

And clean up on shutdown (the `close()` hook) so the tab is freed with the plugin:

```python
def close(self):
    for tab in self.api.virtual_image_tabs:
        if tab.title == "tcBF":
            self.api.close_virtual_image_tab(tab)
```

### 3D volume tabs (v1.6)

A **volume tab** is another kind of plugin tab in the virtual-image pane: instead of a 2D image it renders a **3D volume** with `pyqtgraph`'s OpenGL volume rendering (`pyqtgraph.opengl.GLViewWidget` + `GLVolumeItem`). The user can orbit/pan/zoom with the mouse and adjust the appearance live through two transfer-function editors in the tab, plus a log-scale slider that scales the volume's first (z/stack) axis. Volume tabs are otherwise treated like virtual-image tabs: they are tracked in `virtual_image_tabs`, closed with `close_virtual_image_tab(tab)` / `tab.close()`, and when one is the visible tab `current_virtual_image` returns its 3D array (2D for the built-in tab and image tabs, 3D for volume tabs).

Requires PyOpenGL (a dependency of the browser; if it is somehow absent, `create_volume_tab` raises a `RuntimeError` with an install hint).

* **`create_volume_tab(title, select=False)`** — add a new tab labeled `title` and return a `VolumeTab` object for the plugin to drive. Same default-not-selected behavior as `create_virtual_image_tab` (pass `select=True` to switch the pane to it immediately). Reuse a tab by keeping a reference (or finding it in `virtual_image_tabs` by `title`) and calling `set_volume` on it again.
* **`close_virtual_image_tab(tab)`** / **`tab.close()`** — works for volume tabs exactly as for image tabs (idempotent).

A `VolumeTab` object (returned by `create_volume_tab`) exposes:

* **`set_volume(volume, reset=True, voxel_size=None)`** — set the scalar volume rendered in the tab. `volume` must be a **real 3D** array (x, y, z): non-3D arrays raise `ValueError`, as do complex arrays. With `reset` (the default) the tab's display level range is re-derived from the data using the same autoscale percentile range as the built-in virtual image; pass `reset=False` to keep the tab's current levels (on the very first call, min/max are used). `voxel_size` is the physical size of one voxel: a scalar (isotropic) or a `(dx, dy, dz)` 3-tuple (anisotropic). All values must be finite and positive (anything else raises `ValueError`). When given, the volume and the orientation axes are rendered in these physical units, and the initial camera distance scales to the volume's physical extent; when omitted, the tab's current voxel size is kept.
* **`set_levels(low, high)`** — set the display level range explicitly and re-render.
* **`levels`** — the current `(low, high)` display range (or `None` before any `set_volume`).
* **`voxel_size`** — the current `(dx, dy, dz)` physical voxel size (a 3-tuple of floats), last set via `set_volume`; defaults to `(1.0, 1.0, 1.0)` (unit, isotropic voxels). Read-only. The tab's "z size" slider (log scale, 10⁻³–10³) live-adjusts the first component — for a stack of images whose x/y pixels are equal size, adjusting just the z size is what it takes to scale the volume correctly — and updates this property; `set_volume` keeps the slider in step with whatever it is passed (clamped to the slider's range for display).
* **`volume`** / **`image`** — the last array passed to `set_volume` (or `None`). `image` is an alias so the tab is interchangeable with a `VirtualImageTab` where a tab's array is read; for a volume tab it is the 3D volume.
* **`color_editor`** / **`alpha_editor`** — the two `pyqtgraph` `GradientEditorItem` transfer functions, shown in the tab's bottom strip and adjustable by the user at any time (edits re-render live). In the **color** editor only each tick's *color* matters (its alpha is ignored); in the **alpha** editor only each tick's *alpha* matters (its right-click colormap menu is disabled). A programmatic edit must pass `QtGui.QColor` objects as tick colors (pyqtgraph stores them as-is and later calls `QColor` methods on them). A fresh tab is seeded with the volume tab's default colormap (the `gui/volume_colormap` ini setting, default `thermal` — the same default as the built-in virtual image) and a transparent→opaque alpha ramp.
* **`view`** — the `pyqtgraph.opengl.GLViewWidget` showing the volume. Built-in mouse controls: drag = orbit, Ctrl+drag = pan (view relative), middle-drag = pan (view upright), wheel = zoom, arrow keys = orbit. The camera can also be driven programmatically with `setCameraPosition` / `setCameraParams` / `cameraParams`.
* **`axis`** — the orientation axes (`GLAxisItem`) in the view.
* **`title`** (get/set), **`widget`** (the tab's backing `QWidget`: the 3D view plus the fixed bottom strip — the two transfer-function editors and the z-size slider), **`closed`**.
* **`signal_data_changed`** — emitted (with the tab as its argument) after every `set_volume`.

**What is and isn't 2D-only.** The display-oriented paths assume a 2D rendered image, so they **bail with a status message** when a volume tab is visible: "PNG (display)" / "TIFF (display)" exports, and copying to the clipboard. **"TIFF (raw)" works** — it writes the raw array, so it exports the 3D volume. The built-in virtual-image FFT result modes **skip silently** when the visible image is a 3D volume (they have no 3D equivalent). If you register your own result callback (`register_result_callback`), handle the 3D case yourself.

**Interoperability note.** Code that iterates `virtual_image_tabs` and calls `tab.set_image(...)` on every tab will hit an `AttributeError` on volume tabs (they have `set_volume`, not `set_image`) and vice versa. Distinguish the two kinds with `isinstance(tab, ...)` or a duck check (`hasattr(tab, "set_image")`) before dispatching.

**Sizing guidance.** The RGBA buffer is a full copy of the volume (4 bytes/voxel on top of the volume itself), and the GPU has a 3D-texture size limit — an oversized volume will fail at **first paint**, not at `set_volume`. Volumes in the range of roughly 64³ to 256³ are comfortably safe on typical GPUs; much larger volumes should be downsampled by the plugin before display.

**Example** (render a computed 3D stack, e.g. a tomography reconstruction):

```python
tab = self.api.create_volume_tab("Recon", select=True)   # create once, reuse
tab.set_volume(reconstruction, reset=True, voxel_size=(0.5, 0.5, 2.0))  # physical voxel size
# ... later, adjust the appearance programmatically:
tab.set_levels(0.0, 1.0)
tab.view.setCameraPosition(elevation=30, azimuth=45)
```

And clean up on shutdown (the `close()` hook) the same way as image tabs:

```python
def close(self):
    for tab in self.api.virtual_image_tabs:
        if tab.title == "Recon":
            self.api.close_virtual_image_tab(tab)
```


**Dialog parenting:** `QDialog` requires a `QWidget` parent, so any dialog a plugin creates should be parented with `parent=self.api.qt_window`. This is a dedicated dialog-parent widget — **not** the `DataViewer` itself — so a general plugin cannot reach the viewer's full state through it; do all data and method access through the API object instead. (A plugin that sets `full_access = True` is the exception: it additionally receives the raw `DataViewer` as `parent` and may use it directly.)

## Accessing the detectors

With version 1.3.0, there is a new API for accessing the ROI selections made using the detectors on the two views. Plugins should only interact with the detectors via this API, as the implementation details of the ROI objects themselves are considered internal and subject to change. Calling `get_diffraction_detector` or `get_virtual_image_detector` yields a `DetectorInfo` object containing the properties of the current detector and the information (either a slice or a mask array) needed to produce the selection it represents.

## Namespace packages

Namespace packages are a way to split a package across multiple sources, which can be provided by different distributions. This allows the py4DGUI to import this special namespace and have all plugins, regardless of their source, appear under that import. Details can be found in [PEP 420](https://peps.python.org/pep-0420/).

In order to create a plugin, create a directory called `py4d_browser_plugin` under your `src` directory, and then create a directory for your plugin within that folder. _Do not place an `__init__.py` file in the `py4d_browser_plugin` folder, or the import mechanism will be broken for all plugins._