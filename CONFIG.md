# Configuration

py4DGUI persists its preferences in a single INI file created with
`QSettings` (`QtCore.QSettings.Format.IniFormat`). All settings read and written
by the browser and its built-in plugins are documented here.

## Config file location

The file is `GUI_config.ini` in the platform user-config directory for
organization `py4DSTEM`, application `py4DGUI` — i.e.
`platformdirs.user_config_dir("py4DGUI", "py4DSTEM") / "GUI_config.ini"`.
Typical locations:

| OS      | Path                                                              |
|---------|-------------------------------------------------------------------|
| macOS   | `~/Library/Application Support/py4DSTEM/py4DGUI/GUI_config.ini`   |
| Windows | `%APPDATA%\py4DSTEM\py4DGUI\GUI_config.ini`                       |
| Linux   | `~/.config/py4DSTEM/py4DGUI/GUI_config.ini`                       |

The exact path is printed to stdout on startup
(`Loading configuration from …`). The file is created on first run; delete it
(or reset the relevant section) to return to defaults.

Keys are written as `section/key`.

## Built-in settings

### `[gui]` — user preferences

| Key                    | Type | Default     | Description                                                                                          |
|------------------------|------|-------------|------------------------------------------------------------------------------------------------------|
| `diffraction_colormap` | str  | `inferno`   | Colormap for the diffraction-space pane (any matplotlib / pyqtgraph colormap name).                  |
| `realspace_colormap`   | str  | `thermal`   | Default colormap for real-space (virtual-image) panes, including new tabs.                            |
| `fft_colormap`         | str  | `yellowy`   | Colormap for the result pane (FFT / cepstrum), 2D and 3D.                                             |
| `volume_colormap`      | str  | `thermal`   | Initial color transfer function for 3D volume tabs.                                                   |
| `batch_size_num_rows`  | int  | `32`        | Number of scan rows per slab when computing a masked detector response; bounds peak memory.          |
| `dev_plugins`          | bool | `0` (off)   | Load dev-only plugins (e.g. `tcBF`, `Volume Viewer`). Off by default so development plugins are hidden from end users. |
| `quack`                | bool | `0` (off)   | Quack                         |

Boolean values are written by `QSettings` as `0` / `1`; any value that
`strtobool` accepts is read as truthy (`1`, `true`, `yes`, `on`, …).

### `[last_state]` — automatically saved UI state

These keys are written by the browser as you use it and restored on next
launch. They can be reset by launching with the `--reset-state` flag:

```
py4DGUI --reset-state
```

## Built-in plugin settings

### `py4DGUI.internal.raw_loader` — "Open Raw…" presets

The raw-file loader reads named presets from its own INI section so you can
reuse a set of loading parameters without re-entering them each time. Keys use
the pattern `preset.<name>.<field>`:

| Key pattern                  | Description                                                            |
|------------------------------|------------------------------------------------------------------------|
| `preset.<name>.dtype`        | NumPy dtype name (e.g. `float32`, `uint16`).                            |
| `preset.<name>.endianness`   | `native`, `little`, or `big`.                                          |
| `preset.<name>.offset`       | In-file offset, in items.                                              |
| `preset.<name>.stride`       | Per-pattern pitch, in items (contiguous ⇒ `Qx·Qy`).                    |
| `preset.<name>.Qx` / `Qy`    | Detector shape.                                                        |
| `preset.<name>.Nx` / `Ny`    | Scan shape (optional; may be left unset).                              |
| `preset.<name>.memmap`       | Load as a memmap (bool).                                               |
| `preset.<name>.bin`          | Bin-on-load factor (int).                                             |

Built-in presets `EMPAD-G2-like` and `EMPAD1` ship with the plugin. INI values
overlay them field-by-field (INI wins), and new names add new presets.

```ini
[py4DGUI.internal.raw_loader]
preset.MyFormat.dtype = uint16
preset.MyFormat.offset = 1024
preset.MyFormat.stride = 4096
preset.MyFormat.Qx = 64
preset.MyFormat.Qy = 64
```

## How plugins should use the settings object

The plugin API exposes the *same* `QSettings` object the browser uses as
`api.settings`. To avoid collisions with the built-in sections and with each
other, follow these rules:

1. **Confine all reads and writes to a single top-level section named after
   your plugin's `plugin_id`.** That section is yours; `gui` and `last_state`
   are the browser's and must not be touched.
2. **Reference keys as fully-qualified strings** `"<plugin_id>/<key>"`, e.g.
   `"acme.example.my_plugin/threshold"`. Keep the sub-key flat and
   descriptive; use dots or underscores for multi-part names — just be
   consistent within your section.
3. **Always supply a default** as the second argument to `settings.value(key,
   default)` so your plugin works correctly when the user has no config yet.

Example — reading and writing under your own section:

```python
class MyPlugin:
    plugin_id = "acme.example.my_plugin"

    def __init__(self, api, **kwargs):
        self.api = api
        self.threshold = self.api.settings.value(
            f"{self.plugin_id}/threshold", 0.5, type=float
        )

    def save_threshold(self, value):
        self.api.settings.setValue(f"{self.plugin_id}/threshold", value)
```

The built-in `raw_loader` preset scheme (`preset.<name>.<field>` under
`[py4DGUI.internal.raw_loader]`) is a good reference for this pattern.
