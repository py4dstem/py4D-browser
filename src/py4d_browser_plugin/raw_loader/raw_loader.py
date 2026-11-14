"""
Load an arbitrary raw binary file as a 4D-STEM datacube.

A single-action plugin (Plugins → Open Raw...) that reads a plain binary file
the user fully describes: dtype, endianness, an in-file offset, a per-pattern
stride (pitch), and the full 4D shape. Supports memmap, bin-on-load, and named
presets (built-in defaults plus user entries in the INI file).
"""

import os

import numpy as np
import py4DSTEM
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from py4D_browser.menu_actions import lazy_bin_load
from py4D_browser.utils import strtobool

# All numpy builtin scalar types, in a stable display order.
NUMPY_TYPES = [
    "bool",
    "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
    "float16", "float32", "float64",
    "complex64", "complex128",
]

# (label, value) pairs for the endianness selector.
ENDIANNESS = [
    ("Native", "native"),
    ("Little-endian", "little"),
    ("Big-endian", "big"),
]

# Built-in presets. Users can add/override fields, or whole presets, in the INI
# under this plugin's id with keys `preset.<name>.<field>` (INI wins per field).
DEFAULT_PRESETS = {
    "EMPAD-G2-like": {
        "dtype": "float32",
        "endianness": "native",   # best guess; verify per dataset
        "offset": 0,
        "stride": 128 * 128,      # contiguous (patterns back-to-back)
        "Qx": 128,
        "Qy": 128,
        # Nx, Ny left unset: they vary per dataset
    },
    "EMPAD1": {
        "dtype": "float32",
        "endianness": "native",   # best guess; verify per dataset
        "offset": 0,
        "stride": 128 * 128 + 256,  # 128x128 pattern + 256 per-frame metadata
        "Qx": 128,
        "Qy": 128,
        # Nx, Ny left unset: they vary per dataset
    },
}


class RawLoaderPlugin(QWidget):
    plugin_id = "py4DGUI.internal.raw_loader"
    api_version = (1, 6)
    uses_single_action = True
    display_name = "Open Raw..."

    def __init__(self, api, plugin_action, **kwargs):
        super().__init__()
        self.api = api
        plugin_action.triggered.connect(self.launch)

    def close(self):
        pass

    def launch(self):
        api = self.api
        path, _ = QFileDialog.getOpenFileName(
            api.qt_window, "Open raw file", "", "All files (*)"
        )
        if not path:
            return
        RawLoadDialog(api, path).exec_()


class RawLoadDialog(QDialog):
    def __init__(self, api, path):
        super().__init__(parent=api.qt_window)
        self.api = api
        self.path = path
        self.plugin_id = RawLoaderPlugin.plugin_id
        self.setWindowTitle(f"Open raw file: {os.path.basename(path)}")

        # Stride auto-tracks Qx·Qy until the user sets a custom pitch.
        self._stride_auto = True

        self._build_ui()
        self._load_presets()
        self._wire_signals()
        self._set_defaults()
        self._revalidate()

    # --- UI ---
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Presets
        preset_box = QGroupBox("Preset")
        layout.addWidget(preset_box)
        preset_layout = QHBoxLayout(preset_box)
        self.preset_combo = QComboBox()
        preset_layout.addWidget(self.preset_combo)

        # Reading
        read_box = QGroupBox("Reading")
        layout.addWidget(read_box)
        read_layout = QGridLayout(read_box)

        read_layout.addWidget(QLabel("Data type"), 0, 0)
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(NUMPY_TYPES)
        read_layout.addWidget(self.dtype_combo, 0, 1)

        read_layout.addWidget(QLabel("Endianness"), 1, 0)
        self.endian_combo = QComboBox()
        for label, value in ENDIANNESS:
            self.endian_combo.addItem(label, value)
        read_layout.addWidget(self.endian_combo, 1, 1)

        read_layout.addWidget(QLabel("Offset (pixels)"), 2, 0)
        self.offset_edit = QLineEdit()
        read_layout.addWidget(self.offset_edit, 2, 1)

        read_layout.addWidget(QLabel("Stride (pixels)"), 3, 0)
        self.stride_edit = QLineEdit()
        read_layout.addWidget(self.stride_edit, 3, 1)

        # Shape
        shape_box = QGroupBox("Shape")
        layout.addWidget(shape_box)
        shape_layout = QGridLayout(shape_box)

        shape_layout.addWidget(QLabel("Qx (detector)"), 0, 0)
        self.qx_spin = self._spin(1, 10**6)
        shape_layout.addWidget(self.qx_spin, 0, 1)

        shape_layout.addWidget(QLabel("Qy (detector)"), 1, 0)
        self.qy_spin = self._spin(1, 10**6)
        shape_layout.addWidget(self.qy_spin, 1, 1)

        shape_layout.addWidget(QLabel("Nx (scan)"), 2, 0)
        self.nx_edit = QLineEdit()
        shape_layout.addWidget(self.nx_edit, 2, 1)

        shape_layout.addWidget(QLabel("Ny (scan)"), 3, 0)
        self.ny_edit = QLineEdit()
        shape_layout.addWidget(self.ny_edit, 3, 1)

        # Options
        opt_box = QGroupBox("Options")
        layout.addWidget(opt_box)
        opt_layout = QHBoxLayout(opt_box)
        self.memmap_check = QCheckBox("Memmap")
        opt_layout.addWidget(self.memmap_check)
        opt_layout.addStretch()
        opt_layout.addWidget(QLabel("Bin factor"))
        self.bin_spin = self._spin(1, 1)
        opt_layout.addWidget(self.bin_spin)

        # Status
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        layout.addWidget(buttons)

    @staticmethod
    def _spin(lo, hi):
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setKeyboardTracking(False)
        return spin

    # --- presets ---
    def _load_presets(self):
        presets = {name: dict(fields) for name, fields in DEFAULT_PRESETS.items()}
        settings = self.api.settings
        settings.beginGroup(self.plugin_id)
        try:
            for key in settings.childKeys():
                if not key.startswith("preset."):
                    continue
                name, sep, field = key[len("preset."):].partition(".")
                if not sep or not field:
                    continue
                presets.setdefault(name, {})[field] = settings.value(key)
        finally:
            settings.endGroup()

        self._presets = presets
        self.preset_combo.addItem("(custom)")
        for name in sorted(presets):
            self.preset_combo.addItem(name)

    def _apply_preset(self):
        name = self.preset_combo.currentText()
        if name == "(custom)":
            return
        self._apply_fields(self._presets.get(name, {}))

    def _apply_fields(self, fields):
        # Q dims first, so a preset stride that is the contiguous pitch stays
        # consistent with the new Qx·Qy; stride is applied last.
        order = ("Qx", "Qy", "dtype", "endianness", "offset",
                 "Nx", "Ny", "memmap", "bin", "stride")
        for field in order:
            if field in fields:
                self._set_field(field, fields[field])
        # A custom pitch (≠ Qx·Qy) stops the auto-sync.
        self._set_bin_range()
        try:
            self._stride_auto = (int(self.stride_edit.text())
                                 == self.qx_spin.value() * self.qy_spin.value())
        except ValueError:
            self._stride_auto = True
        self._revalidate()

    def _set_field(self, field, value):
        if field == "dtype":
            self._set_combo(self.dtype_combo, str(value))
        elif field == "endianness":
            i = self.endian_combo.findData(str(value))
            if i >= 0:
                self.endian_combo.setCurrentIndex(i)
        elif field == "offset":
            self._set_edit(self.offset_edit, value)
        elif field == "stride":
            self._set_edit(self.stride_edit, value)
        elif field == "Qx":
            self.qx_spin.setValue(int(value))
        elif field == "Qy":
            self.qy_spin.setValue(int(value))
        elif field == "Nx":
            self._set_edit(self.nx_edit, value)
        elif field == "Ny":
            self._set_edit(self.ny_edit, value)
        elif field == "memmap":
            self.memmap_check.setChecked(self._to_bool(value))
        elif field == "bin":
            self.bin_spin.setValue(int(value))

    # --- field helpers ---
    @staticmethod
    def _set_combo(combo, text):
        i = combo.findText(text)
        if i >= 0:
            combo.setCurrentIndex(i)

    @staticmethod
    def _set_edit(edit, value):
        edit.setText("" if value is None else str(int(value)))

    @staticmethod
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        try:
            return strtobool(str(value).strip())
        except ValueError:
            return False

    # --- defaults ---
    def _set_defaults(self):
        # A sane starting point (mirrors the EMPAD preset); Nx/Ny left blank.
        self.dtype_combo.setCurrentText("float32")
        self.endian_combo.setCurrentIndex(0)
        self.offset_edit.setText("0")
        self.qx_spin.setValue(128)
        self.qy_spin.setValue(128)
        self.stride_edit.setText(str(128 * 128))
        self._stride_auto = True
        self._revalidate()

    # --- signal wiring ---
    def _wire_signals(self):
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        self.dtype_combo.currentTextChanged.connect(self._revalidate)
        self.endian_combo.currentIndexChanged.connect(self._revalidate)
        self.offset_edit.textChanged.connect(self._revalidate)
        self.stride_edit.textChanged.connect(self._on_stride_changed)
        self.qx_spin.valueChanged.connect(self._on_q_changed)
        self.qy_spin.valueChanged.connect(self._on_q_changed)
        self.nx_edit.textChanged.connect(self._revalidate)
        self.ny_edit.textChanged.connect(self._revalidate)
        self.memmap_check.toggled.connect(self._revalidate)
        self.bin_spin.valueChanged.connect(self._revalidate)

    def _on_q_changed(self, *args):
        if self._stride_auto:
            self.stride_edit.setText(
                str(self.qx_spin.value() * self.qy_spin.value())
            )
        self._set_bin_range()
        self._revalidate()

    def _on_stride_changed(self, *args):
        # A manual edit makes the stride a custom pitch.
        self._stride_auto = False
        self._revalidate()

    def _set_bin_range(self):
        self.bin_spin.setRange(
            1, min(self.qx_spin.value(), self.qy_spin.value())
        )

    # --- validation ---
    @staticmethod
    def _int_field(edit, minimum):
        """Return the field's integer value, or None if blank/invalid/<minimum."""
        text = edit.text().strip()
        if not text:
            return None
        try:
            value = int(text)
        except ValueError:
            return None
        return value if value >= minimum else None

    def _revalidate(self, *args):
        """Check the settings fit the file; enable Ok and show a live readout."""
        try:
            dt = np.dtype(self.dtype_combo.currentText())
        except TypeError:
            dt = None
        itemsize = dt.itemsize if dt is not None else 0

        offset = self._int_field(self.offset_edit, 0)
        stride = self._int_field(self.stride_edit, 1)
        qx, qy = self.qx_spin.value(), self.qy_spin.value()
        nx = self._int_field(self.nx_edit, 1)
        ny = self._int_field(self.ny_edit, 1)
        b = self.bin_spin.value()
        per = qx * qy

        problems = []
        if dt is None:
            problems.append("invalid data type")
        if offset is None:
            problems.append("offset must be ≥ 0")
        if stride is None:
            problems.append("stride must be ≥ 1")
        elif stride < per:
            problems.append("stride must be ≥ Qx·Qy")
        if nx is None:
            problems.append("set Nx")
        if ny is None:
            problems.append("set Ny")
        if not 1 <= b <= min(qx, qy):
            problems.append("bin factor out of range")

        # How many full patterns fit after the offset (None if not computable).
        maxn = None
        file_items = None
        if itemsize and offset is not None:
            try:
                file_items = os.path.getsize(self.path) // itemsize
            except OSError:
                file_items = 0
            if stride is not None:
                room = file_items - offset - per
                maxn = room // stride + 1 if room >= 0 else 0

        fits = (
            not problems
            and maxn is not None
            and maxn >= 1
            and nx is not None and ny is not None
            and 1 <= nx * ny <= maxn
        )

        if problems:
            msg = "⚠ " + "; ".join(problems)
        elif maxn is not None:
            msg = f"{file_items:,} pixels · up to {maxn:,} patterns (need {nx * ny:,})"
            if not fits:
                msg += " — exceeds the file"
        else:
            msg = "set a valid data type, offset, and stride"

        self.status_label.setText(msg)
        self.ok_button.setEnabled(fits)

    # --- loading ---
    def accept(self):
        try:
            self._load()
        except Exception as exc:
            self.api.status_bar.showMessage(
                f"Failed to read '{self.path}': {exc}", 5_000
            )
            return
        super().accept()

    def _load(self):
        dt = np.dtype(self.dtype_combo.currentText())
        endian = self.endian_combo.currentData()
        if endian == "little":
            dt = dt.newbyteorder("<")
        elif endian == "big":
            dt = dt.newbyteorder(">")
        itemsize = dt.itemsize

        offset = int(self.offset_edit.text())
        stride = int(self.stride_edit.text())
        qx, qy = self.qx_spin.value(), self.qy_spin.value()
        nx, ny = int(self.nx_edit.text()), int(self.ny_edit.text())
        b = self.bin_spin.value()
        memmap = self.memmap_check.isChecked()

        n = nx * ny
        per = qx * qy

        if stride == per and b == 1:
            # Contiguous, no bin: keep a lazy memmap view (no copy).
            arr = np.memmap(
                self.path, dtype=dt, mode="r",
                shape=(nx, ny, qx, qy), offset=offset * itemsize,
            )
            if not memmap:
                arr = arr[()]
        else:
            span = offset + (n - 1) * stride + per
            flat = np.memmap(self.path, dtype=dt, mode="r", shape=(span,))
            if stride == per:
                win = flat[offset:offset + n * per]
            else:  # strided gather (custom pitch)
                win = flat[(offset + np.arange(n) * stride)[:, None]
                           + np.arange(per)[None, :]]
            arr = win.reshape(nx, ny, qx, qy)

        if b > 1:
            arr = lazy_bin_load(arr, b)

        self.api.set_datacube(py4DSTEM.DataCube(arr))
