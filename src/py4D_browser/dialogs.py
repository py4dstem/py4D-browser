import math

from PyQt5.QtWidgets import QPushButton, QLabel, QDialogButtonBox
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QSpinBox,
)


class ResizeDialog(QDialog):
    def __init__(self, size, parent=None):
        super().__init__(parent=parent)

        self.new_size = size
        Nmax = size[0] * size[1]

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Dataset size unknown. Please enter the shape:"))

        box_layout = QHBoxLayout()
        box_layout.addWidget(QLabel("X:"))

        xbox = QSpinBox()
        xbox.setRange(1, Nmax)
        xbox.setSingleStep(1)
        xbox.setKeyboardTracking(False)
        xbox.valueChanged.connect(self.x_box_changed)
        box_layout.addWidget(xbox)

        box_layout.addStretch()
        box_layout.addWidget(QLabel("Y:"))

        ybox = QSpinBox()
        ybox.setRange(1, Nmax)
        ybox.setSingleStep(1)
        ybox.setValue(Nmax)
        ybox.setKeyboardTracking(False)
        ybox.valueChanged.connect(self.y_box_changed)
        box_layout.addWidget(ybox)

        layout.addLayout(box_layout)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        done_button = QPushButton("Done")
        done_button.pressed.connect(self.close)
        button_layout.addWidget(done_button)
        layout.addLayout(button_layout)

        self.x_box = xbox
        self.y_box = ybox
        self.x_box_last = xbox.value()
        self.y_box_last = ybox.value()
        self.N = Nmax

        self.resize(600, 400)

    @classmethod
    def get_new_size(cls, size, parent=None):
        dialog = cls(size=size, parent=parent)
        dialog.exec_()
        return dialog.new_size

    def x_box_changed(self, new_value):
        if new_value == self.x_box_last:
            return
        x_new, y_new = self.get_next_rect(
            new_value, "down" if new_value < self.x_box_last else "up"
        )

        self.x_box_last = x_new
        self.y_box_last = y_new

        self.x_box.setValue(x_new)
        self.y_box.setValue(y_new)

        self.new_size = [x_new, y_new]

    def y_box_changed(self, new_value):
        if new_value == self.y_box_last:
            return
        y_new, x_new = self.get_next_rect(
            new_value, "down" if new_value < self.y_box_last else "up"
        )

        self.x_box_last = x_new
        self.y_box_last = y_new

        self.x_box.setValue(x_new)
        self.y_box.setValue(y_new)

        self.new_size = [x_new, y_new]

    def get_next_rect(self, current, direction):
        # get the next perfect rectangle
        iterator = (
            range(current, 0, -1) if direction == "down" else range(current, self.N + 1)
        )

        for i in iterator:
            if self.N % i == 0:
                return i, self.N // i

        raise ValueError("Factor finding failed, frustratingly.")


class BinningDialog(QDialog):
    """Dialog to select a binning factor for loading binned data.

    Displays file size, original dimensions, and estimated RAM usage
    both before and after binning.
    """

    def __init__(
        self,
        filepath: str,
        file_size: int,
        shape: tuple | None,
        dtype: str | None,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.setWindowTitle("Select Binning Factor")

        self.filepath = filepath
        self.file_size = file_size
        self.shape = shape
        self.dtype = dtype

        layout = QVBoxLayout(self)

        # File path display
        layout.addWidget(QLabel(f"File: {filepath}"))
        layout.addSpacing(8)

        # File info section
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel("File Information:"))
        info_layout.addWidget(QLabel(f"  File size: {self._format_size(file_size)}"))

        if shape is not None:
            info_layout.addWidget(
                QLabel(f"  Dimensions: {' x '.join(map(str, shape))}")
            )
            if dtype:
                info_layout.addWidget(QLabel(f"  Data type: {dtype}"))
            ram = self._calc_unbinned_ram()
            info_layout.addWidget(QLabel(f"  Un-binned RAM: {self._format_size(ram)}"))
        else:
            info_layout.addWidget(
                QLabel("  Dimensions: unavailable for this file type")
            )

        layout.addLayout(info_layout)
        layout.addSpacing(8)

        # Binning control section
        binning_layout = QVBoxLayout()
        binning_layout.addWidget(QLabel("Binning Control:"))
        binning_layout.addWidget(
            QLabel("  Apply binning factor to detector (last 2) dimensions:")
        )

        spin_layout = QHBoxLayout()
        spin_layout.addWidget(QLabel("  Bin factor:"), 0)

        self.bin_spin = QSpinBox()
        if self.shape is not None:
            max_bin = min(self.shape[-2:])
            self.bin_spin.setRange(1, max(max_bin, 1))
        else:
            self.bin_spin.setRange(1, 100)
        self.bin_spin.setValue(4)
        self.bin_spin.setSingleStep(1)
        self.bin_spin.setAccelerated(True)
        self.bin_spin.setKeyboardTracking(False)
        self.bin_spin.valueChanged.connect(self._update_estimates)
        spin_layout.addWidget(self.bin_spin)
        spin_layout.addStretch()

        binning_layout.addLayout(spin_layout)
        layout.addLayout(binning_layout)
        layout.addSpacing(8)

        # Estimate section
        estimate_layout = QVBoxLayout()
        estimate_layout.addWidget(QLabel("Estimated Result:"))

        self.binned_dims_label = QLabel()
        estimate_layout.addWidget(self.binned_dims_label)

        self.binned_ram_label = QLabel()
        estimate_layout.addWidget(self.binned_ram_label)

        layout.addLayout(estimate_layout)
        layout.addStretch()

        # Crop notice (shown when detector dims aren't even multiples of bin factor)
        self.crop_label = QLabel()
        self.crop_label.setStyleSheet("color: orange;")
        self.crop_label.hide()
        layout.addWidget(self.crop_label)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self.ok_button = btns.button(QDialogButtonBox.Ok)
        layout.addWidget(btns)

        self._update_estimates()

    # ---- public classmethod entry-point ----

    @classmethod
    def get_bin_value(cls, filepath, file_size, shape, dtype, parent=None):
        """Show the dialog and return (accepted, bin_value)."""
        dlg = cls(
            filepath=filepath,
            file_size=file_size,
            shape=shape,
            dtype=dtype,
            parent=parent,
        )
        ok = dlg.exec_() == QDialog.Accepted
        return ok, dlg.bin_spin.value()

    # ---- helpers ----

    def _calc_unbinned_ram(self) -> int:
        if self.shape is None:
            return 0
        itemsize = 1
        if self.dtype:
            itemsize = int(
                self.dtype.replace("uint", "").replace("int", "").replace("float", "")
            )
        return int(math.prod(self.shape)) * itemsize

    def _format_size(self, nbytes: int) -> str:
        if nbytes <= 0:
            return "N/A"
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if nbytes < 1024 or unit == "TB":
                f = nbytes / 1
                if unit == "B" and nbytes == int(nbytes):
                    return f"{int(nbytes)} B"
                return f"{f:.1f} {unit}"
            nbytes /= 1024
        return f"{nbytes:.1f} PB"

    def _update_estimates(self):
        bin_val = self.bin_spin.value()

        if self.shape is not None:
            dx, dy = self.shape[-2:]
            crop_dx = (dx // bin_val) * bin_val
            crop_dy = (dy // bin_val) * bin_val
            binned = (crop_dx // bin_val, crop_dy // bin_val)
            self.binned_dims_label.setText(
                f"  Binned detector dimensions: {' x '.join(map(str, binned))}"
            )

            # Compute RAM based on binned size. Integer sources always
            # produce float32 output (averaging produces non-integer values),
            # so use max(itemsize, 4) for integer dtypes.
            itemsize = 1
            if self.dtype:
                itemsize = int(
                    self.dtype.replace("uint", "")
                    .replace("int", "")
                    .replace("float", "")
                )
                if self.dtype.startswith("uint") or self.dtype.startswith("int"):
                    itemsize = max(itemsize, 4)  # integer → float32 output
            binned_ram = int(math.prod(list(self.shape[:-2]) + list(binned))) * itemsize
            self.binned_ram_label.setText(
                f"  Estimated RAM: {self._format_size(binned_ram)}"
            )

            # Show crop notice if dimensions need cropping
            if dx % bin_val != 0 or dy % bin_val != 0:
                self.crop_label.setText(
                    f"  Note: Detector {dx} x {dy} will be cropped to {crop_dx} x {crop_dy} (dropping edge pixels)."
                )
                self.crop_label.show()
            else:
                self.crop_label.hide()
            self.ok_button.setEnabled(True)
        else:
            self.binned_dims_label.setText("  Binned dimensions: N/A")
            self.binned_ram_label.setText(
                f"  Estimated RAM: N/A (reduces by ~{bin_val**2}x vs un-binned)"
            )
            self.crop_label.hide()
            self.ok_button.setEnabled(True)
