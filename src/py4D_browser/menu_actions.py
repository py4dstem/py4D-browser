import py4DSTEM
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from py4D_browser.help_menu import KeyboardMapMenu
from py4D_browser.dialogs import ResizeDialog, BinningDialog
from py4DSTEM.io.filereaders import read_arina

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py4D_browser import DataViewer


def load_data_auto(self: "DataViewer"):
    filename = self.show_file_dialog()
    self.load_file(filename)


def load_data_mmap(self: "DataViewer"):
    filename = self.show_file_dialog()
    self.load_file(filename, mmap=True)


def load_data_bin(self: "DataViewer"):
    filename = self.show_file_dialog()

    shape, dtype = get_file_info(filename)
    file_size = os.path.getsize(filename)

    ok, bin_value = BinningDialog.get_bin_value(
        filepath=filename,
        file_size=file_size,
        shape=shape,
        dtype=dtype,
        parent=self,
    )
    if not ok:
        return

    self.load_file(filename, mmap=False, binning=bin_value)


def load_data_arina(self: "DataViewer"):
    filename = self.show_file_dialog()
    dataset = read_arina(filename)

    # Warn if the data is not square
    if dataset.data.shape[1] == 1:
        self.statusBar().showMessage(f"Arina data was loaded as 3D, please reshape...")

    self.datacube = dataset
    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(filename)
    self.signal_datacube_changed.emit()


def load_file(self: "DataViewer", filepath, mmap=False, binning=1):
    print(f"Loading file {filepath}")
    extension = os.path.splitext(filepath)[-1].lower()
    print(f"Type: {extension}")

    # mmap + binning are incompatible — binning produces a RAM array
    if mmap and binning > 1:
        QMessageBox.information(
            self,
            "mmap + binning",
            "Binning and mmap cannot be used together. "
            "Data will be loaded with lazy binning into RAM.",
        )
        mmap = False

    if extension in (".h5", ".hdf5", ".py4dstem", ".emd", ".mat"):
        file = h5py.File(filepath, "r")
        datacubes = get_ND(file)
        print(f"Found {len(datacubes)} 4D datasets inside the HDF5 file...")
        if len(datacubes) >= 1:
            print(f"Reading dataset at location {datacubes[0].name}")

            parent = "/".join(datacubes[0].name.split("/")[:-1])
            if len(parent) > 1 and "emd_group_type" in file[parent].attrs:
                print("This appears to be an emdfile... reading natively")
                if binning > 1:
                    # Try to load calibration before closing the file
                    calibration = None
                    try:
                        calibration = py4DSTEM.Calibration.from_h5(
                            file["/datacube_root/metadatabundle/calibration"]
                        )
                    except Exception:
                        pass
                    array = lazy_bin_load(
                        datacubes[0], binning, output_dtype=datacubes[0].dtype
                    )
                    file.close()
                    self.datacube = py4DSTEM.DataCube(array)
                    if calibration is not None:
                        self.datacube.calibration = calibration
                        # Scale Q pixel size by bin factor
                        q_size = self.datacube.calibration.get_Q_pixel_size()
                        if q_size is not None:
                            self.datacube.calibration.set_Q_pixel_size(q_size * binning)
                else:
                    self.datacube = py4DSTEM.DataCube.from_h5(datacubes[0].file[parent])
                    try:
                        calibration = py4DSTEM.Calibration.from_h5(
                            datacubes[0].file[
                                "/datacube_root/metadatabundle/calibration"
                            ]
                        )
                        self.datacube.calibration = calibration
                    except Exception as e:
                        self.statusBar().showMessage(str(e))
            else:
                if binning > 1:
                    array = lazy_bin_load(
                        datacubes[0], binning, output_dtype=datacubes[0].dtype
                    )
                    file.close()
                    self.datacube = py4DSTEM.DataCube(array)
                else:
                    self.datacube = py4DSTEM.DataCube(
                        datacubes[0] if mmap else datacubes[0][()]
                    )

                R_size, R_units, Q_size, Q_units = find_calibrations(
                    datacubes[0] if binning == 1 else self.datacube.data
                )

                # Scale Q pixel size by bin factor
                if binning > 1:
                    Q_size = Q_size * binning if Q_size is not None else None

                self.datacube.calibration.set_R_pixel_size(R_size)
                self.datacube.calibration.set_R_pixel_units(R_units)
                self.datacube.calibration.set_Q_pixel_size(Q_size)
                self.datacube.calibration.set_Q_pixel_units(Q_units)

        else:
            # if no 4D data was found, look for 3D data
            datacubes = get_ND(file, N=3)
            print(f"Found {len(datacubes)} 3D datasets inside the HDF5 file...")
            if len(datacubes) >= 1:
                if binning > 1:
                    array = lazy_bin_load(
                        datacubes[0], binning, output_dtype=datacubes[0].dtype
                    )
                    file.close()
                else:
                    array = datacubes[0] if mmap else datacubes[0][()]
                new_shape = ResizeDialog.get_new_size([1, array.shape[0]], parent=self)
                self.datacube = py4DSTEM.DataCube(
                    array.reshape(*new_shape, *array.shape[1:])
                )
            else:
                raise ValueError("No 4D (or even 3D) data detected in the H5 file!")
    elif extension in [".npy"]:
        if binning > 1:
            memmap = np.load(filepath, mmap_mode="r")
            array = lazy_bin_load(memmap, binning, output_dtype=memmap.dtype)
            self.datacube = py4DSTEM.DataCube(array)
        else:
            self.datacube = py4DSTEM.DataCube(np.load(filepath))
    else:
        self.datacube = py4DSTEM.import_file(
            filepath,
            mem="MEMMAP" if mmap else "RAM",
            binfactor=binning,
        )

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(filepath)
    self.signal_datacube_changed.emit()


def set_datacube(self: "DataViewer", datacube, window_title):
    self.datacube = datacube

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(window_title)
    self.signal_datacube_changed.emit()


def reshape_data(self: "DataViewer"):
    new_shape = ResizeDialog.get_new_size(self.datacube.shape[:2], parent=self)
    self.datacube.data = self.datacube.data.reshape(
        *new_shape, *self.datacube.data.shape[2:]
    )

    print(f"Reshaping data to {new_shape}")

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)


def export_datacube(self: "DataViewer", save_format: str):
    assert save_format in [
        "Raw float32",
        "py4DSTEM HDF5",
        "Plain HDF5",
    ], f"unrecognized format {format}"
    assert self.datacube is not None, "No datacube!"

    # Display RAW format disclaimer
    if save_format == "Raw float32":
        response = QMessageBox.question(
            self,
            "Save RAW file?",
            (
                "Saving raw binary files is not recommended as such files"
                " encode no information about the shape, endianness, or "
                "ordering of the data. Saving to HDF5 is recommended. "
                "Do you wish to continue saving RAW data?"
            ),
            QMessageBox.Cancel,
            QMessageBox.Save,
        )

        if response == QMessageBox.Cancel:
            self.statusBar().showMessage("Cancelling due to user guilt", 5_000)
            return

    try:
        filename = self.get_savefile_name(save_format)

        if save_format == "Raw float32":
            self.datacube.data.astype(np.float32).tofile(filename)

        elif save_format == "py4DSTEM HDF5":
            py4DSTEM.save(filename, self.datacube, mode="o")

        elif save_format == "Plain HDF5":
            with h5py.File(filename, "w") as f:
                f["array"] = self.datacube.data

        self.setWindowTitle(filename)
        self.statusBar().showMessage(f"File saved to {filename}")
    except Exception as exc:
        import traceback

        QMessageBox.critical(
            self,
            "Uh-oh!",
            traceback.format_exc(),
        )

        raise exc


def export_virtual_image(self: "DataViewer", im_format: str, im_type: str):
    assert im_type in ["image", "diffraction", "result"], f"bad image type: {im_type}"

    filename = self.get_savefile_name(im_format)

    if im_type == "image":
        view = self.real_space_widget
        rawimg = self.unscaled_realspace_image
    elif im_type == "diffraction":
        view = self.diffraction_space_widget
        rawimg = self.unscaled_diffraction_image
    elif im_type == "result":
        view = self.fft_widget
        rawimg = self.unscaled_fft_image
    else:
        raise RuntimeError("Unrecognized export image source...")

    vimg = view.image.T
    vmin, vmax = view.getLevels()

    if im_format == "PNG (display)":
        plt.imsave(
            fname=filename, arr=vimg, vmin=vmin, vmax=vmax, format="png", cmap="gray"
        )
    elif im_format == "TIFF (display)":
        plt.imsave(
            fname=filename, arr=vimg, vmin=vmin, vmax=vmax, format="tiff", cmap="gray"
        )
    elif im_format == "TIFF (raw)":
        from tifffile import TiffWriter

        with TiffWriter(filename) as tw:
            tw.write(rawimg)
    else:
        raise RuntimeError("Nothing saved! Format not recognized")


def copy_vimg_to_clipboard(self: "DataViewer"):
    img = self.real_space_widget.getImageItem()

    if img._renderRequired:
        img.render()

    QApplication.clipboard().setImage(img.qimage)


def copy_diff_to_clipboard(self: "DataViewer"):
    img = self.diffraction_space_widget.getImageItem()

    if img._renderRequired:
        img.render()

    QApplication.clipboard().setImage(img.qimage)


def copy_result_to_clipboard(self: "DataViewer"):
    img = self.fft_widget.getImageItem()

    if img._renderRequired:
        img.render()

    QApplication.clipboard().setImage(img.qimage)


def show_keyboard_map(self: "DataViewer"):
    keymap = KeyboardMapMenu(parent=self)
    keymap.open()


def show_file_dialog(self: "DataViewer") -> str:
    filename = QFileDialog.getOpenFileName(
        self,
        "Open 4D-STEM Data",
        "",
        "4D-STEM Data (*.dm3 *.dm4 *.raw *.mib *.gtg *.h5 *.hdf5 *.emd *.py4dstem *.npy *.npz *.mat);;Any file (*)",
    )
    if filename is not None and len(filename[0]) > 0:
        return filename[0]
    else:
        print("File was invalid, or something?")
        raise ValueError("Could not read file")


def get_savefile_name(self: "DataViewer", file_format) -> str:
    filters = {
        "Raw float32": "RAW File (*.raw *.f32);;Any file (*)",
        "py4DSTEM HDF5": "HDF5 File (*.hdf5 *.h5 *.emd *.py4dstem);;Any file (*)",
        "Plain HDF5": "HDF5 File (*.hdf5 *.h5;;Any file (*)",
        "PNG (display)": "PNG File (*.png);;Any file (*)",
        "TIFF (display)": "TIFF File (*.tiff *.tif *.tff);;Any File (*)",
        "TIFF (raw)": "TIFF File (*.tiff *.tif *.tff);;Any File (*)",
    }

    defaults = {
        "Raw float32": ".raw",
        "py4DSTEM HDF5": ".h5",
        "Plain HDF5": ".h5",
        "PNG (display)": ".png",
        "TIFF (display)": ".tiff",
        "TIFF (raw)": ".tiff",
    }

    file_filter = filters.get(file_format, "Any file (*)")

    filename = QFileDialog.getSaveFileName(
        parent=self,
        caption="Select save file",
        directory="",
        filter=file_filter,
    )

    if filename is not None and len(filename[0]) > 0:
        fname = filename[0]
        print(f"Save file picked at {filename}")

        if os.path.splitext(fname)[1] == "":
            fname = fname + defaults.get(file_format, "")
            print(f"Added default extension to get: {fname}")
        return fname
    else:
        print("File was invalid, or something?")
        print(f"QFileDialog returned {filename}")
        raise ValueError("Could get save file")


def get_ND(f, datacubes=None, N=4):
    # Traverse an h5py.File and look for Datasets with N dimensions
    if datacubes is None:
        datacubes = []
    for k in f.keys():
        if isinstance(f[k], h5py.Dataset):
            # we found data
            if len(f[k].shape) == N:
                datacubes.append(f[k])
        elif isinstance(f[k], h5py.Group):
            get_ND(f[k], datacubes)
    return datacubes


def find_calibrations(dset: h5py.Dataset):
    # Attempt to find calibrations from an H5 file
    R_size, R_units, Q_size, Q_units = 1.0, "pixels", 1.0, "pixels"

    # Does it look like a py4DSTEM file?
    try:
        if "emd_group_type" in dset.parent.attrs:
            # EMD files theoretically store this in the Array,
            # but in practice seem to only keep the calibrations
            # in the Metadata object, which is separate

            # R_size = dset.parent["dim0"][1] - dset.parent["dim0"][0]
            # R_units = dset.parent["dim0"].attrs["units"]

            # Q_size = dset.parent["dim3"][1] - dset.parent["dim3"][0]
            # Q_units = dset.parent["dim3"].attrs["units"]
            R_size = dset.parent.parent["metadatabundle"]["calibration"][
                "R_pixel_size"
            ][()]
            R_units = dset.parent.parent["metadatabundle"]["calibration"][
                "R_pixel_units"
            ][()].decode()

            Q_size = dset.parent.parent["metadatabundle"]["calibration"][
                "Q_pixel_size"
            ][()]
            Q_units = dset.parent.parent["metadatabundle"]["calibration"][
                "Q_pixel_units"
            ][()].decode()
    except:
        print(
            "This file looked like a py4DSTEM dataset but the dim vectors appear malformed..."
        )

    # Does it look like an abTEM file?
    try:
        if "sampling" in dset.parent and "units" in dset.parent:
            R_size = dset.parent["sampling"][0]
            R_units = dset.parent["units"][0].decode().replace("Å", "A")

            Q_size = dset.parent["sampling"][3]
            Q_units = dset.parent["units"][3].decode()
    except:
        print(
            "This file looked like an abTEM simulation but the calibrations aren't as expected..."
        )

    return R_size, R_units, Q_size, Q_units


def lazy_bin_load(source, bin_factor, output_dtype=None, chunk_size=10):
    """Read data from a source and bin the last 2 (detector) dimensions lazily.

    Reads the source in chunks along scan-space dimensions so the full unbinned
    array never resides in RAM.  Each chunk's detector frames are binned by
    averaging over NxN blocks.

    Args:
        source: Any subscriptable array-like (h5py.Dataset, np.memmap, np.ndarray).
            Must be at least 3D (scan + detector).
        bin_factor: Integer divisor for each detector dimension.
        output_dtype: Dtype for the output (defaults to float32). Ignored for
            integer source types — those always produce float32 to avoid overflow.
        chunk_size: Maximum number of scan positions to read per chunk.

    Returns:
        numpy array with detector dimensions divided by bin_factor.
    """
    if bin_factor == 1:
        return source[()]

    shape = source.shape
    if len(shape) < 3:
        raise ValueError(f"Source must be at least 3D, got {len(shape)}D")

    Dx, Dy = shape[-2], shape[-1]

    if bin_factor > Dx or bin_factor > Dy:
        raise ValueError(
            f"Bin factor {bin_factor} exceeds detector dimension "
            f"(detector is {Dx} x {Dy}). Reduce bin factor or disable binning."
        )

    # Crop detector dimensions to the largest even multiple of bin_factor
    crop_Dx = (Dx // bin_factor) * bin_factor
    crop_Dy = (Dy // bin_factor) * bin_factor

    scan_shape = shape[:-2]
    binned_Dx = crop_Dx // bin_factor
    binned_Dy = crop_Dy // bin_factor

    # Integer sources can overflow when binned (averaging produces non-integer
    # intermediate values), so always use float32 for integer data types.
    if output_dtype is None:
        output_dtype = np.float32
    if np.issubdtype(source.dtype, np.integer):
        output_dtype = np.float32

    out = np.empty(scan_shape + (binned_Dx, binned_Dy), dtype=output_dtype)

    # Iterate over scan-space in chunks
    _lazy_bin_load_nd(
        source,
        out,
        0,
        binned_Dx,
        binned_Dy,
        bin_factor,
        chunk_size,
        fixed=(),
        crop_Dx=crop_Dx,
        crop_Dy=crop_Dy,
    )

    return out


def _lazy_bin_load_nd(
    source,
    out,
    scan_idx,
    binned_Dx,
    binned_Dy,
    bin_factor,
    chunk_size,
    fixed,
    crop_Dx,
    crop_Dy,
):
    """Recursive helper to iterate over scan-space dimensions in chunks.

    Tracks accumulated slices in `fixed` (a tuple of slice objects) so that
    the leaf level can build a complete ND slice to read the chunk with the
    correct (cropped) detector dimensions.

    Args:
        source: The full data source (not sliced — indexing uses `fixed`).
        out: The full output array (indexing uses `fixed`).
        scan_idx: Index into the scan-space shape (incremented to reach detector dims).
        binned_Dx, binned_Dy: Binned detector dimensions.
        bin_factor: Bin factor for each detector dimension.
        chunk_size: Maximum chunk size for the current dimension.
        fixed: Tuple of slice objects for dimensions already iterated over.
        crop_Dx, crop_Dy: Cropped detector dimensions (largest even multiple of bin_factor).
    """
    N = source.shape[scan_idx]
    start = 0
    while start < N:
        end = min(start + chunk_size, N)
        if scan_idx == len(source.shape) - 3:
            # Last scan dimension — read chunks and bin detector dims
            sl = fixed + (slice(start, end), slice(0, crop_Dx), slice(0, crop_Dy))
            chunk = source[sl]  # (chunk_len, crop_Dx, crop_Dy)
            # Flatten all scan dims, bin detector, reshape back
            flat_len = (
                chunk.shape[:-2].numel()
                if hasattr(chunk.shape[:-2], "numel")
                else int(np.prod(chunk.shape[:-2]))
            )
            binned = (
                chunk.reshape(flat_len, binned_Dx, bin_factor, binned_Dy, bin_factor)
                .mean(axis=-1)
                .mean(axis=-2)
            )
            out_shape = chunk.shape[:-2]  # shape of scan dims in this chunk
            binned = binned.reshape(*out_shape, binned_Dx, binned_Dy)
            out[fixed + (slice(start, end),)] = binned
        else:
            # Intermediate scan dimension — recurse with extended slice
            _lazy_bin_load_nd(
                source,
                out,
                scan_idx + 1,
                binned_Dx,
                binned_Dy,
                bin_factor,
                chunk_size,
                fixed + (slice(start, end),),
                crop_Dx,
                crop_Dy,
            )
        start = end


def get_file_info(filepath):
    """Get (shape, dtype_str) for the primary dataset in a file, if possible.

    Returns ((s0, s1, ...), dtype_string) or (None, None) for unsupported types.
    """
    ext = os.path.splitext(filepath)[-1].lower()
    if ext in (".h5", ".hdf5", ".py4dstem", ".emd", ".mat"):
        with h5py.File(filepath, "r") as f:
            dsets = get_ND(f)
            if not dsets:
                dsets = get_ND(f, N=3)
            if dsets:
                return tuple(dsets[0].shape), str(dsets[0].dtype)
    elif ext == ".npy":
        # Read header without loading full array
        with open(filepath, "rb") as f:
            shape, _, dtype = np.lib.format.read_header(f)
        return shape, str(dtype)
    return None, None
