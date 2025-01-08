from numbers import Real
import py4DSTEM
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from py4D_browser.help_menu import KeyboardMapMenu
from py4D_browser.dialogs import ResizeDialog
from py4DSTEM.io.filereaders import read_arina


def load_data_auto(self):
    filename = self.show_file_dialog()
    self.load_file(filename)


def load_data_mmap(self):
    filename = self.show_file_dialog()
    self.load_file(filename, mmap=True)


def load_data_bin(self):
    # TODO: Ask user for binning level
    filename = self.show_file_dialog()
    self.load_file(filename, mmap=False, binning=4)


def load_data_arina(self):
    filename = self.show_file_dialog()
    dataset = read_arina(filename)

    # Try to reshape the data to be square
    N_patterns = dataset.data.shape[1]
    Nxy = np.sqrt(N_patterns)
    if np.abs(Nxy - np.round(Nxy)) <= 1e-10:
        Nxy = int(Nxy)
        dataset.data = dataset.data.reshape(
            Nxy, Nxy, dataset.data.shape[2], dataset.data.shape[3]
        )
    else:
        self.statusBar().showMessage(
            f"The scan appears to not be square! Found {N_patterns} patterns", 5_000
        )

    self.datacube = dataset
    self.diffraction_scale_bar.pixel_size = self.datacube.calibration.get_Q_pixel_size()
    self.diffraction_scale_bar.units = self.datacube.calibration.get_Q_pixel_units()

    self.real_space_scale_bar.pixel_size = self.datacube.calibration.get_R_pixel_size()
    self.real_space_scale_bar.units = self.datacube.calibration.get_R_pixel_units()

    self.fft_scale_bar.pixel_size = (
        1.0 / self.datacube.calibration.get_R_pixel_size() / self.datacube.R_Ny
    )
    self.fft_scale_bar.units = f"1/{self.datacube.calibration.get_R_pixel_units()}"

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(filename)


def load_file(self, filepath, mmap=False, binning=1):
    print(f"Loading file {filepath}")
    extension = os.path.splitext(filepath)[-1].lower()
    print(f"Type: {extension}")
    if extension in (".h5", ".hdf5", ".py4dstem", ".emd", ".mat"):
        file = h5py.File(filepath, "r")
        datacubes = get_ND(file)
        print(f"Found {len(datacubes)} 4D datasets inside the HDF5 file...")
        if len(datacubes) >= 1:
            # Read the first datacube in the HDF5 file into RAM
            print(f"Reading dataset at location {datacubes[0].name}")
            self.datacube = py4DSTEM.DataCube(
                datacubes[0] if mmap else datacubes[0][()]
            )

            R_size, R_units, Q_size, Q_units = find_calibrations(datacubes[0])

            self.datacube.calibration.set_R_pixel_size(R_size)
            self.datacube.calibration.set_R_pixel_units(R_units)
            self.datacube.calibration.set_Q_pixel_size(Q_size)
            self.datacube.calibration.set_Q_pixel_units(Q_units)

        else:
            # if no 4D data was found, look for 3D data
            datacubes = get_ND(file, N=3)
            print(f"Found {len(datacubes)} 3D datasets inside the HDF5 file...")
            if len(datacubes) >= 1:
                array = datacubes[0] if mmap else datacubes[0][()]
                new_shape = ResizeDialog.get_new_size([1, array.shape[0]], parent=self)
                self.datacube = py4DSTEM.DataCube(
                    array.reshape(*new_shape, *array.shape[1:])
                )
            else:
                raise ValueError("No 4D (or even 3D) data detected in the H5 file!")
    elif extension in [".npy"]:
        self.datacube = py4DSTEM.DataCube(np.load(filepath))
    else:
        self.datacube = py4DSTEM.import_file(
            filepath,
            mem="MEMMAP" if mmap else "RAM",
            binfactor=binning,
        )

    self.update_scalebars()

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(filepath)


def set_datacube(self, datacube, window_title):
    self.datacube = datacube

    self.update_scalebars()

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(window_title)


def update_scalebars(self):

    realspace_translation = {
        "A": "Å",
    }
    reciprocal_translation = {
        "A^-1": "Å⁻¹",
    }

    self.diffraction_scale_bar.pixel_size = self.datacube.calibration.get_Q_pixel_size()
    q_units = self.datacube.calibration.get_Q_pixel_units()
    self.diffraction_scale_bar.units = (
        reciprocal_translation[q_units]
        if q_units in reciprocal_translation.keys()
        else q_units
    )

    self.real_space_scale_bar.pixel_size = self.datacube.calibration.get_R_pixel_size()
    r_units = self.datacube.calibration.get_R_pixel_units()
    self.real_space_scale_bar.units = (
        realspace_translation[r_units]
        if r_units in realspace_translation.keys()
        else r_units
    )

    self.fft_scale_bar.pixel_size = (
        1.0 / self.datacube.calibration.get_R_pixel_size() / self.datacube.R_Ny
    )
    self.fft_scale_bar.units = f"{self.datacube.calibration.get_R_pixel_units()}⁻¹"

    self.diffraction_scale_bar.updateBar()
    self.real_space_scale_bar.updateBar()
    self.fft_scale_bar.updateBar()


def reshape_data(self):
    new_shape = ResizeDialog.get_new_size(self.datacube.shape[:2], parent=self)
    self.datacube.data = self.datacube.data.reshape(
        *new_shape, *self.datacube.data.shape[2:]
    )

    print(f"Reshaping data to {new_shape}")

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)


def export_datacube(self, save_format: str):
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

    filename = self.get_savefile_name(save_format)

    if save_format == "Raw float32":
        self.datacube.data.astype(np.float32).tofile(filename)

    elif save_format == "py4DSTEM HDF5":
        py4DSTEM.save(filename, self.datacube, mode="o")

    elif save_format == "Plain HDF5":
        with h5py.File(filename, "w") as f:
            f["array"] = self.datacube.data


def export_virtual_image(self, im_format: str, im_type: str):
    assert im_type in ["image", "diffraction"], f"bad image type: {im_type}"

    filename = self.get_savefile_name(im_format)

    view = (
        self.real_space_widget if im_type == "image" else self.diffraction_space_widget
    )

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

        vimg = (
            self.unscaled_realspace_image
            if im_type == "image"
            else self.unscaled_diffraction_image
        )
        with TiffWriter(filename) as tw:
            tw.write(vimg)
    else:
        raise RuntimeError("Nothing saved! Format not recognized")


def show_keyboard_map(self):
    keymap = KeyboardMapMenu(parent=self)
    keymap.open()


def show_file_dialog(self) -> str:
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


def get_savefile_name(self, file_format) -> str:
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
