import py4DSTEM
from PyQt5.QtWidgets import QFileDialog
import h5py


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


def load_file(self, filepath, mmap=False, binning=1):
    print(f"Loading file {filepath}")

    if py4DSTEM.io.utils.parse_filetype(filepath) == "py4DSTEM":
        datacubes = get_4D(h5py.File(filepath, "r"))
        print(f"Found {len(datacubes)} 4D datasets inside the HDF5 file...")
        if len(datacubes) >= 1:
            # Read the first datacube in the HDF5 file into RAM
            print(f"Reading dataset at location {datacubes[0].name}")
            self.datacube = py4DSTEM.io.DataCube(
                datacubes[0] if mmap else datacubes[0][()]
            )
    else:
        self.datacube = py4DSTEM.import_file(
            filepath,
            mem="MEMMAP" if mmap else "RAM",
            binfactor=binning,
        )

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(filepath)


def show_file_dialog(self):
    filename = QFileDialog.getOpenFileName(
        self,
        "Open 4D-STEM Data",
        "",
        "4D-STEM Data (*.dm3 *.dm4 *.raw *.mib *.gtg);;Any file (*)",
    )
    if filename is not None and len(filename[0]) > 0:
        return filename[0]
    else:
        print("File was invalid, or something?")
        raise ValueError("Could not read file")


def get_4D(f, datacubes=None):
    if datacubes is None:
        datacubes = []
    for k in f.keys():
        if isinstance(f[k], h5py.Dataset):
            # we found data
            if len(f[k].shape) == 4:
                datacubes.append(f[k])
        elif isinstance(f[k], h5py.Group):
            get_4D(f[k], datacubes)
    return datacubes
