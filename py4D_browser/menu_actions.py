import py4DSTEM


def load_data_auto(self):
    pass


def load_data_mmap(self):
    pass


def load_data_bin(self):
    pass


def load_file(self, filepath, mmap=False, binning=1):
    print(f"Loading file {filepath}")

    self.datacube = py4DSTEM.import_file(
        filepath,
        mem="MEMMAP" if mmap else "RAM",
        binning=binning,
    )

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(filepath)

