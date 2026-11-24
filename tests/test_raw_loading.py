from types import SimpleNamespace

import numpy as np
import pytest

from py4D_browser.menu_actions import load_file
from py4d_browser_plugin.raw_loader.raw_loader import RawLoadDialog


@pytest.fixture
def loader():
    return SimpleNamespace(
        datacube=None,
        update_diffraction_space_view=lambda **kwargs: None,
        update_real_space_view=lambda **kwargs: None,
        setWindowTitle=lambda title: None,
        signal_datacube_changed=SimpleNamespace(emit=lambda: None),
    )


@pytest.mark.parametrize("partial_bytes", [b"", b"partial frame"])
def test_invalid_empad_leaves_datacube_intact(loader, tmp_path, partial_bytes):
    path = tmp_path / "incomplete.raw"
    if partial_bytes:
        np.ones((130, 128), dtype=np.float32).tofile(path)
        with path.open("ab") as raw_file:
            raw_file.write(partial_bytes)
    else:
        path.touch()
    original_cube = object()
    loader.datacube = original_cube
    messages = []
    loader.statusBar = lambda: SimpleNamespace(
        showMessage=lambda message, timeout: messages.append(message)
    )

    load_file(loader, path)

    assert loader.datacube is original_cube
    assert "Open Raw" in messages[0]


@pytest.mark.parametrize(
    ("mmap", "binning", "shape", "intensity"),
    [
        (False, 1, (1, 1, 128, 128), 2),
        (True, 1, (1, 1, 128, 128), 2),
        (False, 2, (1, 1, 64, 64), 8),
    ],
)
def test_empad_ram_and_binned_load(loader, tmp_path, mmap, binning, shape, intensity):
    path = tmp_path / "empad.raw"
    frame = np.full((130, 128), 2, dtype=np.float32)
    frame[128:] = 99
    frame.tofile(path)

    load_file(loader, path, mmap=mmap, binning=binning)

    data = loader.datacube.data
    assert data.shape == shape
    assert data.flags.writeable is not mmap
    assert isinstance(data, np.memmap) is mmap
    np.testing.assert_array_equal(data, intensity)


@pytest.mark.parametrize("stride", [6, 10])
@pytest.mark.parametrize("mmap", [False, True])
def test_open_raw_dialog_respects_memmap(viewer, tmp_path, stride, mmap):
    path = tmp_path / "arbitrary.bin"
    frames = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    file_data = np.full(2 * stride, 99, dtype=np.float32)
    file_data[:6] = frames[0].ravel()
    file_data[stride : stride + 6] = frames[1].ravel()
    file_data.tofile(path)

    loaded = []
    api = SimpleNamespace(
        qt_window=None,
        settings=viewer.settings,
        status_bar=viewer.statusBar(),
        set_datacube=loaded.append,
    )
    dialog = RawLoadDialog(api, path)
    dialog.qx_spin.setValue(2)
    dialog.qy_spin.setValue(3)
    dialog.nx_edit.setText("1")
    dialog.ny_edit.setText("2")
    dialog.stride_edit.setText(str(stride))
    dialog.memmap_check.setChecked(mmap)

    assert dialog.ok_button.isEnabled()
    dialog.accept()
    data = loaded[0].data
    assert data.flags.writeable is not mmap
    np.testing.assert_array_equal(data[0], frames)
