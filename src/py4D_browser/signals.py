from typing import Callable, Optional

"""
Registration and marshalling for callbacks/signals
"""

__all__ = ["register_result_callback", "set_internal_result_callback"]

_registered_result_callbacks: dict[str, Optional[Callable]] = {
    "diffraction": None,
    "virtual_image": None,
    "datacube": None,
    "cleanup": None,
}


def register_result_callback(
    self,
    title: str,
    cleanup: Optional[Callable],
    callback_diffraction_pattern_changed: Optional[Callable] = None,
    callback_virtual_image_changed: Optional[Callable] = None,
    callback_datacube_changed: Optional[Callable] = None,
):
    # Only one plugin should use these callbacks at a time, and the menu
    # must be updated to indicate which plugin is actively recieving
    # the signals.

    # The cleanup function is called when the plugin/handler is removed,
    # and should be used to restore the GUI to its original state (i.e.
    # by removing any additional ROIs or windows).
    self.result_other_action.setText(title)
    # self.fft_widget_text.setText(title) # this is handled in set_result_image now
    self.result_other_action.setChecked(True)

    _replace_result_callbacks(
        self,
        cleanup=cleanup,
        callback_diffraction_pattern_changed=callback_diffraction_pattern_changed,
        callback_virtual_image_changed=callback_virtual_image_changed,
        callback_datacube_changed=callback_datacube_changed,
    )


def set_internal_result_callback(self):
    # Set the callbacks back to the internal ones for FFT/EWPC
    # and use the default renderer

    # TODO: In a future release, these actions will also
    # be handed as a plugin, but new plumbing is required
    # for plugins to add to this menu directly...

    self.result_other_action.setText("Plugin")

    _replace_result_callbacks(
        self,
        callback_diffraction_pattern_changed=self.update_fft_view,
        callback_virtual_image_changed=self.update_fft_view,
        callback_datacube_changed=None,
    )

    self.update_fft_view()


def _replace_result_callbacks(
    self,
    cleanup: Optional[Callable] = None,
    callback_diffraction_pattern_changed: Optional[Callable] = None,
    callback_virtual_image_changed: Optional[Callable] = None,
    callback_datacube_changed: Optional[Callable] = None,
):
    # unregister any previously set callbacks
    if _registered_result_callbacks["diffraction"] is not None:
        self.signal_diffraction_data_changed.disconnect(
            _registered_result_callbacks["diffraction"]
        )
    _registered_result_callbacks["diffraction"] = None

    if _registered_result_callbacks["virtual_image"] is not None:
        self.signal_virtual_image_data_changed.disconnect(
            _registered_result_callbacks["virtual_image"]
        )
    _registered_result_callbacks["virtual_image"] = None

    if _registered_result_callbacks["datacube"] is not None:
        self.signal_datacube_changed.disconnect(
            _registered_result_callbacks["datacube"]
        )
    _registered_result_callbacks["datacube"] = None

    # call the cleanup function for the old plugin/handler
    if _registered_result_callbacks["cleanup"] is not None:
        _registered_result_callbacks["cleanup"]()
    _registered_result_callbacks["cleanup"] = cleanup

    # register any supplied callbacks
    if callback_diffraction_pattern_changed is not None:
        _registered_result_callbacks["diffraction"] = (
            self.signal_diffraction_data_changed.connect(
                callback_diffraction_pattern_changed
            )
        )

    if callback_virtual_image_changed is not None:
        _registered_result_callbacks["virtual_image"] = (
            self.signal_virtual_image_data_changed.connect(
                callback_virtual_image_changed
            )
        )

    if callback_datacube_changed is not None:
        _registered_result_callbacks["datacube"] = self.signal_datacube_changed.connect(
            callback_datacube_changed
        )
