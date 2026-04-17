from typing import Callable, Optional

"""
Registration and marshalling for callbacks/signals
"""

__all__ = ["register_result_callback", "set_internal_result_callback"]

_registered_result_callbacks = {
    "diffraction": None,
    "virtual_image": None,
    "datacube": None,
}


def register_result_callback(
    self,
    title: str,
    callback_diffraction_pattern_changed: Optional[Callable] = None,
    callback_virtual_image_changed: Optional[Callable] = None,
    callback_datacube_changed: Optional[Callable] = None,
):
    # Only one plugin should use these callbacks at a time, and the menu
    # must be updated to indicate which plugin is actively recieving
    # the signals.
    self.result_other_action.setText(title)
    self.fft_widget_text.setText(title)
    self.result_other_action.setChecked(True)

    _replace_result_callbacks(
        self,
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


def _replace_result_callbacks(
    self,
    callback_diffraction_pattern_changed: Optional[Callable] = None,
    callback_virtual_image_changed: Optional[Callable] = None,
    callback_datacube_changed: Optional[Callable] = None,
):
    # unregister any previously set callbacks
    if _registered_result_callbacks["diffraction"] is not None:
        self.signal_diffraction_data_changed.disconnect(
            _registered_result_callbacks["diffraction"]
        )

    if _registered_result_callbacks["virtual_image"] is not None:
        self.signal_virtual_image_data_changed.disconnect(
            _registered_result_callbacks["virtual_image"]
        )

    if _registered_result_callbacks["datacube"] is not None:
        self.signal_datacube_changed.disconnect(
            _registered_result_callbacks["datacube"]
        )

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
