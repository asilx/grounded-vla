import numpy as np
from grounded_vla.backends.rtx_camera import RtxCameraBridge


class Sensor:
    def __init__(self, data):
        self.data = data

    def get_data(self, name):
        assert name == "rgb"
        return self.data, {}


class Timer:
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data


def test_warp_rgb_conversion_preserves_pixels_and_capture_timestamp():
    rgb = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)

    class WarpArray:
        def numpy(self):
            return rgb

    camera = RtxCameraBridge(Sensor(WarpArray()), Timer({"simulationTime": 1.25}))
    frame = camera.get_current_frame(clone=True)
    assert frame["rendering_time"] == 1.25
    assert np.array_equal(frame["rgba"][..., :3], rgb)
    assert (frame["rgba"][..., 3] == 255).all()
    rgb[:] = 0
    assert frame["rgba"][0, 0, 1] == 1


def test_missing_timestamp_or_rgb_never_substitutes_a_frame():
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    assert RtxCameraBridge(Sensor(rgb), Timer({})).get_current_frame() == {}
    assert RtxCameraBridge(Sensor(None), Timer({"simulationTime": 2})).get_current_frame() == {}
    assert RtxCameraBridge(Sensor(rgb.astype(float)), Timer({"simulationTime": 2})).get_current_frame() == {}
