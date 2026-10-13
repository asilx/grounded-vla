"""Isaac 6 RTX camera bridge using render-product simulation timestamps."""

import numpy as np


class RtxCameraBridge:
    def __init__(self, sensor, timer):
        self.sensor, self.timer = sensor, timer

    @classmethod
    def attach(cls, path, *, tick_rate):
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        import omni.replicator.core as rep

        sensor = CameraSensor(
            RtxCamera(path, tick_rate=tick_rate),
            resolution=(224, 224),
            annotators=["rgb"],
        )
        timer = rep.AnnotatorRegistry.get_annotator("IsaacReadSimulationTime")
        timer.attach([str(sensor.render_product.GetPath())])
        return cls(sensor, timer)

    def initialize(self):
        # CameraSensor attaches its render product during construction.
        pass

    def get_current_frame(self, *, clone=True):
        data, _ = self.sensor.get_data("rgb")
        timing = self.timer.get_data()
        if data is None or not isinstance(timing, dict) or "simulationTime" not in timing:
            return {}
        rgb = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
        if rgb.dtype != np.uint8 or rgb.shape != (224, 224, 3):
            return {}
        # The existing environment validates age and duplicate timestamps.
        rgba = np.concatenate(
            (rgb, np.full((224, 224, 1), 255, dtype=np.uint8)), axis=-1
        )
        return {"rgba": rgba, "rendering_time": timing["simulationTime"]}
