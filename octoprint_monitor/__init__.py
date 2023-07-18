from __future__ import absolute_import
import cv2
import numpy as np
import octoprint.plugin
from .real_time_monitoring import ImageProcessing, PerspectiveProjection, GcodeParser


image_file_path = r"C:\Users\georg\OneDrive - RMIT University\Master by Research\Assets\Images\Geometric Distortion Detection\Frustum\frustum_5.jpg"


class MonitorPlugin(
    octoprint.plugin.StartupPlugin, octoprint.plugin.EventHandlerPlugin
):
    def __init__(self):
        self.layer_height = 0.2
        # self.capture_device = cv2.VideoCapture("http://192.168.0.23:8080/video")
        super().__init__()

    def on_after_startup(self):
        self._logger.info("Temperature Monitor Plugin Started")
        return super().on_after_startup()

    def on_event(self, event, payload):
        if event == "PrintStarted":
            file_path = self._file_manager.path_on_disk(
                payload["origin"], payload["path"]
            )
            self.gp = GcodeParser(file_path)
            self.model_ar = PerspectiveProjection(image_file_path, self.gp.coordinates)
        if (
            event == "ZChange"
            and self._printer.is_printing()
            and payload["old"] is not None
        ):
            if payload["old"] == self.layer_height:
                self.current_layer = self.gp.coordinates_dic[payload["old"]]
                self.capture_image(self.current_layer)

            elif round(payload["new"] - payload["old"], 1) == self.layer_height:
                new_layer = self.gp.coordinates_dic[payload["new"]]
                self.current_layer = np.concatenate((self.current_layer, new_layer))
                self.capture_image(self.current_layer)

        if event == "PrintDone":
            self._logger.info("Print Done")
            cv2.destroyAllWindows()

    def capture_image(self, layer):
        self._printer.pause_print()
        self._printer.commands("G1 X0 Y0 Z200 F3000")

        pp = PerspectiveProjection(image_file_path, layer)
        # ret, frame = self.capture_device.read()
        ip = ImageProcessing(pp.image, pp.mask, self.model_ar.image_points)
        cv2.imshow("frame ", ip.image_difference)
        cv2.waitKey(10)
        # save the image
        self._printer.resume_print()


__plugin_name__ = "Monitor Plugin"
__plugin_pythoncompat__ = ">=3,<4"  # Only Python 3


def __plugin_load__():
    global __plugin_implementation__
    __plugin_implementation__ = MonitorPlugin()

    global __plugin_hooks__
    __plugin_hooks__ = {}
