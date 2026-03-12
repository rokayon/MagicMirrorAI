"""
vision/camera.py
================
Camera capture module.
Wraps OpenCV VideoCapture with config-driven settings.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("vision.camera")


def _load_cfg() -> dict:
    path = Path("config/settings.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("vision", {})


class Camera:
    """
    Manages webcam/USB/RPi camera capture.

    Args:
        camera_index: Device index (default from settings.yaml)
        width:        Frame width
        height:       Frame height
        fps:          Target FPS
    """

    def __init__(
        self,
        camera_index: int | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> None:
        cfg = _load_cfg()
        self._index = camera_index if camera_index is not None else cfg.get("camera_index", 0)
        self._width = width or cfg.get("frame_width", 640)
        self._height = height or cfg.get("frame_height", 480)
        self._fps = fps or cfg.get("target_fps", 30)

        self.cap = cv2.VideoCapture(self._index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self._index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self.cap.set(cv2.CAP_PROP_FPS, self._fps)
        logger.info(f"Camera opened: index={self._index} {self._width}x{self._height}@{self._fps}fps")

    def get_frame(self) -> np.ndarray:
        """
        Capture and return one BGR frame.

        Returns:
            numpy array (H, W, 3) BGR

        Raises:
            RuntimeError: if frame read fails
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def release(self) -> None:
        """Release the camera resource."""
        self.cap.release()
        logger.info("Camera released")

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_) -> None:
        self.release()
