"""
vision/face_detection.py
========================
MediaPipe-based face detection (CPU-optimized, ONNX-based internally).
Target: < 100ms per frame on Raspberry Pi 5.
"""

import cv2
import mediapipe as mp
import numpy as np
import yaml
from pathlib import Path
from typing import Optional
from utils.logger import get_logger

logger = get_logger("vision.face_detection")


def _load_cfg() -> dict:
    path = Path("config/settings.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("face_detection", {})


class FaceDetector:
    """
    Wraps MediaPipe FaceDetection for lightweight, CPU-optimized face detection.

    Args:
        model_selection:         0 = short range (<2m, best for mirror)
                                 1 = full range (<5m)
        min_detection_confidence: Minimum confidence threshold (0.0–1.0)
    """

    def __init__(
        self,
        model_selection: Optional[int] = None,
        min_detection_confidence: Optional[float] = None,
    ) -> None:
        cfg = _load_cfg()
        self._model_selection = model_selection if model_selection is not None else cfg.get("model_selection", 0)
        self._min_conf = min_detection_confidence if min_detection_confidence is not None else cfg.get("min_detection_confidence", 0.6)

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=self._model_selection,
            min_detection_confidence=self._min_conf,
        )
        self._draw_utils = mp.solutions.drawing_utils
        logger.info(f"FaceDetector ready (model={self._model_selection}, conf={self._min_conf})")

    def detect(self, frame: np.ndarray, draw: bool = False) -> list[dict]:
        """
        Detect faces in a BGR frame.

        Args:
            frame: BGR numpy array from camera
            draw:  If True, draw landmarks/boxes on frame in-place

        Returns:
            List of dicts:
              {
                "bbox":       (x, y, w, h) in pixels,
                "confidence": float,
                "detection":  raw MediaPipe detection object
              }
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._detector.process(rgb)

        faces = []
        if results.detections:
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x = max(0, int(bb.xmin * w))
                y = max(0, int(bb.ymin * h))
                bw = int(bb.width * w)
                bh = int(bb.height * h)
                conf = float(det.score[0]) if det.score else 0.0

                faces.append({
                    "bbox": (x, y, bw, bh),
                    "confidence": conf,
                    "detection": det,
                })

                if draw:
                    self._draw_utils.draw_detection(frame, det)
                    cv2.putText(
                        frame, f"{conf:.0%}",
                        (x, max(y - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2,
                    )

        return faces

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()
        logger.info("FaceDetector closed")

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()
