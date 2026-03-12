"""
vision/face_alignment.py
========================
Crops and aligns a detected face from a frame.
Output: 112×112 BGR image (standard ArcFace input size).
Uses eye landmarks from MediaPipe FaceMesh for rotation correction.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("vision.face_alignment")


def _load_cfg() -> dict:
    path = Path("config/settings.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("face_alignment", {})


class FaceAligner:
    """
    Aligns and crops a face region to a fixed output size.

    Pipeline:
      1. Rotate frame so eyes are horizontal
      2. Crop face bounding box (with padding)
      3. Resize to output_size × output_size

    Args:
        output_size: Output image size in pixels (square)
        padding:     Extra crop margin around bounding box (fraction)
    """

    def __init__(
        self,
        output_size: int | None = None,
        padding: float | None = None,
    ) -> None:
        cfg = _load_cfg()
        self.output_size = output_size or cfg.get("output_size", 112)
        self.padding = padding if padding is not None else cfg.get("padding", 0.3)
        self._eye_l_idx = cfg.get("eye_landmark_left", 33)
        self._eye_r_idx = cfg.get("eye_landmark_right", 263)
        logger.info(f"FaceAligner ready (size={self.output_size}, padding={self.padding})")

    def align(
        self,
        frame: np.ndarray,
        face_landmarks,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        """
        Align and crop face from frame using MediaPipe FaceMesh landmarks.

        Args:
            frame:          BGR frame
            face_landmarks: MediaPipe FaceMesh landmarks object
            bbox:           (x, y, w, h) bounding box in pixels

        Returns:
            Aligned BGR face image (output_size × output_size), or None on failure
        """
        h, w = frame.shape[:2]

        try:
            lms = face_landmarks.landmark

            # Eye center coordinates
            eye_l = np.array([lms[self._eye_l_idx].x * w, lms[self._eye_l_idx].y * h])
            eye_r = np.array([lms[self._eye_r_idx].x * w, lms[self._eye_r_idx].y * h])

            # Rotation angle to level eyes
            dy = float(eye_r[1] - eye_l[1])
            dx = float(eye_r[0] - eye_l[0])
            angle = float(np.degrees(np.arctan2(dy, dx)))

            # Rotate around center of eyes
            eye_center = ((eye_l + eye_r) / 2).astype(int)
            M = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0)
            rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)

            # Rotate bounding box center too
            x, y, bw, bh = bbox
            cx, cy = x + bw // 2, y + bh // 2
            rotated_center = M @ np.array([cx, cy, 1.0])
            cx_r, cy_r = int(rotated_center[0]), int(rotated_center[1])

            # Apply padding
            pad_w = int(bw * self.padding)
            pad_h = int(bh * self.padding)
            x1 = max(cx_r - bw // 2 - pad_w, 0)
            y1 = max(cy_r - bh // 2 - pad_h, 0)
            x2 = min(cx_r + bw // 2 + pad_w, w)
            y2 = min(cy_r + bh // 2 + pad_h, h)

            if x2 <= x1 or y2 <= y1:
                return None

            crop = rotated[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            return cv2.resize(crop, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)

        except Exception as e:
            logger.warning(f"Face alignment failed: {e}")
            return None

    def crop_only(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        """
        Fallback: crop face without rotation alignment.

        Args:
            frame: BGR frame
            bbox:  (x, y, w, h)

        Returns:
            Cropped and resized BGR face image, or None
        """
        x, y, bw, bh = bbox
        h, w = frame.shape[:2]

        pad_w = int(bw * self.padding)
        pad_h = int(bh * self.padding)
        x1 = max(x - pad_w, 0)
        y1 = max(y - pad_h, 0)
        x2 = min(x + bw + pad_w, w)
        y2 = min(y + bh + pad_h, h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return cv2.resize(crop, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
