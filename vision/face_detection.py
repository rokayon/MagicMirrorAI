"""
face_detection.py
=================
Production face detector using MediaPipe.
CPU-optimized, ARM64 compatible — same code runs on laptop and RPi 5.

Usage:
    from vision.face_detection import FaceDetector, DetectionResult
"""

import cv2
import mediapipe as mp
import time
import logging
import yaml
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Structured output from one frame's detection pass."""
    detections: list          # raw MediaPipe detection objects
    latency_ms: float         # inference time in ms
    face_count: int           # number of faces found


class FaceDetector:
    """
    MediaPipe-based face detector.
    model_selection=0 → short range (<2m) — perfect for mirror.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        det_cfg = cfg["face_detection"]
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=det_cfg["model_selection"],
            min_detection_confidence=det_cfg["min_detection_confidence"]
        )
        self._draw_utils     = mp.solutions.drawing_utils
        self._show_landmarks = det_cfg["show_landmarks"]
        self._show_fps       = det_cfg["show_fps"]
        self._target_ms      = det_cfg["latency_target_ms"]
        logger.info("FaceDetector ready (MediaPipe)")

    # ── CORE ──────────────────────────────────────────────

    def detect(self, frame) -> DetectionResult:
        """
        Run detection on a single BGR frame.
        Returns DetectionResult (detections, latency_ms, face_count).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        t0 = time.perf_counter()
        results = self._detector.process(rgb)
        elapsed = (time.perf_counter() - t0) * 1000

        detections = results.detections or []

        if elapsed > self._target_ms:
            logger.warning(f"Detection {elapsed:.1f}ms > target {self._target_ms}ms")

        return DetectionResult(
            detections=detections,
            latency_ms=elapsed,
            face_count=len(detections)
        )

    def has_face(self, frame) -> bool:
        """Quick check — is there at least one face in frame?"""
        return self.detect(frame).face_count > 0

    # ── DRAW (dev/debug only) ──────────────────────────────

    def annotate(self, frame, result: DetectionResult) -> None:
        """
        Draw boxes, landmarks, confidence, FPS on frame (in-place).
        Call only during development — skip on RPi if CPU is tight.
        """
        h, w = frame.shape[:2]

        for det in result.detections:
            if self._show_landmarks:
                self._draw_utils.draw_detection(frame, det)

            score = det.score[0]
            bbox  = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            cv2.putText(frame, f"{score:.0%}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self._show_fps:
            fps = 1000 / result.latency_ms if result.latency_ms > 0 else 0
            cv2.putText(frame,
                        f"FPS: {fps:.1f}  |  Detection: {result.latency_ms:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        color = (0,255,0) if result.face_count==1 else \
                (0,165,255) if result.face_count>1 else (0,0,255)
        cv2.putText(frame, f"Faces: {result.face_count}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ── LIFECYCLE ─────────────────────────────────────────

    def close(self) -> None:
        self._detector.close()
        logger.info("FaceDetector closed")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()