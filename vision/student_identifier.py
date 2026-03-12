"""
student_identifier.py
=====================
Combines FaceDetector + FaceRecognizer into one pipeline.
Also handles embedding registration and persistence.

Usage:
    from vision.student_identifier import StudentIdentifier
"""

import cv2
import numpy as np
import pickle
import logging
import yaml
import os
from typing import Optional
from dataclasses import dataclass

from vision.face_detection import FaceDetector, DetectionResult
from vision.face_recognition import FaceRecognizer, RecognitionResult

logger = logging.getLogger(__name__)


@dataclass
class IdentityResult:
    """Final output of the full vision pipeline for one frame."""
    name: str               # student name or "Unknown"
    similarity: float       # match confidence
    face_found: bool        # was a face detected at all?
    det_latency_ms: float   # detection time
    rec_latency_ms: float   # recognition time

    @property
    def total_latency_ms(self) -> float:
        return self.det_latency_ms + self.rec_latency_ms

    @property
    def is_known(self) -> bool:
        return self.name != "Unknown"


class StudentIdentifier:
    """
    Full vision pipeline: detect → recognize → identify student.
    Manages embedding storage (load/save/register).
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self._embeddings_path  = cfg["student"]["embeddings_path"]
        self._register_frames  = cfg["face_recognition"]["register_frames"]

        self._detector   = FaceDetector(config_path)
        self._recognizer = FaceRecognizer(config_path)
        self._embeddings = self._load_embeddings()

        logger.info(f"StudentIdentifier ready — {len(self._embeddings)} students registered")

    # ── CORE PIPELINE ─────────────────────────────────────

    def identify(self, frame) -> IdentityResult:
        """
        Run full pipeline on one BGR frame.
        Returns IdentityResult with student name + latencies.
        """
        det_result = self._detector.detect(frame)

        if det_result.face_count == 0:
            return IdentityResult(
                name="Unknown", similarity=0.0, face_found=False,
                det_latency_ms=det_result.latency_ms, rec_latency_ms=0.0
            )

        rec_result = self._recognizer.recognize(frame, self._embeddings)

        return IdentityResult(
            name=rec_result.name,
            similarity=rec_result.similarity,
            face_found=True,
            det_latency_ms=det_result.latency_ms,
            rec_latency_ms=rec_result.latency_ms
        )

    # ── REGISTRATION ──────────────────────────────────────

    def register_student(self, name: str, cap) -> bool:
        """
        Register a new student by capturing N frames from camera.
        Averages embeddings for robustness.
        Returns True if registration succeeded.
        """
        logger.info(f"Registering student: {name}")
        collected = []

        while len(collected) < self._register_frames:
            ret, frame = cap.read()
            if not ret:
                break

            emb = self._recognizer.extract_embedding_only(frame)
            if emb is not None:
                collected.append(emb)

            # Progress indicator
            progress = len(collected) / self._register_frames
            bar_w = int(progress * 400)
            cv2.rectangle(frame, (120, 220), (520, 260), (40, 40, 40), -1)
            cv2.rectangle(frame, (120, 220), (120 + bar_w, 260), (0, 255, 100), -1)
            cv2.putText(frame, f"Registering: {name} ({len(collected)}/{self._register_frames})",
                        (100, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            cv2.imshow("Registering...", frame)
            cv2.waitKey(1)

        cv2.destroyWindow("Registering...")

        if len(collected) < 10:
            logger.warning(f"Registration failed for {name} — only {len(collected)} valid frames")
            return False

        avg = np.mean(collected, axis=0)
        avg /= np.linalg.norm(avg)
        self._embeddings[name] = avg
        self.save_embeddings()
        logger.info(f"✅ {name} registered successfully ({len(collected)} frames)")
        return True

    # ── EMBEDDINGS STORAGE ────────────────────────────────

    def _load_embeddings(self) -> dict:
        if os.path.exists(self._embeddings_path):
            with open(self._embeddings_path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded {len(data)} embeddings from {self._embeddings_path}")
            return data
        return {}

    def save_embeddings(self) -> None:
        os.makedirs(os.path.dirname(self._embeddings_path), exist_ok=True)
        with open(self._embeddings_path, "wb") as f:
            pickle.dump(self._embeddings, f)
        logger.info(f"Embeddings saved → {self._embeddings_path}")

    def list_students(self) -> list:
        return list(self._embeddings.keys())

    def remove_student(self, name: str) -> bool:
        if name in self._embeddings:
            del self._embeddings[name]
            self.save_embeddings()
            return True
        return False

    # ── LIFECYCLE ─────────────────────────────────────────

    def close(self) -> None:
        self._detector.close()
        self._recognizer.close()
        logger.info("StudentIdentifier closed")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()