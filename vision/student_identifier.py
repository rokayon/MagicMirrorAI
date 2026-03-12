"""
vision/student_identifier.py
=============================
Orchestrates the full vision pipeline:
  Camera -> FaceDetector -> FaceAligner -> FaceRecognizer -> Student match

Usage:
    sid = StudentIdentifier()
    result = sid.identify(frame)
    print(result.name, result.similarity)
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional

from vision.face_detection import FaceDetector
from vision.face_alignment import FaceAligner
from vision.face_recognition import FaceRecognizer
from database.embeddings_db import EmbeddingsDB
from utils.logger import get_logger

logger = get_logger("vision.student_identifier")


@dataclass
class IdentityResult:
    """Final output of the full vision pipeline for one frame."""
    name: str               # student name or "Unknown"
    similarity: float       # match confidence (0.0-1.0)
    face_found: bool        # was a face detected at all?
    det_latency_ms: float   # FaceDetector time
    rec_latency_ms: float   # FaceRecognizer time

    @property
    def total_latency_ms(self) -> float:
        """Combined detection + recognition latency."""
        return self.det_latency_ms + self.rec_latency_ms

    @property
    def is_known(self) -> bool:
        """True if a registered student was matched."""
        return self.name != "Unknown"


class StudentIdentifier:
    """
    Full vision pipeline: detect -> align -> recognize -> identify student.

    Args:
        db: EmbeddingsDB instance (loads from settings.yaml path by default)
    """

    def __init__(self, db: Optional[EmbeddingsDB] = None) -> None:
        self._detector   = FaceDetector()
        self._aligner    = FaceAligner()
        self._recognizer = FaceRecognizer()
        self._db         = db or EmbeddingsDB()
        logger.info(f"StudentIdentifier ready -- {len(self._db)} students registered: {self._db.names()}")

    def identify(self, frame: np.ndarray) -> IdentityResult:
        """
        Run the full pipeline on one BGR frame.

        Args:
            frame: BGR numpy array from Camera

        Returns:
            IdentityResult with student name, confidence, and latency stats
        """
        t0 = time.perf_counter()
        faces = self._detector.detect(frame)
        det_ms = (time.perf_counter() - t0) * 1000

        if not faces:
            return IdentityResult(
                name="Unknown", similarity=0.0, face_found=False,
                det_latency_ms=det_ms, rec_latency_ms=0.0,
            )

        best_face = max(faces, key=lambda f: f["confidence"])
        bbox = best_face["bbox"]

        face_crop = self._aligner.crop_only(frame, bbox)
        if face_crop is None:
            return IdentityResult(
                name="Unknown", similarity=0.0, face_found=True,
                det_latency_ms=det_ms, rec_latency_ms=0.0,
            )

        t1 = time.perf_counter()
        embedding = self._recognizer.extract_embedding(face_crop)
        rec_ms = (time.perf_counter() - t1) * 1000

        if embedding is None:
            return IdentityResult(
                name="Unknown", similarity=0.0, face_found=True,
                det_latency_ms=det_ms, rec_latency_ms=rec_ms,
            )

        name, score = self._recognizer.match(embedding, self._db.all())
        logger.debug(f"Identified: {name} ({score:.2f}) -- det={det_ms:.1f}ms rec={rec_ms:.1f}ms")

        return IdentityResult(
            name=name, similarity=score, face_found=True,
            det_latency_ms=det_ms, rec_latency_ms=rec_ms,
        )

    def register_student(self, name: str, cap, num_frames: int = 30) -> bool:
        """
        Register a new student by capturing multiple frames from camera.

        Args:
            name:       Student name
            cap:        Camera instance (.get_frame() method)
            num_frames: Number of frames to collect

        Returns:
            True if registration succeeded
        """
        logger.info(f"Registering: '{name}' -- collecting {num_frames} frames...")
        collected: list[np.ndarray] = []

        while len(collected) < num_frames:
            frame = cap.get_frame()
            faces = self._detector.detect(frame)
            if faces:
                bbox = max(faces, key=lambda f: f["confidence"])["bbox"]
                crop = self._aligner.crop_only(frame, bbox)
                if crop is not None:
                    emb = self._recognizer.extract_embedding(crop)
                    if emb is not None:
                        collected.append(emb)

            progress = len(collected) / num_frames
            bar_w = int(progress * 400)
            cv2.rectangle(frame, (120, 220), (520, 260), (40, 40, 40), -1)
            cv2.rectangle(frame, (120, 220), (120 + bar_w, 260), (0, 255, 100), -1)
            cv2.putText(
                frame, f"Registering: {name} ({len(collected)}/{num_frames})",
                (100, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2,
            )
            cv2.imshow("Magic Mirror -- Register", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyWindow("Magic Mirror -- Register")

        if len(collected) < 10:
            logger.warning(f"Registration failed for '{name}' -- only {len(collected)} valid frames")
            return False

        avg_emb = self._recognizer.build_average_embedding(collected)
        self._db.add(name, avg_emb)
        self._db.save()
        logger.info(f"Student '{name}' registered successfully ({len(collected)} frames)")
        return True

    def list_students(self) -> list[str]:
        """Return all registered student names."""
        return self._db.names()

    def remove_student(self, name: str) -> bool:
        """Remove a registered student and persist change."""
        removed = self._db.remove(name)
        if removed:
            self._db.save()
        return removed

    def close(self) -> None:
        """Release all resources."""
        self._detector.close()
        self._recognizer.close()
        logger.info("StudentIdentifier closed")

    def __enter__(self) -> "StudentIdentifier":
        return self

    def __exit__(self, *_) -> None:
        self.close()
