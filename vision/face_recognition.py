"""
face_recognition.py
===================
MediaPipe FaceMesh-based face recognizer.
Extracts normalized landmark embeddings, matches via cosine similarity.
CPU-optimized, ARM64 compatible — RPi 5 ready.

Usage:
    from vision.face_recognition import FaceRecognizer, RecognitionResult
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import logging
import yaml
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Key landmark indices (eyes, nose, mouth, chin, brows)
_KEY_LANDMARKS = [
    33, 133, 362, 263,
    1, 4,
    61, 291,
    199,
    70, 300,
    168, 6, 197, 195,
]


@dataclass
class RecognitionResult:
    """Structured output from one recognition pass."""
    name: str               # matched name or "Unknown"
    similarity: float       # cosine similarity score (0.0–1.0)
    latency_ms: float       # inference time in ms
    embedding: Optional[np.ndarray] = None   # raw embedding (for registration)


class FaceRecognizer:
    """
    Landmark-based face recognizer using MediaPipe FaceMesh.
    Matches faces against a dict of registered embeddings.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        rec_cfg = cfg["face_recognition"]
        stu_cfg = cfg["student"]

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=rec_cfg["max_num_faces"],
            refine_landmarks=rec_cfg["refine_landmarks"],
            min_detection_confidence=rec_cfg["min_detection_confidence"],
            min_tracking_confidence=rec_cfg["min_tracking_confidence"]
        )
        self._threshold    = rec_cfg["similarity_threshold"]
        self._unknown      = stu_cfg["unknown_label"]
        self._key_lms      = rec_cfg.get("key_landmarks", _KEY_LANDMARKS)
        logger.info("FaceRecognizer ready (MediaPipe FaceMesh)")

    # ── EMBEDDING ─────────────────────────────────────────

    def _extract_embedding(self, face_landmarks, frame_shape: tuple) -> Optional[np.ndarray]:
        """
        Build a normalized embedding vector from key facial landmarks.
        Normalized to face width → scale/distance invariant.
        Returns None if face geometry is invalid.
        """
        h, w = frame_shape[:2]
        coords = np.array(
            [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark],
            dtype=np.float32
        )

        key = coords[self._key_lms]

        # Normalize: center on nose tip, scale by eye distance
        nose      = key[4]
        eye_dist  = np.linalg.norm(key[2] - key[0]) + 1e-6
        normalized = (key - nose) / eye_dist

        embedding = normalized.flatten()
        norm = np.linalg.norm(embedding)
        if norm < 1e-6:
            return None
        return embedding / norm

    # ── CORE ──────────────────────────────────────────────

    def recognize(self, frame, embeddings: dict) -> RecognitionResult:
        """
        Run recognition on a single BGR frame.
        Matches against provided embeddings dict {name: np.ndarray}.
        Returns RecognitionResult.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        t0      = time.perf_counter()
        results = self._face_mesh.process(rgb)
        elapsed = (time.perf_counter() - t0) * 1000

        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return RecognitionResult(
                name=self._unknown, similarity=0.0,
                latency_ms=elapsed, embedding=None
            )

        face_lms = results.multi_face_landmarks[0]
        emb = self._extract_embedding(face_lms, frame.shape)

        if emb is None:
            return RecognitionResult(
                name=self._unknown, similarity=0.0,
                latency_ms=elapsed, embedding=None
            )

        name, score = self._match(emb, embeddings)

        return RecognitionResult(
            name=name, similarity=score,
            latency_ms=elapsed, embedding=emb
        )

    def extract_embedding_only(self, frame) -> Optional[np.ndarray]:
        """
        Extract embedding without matching — used during registration.
        Returns embedding array or None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return None
        return self._extract_embedding(
            results.multi_face_landmarks[0], frame.shape
        )

    # ── MATCHING ──────────────────────────────────────────

    def _match(self, embedding: np.ndarray, embeddings: dict) -> Tuple[str, float]:
        """Find best cosine similarity match in embeddings dict."""
        if not embeddings:
            return self._unknown, 0.0

        best_name  = self._unknown
        best_score = 0.0

        for name, stored in embeddings.items():
            score = float(np.dot(embedding, stored))
            if score > best_score:
                best_score = score
                best_name  = name

        if best_score < self._threshold:
            return self._unknown, best_score

        return best_name, best_score

    # ── DRAW (dev/debug only) ──────────────────────────────

    def annotate(self, frame, result: RecognitionResult, face_landmarks=None) -> None:
        """Draw recognition result on frame (in-place)."""
        h, w = frame.shape[:2]

        if face_landmarks:
            xs = [lm.x * w for lm in face_landmarks.landmark]
            ys = [lm.y * h for lm in face_landmarks.landmark]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            color = (0, 255, 0) if result.name != self._unknown else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{result.name} ({result.similarity:.0%})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"Recog: {result.latency_ms:.1f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # ── LIFECYCLE ─────────────────────────────────────────

    def close(self) -> None:
        self._face_mesh.close()
        logger.info("FaceRecognizer closed")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()