"""
vision/face_recognition.py
==========================
Face recognition using MediaPipe FaceMesh landmark embeddings.
Called ONLY when FaceDetector confirms a face is present (performance optimization).

Pipeline:
  FaceDetected → FaceMesh → extract_embedding → cosine_similarity → match
"""

import numpy as np
import mediapipe as mp
import yaml
from pathlib import Path
from typing import Optional
from utils.logger import get_logger

logger = get_logger("vision.face_recognition")


def _load_cfg() -> dict:
    path = Path("config/settings.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("face_recognition", {})


class FaceRecognizer:
    """
    Extracts normalized landmark embeddings from a face using MediaPipe FaceMesh,
    then matches them against registered embeddings using cosine similarity.

    Args:
        similarity_threshold: Minimum score to consider a match (0.0–1.0)
    """

    def __init__(self, similarity_threshold: Optional[float] = None) -> None:
        cfg = _load_cfg()
        self._threshold = similarity_threshold if similarity_threshold is not None else cfg.get("similarity_threshold", 0.75)
        self._key_landmarks: list[int] = cfg.get(
            "key_landmarks",
            [33, 133, 362, 263, 1, 4, 61, 291, 199, 70, 300, 168, 6, 197, 195],
        )

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=cfg.get("max_num_faces", 1),
            refine_landmarks=cfg.get("refine_landmarks", False),
            min_detection_confidence=cfg.get("min_detection_confidence", 0.6),
            min_tracking_confidence=cfg.get("min_tracking_confidence", 0.5),
        )
        logger.info(f"FaceRecognizer ready (threshold={self._threshold}, key_landmarks={len(self._key_landmarks)})")

    # ── Embedding extraction ──────────────────────────────────────────────

    def extract_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run FaceMesh on a BGR frame and extract a normalized embedding vector.

        Args:
            frame: BGR numpy array (should already have a detected face)

        Returns:
            L2-normalized 1D embedding array, or None if no face found
        """
        import cv2
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        return self._landmarks_to_embedding(results.multi_face_landmarks[0], h, w)

    def _landmarks_to_embedding(self, face_landmarks, h: int, w: int) -> Optional[np.ndarray]:
        """Convert FaceMesh landmarks to a normalized embedding vector."""
        coords = np.array(
            [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark],
            dtype=np.float32,
        )
        key_coords = coords[self._key_landmarks]

        # Normalize: center on nose tip (index 4 in key_landmarks), scale by inter-eye distance
        nose = key_coords[4]
        eye_l = key_coords[0]
        eye_r = key_coords[2]
        face_width = float(np.linalg.norm(eye_r - eye_l)) + 1e-6

        normalized = (key_coords - nose) / face_width
        embedding = normalized.flatten()

        norm = float(np.linalg.norm(embedding))
        if norm < 1e-6:
            return None
        return embedding / norm

    # ── Matching ─────────────────────────────────────────────────────────

    def match(
        self,
        embedding: np.ndarray,
        embeddings_db: dict[str, np.ndarray],
    ) -> tuple[str, float]:
        """
        Find the best matching name for a given embedding.

        Args:
            embedding:     Query embedding (L2-normalized)
            embeddings_db: {name: embedding} dict from EmbeddingsDB

        Returns:
            (name, similarity_score) — name is "Unknown" if below threshold
        """
        if not embeddings_db:
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = 0.0

        for name, stored in embeddings_db.items():
            score = float(np.dot(embedding, stored))  # cosine sim (both L2-normalized)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self._threshold:
            return "Unknown", best_score

        return best_name, best_score

    # ── Registration helper ───────────────────────────────────────────────

    def build_average_embedding(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """
        Average multiple embeddings (collected over several frames) into one.

        Args:
            embeddings: List of L2-normalized embedding arrays

        Returns:
            Single averaged and re-normalized embedding
        """
        avg = np.mean(embeddings, axis=0)
        norm = float(np.linalg.norm(avg))
        if norm < 1e-6:
            return avg
        return avg / norm

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release FaceMesh resources."""
        self._face_mesh.close()
        logger.info("FaceRecognizer closed")

    def __enter__(self) -> "FaceRecognizer":
        return self

    def __exit__(self, *_) -> None:
        self.close()
