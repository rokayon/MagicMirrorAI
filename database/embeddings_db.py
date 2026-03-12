"""
database/embeddings_db.py
=========================
Persistent storage for face embeddings.
Stores: {student_name: L2-normalized embedding (np.ndarray)}
Backend: pickle file (lightweight, offline, RPi-friendly).
"""

import pickle
import numpy as np
from pathlib import Path
import yaml
from utils.logger import get_logger

logger = get_logger("database.embeddings_db")


def _default_path() -> Path:
    cfg_path = Path("config/settings.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Path(cfg.get("student", {}).get("embeddings_path", "database/embeddings.pkl"))


class EmbeddingsDB:
    """
    Simple key-value store for face embeddings.

    Usage:
        db = EmbeddingsDB()
        db.add("Rahim", embedding_array)
        db.save()
        name, score = recognizer.match(query_emb, db.all())
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._path = Path(db_path) if db_path else _default_path()
        self._data: dict[str, np.ndarray] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load embeddings from disk if file exists."""
        if self._path.exists():
            with open(self._path, "rb") as f:
                self._data = pickle.load(f)
            logger.info(f"Loaded {len(self._data)} embeddings from {self._path}")
        else:
            logger.info(f"No embeddings file found at {self._path} — starting fresh")

    def save(self) -> None:
        """Persist all embeddings to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(self._data, f)
        logger.info(f"Saved {len(self._data)} embeddings → {self._path}")

    # ── CRUD ─────────────────────────────────────────────────────────────

    def add(self, name: str, embedding: np.ndarray) -> None:
        """
        Register or update a face embedding.

        Args:
            name:      Student name / ID
            embedding: L2-normalized 1D numpy array
        """
        self._data[name] = embedding
        logger.info(f"Registered face: '{name}'")

    def remove(self, name: str) -> bool:
        """
        Remove a registered face.

        Returns:
            True if removed, False if not found
        """
        if name in self._data:
            del self._data[name]
            logger.info(f"Removed face: '{name}'")
            return True
        return False

    def all(self) -> dict[str, np.ndarray]:
        """Return all embeddings as {name: embedding} dict."""
        return dict(self._data)

    def names(self) -> list[str]:
        """Return list of registered student names."""
        return list(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, name: str) -> bool:
        return name in self._data
