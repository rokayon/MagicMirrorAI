"""
face_recognition_test.py
========================
Standalone test for face recognition using MediaPipe FaceMesh.
Two modes:
  REGISTER mode : 'R' চাপো → webcam থেকে face register করো
  RECOGNIZE mode: live video তে registered face match করো

Test: conda activate mm → python vision/face_recognition_test.py
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import pickle
import os
from typing import Optional

# ── CONFIG ──────────────────────────────────────────────
CAMERA_INDEX       = 0
FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480
TARGET_FPS         = 30
SIMILARITY_THRESHOLD = 0.75   # 0.0–1.0 (higher = stricter match)
REGISTER_FRAMES    = 30       # কতগুলো frame average করে embedding বানাবে
EMBEDDINGS_FILE    = "test_embeddings.pkl"  # temp storage for test
# ────────────────────────────────────────────────────────


# ── MEDIAPIPE SETUP ──────────────────────────────────────
mp_face_mesh    = mp.solutions.face_mesh
mp_face_detect  = mp.solutions.face_detection
mp_drawing      = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Key landmark indices for embedding (eyes, nose, mouth corners, chin)
KEY_LANDMARKS = [
    33, 133, 362, 263,   # eye corners
    1, 4,                # nose tip, bridge
    61, 291,             # mouth corners
    199,                 # chin
    70, 300,             # eyebrows
    168, 6, 197, 195,    # nose bridge
]
# ─────────────────────────────────────────────────────────


def extract_embedding(face_landmarks, frame_shape: tuple) -> Optional[np.ndarray]:
    """
    Extract normalized 128D-like embedding from MediaPipe landmarks.
    Uses key facial landmark positions, normalized to face bounding box.
    Returns None if landmarks invalid.
    """
    h, w = frame_shape[:2]

    # Get all landmark coords
    coords = np.array([
        [lm.x * w, lm.y * h]
        for lm in face_landmarks.landmark
    ], dtype=np.float32)

    # Use key landmarks only
    key_coords = coords[KEY_LANDMARKS]

    # Normalize: center around nose tip (landmark 1), scale by face width
    nose    = key_coords[4]  # nose tip
    eye_l   = key_coords[0]  # left eye corner
    eye_r   = key_coords[2]  # right eye corner
    face_width = np.linalg.norm(eye_r - eye_l) + 1e-6

    normalized = (key_coords - nose) / face_width

    # Flatten → embedding vector
    embedding = normalized.flatten()

    # L2 normalize for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm < 1e-6:
        return None
    return embedding / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def load_embeddings(path: str) -> dict:
    """Load registered face embeddings from file."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_embeddings(embeddings: dict, path: str) -> None:
    """Save embeddings to file."""
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"[INFO] Embeddings saved → {path}")


def register_face(name: str, cap, face_mesh, embeddings: dict) -> dict:
    """
    Capture REGISTER_FRAMES frames, extract embeddings, average them.
    Returns updated embeddings dict.
    """
    print(f"\n[REGISTER] Registering: {name}")
    print(f"[REGISTER] Look at camera — collecting {REGISTER_FRAMES} frames...")

    collected = []
    frame_count = 0

    while len(collected) < REGISTER_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        if results.multi_face_landmarks:
            emb = extract_embedding(results.multi_face_landmarks[0], frame.shape)
            if emb is not None:
                collected.append(emb)

        # Progress bar on frame
        progress = len(collected) / REGISTER_FRAMES
        bar_w    = int(progress * 400)
        cv2.rectangle(frame, (120, 220), (520, 260), (50, 50, 50), -1)
        cv2.rectangle(frame, (120, 220), (120 + bar_w, 260), (0, 255, 100), -1)
        cv2.putText(frame, f"Registering: {name} ({len(collected)}/{REGISTER_FRAMES})",
                    (100, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

        cv2.imshow("Magic Mirror — Face Recognition Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    if len(collected) >= 10:
        # Average embedding
        avg_embedding = np.mean(collected, axis=0)
        avg_embedding /= np.linalg.norm(avg_embedding)
        embeddings[name] = avg_embedding
        print(f"[REGISTER] ✅ {name} registered! ({len(collected)} frames used)")
    else:
        print(f"[REGISTER] ❌ Failed — not enough valid frames ({len(collected)})")

    return embeddings


def match_face(embedding: np.ndarray, embeddings: dict) -> tuple[str, float]:
    """
    Find best matching name for given embedding.
    Returns (name, similarity) or ("Unknown", score).
    """
    if not embeddings:
        return "Unknown", 0.0

    best_name  = "Unknown"
    best_score = 0.0

    for name, stored_emb in embeddings.items():
        score = cosine_similarity(embedding, stored_emb)
        if score > best_score:
            best_score = score
            best_name  = name

    if best_score < SIMILARITY_THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score


def draw_recognition(frame, name: str, score: float,
                     face_landmarks, latency_ms: float) -> None:
    """Draw recognition result on frame."""
    h, w = frame.shape[:2]

    # Get face bounding box from landmarks
    xs = [lm.x * w for lm in face_landmarks.landmark]
    ys = [lm.y * h for lm in face_landmarks.landmark]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))

    # Box color: green=known, red=unknown
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Name + score label
    label = f"{name} ({score:.0%})" if name != "Unknown" else "Unknown"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Latency overlay
    cv2.putText(frame, f"Recognition: {latency_ms:.1f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)


def run_test() -> None:
    """
    Main test loop.
    Controls:
      R = register new face (prompts for name in terminal)
      S = save embeddings
      Q = quit
    """
    print("=" * 50)
    print("  Magic Mirror — Face Recognition Test")
    print("=" * 50)
    print("  R = Register new face")
    print("  S = Save embeddings")
    print("  Q = Quit")
    print("=" * 50)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    embeddings = load_embeddings(EMBEDDINGS_FILE)
    print(f"[INFO] Loaded {len(embeddings)} registered faces: {list(embeddings.keys())}")

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,    # False = faster, enough for recognition
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    frame_count   = 0
    total_latency = 0.0

    with face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame capture failed")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            t0      = time.perf_counter()
            results = face_mesh.process(rgb)
            elapsed = (time.perf_counter() - t0) * 1000

            rgb.flags.writeable = True

            if results.multi_face_landmarks:
                face_lms = results.multi_face_landmarks[0]
                emb = extract_embedding(face_lms, frame.shape)

                if emb is not None:
                    name, score = match_face(emb, embeddings)
                    draw_recognition(frame, name, score, face_lms, elapsed)

            else:
                cv2.putText(frame, "No face detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            # Instructions overlay
            cv2.putText(frame, "R=Register  S=Save  Q=Quit",
                        (10, FRAME_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            cv2.imshow("Magic Mirror — Face Recognition Test", frame)

            frame_count   += 1
            total_latency += elapsed

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r'):
                # Register new face
                name = input("\n[INPUT] Enter name to register: ").strip()
                if name:
                    embeddings = register_face(name, cap, face_mesh, embeddings)

            elif key == ord('s'):
                save_embeddings(embeddings, EMBEDDINGS_FILE)

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    if frame_count > 0:
        avg = total_latency / frame_count
        print(f"\n[RESULT] Frames processed  : {frame_count}")
        print(f"[RESULT] Avg recognition   : {avg:.2f}ms")
        print(f"[RESULT] Registered faces  : {list(embeddings.keys())}")
        print(f"[RESULT] Target (<50ms)    : {'✅ PASS' if avg < 50 else '❌ FAIL'}")


if __name__ == "__main__":
    run_test()