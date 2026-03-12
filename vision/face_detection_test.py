"""
face_detection_test.py
======================
Standalone test for face detection using MediaPipe.
Development only — runs on laptop webcam.
RPi deployment: same code, same library (MediaPipe is ARM64 compatible).

Test: conda activate mm → python vision/face_detection_test.py
"""

import cv2
import mediapipe as mp
import time
import sys

# ── CONFIG ──────────────────────────────────────────────
CAMERA_INDEX      = 0       # 0 = default webcam
TARGET_FPS        = 30
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
MIN_DETECTION_CONF = 0.6    # 0.0–1.0 (higher = stricter)
MIN_TRACKING_CONF  = 0.5
SHOW_FPS          = True
SHOW_LANDMARKS    = True
# ────────────────────────────────────────────────────────


def init_detector() -> mp.solutions.face_detection.FaceDetection:
    """Initialize MediaPipe face detector (CPU-optimized, ONNX-based internally)."""
    return mp.solutions.face_detection.FaceDetection(
        model_selection=0,              # 0 = short range (< 2m) — best for mirror use
        min_detection_confidence=MIN_DETECTION_CONF
    )


def draw_detections(frame, detections, draw_utils, elapsed_ms: float) -> None:
    """Draw bounding boxes, keypoints, and stats on frame."""
    h, w = frame.shape[:2]

    if detections:
        for detection in detections:
            # Draw MediaPipe default landmarks
            if SHOW_LANDMARKS:
                draw_utils.draw_detection(frame, detection)

            # Draw confidence score
            score = detection.score[0]
            bbox  = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)

            cv2.putText(
                frame,
                f"{score:.0%}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

    # FPS + latency overlay
    if SHOW_FPS:
        fps = 1000 / elapsed_ms if elapsed_ms > 0 else 0
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}  |  Detection: {elapsed_ms:.1f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 200, 255), 2
        )

    # Face count
    count = len(detections) if detections else 0
    color = (0, 255, 0) if count == 1 else (0, 165, 255) if count > 1 else (0, 0, 255)
    cv2.putText(
        frame,
        f"Faces: {count}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, color, 2
    )


def run_test() -> None:
    """Main test loop — press Q to quit."""
    print("[INFO] Starting face detection test...")
    print("[INFO] Press Q to quit\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    detector   = init_detector()
    draw_utils = mp.solutions.drawing_utils

    frame_count    = 0
    total_latency  = 0.0

    with detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame capture failed")
                break

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False         # minor perf boost

            t0      = time.perf_counter()
            results = detector.process(rgb)
            elapsed = (time.perf_counter() - t0) * 1000   # ms

            rgb.flags.writeable = True

            draw_detections(frame, results.detections, draw_utils, elapsed)

            cv2.imshow("Magic Mirror — Face Detection Test", frame)

            frame_count   += 1
            total_latency += elapsed

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    if frame_count > 0:
        avg = total_latency / frame_count
        print(f"\n[RESULT] Frames processed : {frame_count}")
        print(f"[RESULT] Avg detection time: {avg:.2f}ms")
        print(f"[RESULT] Target (<100ms)   : {'✅ PASS' if avg < 100 else '❌ FAIL'}")


if __name__ == "__main__":
    run_test()