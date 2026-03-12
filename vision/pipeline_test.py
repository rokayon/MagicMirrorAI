"""
vision/pipeline_test.py
=======================
Live test of the full vision pipeline with on-screen display.
Shows: camera feed + face detection box + student name + latency.

Run: conda activate mm → python vision/pipeline_test.py
Controls:
  R = Register new face (type name in terminal)
  Q = Quit
"""

import sys
import cv2

sys.path.insert(0, ".")

from vision.camera import Camera
from vision.student_identifier import StudentIdentifier


def run() -> None:
    print("=" * 50)
    print("  Magic Mirror — Full Pipeline Live Test")
    print("=" * 50)
    print("  R = Register new face")
    print("  Q = Quit")
    print("=" * 50)

    cam = Camera()
    sid = StudentIdentifier()

    while True:
        frame = cam.get_frame()
        result = sid.identify(frame)

        # ── Draw result on frame ──────────────────────────
        if result.face_found:
            # Re-run detection just to get bbox for drawing
            faces = sid._detector.detect(frame, draw=True)
            if faces:
                x, y, w, h = max(faces, key=lambda f: f["confidence"])["bbox"]
                color = (0, 255, 0) if result.is_known else (0, 0, 255)
                label = f"{result.name} ({result.similarity:.0%})" if result.is_known else "Unknown"
                cv2.putText(frame, label, (x, max(y - 12, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ── Stats overlay ─────────────────────────────────
        face_status = "Face: YES" if result.face_found else "Face: NO"
        cv2.putText(frame, face_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"Pipeline: {result.total_latency_ms:.1f}ms",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"Students: {sid.list_students()}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, "R=Register  Q=Quit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Magic Mirror — Pipeline Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            name = input("\n[INPUT] Enter student name to register: ").strip()
            if name:
                sid.register_student(name, cam)

    cam.release()
    sid.close()
    cv2.destroyAllWindows()
    print("\n[DONE] Pipeline test complete.")
    print(f"[INFO] Registered students: {sid.list_students()}")


if __name__ == "__main__":
    run()
