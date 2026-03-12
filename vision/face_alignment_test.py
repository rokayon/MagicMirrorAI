"""
face_alignment_test.py
======================
Standalone test for face alignment.
Shows: original frame + aligned face thumbnail + rotation angle.

Test: conda activate mm → python vision/face_alignment_test.py
"""

import cv2
import sys
from vision.face_alignment import FaceAligner

CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

def run_test() -> None:
    print("[INFO] Face Alignment Test — press Q to quit")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    success_count = 0
    total_count   = 0

    with FaceAligner() as aligner:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = aligner.align(frame)
            aligner.annotate(frame, result)

            total_count += 1
            if result.success:
                success_count += 1

            cv2.imshow("Magic Mirror — Face Alignment Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if total_count > 0:
        rate = success_count / total_count * 100
        print(f"\n[RESULT] Alignment success rate: {rate:.1f}%")
        print(f"[RESULT] Target (>90%)          : {'✅ PASS' if rate > 90 else '❌ FAIL'}")

if __name__ == "__main__":
    run_test()