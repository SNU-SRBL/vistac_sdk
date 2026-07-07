"""Quick test of DIGIT camera using cv2.VideoCapture (matches vistac_device.Camera).

Press 'q' to exit.
"""
import cv2
import numpy as np


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    # Warmup
    for _ in range(10):
        cap.read()

    print("Camera ready. Press 'q' to quit.")
    reads = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"cap.read() returned False at frame {reads}")
                break
            reads += 1
            frame = frame.copy()

            # Row-continuity validation (same as vistac_device)
            if reads > 1:
                rd = np.sum(np.abs(
                    frame[:-1].astype(np.int16) - frame[1:].astype(np.int16)
                ), axis=(1, 2))
                med = float(np.median(rd))
                mx = float(np.max(rd))
                if med > 0 and mx / med > 3.0:
                    print(f"RAW-SHIFT at frame {reads}: ratio={mx/med:.1f}")
                    continue

            cv2.putText(frame, f"frame={reads}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imshow("DIGIT Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Test done. {reads} frames captured.")
