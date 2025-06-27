import ffmpeg
import subprocess
import numpy as np
import cv2

'''This script captures video frames from a camera using ffmpeg and displays them using OpenCV.
It reads raw video frames in YUYV format, converts them to BGR format, and displays them in a window.
Press 'q' to exit the display window.'
'''

if __name__ == "__main__":
    device = "/dev/video0"
    width, height = 320, 240
    ffmpeg_command = (
        ffmpeg.input(device, format="v4l2", framerate=60, video_size=f"{width}x{height}", pix_fmt="yuyv422")
        .output("pipe:", format="rawvideo", pix_fmt="bgr24")
        .compile()
    )
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    try:
        while True:
            raw_frame = process.stdout.read(width * height * 3)
            if len(raw_frame) != width * height * 3:
                print("Incomplete frame read, skipping...")
                break
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            cv2.imshow("Test Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        process.terminate()
        process.wait()
        cv2.destroyAllWindows()