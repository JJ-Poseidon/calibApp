import cv2
import subprocess
import numpy as np
import time
from cvDetector import detector

tagDet = detector()
persistent_corners = []
detection_enabled = True
WIDTH = 1920
HEIGHT = 1080

def run_live_feedback(camera_url):
    cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-i', camera_url,
        # '-loglevel', 'quiet',
        '-an',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-'
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    while True:
        raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
        if len(raw_frame) != WIDTH * HEIGHT * 3:
            continue

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))

        if detection_enabled:
            # Detect markers
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = tagDet.detectMarkers(gray)

            # If tags detected, update persistent markers
            if ids is not None:
                persistent_corners.extend([corner[0] for corner in corners])

        frame = cv2.resize(frame, (1920, 1080))

        # Reduce JPEG quality for faster encoding
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpg\r\n\r\n' + frame_bytes + b'\r\n')