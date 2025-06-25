from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import subprocess
import threading
import uvicorn
import cv2
import numpy as np

from cvDetector import detector

# Configure your camera URL here!
# camera_url = "http://root:tJe9R87pDxG6b62@192.168.110.150/axis-cgi/mjpg/video.cgi" # Axis Camera
camera_url = "rtsp://root:tJe9R87pDxG6b62@192.168.110.150/axis-media/media.amp" # Axis Camera
# camera_url = "rtsp://admin:smB@ston!@192.168.50.175/profile2/media.smp"
# camera_url = "rtsp://192.168.50.175/profile2/media.smp" # Wisenet Camera
# camera_url = "rtsp://service:smB0ston!@10.200.11.33:554/" # Bosch Camera
# camera_url = "rtsp://10.200.11.33/axis-media/media.amp"
# camera_url = 0 # For local built-in camera
# camera_url = "http://localhost:8080/video"

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")
lock = threading.Lock()

# ========== Setting up variables ==========
# frame_count = 0
tagDet = detector()
WIDTH = 3840
HEIGHT = 2160
frame_size = WIDTH * HEIGHT * 3  # 3 channels for RGB
tagDet = detector()
persistent_corners = []
detection_enabled = False
lock = threading.Lock()

# ========== Focus Helper ==========
def run_focusHelper():
    cmd = [
    'ffmpeg',
    '-rtsp_transport', 'tcp',
    '-fflags', 'nobuffer',
    '-flags', 'low_delay',
    '-probesize', '32',
    '-analyzeduration', '0',
    '-i', camera_url,
    '-vf', 'fps=8',           # Drop all but 8 frames/sec
    '-an',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-'
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            raw_frame = pipe.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                continue
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()
            # ======== Focus Measurement ========
            # h_crop = int(HEIGHT * 0.01)
            # w_crop = int(WIDTH * 0.01)
            y1 = HEIGHT // 2 - 10 # h_crop // 2
            y2 = HEIGHT // 2 + 10 # h_crop // 2
            x1 = WIDTH // 2 - 10 # w_crop // 2
            x2 = WIDTH // 2 + 10 # w_crop // 2
            center_crop = frame[y1:y2, x1:x2]
            gray_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
            # Threshold to consider it "in focus"
            focus_threshold = 550.0
            focus_color = (0, 255, 0) if laplacian_var > focus_threshold else (0, 0, 255)
            # Draw rectangle for visualizing the crop
            cv2.rectangle(frame, (x1, y1), (x2, y2), focus_color, 2)
            # Put Laplacian variance at top of screen
            text = f"Focus: {laplacian_var:.2f}"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, focus_color, 3)
            frame = cv2.resize(frame, (1920, 1080))
            # Encode frame to JPEG
            ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
    except GeneratorExit:
        print("[INFO] Client disconnected, shutting down FFmpeg")
    finally:
        pipe.terminate()
        pipe.wait()
        print("[INFO] FFmpeg process terminated")

# ========== Calculate Calibration Distance ==========
def compute_z(x, y):
    return (-0.0339 * x**2
            - 2.0089 * y**2
            + 0.6164 * x * y
            - 19.841 * x
            + 10.4239 * y
            + 9.0062)        

# ========== Live Feedback Stream ==========
def run_liveFeedback():
    cmd = [
    'ffmpeg',
    '-rtsp_transport', 'tcp',
    '-fflags', 'nobuffer',
    '-flags', 'low_delay',
    '-probesize', '32',
    '-analyzeduration', '0',
    '-i', camera_url,
    '-vf', 'fps=8',           # Drop all but 8 frames/sec
    '-an',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-'
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_count = 0

    try:
        while True:
            raw_frame = pipe.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                continue

            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()
            frame_count += 1

            if detection_enabled:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = tagDet.detectMarkers(gray)
                if frame_count % 5 == 0 and ids is not None:
                    with lock:
                        persistent_corners.extend([corner[0] for corner in corners])

            if frame_count % 5 == 0:
                for marker_corners in persistent_corners:
                    for x, y in marker_corners:
                        cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), -1)

            frame = cv2.resize(frame, (1920, 1080))

            ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )

    except GeneratorExit:
        print("[INFO] Client disconnected, shutting down FFmpeg")
    finally:
        pipe.terminate()
        pipe.wait()
        print("[INFO] FFmpeg process terminated")

# ========== Routes ==========
@app.get("/video_focus")
def video_focus():
    return StreamingResponse(run_focusHelper(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(run_liveFeedback(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/toggle_detection")
def toggle_detection():
    global detection_enabled
    detection_enabled = not detection_enabled
    return {"status": "detection_enabled" if detection_enabled else "detection_disabled"}

@app.post("/clear_corners")
def clear_corners():
    global persistent_corners
    with lock:
        persistent_corners.clear()
    return {"status": "corners_cleared"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/function1", response_class=HTMLResponse)
def function1_screen(request: Request):
    return templates.TemplateResponse("function1.html", {"request": request})

@app.get("/function2", response_class=HTMLResponse)
def function2_screen(request: Request):
    return templates.TemplateResponse("function2.html", {"request": request})

@app.post("/function2", response_class=HTMLResponse)
async def function2_result(request: Request, focal_length: float = Form(...)):
    y_values = np.linspace(2.5, 9.0, 14)
    z_values = compute_z(focal_length, y_values)

    max_index = np.argmax(z_values)
    best_y = y_values[max_index]
    max_z = z_values[max_index]

    return templates.TemplateResponse("function2.html", {
        "request": request,
        "result": {
            "focal_length": focal_length,
            "best_y": round(best_y, 2),
            "max_z": round(max_z, 4)
        }
    })

@app.get("/function3", response_class=HTMLResponse)
def function3_screen(request: Request):
    return templates.TemplateResponse("function3.html", {"request": request})

# ========== Entry Point ==========
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6800)