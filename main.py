from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import subprocess
import threading
import uvicorn
import time
import cv2
import numpy as np

from cvDetector import detector

# Configure your camera URL here!
# camera_url = "http://root:tJe9R87pDxG6b62@192.168.110.150/axis-cgi/mjpg/video.cgi" # Axis Camera
# camera_url = "rtsp://root:tJe9R87pDxG6b62@192.168.110.150/axis-media/media.amp" # Axis Camera
# camera_url = "rtsp://admin:smB@ston!@192.168.50.175/profile2/media.smp"
# camera_url = "rtsp://192.168.50.175/profile2/media.smp" # Wisenet Camera
# camera_url = "rtsp://service:smB0ston!@10.200.11.33:554/" # Bosch Camera
# camera_url = "rtsp://10.200.11.33/axis-media/media.amp"
# camera_url = "http://localhost:8080/video" # Local test stream

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")
lock = threading.Lock()

# ========== Setting up variables ==========
tagDet = detector()
WIDTH = 3840
HEIGHT = 2160
frame_size = WIDTH * HEIGHT * 3  # 3 channels for RGB
detection_enabled = False  # Flag to control tag detection
focus_enabled = False  # Flag to control focus measurement
persistent_corners = []
focus_data = {
    "bbox": None,  # Bounding box for detected markers
    "laplacian": None,  # Focus measure
    "focus_max": 0.0
}
lock = threading.Lock()

# ---- ArUco Board Setup ----
board_def = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
board = cv2.aruco.GridBoard(
    size=(6,6),
    markerLength=0.0873125,      # Adjust to match your board in meters or relative scale
    markerSeparation=0.308,
    dictionary=board_def
)

# Update description
def focus_worker(get_frame_func):
    while True:
        if focus_enabled:
            frame = get_frame_func()
            if frame is None:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = tagDet.detectMarkers(gray)

            with lock:
                if ids is not None and len(corners) > 0:
                    all_pts = np.vstack(corners).reshape(-1, 2).astype(int)
                    x, y, w, h = cv2.boundingRect(all_pts)
                    crop = gray[y:y+h, x:x+w]
                    lap = cv2.Laplacian(crop, cv2.CV_64F).var()

                    # Update focus data
                    focus_data["bbox"] = (x, y, w, h)
                    focus_data["laplacian"] = lap

                    # Update maximum focus if current focus is better
                    if lap > focus_data["focus_max"]:
                        focus_data["focus_max"] = lap
                        focus_data["focus_threshold"] = lap * 0.8333  # New threshold
                else:
                    focus_data["bbox"] = None
                    focus_data["laplacian"] = None
        else:
            time.sleep(0.1)  # Idle when focus is disabled

# ========== Focus Helper ==========
def run_focusHelper():
    cmd = [
    'ffmpeg',
    '-loglevel', 'quiet',
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
    focus_threshold = 4500.0
    last_frame = [None]

    def get_latest_frame():
        with lock:
            return last_frame[0].copy() if last_frame[0] is not None else None

    threading.Thread(target=focus_worker, args=(get_latest_frame,), daemon=True).start()

    try:
        while not end_stream:
            raw = pipe.stdout.read(frame_size)
            if len(raw) != frame_size:
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()

            with lock:
                last_frame[0] = frame.copy()  # Let thread read this

                bbox = focus_data["bbox"]
                lap = focus_data["laplacian"]
                focus_threshold = focus_data["focus_max"]
            
            if focus_enabled:
                if bbox is not None and lap is not None:
                    x, y, w, h = bbox
                    focus_color = (0, 255, 0) if lap > focus_threshold else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), focus_color, 2)

                    # Draw max focus at the top-left
                    cv2.putText(frame, f"Max Focus: {focus_data['focus_max']:.2f}", (70, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

                    # Draw current focus below max focus
                    cv2.putText(frame, f"Current: {lap:.2f}", (70, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, focus_color, 2)
                else:
                    cv2.putText(frame, "Detecting ArUco...", (70, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 2)
            
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
        end_stream = False
        print("[INFO] FFmpeg process terminated")

# ========== Calculate Calibration Distance ==========
def compute_CD(x, y):
    return (-0.0339 * x**2 - 2.0089 * y**2 + 0.6164 * x * y - 19.841 * x + 10.4239 * y + 9.0062)        

# ========== Tag Coordinate Record ==========
def detect_tags(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = tagDet.detectMarkers(gray)
    if ids is not None:
        with lock:
            persistent_corners.extend(c[0] for c in corners)

# ========== Live Feedback Stream ==========
def run_liveFeedback():
    cmd = [
    'ffmpeg',
    '-loglevel', 'quiet',
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
        while not end_stream:
            raw_frame = pipe.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                continue

            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()
            frame_count += 1

            if detection_enabled and frame_count % 4 == 0:
                threading.Thread(target=detect_tags, args=(frame,)).start()

            if frame_count % 1 == 0:
                for marker_corners in persistent_corners:
                    for x, y in marker_corners:
                        cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), -1)

            frame = cv2.resize(frame, (1280, 720))

            ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])

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
        end_stream = False
        print("[INFO] FFmpeg process terminated")

# ========== HTML Routes ==========
@app.get("/video_focus")
def video_focus():
    return StreamingResponse(run_focusHelper(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/record_focus")
def record_focus():
    global start_recording
    start_recording = not start_recording
    return {"status": "recording_started" if start_recording else "recording_stopped"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(run_liveFeedback(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/toggle_focus")
def toggle_focus():
    global focus_enabled
    with lock:
        focus_enabled = not focus_enabled
        # Always reset focus data on toggle (both directions)
        focus_data["bbox"] = None
        focus_data["laplacian"] = None
        focus_data["focus_threshold"] = 0.0
    return {"status": "focus_enabled" if focus_enabled else "focus_disabled"}

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
    return templates.TemplateResponse("home.html", {"request": request, "camera_status": None})

@app.post("/stop_stream")
def stop_stream():
    global end_stream
    end_stream = True
    return Response(status_code=204)

@app.post("/test_url")
def test_url(request: Request, camera_url_input: str = Form(...)):
    global camera_url

    # Try to connect to the RTSP stream
    cap = cv2.VideoCapture(camera_url_input)
    if cap.isOpened():
        camera_url = camera_url_input
        cap.release()
        status_message = "Camera Connected!"
    else:
        status_message = "Camera unreachable. Please check the URL."

    return templates.TemplateResponse("home.html", {"request": request, "camera_status": status_message})

@app.get("/focusHelp", response_class=HTMLResponse)
def focusHelp_screen(request: Request):
    return templates.TemplateResponse("focusHelp.html", {"request": request})

@app.get("/calculateCD", response_class=HTMLResponse)
def calculateCD_screen(request: Request):
    return templates.TemplateResponse("calculateCD.html", {"request": request})

@app.post("/calculateCD", response_class=HTMLResponse)
async def calculateCD_result(request: Request, focal_length: float = Form(...)):
    y_values = np.linspace(2.5, 9.0, 14)
    z_values = compute_CD(focal_length, y_values)

    max_index = np.argmax(z_values)
    best_y = y_values[max_index]

    return templates.TemplateResponse("calculateCD.html", {
        "request": request,
        "result": {
            "focal_length": focal_length,
            "best_y": round(best_y, 2),
        }
    })

@app.get("/liveDetect", response_class=HTMLResponse)
def liveDetect_screen(request: Request):
    return templates.TemplateResponse("liveDetect.html", {"request": request})

# ========== Entry Point ==========
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6800)