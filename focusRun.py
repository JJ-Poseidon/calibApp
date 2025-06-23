from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uvicorn
import cv2
import numpy as np

from cvDetector import detector

# camera_url = "rtsp://192.168.50.175/profile2/media.smp" # Wisenet Camera
camera_url = "rtsp://root:tJe9R87pDxG6b62@192.168.110.150/axis-media/media.amp" # Axis Camera

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

tagDet = detector()
WIDTH = 3840
HEIGHT = 2160

def run_focusHelper(camera_url):
    cap = cv2.VideoCapture(camera_url)

    # Create a window and trackbar
    window_name = "AprilGrid Focus Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Dial (Blur)", window_name, 0, 50, lambda x: None)

# ======== Focus Measurement ========
    # h_crop = int(HEIGHT * 0.01)
    # w_crop = int(WIDTH * 0.01)
    y1 = HEIGHT // 2 - 5 # h_crop // 2
    y2 = HEIGHT // 2 + 5 # h_crop // 2
    x1 = WIDTH // 2 - 5 # w_crop // 2
    x2 = WIDTH // 2 + 5 # w_crop // 2

    center_crop = cap[y1:y2, x1:x2]
    gray_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

    # Threshold to consider it "in focus"
    focus_threshold = 550.0
    focus_color = (0, 255, 0) if laplacian_var > focus_threshold else (0, 0, 255)

    # Draw rectangle for visualizing the crop
    cv2.rectangle(cap, (x1, y1), (x2, y2), focus_color, 2)

    # Put Laplacian variance at top of screen
    text = f"Focus: {laplacian_var:.2f}"
    cv2.putText(cap, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, focus_color, 3)

    cv2.resize(cap, (1920, 1080))

    # Encode frame to JPEG
    ret, jpeg = cv2.imencode(".jpg", cap, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ret:
        raise RuntimeError("Failed to encode frame to JPEG")

    yield (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
    )

@app.get("/video_focus")
def video_focus():
    return StreamingResponse(run_focusHelper(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/function1", response_class=HTMLResponse)
def function1_screen(request: Request):
    return templates.TemplateResponse("function1.html", {"request": request})

# ========== Entry Point ==========
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6800)