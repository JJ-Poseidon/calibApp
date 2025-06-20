import cv2
import numpy as np
import time
from cvDetector import detector

tagDet = detector()

def run_apriltag_capture(camera_url, F_MM, FOV_DEG, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM):
    NUM_TAGS_X         = 6
    NUM_TAGS_Y         = 6
    TAG_REAL_WIDTH_M   = 0.088
    TAG_REAL_SPACING_M = 0.0265

    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source {camera_url!r}")

    print("→ Press 'c' to schedule a capture in 10 seconds…")
    capture_requested = False
    capture_time = 0
    frame_capture = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and not capture_requested:
            capture_requested = True
            capture_time = time.time() + 10
            print("Capture scheduled in 10 seconds…")

        if capture_requested:
            remaining = int(capture_time - time.time())
            if remaining > 0:
                cv2.putText(frame, f"Capturing in {remaining}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                frame_capture = frame.copy()
                print("Frame captured, processing…")
                break
        
        cv2.namedWindow('Live Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('Live Feed', frame)

    cap.release()
    cv2.destroyAllWindows()

    if frame_capture is None:
        raise RuntimeError("Failed to capture frame")

    gray = cv2.cvtColor(frame_capture, cv2.COLOR_BGR2GRAY)
    res_y, res_x = gray.shape

    corners, ids, _ = tagDet.detectMarkers(gray)
    if ids is None or len(ids) < 2:
        raise RuntimeError("Not enough tags detected.")

    tag_ws = [np.linalg.norm(c[0][1] - c[0][0]) for c in corners]
    tag_hs = [np.linalg.norm(c[0][2] - c[0][1]) for c in corners]
    avg_tag_w = np.mean(tag_ws)
    avg_tag_h = np.mean(tag_hs)

    px_per_m = avg_tag_w / TAG_REAL_WIDTH_M
    gap_px = TAG_REAL_SPACING_M * px_per_m
    total_px_w = NUM_TAGS_X * avg_tag_w + (NUM_TAGS_X-1)*gap_px
    total_px_h = NUM_TAGS_Y * avg_tag_h + (NUM_TAGS_Y-1)*gap_px

    print(f"1) Average Tag: {avg_tag_w:.1f}px × {avg_tag_h:.1f}px")
    print(f"   Grid size:    {total_px_w:.1f}px × {total_px_h:.1f}px")

    proj_w_mm = (total_px_w / res_x) * SENSOR_WIDTH_MM
    board_w_mm = (NUM_TAGS_X*TAG_REAL_WIDTH_M + (NUM_TAGS_X-1)*TAG_REAL_SPACING_M)*1000
    mag = proj_w_mm / board_w_mm
    wd_mm = (-F_MM * (mag - 1) / mag)
    wd_m  = wd_mm / 1000

    print("\n2) Working distance:")
    print(f"   WD = {wd_mm:.1f}mm  ≃ {wd_m:.3f}m")
    print(f"   Magnification: {mag:.5f}x")

    horiz_dist = wd_m * np.tan(np.radians(FOV_DEG / 2))
    print("\n3) Perpendicular distance @ Working Distance:")
    print(f"   Horizontal distance traveled: {horiz_dist:.3f}m")