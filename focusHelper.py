import numpy as np
import cv2
from cvDetector import detector

tagDet = detector()

def calculate_focus_measure(gray_roi):
    """Calculate sharpness using Laplacian variance."""
    lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
    return lap.var()

def apply_simulated_blur(frame, blur_level):
    """Simulate blur based on 'dial' input."""
    if blur_level == 0:
        return frame
    ksize = max(1, (blur_level // 2) * 2 + 1)  # Must be odd and >=1
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)

def run_focusApp(camera_url):

    cap = cv2.VideoCapture(camera_url)
    
    # Create a window and trackbar
    window_name = "AprilGrid Focus Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Dial (Blur)", window_name, 0, 50, lambda x: None)

    focus_threshold = 150.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simulate blur based on slider
        blur_level = cv2.getTrackbarPos("Dial (Blur)", window_name)
        blurred_frame = apply_simulated_blur(frame, blur_level)
        gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        corners, ids, rejected = tagDet.detectMarkers(gray)

        focus_measures = []
        if ids is not None:
            for i in range(len(ids)):
                tag_corners = corners[i].reshape(-1, 2).astype(int)
                x, y, w, h = cv2.boundingRect(tag_corners)
                roi = gray[y:y+h, x:x+w]

                if roi.size == 0:
                    continue

                focus = calculate_focus_measure(roi)
                focus_measures.append(focus)

                color = (0, 255, 0) if focus > focus_threshold else (0, 0, 255)
                cv2.polylines(blurred_frame, [tag_corners], isClosed=True, color=color, thickness=2)
                cv2.putText(blurred_frame, f"{focus:.1f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Final status
        if focus_measures:
            avg_focus = np.mean(focus_measures)
            status = "Grid Focused" if avg_focus > focus_threshold else "Grid Not Focused"
        else:
            status = "No Tags Detected"

        cv2.putText(blurred_frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow(window_name, blurred_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()