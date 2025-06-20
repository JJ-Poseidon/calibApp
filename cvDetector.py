import cv2

def detector():
    # Use a standard ArUco dictionary — change if you're using a different one
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    # Set up detector parameters
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 25
    params.markerBorderBits = 2
    params.polygonalApproxAccuracyRate = 0.03
    params.minCornerDistanceRate = 0.05
    params.perspectiveRemovePixelPerCell = 1
    params.perspectiveRemoveIgnoredMarginPerCell = 0

    # Set up detector once outside the loop
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return detector