import cv2
import numpy as np

# Chessboard config
chessboard_size = (9, 6)
square_size = 0.025  # meters

# Prepare 3D object points
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

# Storage
objpoints = []
imgpointsL = []
imgpointsR = []

# Open two cameras
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

print("Press [SPACE] when chessboard is detected in both cameras to capture")
print("Press [ESC] to start calibration")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    retL_c, cornersL = cv2.findChessboardCorners(grayL, chessboard_size)
    retR_c, cornersR = cv2.findChessboardCorners(grayR, chessboard_size)

    dispL = frameL.copy()
    dispR = frameR.copy()

    if retL_c:
        cv2.drawChessboardCorners(dispL, chessboard_size, cornersL, retL_c)
    if retR_c:
        cv2.drawChessboardCorners(dispR, chessboard_size, cornersR, retR_c)

    cv2.imshow("Left Camera", dispL)
    cv2.imshow("Right Camera", dispR)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break
    elif key == 32 and retL_c and retR_c:  # Spacebar
        print("Captured pair.")
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

capL.release()
capR.release()
cv2.destroyAllWindows()

# Calibrate both cameras
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1], criteria=criteria, flags=flags
)

print("\nCalibration Done!")
print("Focal Length fx:", mtxL[0, 0])
print("Baseline (meters):", T[0])
