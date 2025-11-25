import cv2
import numpy as np
import open3d as o3d

# --- Camera Parameters ---
baseline = 0.07  # meters
fx = 930.0      # calibrated focal length
cx, cy = 320, 240  # optical center (assumed for 640x480)

# --- Open Cameras ---
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- HSV Range for Red ---
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# --- Store 3D points ---
points_3d = []

def detect_red_center(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center = (cx, cy)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
    return frame, center

print("[INFO] Press 'q' to quit and view point cloud...")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    frameL, centerL = detect_red_center(frameL)
    frameR, centerR = detect_red_center(frameR)

    if centerL and centerR:
        xL, yL = centerL
        xR, yR = centerR
        disparity = xL - xR
        if disparity != 0:
            Z = (fx * baseline) / disparity
            X = ((xL - cx) * Z) / fx
            Y = ((yL - cy) * Z) / fx
            points_3d.append([X, Y, Z])
            cv2.putText(frameL, f"Z={Z:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # Display camera windows
    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()

# --- Open3D Visualization ---
print(f"[INFO] Visualizing {len(points_3d)} points...")
if points_3d:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points_3d))
    o3d.visualization.draw_geometries([pcd])
else:
    print("[WARNING] No 3D points were collected.")
