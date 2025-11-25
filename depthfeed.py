import cv2
import numpy as np
import open3d as o3d

# === Stereo camera parameters ===
focal_length = 1479.2  # in pixels
baseline = 0.194       # in meters

# === Open stereo cameras ===
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

# === Create StereoSGBM matcher ===
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*6,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# === Start GUI loop ===
print("Press SPACE to show 3D point cloud. Press ESC to exit.")

while True:
    retL, imgL = capL.read()
    retR, imgR = capR.read()
    if not retL or not retR:
        break

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Compute disparity
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # Show disparity feed
    cv2.imshow("Depth Feed", disp_vis)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE: build and show point cloud
        h, w = disparity.shape
        Q = np.float32([
            [1, 0, 0, -w / 2],
            [0, -1, 0, h / 2],
            [0, 0, 0, -focal_length],
            [0, 0, 1 / baseline, 0]
        ])

        points_3D = cv2.reprojectImageTo3D(disparity, Q)
        mask = disparity > 0
        output_points = points_3D[mask]
        output_colors = imgL[mask]

        # Open3D visualization
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(output_points)
        pc.colors = o3d.utility.Vector3dVector(output_colors.astype(np.float32) / 255.0)

        o3d.visualization.draw_geometries([pc], window_name="3D Point Cloud")

# Cleanup
capL.release()
capR.release()
cv2.destroyAllWindows()
