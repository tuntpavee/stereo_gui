import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares

# --- Calibration parameters ---
focal_length = 1479.2
baseline = 0.155  # meters

K = np.array([[focal_length, 0, 640],
              [0, focal_length, 360],
              [0, 0, 1]])

P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((np.eye(3), np.array([[-baseline], [0], [0]])))

def optical_center(P):
    M = P[:, :3]
    p4 = P[:, 3]
    return -np.linalg.inv(M) @ p4

def get_ray(P, x):
    M = P[:, :3]
    K, _ = np.linalg.qr(M)
    x_h = np.array([x[0], x[1], 1.0])
    ray = np.linalg.inv(K) @ x_h
    return ray / np.linalg.norm(ray)

def mvmp_initial_estimate(P1, P2, x1, x2):
    O1 = optical_center(P1)
    O2 = optical_center(P2)
    b1 = get_ray(P1, x1)
    b2 = get_ray(P2, x2)
    A = np.eye(3) - np.outer(b1, b1)
    B = np.eye(3) - np.outer(b2, b2)
    return np.linalg.inv(A + B) @ (A @ O1 + B @ O2)

def reprojection_error(X, P_list, x_list):
    X_hom = np.append(X, 1)
    error = []
    for P, x in zip(P_list, x_list):
        x_proj = P @ X_hom
        x_proj /= x_proj[2]
        error.append(x_proj[:2] - x)
    return np.concatenate(error)

def mplm_triangulation(P1, P2, x1, x2):
    X_init = mvmp_initial_estimate(P1, P2, x1, x2)
    P_list = [P1, P2]
    x_list = [x1, x2]
    result = least_squares(reprojection_error, X_init, args=(P_list, x_list))
    return result.x

# --- Init video and matcher ---
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- Init Open3D viewer ---
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Live 3D Point Cloud", width=960, height=720)
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

print("Press ESC in Open3D window or Ctrl+C to exit")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(grayL, None)
    kp2, des2 = orb.detectAndCompute(grayR, None)

    matches = []
    if des1 is not None and des2 is not None and des1.shape[1] == des2.shape[1]:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

    points = []
    colors = []

    for m in matches[:300]:
        pt1 = np.array(kp1[m.queryIdx].pt)
        pt2 = np.array(kp2[m.trainIdx].pt)

        if abs(pt1[1] - pt2[1]) > 2:
            continue

        try:
            X = mplm_triangulation(P1, P2, pt1, pt2)
            depth = X[2]
            if 0 < depth < 10:
                points.append(X)
                color = frameL[int(pt1[1]), int(pt1[0])] / 255.0
                colors.append(color)
        except:
            continue

    if points:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

capL.release()
capR.release()
vis.destroy_window()
