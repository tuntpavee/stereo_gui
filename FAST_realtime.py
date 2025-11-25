import cv2
import numpy as np
import open3d as o3d

# Stereo camera parameters
fx = 1479.20 * (640 / 1280)
baseline = 0.07  # meters
cx, cy = 320, 240  # for 640x480

K1 = np.array([[fx, 0, cx],
               [0, fx, cy],
               [0,  0,  1]])
K2 = K1.copy()
R = np.eye(3)
T = np.array([[baseline], [0.0], [0.0]])

# IRMP triangulation
def compute_irmp(X_init, camera_centers, ray_directions, max_iter=5, tol=1e-6):
    X = X_init.copy()
    for _ in range(max_iter):
        A = np.zeros((3, 3))
        b = np.zeros((3,))
        for O, b_i in zip(camera_centers, ray_directions):
            I_bb = np.eye(3) - np.outer(b_i, b_i)
            w = 1.0 / (np.linalg.norm(X - O) + 1e-6)
            A += w * I_bb
            b += w * I_bb @ O
        X_new = np.linalg.solve(A, b)
        if np.linalg.norm(X_new - X) < tol:
            break
        X = X_new
    return X

def triangulate_match(pt1, pt2):
    x1_h = np.array([pt1[0], pt1[1], 1.0])
    x2_h = np.array([pt2[0], pt2[1], 1.0])

    O1 = np.array([0.0, 0.0, 0.0])
    O2 = -R.T @ T.flatten()
    camera_centers = [O1, O2]

    b1 = np.linalg.inv(K1) @ x1_h
    b1 /= np.linalg.norm(b1)

    b2 = np.linalg.inv(K2) @ x2_h
    b2 = R.T @ b2
    b2 /= np.linalg.norm(b2)

    ray_directions = [b1, b2]
    X_init = 0.5 * (O1 + O2)

    return compute_irmp(X_init, camera_centers, ray_directions)

def find_red_centroid(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Wider red color range
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological filtering
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    cv2.imshow("Red Mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), largest
    return None, None

def run_color_cube_irmp():
    capL = cv2.VideoCapture(0)
    capR = cv2.VideoCapture(1)

    capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Red Cube IRMP 3D')
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 5
    vis.get_view_control().set_zoom(0.8)

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("Camera read error")
            break

        pt1, cnt1 = find_red_centroid(frameL)
        pt2, cnt2 = find_red_centroid(frameR)

        if pt1 and pt2 and abs(pt1[1] - pt2[1]) < 5:
            X = triangulate_match(pt1, pt2)
            if X[2] > 0 and np.isfinite(X).all():
                print(f"3D position: {X}, Depth: {X[2]:.2f} m")
                cv2.drawContours(frameL, [cnt1], -1, (0, 255, 0), 2)
                cv2.drawContours(frameR, [cnt2], -1, (0, 255, 0), 2)
                cv2.putText(frameL, f"{X[2]:.2f} m", (pt1[0] + 5, pt1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                pcd.points = o3d.utility.Vector3dVector([X])
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

        cv2.imshow("Left", frameL)
        cv2.imshow("Right", frameR)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
    vis.destroy_window()

# Run the red object triangulation system
run_color_cube_irmp()
