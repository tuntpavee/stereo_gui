import cv2
import numpy as np

# ---- Stereo Camera Parameters (from user calibration) ----
fx = 1479.20
baseline = 0.15  # meters
cx, cy = 320, 240

K1 = np.array([[fx, 0, cx],
               [0, fx, cy],
               [0,  0,  1]])
K2 = K1.copy()
R = np.eye(3)
T = np.array([[baseline], [0.0], [0.0]])

# ---- IRMP triangulation ----
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

# ---- Main real-time stereo pipeline ----
def run_stereo_irmp_live():
    capL = cv2.VideoCapture(0)
    capR = cv2.VideoCapture(1)

    capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    orb = cv2.ORB_create(500)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("Camera read error")
            break

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        kp1, des1 = orb.detectAndCompute(grayL, None)
        kp2, des2 = orb.detectAndCompute(grayR, None)

        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:10]

            for m in matches:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                if abs(pt1[1] - pt2[1]) < 2:
                    X = triangulate_match(pt1, pt2)


                u1, v1 = int(pt1[0]), int(pt1[1])
                u2, v2 = int(pt2[0]), int(pt2[1])

                cv2.circle(frameL, (u1, v1), 4, (0, 255, 0), -1)
                cv2.circle(frameR, (u2, v2), 4, (0, 255, 0), -1)

                depth = X[2]
                label = f"Z={depth:.2f}m"
                cv2.putText(frameL, label, (u1+5, v1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        cv2.imshow("Left Camera", frameL)
        cv2.imshow("Right Camera", frameR)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

# Run live stereo IRMP
run_stereo_irmp_live()
