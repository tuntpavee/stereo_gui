import numpy as np
from scipy.optimize import least_squares

def optical_center(P):
    # Get optical center from projection matrix
    M = P[:, :3]
    p4 = P[:, 3]
    return -np.linalg.inv(M) @ p4

def mvmp_initial_estimate(P1, P2, x1, x2):
    # Get optical centers
    O1 = optical_center(P1)
    O2 = optical_center(P2)

    # Normalize rays from image points
    def get_ray(P, x):
        M = P[:, :3]
        K, R = np.linalg.qr(M)
        x_h = np.array([x[0], x[1], 1.0])
        ray = np.linalg.inv(K) @ x_h
        return ray / np.linalg.norm(ray)

    b1 = get_ray(P1, x1)
    b2 = get_ray(P2, x2)

    # Solve for closest point between skew rays (MVMP)
    A = np.eye(3) - np.outer(b1, b1)
    B = np.eye(3) - np.outer(b2, b2)
    X0 = np.linalg.inv(A + B) @ (A @ O1 + B @ O2)

    return X0

def reprojection_error(X, P_list, x_list):
    X_hom = np.append(X, 1)
    error = []
    for P, x in zip(P_list, x_list):
        x_proj = P @ X_hom
        x_proj /= x_proj[2]
        err = x_proj[:2] - x
        error.append(err)
    return np.concatenate(error)

def mplm_triangulation(P1, P2, x1, x2):
    # Initial 3D point estimate from MVMP
    X_init = mvmp_initial_estimate(P1, P2, x1, x2)

    # Refine with LM optimization
    P_list = [P1, P2]
    x_list = [x1, x2]
    result = least_squares(reprojection_error, X_init, args=(P_list, x_list))

    return result.x  # Refined 3D point

# ==== Example Usage ====
if __name__ == "__main__":
    # Example projection matrices and pixel coordinates
    # Replace with actual calibration and image points
    P1 = np.array([[1479.2, 0, 640, 0],
                   [0, 1479.2, 360, 0],
                   [0, 0, 1, 0]])

    T = np.array([0.155, 0, 0])  # 15.5 cm baseline
    P2 = np.array([[1479.2, 0, 640, -1479.2 * T[0]],
                   [0, 1479.2, 360, 0],
                   [0, 0, 1, 0]])

    x1 = np.array([650, 370])  # point in left image
    x2 = np.array([620, 370])  # point in right image

    X = mplm_triangulation(P1, P2, x1, x2)
    print("Triangulated 3D point:", X)
