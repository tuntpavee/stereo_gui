import cv2
import numpy as np
import open3d as o3d

# --- Settings ---
show_2d_coords = True

# --- Camera Parameters ---
baseline = 0.07  # meters
fx = 930.0      # Focal length (calibrate this!)
cx, cy = 320, 240  # Optical center (for 640x480)

# --- HSV Range for Red ---
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# --- Initialize Open3D ---
vis = o3d.visualization.Visualizer()
vis.create_window("Real-Time 3D Points")
pcd = o3d.geometry.PointCloud()
points_3d = []

# --- Disparity Map (Sparse) ---
disparity_map = np.zeros((480, 640), dtype=np.uint8)

# --- Open Cameras ---
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)
capL.set(cv2.CAP_PROP_BUFFERSIZE, 1)
capR.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not capL.isOpened() or not capR.isOpened():
    print("[ERROR] Could not open cameras!")
    exit()

# Set resolution
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def detect_red_pixels(frame):
    """Detects red object and returns pixels within bounding box + annotated frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        mask_box = np.zeros_like(mask)
        mask_box[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
        red_pixels = np.column_stack(np.where(mask_box > 0))  # [y, x]

        for (py, px) in red_pixels[::20]:
            cv2.circle(frame, (px, py), 1, (0, 255, 255), -1)

        return frame, red_pixels
    else:
        return frame, np.array([])

print("[INFO] Press 'q' to quit. Detected objects will be logged below:")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    frameL, red_pixelsL = detect_red_pixels(frameL)
    frameR, red_pixelsR = detect_red_pixels(frameR)

    # Reset disparity map each frame
    disparity_map[:] = 0

    # Print 2D coordinates from left
    if red_pixelsL.size > 0:
        print("\n=== Left Camera Red Pixels ===")
        for i, (y, x) in enumerate(red_pixelsL):
            print(f"Pixel {i+1}: (x={x}, y={y})")
            if show_2d_coords:
                cv2.putText(frameL, f"({x}, {y})", (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Stereo triangulation and sparse disparity map
    if red_pixelsL.shape[0] > 0 and red_pixelsR.shape[0] > 0:
        min_len = min(len(red_pixelsL), len(red_pixelsR))
        print("\n=== 3D Depth & Disparity ===")
        for i, ((yL, xL), (yR, xR)) in enumerate(zip(red_pixelsL[:min_len], red_pixelsR[:min_len])):
            disparity = xL - xR
            if disparity > 0:
                Z = (fx * baseline) / disparity
                X = ((xL - cx) * Z) / fx
                Y = ((yL - cy) * Z) / fx
                points_3d.append([X, Y, Z])
                cv2.putText(frameL, f"{Z:.2f}m", (xL, yL - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                disparity_map[yL, xL] = np.clip(disparity * 3, 0, 255)  # Scale for visualization
                print(f"Pixel {i+1}: (x={xL}, y={yL}) → Disparity = {disparity:.2f}, Depth = {Z:.2f} m")

    # Open3D visualization
    if points_3d:
        pcd.points = o3d.utility.Vector3dVector(np.array(points_3d))
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    # Display
    cv2.imshow("Left Camera", frameL)
    cv2.imshow("Right Camera", frameR)
    cv2.imshow("Sparse Disparity Map", disparity_map)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
capL.release()
capR.release()
cv2.destroyAllWindows()
vis.destroy_window()
