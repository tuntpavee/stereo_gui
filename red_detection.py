import cv2
import numpy as np

# Stereo camera parameters
baseline = 0.07  # 7 cm in meters
# fx = 1479.20 * (640 / 1280)  # Adjusted focal length for 640px width
fx = 1000.0
cx, cy = 320, 240  # Assuming 640x480 resolution

# Open both cameras
cap_left = cv2.VideoCapture(0)  # Left
cap_right = cv2.VideoCapture(1)  # Right

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# HSV range for red
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

def detect_red_center(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Noise reduction
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Find contours and centroid
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
            cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
    return frame, center

while True:
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()
    if not retL or not retR:
        break

    frameL, centerL = detect_red_center(frameL)
    frameR, centerR = detect_red_center(frameR)

    if centerL and centerR:
        xL, yL = centerL
        xR, yR = centerR

        # Disparity
        disparity = xL - xR
        if disparity != 0:
            Z = (fx * baseline) / disparity  # depth in meters
            X = ((xL - cx) * Z) / fx
            Y = ((yL - cy) * Z) / fx

            # Display result
            cv2.putText(frameL, f"X: {X:.2f}m Y: {Y:.2f}m Z: {Z:.2f}m", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show video
    cv2.imshow("Left Camera", frameL)
    cv2.imshow("Right Camera", frameR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
