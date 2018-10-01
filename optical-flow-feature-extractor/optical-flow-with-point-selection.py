# Code from:
# https://pysource.com/2018/05/14/optical-flow-with-lucas-kanade-method-opencv-3-4-with-python-3-tutorial-31/
# + my comments

import cv2
import numpy as np

# Lucas-Kanade parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Mouse function
def select_point(event, x, y):
    # Some useful global variables
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)


# Starts capturing
cap = cv2.VideoCapture(0)

# Open new window capture and set the mouse callback
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)

point_selected = False
point = ()
old_points = np.array([[]])

# Create the very first 'old frame'
_, frame = cap.read()
old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Start infinite loop
while True:
    _, frame = cap.read()
    # Retrieve the current frame
    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_selected is True:
        # Add red circle on the selected point
        cv2.circle(frame, point, 5, (0, 0, 255), 2)

        # Compute Optical Flow
        current_points, status, error = cv2.calcOpticalFlowPyrLK(old_frame, current_frame, old_points, None, **lk_params)

        # Update values
        old_frame = current_frame.copy()
        old_points = current_points

        # Find the new position of the point and add a green circle
        x, y = current_points.ravel()
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
