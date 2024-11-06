import cv2
import numpy as np

# Load the ROI image and the green tick template
roi = cv2.imread("images/test_false_tick.jpg")  # Replace with the path to your ROI
green_tick = cv2.imread("images/green_tick.jpg")

# Convert both images to grayscale (template matching works better with grayscale images)
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray_tick = cv2.cvtColor(green_tick, cv2.COLOR_BGR2GRAY)

# Perform template matching
result = cv2.matchTemplate(gray_roi, gray_tick, cv2.TM_CCOEFF_NORMED)

# Set a threshold for detection
threshold = 0.8  # You may need to adjust this threshold based on testing

# Find locations in the result matrix that exceed the threshold
locations = np.where(result >= threshold)

# Check if we have at least one match
if len(locations[0]) > 0:
    print("Green tick detected in the ROI.")
    # Optionally, draw rectangles around matched regions
    for pt in zip(*locations[::-1]):
        cv2.rectangle(roi, pt, (pt[0] + green_tick.shape[1], pt[1] + green_tick.shape[0]), (0, 255, 0), 2)
else:
    print("Green tick not found in the ROI.")

# Display the ROI with rectangles around detected ticks (if any)
cv2.imshow("ROI with detected green tick", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
