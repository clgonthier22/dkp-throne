import cv2
import pytesseract
import sys


# Load the screenshot image (adjust path if necessary)
image = cv2.imread("images/contribution.jpg")

# Define the coordinates and dimensions for the ROI
x, y = 637, 307
width, height = 506, 62

# Extract the first ROI
first_roi = image[y:y+height, x:x+width]

# Coordinates for the sub-ROIs within the first ROI
x1, y1, w1, h1 = 369, 21, 484 - 369, 41 - 21
sub_roi1 = first_roi[y1:y1+h1, x1:x1+w1]

# Convert the sub-ROI to grayscale
gray_roi = cv2.cvtColor(sub_roi1, cv2.COLOR_BGR2GRAY)

# Get the threshold value from the command line
threshold_value = int(sys.argv[1])

# Apply the threshold
_, thresholded_roi = cv2.threshold(gray_roi, threshold_value, 255, cv2.THRESH_BINARY)

# Perform OCR
text = pytesseract.image_to_string(thresholded_roi, config='--psm 7')
print(f"Threshold: {threshold_value}, OCR Result: {text}")
