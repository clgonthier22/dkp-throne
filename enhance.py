import cv2
import pytesseract

# Load the original image
image = cv2.imread("images/first_roi.jpg")

# Display the original image
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# Step 1: Resize the image to increase character size
image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Step 2: Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Gaussian blur to reduce noise (optional)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Step 4: Apply binary thresholding
_, thresholded_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY)

# Step 5: Apply morphological transformations to improve text structure
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
processed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)

# Display the processed image to verify quality before OCR
cv2.imshow("Processed Image for OCR", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()