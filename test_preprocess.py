import cv2
import pytesseract


# Load the full contribution image
image = cv2.imread("images/contribution.jpg")

# Step 1: Resize the image to increase character size
resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Step 2: Convert to grayscale
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Gaussian blur to reduce noise (optional)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Step 4: Apply binary thresholding to increase contrast
_, thresholded_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY)

# Step 5: Apply morphological transformations to improve text structure
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
processed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)

# Display the processed image to verify quality before OCR
cv2.imshow("Processed Contribution Image", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Run OCR on the processed image
text = pytesseract.image_to_string(processed_image, config='--psm 6')  # Use PSM 6 for blocks of text
print("OCR Result for Entire Image:")
print(text)
