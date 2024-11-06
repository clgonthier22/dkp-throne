import cv2
import pytesseract

# Set the path to Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path if necessary

# Load the screenshot image
image = cv2.imread("images/contribution.jpg")

# Define the initial coordinates and dimensions for the ROI column
x, y = 637, 307
width, height = 506, 62

# Coordinates for sub-ROIs within each main ROI
sub1_x, sub1_y, sub1_width, sub1_height = 19, 21, 234 - 19, 41 - 21
sub2_x, sub2_y, sub2_width, sub2_height = 369, 21, 484 - 369, 41 - 21

# Preprocessing function to improve OCR accuracy
def preprocess_for_ocr(roi):
    # Step 1: Resize the image to increase character size
    resized_roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Step 2: Convert to grayscale
    gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Apply Gaussian blur to reduce noise (optional)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    
    # Step 4: Apply binary thresholding
    _, thresholded_roi = cv2.threshold(blurred_roi, 70, 255, cv2.THRESH_BINARY)
    
    # Step 5: Apply morphological transformations to improve text structure
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed_roi = cv2.morphologyEx(thresholded_roi, cv2.MORPH_CLOSE, kernel)
    
    return processed_roi

# Loop to extract and process each ROI in the column
roi_index = 1
while y + height <= image.shape[0]:  # Ensure we stay within the image bounds
    # Extract the current main ROI
    roi = image[y:y+height, x:x+width]
    
    # Extract the first sub-ROI within the current ROI
    sub_roi1 = roi[sub1_y:sub1_y + sub1_height, sub1_x:sub1_x + sub1_width]
    # Extract the second sub-ROI within the current ROI
    sub_roi2 = roi[sub2_y:sub2_y + sub2_height, sub2_x:sub2_x + sub2_width]
    
    # Apply preprocessing
    processed_sub_roi1 = preprocess_for_ocr(sub_roi1)
    processed_sub_roi2 = preprocess_for_ocr(sub_roi2)

    # Display the processed sub-ROIs for visual verification
    cv2.imshow(f"Thresholded Sub-ROI 1 - ROI {roi_index}", processed_sub_roi1)
    cv2.imshow(f"Thresholded Sub-ROI 2 - ROI {roi_index}", processed_sub_roi2)
    
    # Perform OCR on each processed sub-ROI
    text1 = pytesseract.image_to_string(processed_sub_roi1, config='--psm 7')
    text2 = pytesseract.image_to_string(processed_sub_roi2, config='--psm 7')
    
    # Print the OCR results
    print(f"OCR Result for Sub-ROI 1 in ROI {roi_index}: {text1}")
    print(f"OCR Result for Sub-ROI 2 in ROI {roi_index}: {text2}")
    
    # Move to the next ROI position
    y += height
    roi_index += 1

cv2.waitKey(0)
cv2.destroyAllWindows()
