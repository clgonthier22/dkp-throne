import cv2
import pytesseract
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import re

# Global variables
rect_start = None
rect_end = None
drawing = False
image = None
roi = None

def mouse_callback(event, x, y, flags, param):
    global rect_start, rect_end, drawing, image, roi

    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = image.copy()
            cv2.rectangle(temp_img, rect_start, (x, y), (0, 255, 0), 2)
            cv2.imshow("Select ROI", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        rect_end = (x, y)
        drawing = False
        cv2.rectangle(image, rect_start, rect_end, (0, 255, 0), 2)
        cv2.imshow("Select ROI", image)
        roi = image[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]]

def select_roi():
    global image, roi
    # Load the full contribution image
    image = cv2.imread("images/contribution.jpg")
    
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)
    
    while True:
        cv2.imshow("Select ROI", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break
    
    cv2.destroyAllWindows()
    return roi

# Select ROI
roi = select_roi()

if roi is not None:
    # Step 1: Resize the ROI to increase character size
    resized_image = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Step 2: Convert to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian blur to reduce noise (optional)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 4: Apply binary thresholding to increase contrast
    _, thresholded_image = cv2.threshold(blurred_image, 70, 255, cv2.THRESH_BINARY)

    # Step 5: Apply morphological transformations to improve text structure
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)

    # Display the processed image to verify quality before OCR
    cv2.imshow("Processed ROI", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Run OCR on the processed image with number and comma-specific configuration
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,'
    text = pytesseract.image_to_string(processed_image, config=custom_config)
    
    # Post-process the result to ensure only numbers and commas
    numbers_with_commas = re.sub(r'[^\d,]', '', text)  # Remove all characters except digits and commas
    
    # Convert the string to a number (removing commas)
    try:
        number = int(numbers_with_commas.replace(',', ''))
        print("OCR Result for Selected ROI (as number):")
        print(text)

    except ValueError:
        print("Could not convert the detected text to a number.")

else:
    print("No ROI selected. Exiting.")