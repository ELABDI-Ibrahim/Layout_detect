

import cv2
import pytesseract

# If on Windows, set tesseract.exe path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 1. Load your image
image_path = r'output_yolo\page_1_layout.jpg'
image = cv2.imread(image_path)

# Optional: convert to grayscale (helps OCR sometimes)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Perform OCR
text = pytesseract.image_to_string(gray)  # returns all detected text

# 3. Print results
print(text)