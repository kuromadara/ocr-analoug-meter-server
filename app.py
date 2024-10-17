import cv2
import numpy as np
import pytesseract
from flask import Flask, request, render_template, send_file
import os
import base64

app = Flask(__name__)

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract.exe'

# def extract_meter_digits(image_path):
#     # Read the image
#     img = cv2.imread(image_path)
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply adaptive thresholding or Otsu's thresholding to better segment digits
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # Apply morphological operations to remove noise
#     kernel = np.ones((2,2), np.uint8)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
#     # Try increasing image resolution to improve OCR accuracy
#     height, width = binary.shape
#     enlarged = cv2.resize(binary, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
#     # Save preprocessed image for visualization
#     roi_path = os.path.join('debug', 'enhanced_preprocessed_roi.png')
#     cv2.imwrite(roi_path, enlarged)
    
#     # Use Tesseract to extract text from the processed ROI
#     config1 = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
#     text1 = pytesseract.image_to_string(enlarged, config=config1).strip()
    
#     return [("Enhanced Extracted Digits", text1)], roi_path
def extract_meter_digits(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold for detecting white digits on black background
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_roi = None
    largest_area = 0
    
    # Loop through all contours to find the largest rectangular contour (meter display area)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Use size and aspect ratio filtering to detect the digit region
        aspect_ratio = w / float(h)
        area = w * h
        
        # Assuming the meter display is rectangular and sufficiently large
        if 2.5 < aspect_ratio < 6 and area > largest_area:
            largest_area = area
            digit_roi = (x, y, w, h)
    
    if digit_roi:
        x, y, w, h = digit_roi
        
        # Crop the detected digit region
        roi = binary[y:y+h, x:x+w]
        
        # Apply morphological operations to enhance digit visibility
        kernel = np.ones((2,2), np.uint8)
        roi = cv2.dilate(roi, kernel, iterations=1)
        roi = cv2.erode(roi, kernel, iterations=1)
        
        # Enhance contrast
        roi = cv2.convertScaleAbs(roi, alpha=1.5, beta=0)
        
        # Save preprocessed ROI for visualization
        roi_path = os.path.join('debug', 'preprocessed_roi.png')
        cv2.imwrite(roi_path, roi)
        
        # Use Tesseract to extract text from the dynamically detected ROI
        config1 = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        text1 = pytesseract.image_to_string(roi, config=config1).strip()
        
        return [("Extracted Digits", text1)], roi_path
    else:
        return [("Error", "No suitable digit region found")], None

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

@app.route('/')
def upload_form():
    return '''
    <html>
        <body>
            <h2>Upload Meter Image</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <input type="submit" value="Extract Digits">
            </form>
        </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No file uploaded'
    
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'
    
    # Create directories if they don't exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('debug', exist_ok=True)
    
    # Save uploaded image
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)
    
    # Process image and extract text
    results, roi_path = extract_meter_digits(image_path)
    
    # Convert images to base64 for display
    original_image = get_image_base64(image_path)
    roi_image = get_image_base64(roi_path)
    
    # Create HTML response
    html_response = f'''
    <h2>Results:</h2>
    
    <h3>Extracted Text:</h3>
    <ul>
    '''
    
    # Add all OCR results
    for config_name, text in results:
        html_response += f'<li><strong>{config_name}:</strong> "{text}"</li>'
    
    html_response += f'''
    </ul>
    
    <h3>Original Image:</h3>
    <img src="data:image/jpeg;base64,{original_image}" style="max-width: 500px"><br>
    
    <h3>Preprocessed ROI (Region provided to OCR):</h3>
    <img src="data:image/jpeg;base64,{roi_image}" style="max-width: 500px"><br>
    '''
    
    return html_response

if __name__ == '__main__':
    app.run(debug=True)