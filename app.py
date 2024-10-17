import cv2
import numpy as np
import pytesseract
from flask import Flask, request, render_template, send_file
import os
import base64

app = Flask(__name__)

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract.exe'

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
    
    results = []
    
    if digit_roi:
        x, y, w, h = digit_roi
        
        # Crop the detected digit region
        roi = binary[y:y+h, x:x+w]
        
        # Apply denoising to the ROI
        denoised = cv2.fastNlMeansDenoising(roi, None, 30, 7, 21)
        
        # Apply Gaussian blur to smooth the ROI
        blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
        
        # Invert the ROI to make digits white on black background (if needed)
        inverted = cv2.bitwise_not(blurred)
        
        # Save the preprocessed ROI for visualization
        roi_path = os.path.join('debug', 'preprocessed_roi.png')
        cv2.imwrite(roi_path, inverted)
        
        # Use Tesseract to extract text from the enhanced ROI
        config1 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        text1 = pytesseract.image_to_string(inverted, config=config1).strip()
        results.append(("Extracted Digits", text1))
    else:
        results.append(("Error", "No suitable digit region found"))
        roi_path = None
    
    # Extract text from the entire image without preprocessing
    config2 = r'--oem 3 --psm 6'
    full_text = pytesseract.image_to_string(gray, config=config2).strip()
    results.append(("Full Image Text", full_text))
    
    return results, roi_path


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
    roi_image = get_image_base64(roi_path) if roi_path else None
    
    # Create HTML response
    html_response = f'''
    <h2>Results:</h2>
    
    <h3>Extracted Digits (Meter Display):</h3>
    <ul>
    '''
    
    # Add only the "Extracted Digits" result
    for config_name, text in results:
        if config_name == "Extracted Digits":
            html_response += f'<li><strong>{config_name}:</strong> "{text}"</li>'
    
    html_response += f'''
    </ul>
    
    <h3>Full Image Text (Without Preprocessing):</h3>
    <ul>
    '''
    
    # Add only the "Full Image Text" result
    for config_name, text in results:
        if config_name == "Full Image Text":
            html_response += f'<li><strong>{config_name}:</strong> "{text}"</li>'
    
    html_response += f'''
    </ul>
    
    <h3>Original Image:</h3>
    <img src="data:image/jpeg;base64,{original_image}" style="max-width: 500px"><br>
    '''
    
    if roi_image:
        html_response += f'''
        <h3>Preprocessed ROI (Region provided to OCR):</h3>
        <img src="data:image/jpeg;base64,{roi_image}" style="max-width: 500px"><br>
        '''
    
    return html_response

if __name__ == '__main__':
    app.run(debug=True)
