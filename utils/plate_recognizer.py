import cv2
import numpy as np
import pytesseract
import re

class PlateRecognizer:
    def __init__(self):
        self.min_confidence = 0.5
    
    def preprocess_plate_image(self, image):
        """Enhanced preprocessing for better OCR results"""
        if image.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_plate_region(self, bike_roi):
        """Try to locate the license plate region"""
        if bike_roi.size == 0:
            return None
        
        height, width = bike_roi.shape[:2]
        
        # Common plate positions (rear of motorcycle)
        plate_regions = [
            (int(width*0.2), int(height*0.6), int(width*0.8), int(height*0.9)),  # Rear
            (int(width*0.1), int(height*0.1), int(width*0.4), int(height*0.3)),  # Front
        ]
        
        for x1, y1, x2, y2 in plate_regions:
            region = bike_roi[y1:y2, x1:x2]
            if region.size > 0:
                # Check if this looks like a plate (aspect ratio, edges)
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                if np.sum(edges) > 1000:  # Has significant edges
                    return region
        
        return bike_roi[int(height*0.6):int(height*0.9), int(width*0.2):int(width*0.8)]
    
    def read_plate(self, bike_roi):
        """Improved plate recognition with better preprocessing"""
        if bike_roi is None or bike_roi.size == 0:
            return "UNKNOWN", 0.0
        
        try:
            # Find plate region
            plate_region = self.detect_plate_region(bike_roi)
            
            if plate_region is None or plate_region.size == 0:
                return "UNKNOWN", 0.0
            
            # Preprocess for OCR
            processed = self.preprocess_plate_image(plate_region)
            
            if processed is None:
                return "UNKNOWN", 0.0
            
            # Use Tesseract with custom configuration
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(processed, config=custom_config)
            
            # Clean and validate text
            cleaned_text = self.clean_plate_text(text)
            
            if cleaned_text and len(cleaned_text) >= 4:
                confidence = min(len(cleaned_text) / 10.0, 0.9)
                return cleaned_text, confidence
            else:
                return "UNKNOWN", 0.0
                
        except Exception as e:
            return "UNKNOWN", 0.0
    
    def clean_plate_text(self, text):
        """Clean and validate license plate text"""
        # Remove unwanted characters and whitespace
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
        
        # Validate plate format (basic check)
        if 4 <= len(cleaned) <= 10:
            # Check for common patterns
            if re.match(r'^[A-Z]{2,3}\d{1,4}[A-Z]?$', cleaned):
                return cleaned
            elif re.match(r'^\d{1,4}[A-Z]{2,3}$', cleaned):
                return cleaned
        
        return ""

def create_plate_recognizer():
    return PlateRecognizer()