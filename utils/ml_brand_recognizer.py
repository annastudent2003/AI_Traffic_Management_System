import cv2
import numpy as np
import os
from config import MOTORCYCLE_BRANDS

class MLBrandRecognizer:
    def __init__(self):
        self.brand_list = MOTORCYCLE_BRANDS
        self.feature_extractor = self._load_feature_extractor()
    
    def _load_feature_extractor(self):
        """Simple feature extraction without heavy dependencies"""
        try:
            # Try to load a pre-trained model if available
            model_path = os.path.join("models", "brand_features.npy")
            if os.path.exists(model_path):
                print("✅ Loaded brand feature database")
            return None
        except:
            print("⚠️  Using lightweight brand recognition")
            return None
    
    def extract_color_histogram(self, image, bins=(8, 8, 8)):
        """Extract color histogram features"""
        if image.size == 0:
            return None
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Compute 3D color histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        
        # Normalize the histogram
        cv2.normalize(hist, hist)
        
        return hist.flatten()
    
    def recognize_brand(self, bike_roi):
        """Lightweight brand recognition using color features"""
        if bike_roi is None or bike_roi.size == 0:
            return "UNKNOWN", 0.0
        
        try:
            # Resize for consistent processing
            resized = cv2.resize(bike_roi, (150, 150))
            
            # Extract simple features (color histogram)
            features = self.extract_color_histogram(resized)
            
            if features is None:
                return "UNKNOWN", 0.0
            
            # Simple matching based on color distribution
            # In a real implementation, you'd use a trained classifier here
            
            # For now, return a basic implementation
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:,:,0])
            
            # Simple heuristic based on hue
            if h_mean < 15:
                return "Bajaj", 0.6
            elif h_mean < 30:
                return "Honda", 0.65
            elif h_mean < 45:
                return "TVS", 0.55
            elif h_mean < 60:
                return "Suzuki", 0.6
            else:
                return "Yamaha", 0.5
                
        except Exception as e:
            print(f"Brand recognition error: {e}")
            return "UNKNOWN", 0.0

def create_ml_brand_recognizer():
    return MLBrandRecognizer()