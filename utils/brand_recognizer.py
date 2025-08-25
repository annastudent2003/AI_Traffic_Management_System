import cv2
import numpy as np

class BrandRecognizer:
    def __init__(self):
        # Precompute color profiles for common brands
        self.brand_profiles = {
            "Honda": {"h_range": (0, 30), "s_range": (50, 200), "v_range": (100, 220)},
            "Yamaha": {"h_range": (20, 40), "s_range": (80, 220), "v_range": (120, 240)},
            "Suzuki": {"h_range": (10, 25), "s_range": (70, 180), "v_range": (90, 200)},
            "Bajaj": {"h_range": (0, 15), "s_range": (40, 150), "v_range": (80, 180)},
            "TVS": {"h_range": (5, 20), "s_range": (60, 170), "v_range": (100, 190)}
        }
    
    def recognize_brand(self, bike_roi):
        if bike_roi is None or bike_roi.size == 0:
            return "UNKNOWN", 0.0
        
        try:
            # Resize for faster processing
            small_roi = cv2.resize(bike_roi, (100, 100))
            
            # Convert to HSV
            hsv = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV)
            
            # Calculate average values
            h_mean = np.mean(hsv[:,:,0])
            s_mean = np.mean(hsv[:,:,1])
            v_mean = np.mean(hsv[:,:,2])
            
            # Find best matching brand
            best_match = "UNKNOWN"
            best_score = 0.0
            
            for brand, profile in self.brand_profiles.items():
                h_score = 1.0 - min(abs(h_mean - np.mean(profile["h_range"])) / 90.0, 1.0)
                s_score = 1.0 - min(abs(s_mean - np.mean(profile["s_range"])) / 255.0, 1.0)
                v_score = 1.0 - min(abs(v_mean - np.mean(profile["v_range"])) / 255.0, 1.0)
                
                total_score = (h_score + s_score + v_score) / 3.0
                
                if total_score > best_score:
                    best_score = total_score
                    best_match = brand
            
            # Apply minimum confidence threshold
            if best_score < 0.4:
                return "UNKNOWN", best_score
            
            return best_match, min(best_score * 1.2, 0.95)  # Scale confidence
                
        except Exception as e:
            return "UNKNOWN", 0.0

def create_brand_recognizer():
    return BrandRecognizer()