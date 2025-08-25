import cv2
import numpy as np
import time
import os

try:
    from config import (HELMET_MIN_CONFIDENCE, HELMET_CONSECUTIVE_FRAMES, 
                       HELMET_DEBUG_MODE, DEVICE)
except ImportError:
    HELMET_MIN_CONFIDENCE = 0.3
    HELMET_CONSECUTIVE_FRAMES = 2
    HELMET_DEBUG_MODE = True
    DEVICE = "cpu"

class HelmetDetector:
    def __init__(self):
        self.min_confidence = HELMET_MIN_CONFIDENCE
        self.consecutive_frames = HELMET_CONSECUTIVE_FRAMES
        self.debug_mode = HELMET_DEBUG_MODE
        self.violation_buffer = {}
        
        print(f"✅ Enhanced helmet detector loaded (Confidence: {self.min_confidence})")
        if self.debug_mode:
            print("🔧 Debug mode enabled")
    
    def detect_helmet_violation(self, frame, motorcycle_bbox, track_id):
        """Enhanced helmet detection using multiple computer vision techniques"""
        try:
            x1, y1, x2, y2 = map(int, motorcycle_bbox)
            
            # Extract head region (upper portion of motorcycle)
            head_height = int((y2 - y1) * 0.35)  # 35% of motorcycle height for head
            head_y1 = max(0, y1 - 20)  # Extend upward
            head_y2 = head_y1 + head_height
            head_x1 = max(0, x1 - 10)
            head_x2 = min(frame.shape[1], x2 + 10)
            
            head_roi = frame[head_y1:head_y2, head_x1:head_x2]
            
            if head_roi.size == 0 or head_roi.shape[0] < 25 or head_roi.shape[1] < 25:
                if self.debug_mode:
                    print(f"❌ Invalid ROI for track {track_id}")
                return False, 0.0, {}, None
            
            # Apply multiple detection techniques
            has_helmet, confidence = self._multi_method_detection(head_roi)
            
            if self.debug_mode:
                print(f"🔍 Track {track_id}: Helmet={has_helmet}, Confidence={confidence:.2f}")
            
            if not has_helmet and confidence >= self.min_confidence:
                violation_data = {
                    'has_helmet': False,
                    'helmet_confidence': float(confidence),
                    'plate_number': "DETECTED",
                    'plate_confidence': 0.7,
                    'bike_brand': "Unknown",
                    'brand_confidence': 0.6,
                    'bike_color': self._detect_bike_color(frame, motorcycle_bbox),
                    'track_id': track_id,
                    'timestamp': time.time()
                }
                
                # Draw visual feedback
                color = (0, 0, 255)  # Red for violation
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"NO HELMET {confidence:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show debug info
                if self.debug_mode:
                    cv2.imshow(f"Head ROI {track_id}", head_roi)
                    cv2.waitKey(1)
                
                return self._confirm_violation(track_id, violation_data, head_roi)
            else:
                # Draw green box for helmets
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                return False, 0.0, {}, None
                
        except Exception as e:
            print(f"❌ Error in helmet detection: {e}")
            return False, 0.0, {}, None
    
    def _multi_method_detection(self, head_roi):
        """Use multiple computer vision techniques for better accuracy"""
        try:
            # Resize for consistent processing
            resized = cv2.resize(head_roi, (150, 150))
            
            # Convert color spaces
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Skin tone detection
            skin_mask = self._detect_skin(hsv)
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            
            # Method 2: Hair detection (dark regions)
            _, hair_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
            hair_ratio = np.sum(hair_mask > 0) / hair_mask.size
            
            # Method 3: Edge density (faces have more edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            
            # Method 4: Color consistency (helmets are more uniform)
            color_std = np.std(resized)
            
            # Method 5: Brightness analysis
            brightness = np.mean(hsv[:,:,2])
            
            # Debug information
            if self.debug_mode:
                print(f"📊 Skin: {skin_ratio:.2f}, Hair: {hair_ratio:.2f}, Edges: {edge_ratio:.2f}")
                print(f"📊 Color STD: {color_std:.1f}, Brightness: {brightness:.1f}")
            
            # Weighted decision making
            no_helmet_confidence = 0.0
            
            # Strong indicators of no helmet
            if skin_ratio > 0.12:  # Skin detected
                no_helmet_confidence += 0.4
            if 0.08 < hair_ratio < 0.35:  # Reasonable hair amount
                no_helmet_confidence += 0.3
            if edge_ratio > 0.07:  # Facial features
                no_helmet_confidence += 0.2
            if color_std > 40:  # Natural color variation
                no_helmet_confidence += 0.1
            if brightness < 110:  # Dark hair/head
                no_helmet_confidence += 0.1
            
            # Final decision
            has_helmet = no_helmet_confidence < 0.6
            confidence = max(no_helmet_confidence, 1.0 - no_helmet_confidence)
            
            return has_helmet, confidence
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return True, 0.0  # Assume helmet on error
    
    def _detect_skin(self, hsv_image):
        """Detect skin tones in HSV image"""
        # Define HSV ranges for skin tones
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([25, 255, 255], dtype=np.uint8)
        lower_skin2 = np.array([160, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        
        skin_mask1 = cv2.inRange(hsv_image, lower_skin1, upper_skin1)
        skin_mask2 = cv2.inRange(hsv_image, lower_skin2, upper_skin2)
        
        return cv2.bitwise_or(skin_mask1, skin_mask2)
    
    def _detect_bike_color(self, frame, bbox):
        """Detect dominant bike color"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            bike_roi = frame[y1:y2, x1:x2]
            
            if bike_roi.size == 0:
                return "Unknown"
            
            # Convert to HSV and get dominant color
            hsv = cv2.cvtColor(bike_roi, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:,:,0])
            
            # Map to color names
            if h_mean < 15: return "Red"
            elif h_mean < 30: return "Orange"
            elif h_mean < 45: return "Yellow" 
            elif h_mean < 75: return "Green"
            elif h_mean < 105: return "Blue"
            elif h_mean < 135: return "Purple"
            else: return "Red"
            
        except:
            return "Unknown"
    
    def _confirm_violation(self, track_id, violation_data, evidence_img):
        """Confirm violation across multiple frames"""
        if track_id not in self.violation_buffer:
            self.violation_buffer[track_id] = []
        
        self.violation_buffer[track_id].append(violation_data)
        
        # Keep only recent frames
        if len(self.violation_buffer[track_id]) > 5:
            self.violation_buffer[track_id] = self.violation_buffer[track_id][-3:]
        
        # Check for consistent violations
        if len(self.violation_buffer[track_id]) >= self.consecutive_frames:
            violation_count = sum(1 for v in self.violation_buffer[track_id] 
                                if not v['has_helmet'] and v['helmet_confidence'] >= self.min_confidence)
            
            if violation_count >= self.consecutive_frames:
                confirmed_violation = self.violation_buffer[track_id][-1]
                del self.violation_buffer[track_id]
                
                print(f"🚨 CONFIRMED VIOLATION: Track {track_id}, Confidence: {confirmed_violation['helmet_confidence']:.3f}")
                return True, confirmed_violation['helmet_confidence'], confirmed_violation, evidence_img
        
        return False, 0.0, {}, None
    
    def cleanup_buffer(self):
        """Clean up old buffer entries"""
        current_time = time.time()
        keys_to_remove = []
        
        for track_id, violations in self.violation_buffer.items():
            if violations and current_time - violations[-1].get('timestamp', 0) > 10.0:
                keys_to_remove.append(track_id)
        
        for key in keys_to_remove:
            del self.violation_buffer[key]

def create_helmet_detector():
    return HelmetDetector()