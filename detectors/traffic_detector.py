from ultralytics import YOLO
import numpy as np
import time
import cv2

try:
    from config import MODEL_PATH, VEHICLE_CLASSES, CONFIDENCE_THRESHOLD, DEVICE, SKIP_FRAMES, MAX_FRAME_WIDTH
except ImportError:
    MODEL_PATH = "yolov8m.pt"
    VEHICLE_CLASSES = [2, 3, 5, 7]
    CONFIDENCE_THRESHOLD = 0.25
    DEVICE = "cpu"
    SKIP_FRAMES = 1
    MAX_FRAME_WIDTH = 1920

class VehicleDetector:
    def __init__(self):
        self.model = self._load_optimized_model()
        self.frame_count = 0
        self.cached_results = None
        self.cached_lane_counts = [0, 0]
        self.cache_time = time.time()
        
    def _load_optimized_model(self):
        """Load optimized YOLO model for better accuracy"""
        try:
            model = YOLO(MODEL_PATH)
            print(f"✅ Loaded YOLO model: {MODEL_PATH}")
            print(f"📊 Confidence threshold: {CONFIDENCE_THRESHOLD}")
            return model
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def detect_vehicles(self, frame):
        """Vehicle detection with optimized accuracy"""
        current_time = time.time()
        
        # Skip frames if configured
        if (self.frame_count % SKIP_FRAMES != 0 and 
            current_time - self.cache_time < 0.1):
            return self._get_cached_results(frame), self.cached_lane_counts
        
        try:
            # Preprocess frame for better accuracy
            processed_frame = self._preprocess_frame(frame)
            
            # Run inference with optimized parameters
            results = self.model(
                processed_frame, 
                classes=VEHICLE_CLASSES,
                conf=CONFIDENCE_THRESHOLD,
                device=DEVICE,
                verbose=False,
                imgsz=640
            )
            
            self.frame_count += 1
            
            # Lane counting with better accuracy
            if results[0].boxes is not None:
                width = frame.shape[1]
                left_count = sum(1 for box in results[0].boxes 
                               if (box.xyxy[0][0] + box.xyxy[0][2]) / 2 < width / 2)
                right_count = len(results[0].boxes) - left_count
                lane_counts = [left_count, right_count]
                
                # Detailed logging for accuracy monitoring
                motorcycle_count = sum(1 for box in results[0].boxes 
                                     if int(box.cls[0]) == 3)
                if motorcycle_count > 0:
                    print(f"🏍️  Detected {motorcycle_count} motorcycle(s)")
                    
                # Cache results
                self._cache_results(results, lane_counts)
                return results, lane_counts
            else:
                self._cache_results(results, [0, 0])
                return results, [0, 0]
                
        except Exception as e:
            print(f"Detection error: {e}")
            return self._get_cached_results(frame), self.cached_lane_counts
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for better accuracy"""
        # Maintain higher resolution for better detection
        if frame.shape[1] > MAX_FRAME_WIDTH:
            scale = MAX_FRAME_WIDTH / frame.shape[1]
            new_width = MAX_FRAME_WIDTH
            new_height = int(frame.shape[0] * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame
    
    def _cache_results(self, results, lane_counts):
        """Cache detection results"""
        self.cached_results = results
        self.cached_lane_counts = lane_counts
        self.cache_time = time.time()
    
    def _get_cached_results(self, frame):
        """Return cached results"""
        if (self.cached_results is not None and 
            time.time() - self.cache_time < 1.0):
            return self.cached_results
        
        # Return empty results
        from ultralytics.engine.results import Results
        return Results(
            orig_img=frame,
            path="",
            names={},
            boxes=None
        )

def create_vehicle_detector():
    return VehicleDetector()