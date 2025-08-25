import cv2
import time
import sys
import os
import numpy as np
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Global variables
ADVANCED_MODE = False
PERSPECTIVE_AVAILABLE = False

# Import configuration with fallback values
try:
    from config import (VIDEO_PATH, USE_WEBCAM, CAMERA_ID, CAMERA_RESOLUTION,
                       DISPLAY_WIDTH, DISPLAY_HEIGHT, CONSOLE_LOGGING, EXIT_KEY,
                       VIDEO_RESTART, LOG_LEVEL, LANE_COUNT, SKIP_FRAMES,
                       DATA_FOLDER, VIOLATIONS_DIR, EVIDENCE_QUALITY,
                       HELMET_MIN_CONFIDENCE, HELMET_MODEL_PATH)
except ImportError as e:
    print(f"⚠️  Config import error: {e}")
    print("🔄 Using default configuration values")
    
    # Fallback values
    VIDEO_PATH = "no_helmet_biker.webm"
    USE_WEBCAM = False
    CAMERA_ID = 0
    CAMERA_RESOLUTION = (640, 480)
    DISPLAY_WIDTH = 800
    DISPLAY_HEIGHT = 600
    CONSOLE_LOGGING = True
    EXIT_KEY = "q"
    VIDEO_RESTART = True
    LOG_LEVEL = "INFO"
    LANE_COUNT = 2
    SKIP_FRAMES = 1
    DATA_FOLDER = "data"
    VIOLATIONS_DIR = "violations"
    EVIDENCE_QUALITY = 95
    HELMET_MIN_CONFIDENCE = 0.6
    HELMET_MODEL_PATH = "models/yolov8s-helmet.pt"

def download_helmet_model():
    """Download a pre-trained helmet detection model if not available"""
    model_path = HELMET_MODEL_PATH
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    
    if not os.path.exists(model_path):
        print("⚠️  No helmet detection model found. Attempting to download...")
        os.makedirs("models", exist_ok=True)
        
        try:
            import requests
            from tqdm import tqdm
            
            print("📥 Downloading YOLOv8s model (general purpose)...")
            response = requests.get(model_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Model downloaded to {model_path}")
            print("⚠️  Note: This is a general YOLOv8s model, not specialized for helmet detection")
            print("🔄 For better results, train a custom helmet detection model")
            
        except ImportError:
            print("❌ 'requests' or 'tqdm' package not available. Please install: pip install requests tqdm")
            return False
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            print("🔄 Using enhanced detection without specialized model")
            return False
    
    return True

# Import components with error handling
try:
    from detectors.traffic_detector import create_vehicle_detector
    from detectors.helmet_detector import create_helmet_detector
    from logic.traffic_light import TrafficLightSystem
    from logic.violation_manager import create_violation_manager
    
    # Try to import PerspectiveTransformer
    try:
        from utils.perspective_transform import PerspectiveTransformer
        PERSPECTIVE_AVAILABLE = True
    except ImportError:
        print("⚠️  PerspectiveTransformer not available")
        PERSPECTIVE_AVAILABLE = False
        # Create a simple fallback class
        class PerspectiveTransformer:
            def get_lane_regions(self, frame, lane_count=4):
                height, width = frame.shape[:2]
                lane_width = width / lane_count
                return [frame] * lane_count, lane_width
            def transform(self, frame):
                return frame
            def inverse_transform(self, frame):
                return frame
    
    ADVANCED_MODE = True
    print("✅ All advanced modules loaded successfully")
    
except ImportError as e:
    print(f"⚠️  Advanced modules not available: {e}")
    print("⚠️  Running in basic mode")
    
    # Create fallback classes
    class PerspectiveTransformer:
        def get_lane_regions(self, frame, lane_count=4):
            height, width = frame.shape[:2]
            lane_width = width / lane_count
            return [frame] * lane_count, lane_width
        def transform(self, frame):
            return frame
        def inverse_transform(self, frame):
            return frame

    class TrafficLightSystem:
        def __init__(self):
            self.current_signal = "LEFT"
            self.signal_start_time = time.time()
            self.violation_detected = False
            self.violation_lane = None
            self.violation_time = 0
        
        def decide_traffic_signal(self, lane_vehicle_counts, violation_data=None):
            left_count, right_count = lane_vehicle_counts
            current_time = time.time()
            
            # Check if we need to handle a violation
            if violation_data and violation_data.get('has_violation', False):
                self.violation_detected = True
                self.violation_lane = violation_data.get('lane', 'LEFT')
                self.violation_time = current_time
                print(f"🚨 VIOLATION DETECTED in {self.violation_lane} lane!")
            
            # Handle violation priority (stop traffic in violation lane)
            if self.violation_detected and current_time - self.violation_time < 10.0:
                if self.violation_lane == "LEFT":
                    signal_text = "RED: LEFT (VIOLATION)"
                    signal_color = (0, 0, 255)
                    new_signal = "LEFT"
                else:
                    signal_text = "RED: RIGHT (VIOLATION)"
                    signal_color = (0, 0, 255)
                    new_signal = "RIGHT"
                
                # Reset violation after handling period
                if current_time - self.violation_time > 8.0:
                    self.violation_detected = False
                    print("⚠️  Violation handling complete")
            
            # Normal traffic flow decision
            elif left_count > right_count + 2:
                signal_text = "GREEN: LEFT"
                signal_color = (0, 255, 0)
                new_signal = "LEFT"
            elif right_count > left_count + 2:
                signal_text = "GREEN: RIGHT"
                signal_color = (0, 255, 0)
                new_signal = "RIGHT"
            else:
                # Maintain current signal if counts are balanced
                new_signal = self.current_signal
                signal_text = f"GREEN: {new_signal}"
                signal_color = (0, 255, 0)
            
            # Check if signal needs to change
            if new_signal != self.current_signal:
                self.current_signal = new_signal
                self.signal_start_time = current_time
            
            return signal_text, signal_color, new_signal

    def create_helmet_detector():
        class DummyHelmetDetector:
            def detect_helmet_violation(self, frame, bbox, track_id):
                return False, 0.0, {}, None
            def cleanup_buffer(self):
                pass
        return DummyHelmetDetector()

    def create_violation_manager():
        class DummyViolationManager:
            def log_violation(self, violation_data, evidence_img):
                return False, "N/A"
            def get_violation_stats(self):
                return {'total': 0, 'pending': 0, 'approved': 0, 'rejected': 0}
        return DummyViolationManager()

class TrafficManagementSystem:
    def __init__(self):
        # Download helmet model if needed
        if ADVANCED_MODE:
            download_helmet_model()
        
        # Initialize core components
        self.vehicle_detector = create_vehicle_detector()
        self.traffic_system = TrafficLightSystem()
        
        # Initialize perspective transformer (with fallback)
        self.perspective_transformer = PerspectiveTransformer()
        
        # Initialize advanced components if available
        if ADVANCED_MODE:
            try:
                self.helmet_detector = create_helmet_detector()
                self.violation_manager = create_violation_manager()
                print("✅ Helmet detection and violation management enabled")
            except Exception as e:
                print(f"⚠️  Failed to initialize advanced features: {e}")
                self.helmet_detector = None
                self.violation_manager = None
        else:
            # Create dummy components for basic mode
            self.helmet_detector = None
            self.violation_manager = None
        
        # System state
        self.cap = None
        self.current_signal = None
        self.signal_start_time = time.time()
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.fps = 0
        
        # Statistics
        self.total_vehicles = 0
        self.total_violations = 0
        self.lane_stats = [0] * LANE_COUNT
        
        # Create necessary directories
        os.makedirs(DATA_FOLDER, exist_ok=True)
        os.makedirs(VIOLATIONS_DIR, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
    def initialize_video_capture(self):
        """Initialize video capture from webcam or file"""
        try:
            if USE_WEBCAM:
                self.cap = cv2.VideoCapture(CAMERA_ID)
                if CAMERA_RESOLUTION:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                source_name = "webcam"
            else:
                self.cap = cv2.VideoCapture(VIDEO_PATH)
                source_name = VIDEO_PATH
                
            if not self.cap.isOpened():
                # Try to use webcam as fallback
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception(f"Cannot open video source: {source_name}")
                source_name = "webcam (fallback)"
                    
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Video source: {source_name}")
            print(f"📏 Resolution: {width}x{height}")
            if fps > 0:
                print(f"🎞️  FPS: {fps:.1f}")
            
            return width, height
            
        except Exception as e:
            print(f"❌ Failed to initialize video capture: {e}")
            # Create a dummy black frame for testing
            print("🔄 Using test pattern instead")
            return 640, 480
    
    def process_frame(self, frame):
        """Process a single frame"""
        # If no real camera, create a test pattern
        if frame is None:
            frame = self._create_test_pattern()
        
        # Calculate FPS
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            self.fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        # Skip frames if configured
        if SKIP_FRAMES > 1 and self.frame_count % SKIP_FRAMES != 0:
            return frame, (0, 0), 0
        
        # Detect vehicles
        results = self.vehicle_detector.detect_vehicles(frame)
        
        # Get lane counts
        if hasattr(results, '__len__') and len(results) > 1:
            lane_vehicle_counts = results[1]
            # Extract left and right counts if it's a list
            if isinstance(lane_vehicle_counts, (list, tuple)) and len(lane_vehicle_counts) >= 2:
                left_count = lane_vehicle_counts[0]
                right_count = lane_vehicle_counts[1]
            else:
                # Fallback if not a list
                left_count = lane_vehicle_counts
                right_count = 0
        else:
            # Fallback: simple left/right counting
            width = frame.shape[1]
            if hasattr(results, 'boxes') and results[0].boxes is not None:
                left_count = sum(1 for box in results[0].boxes if (box.xyxy[0][0] + box.xyxy[0][2]) / 2 < width / 2)
                right_count = len(results[0].boxes) - left_count
            else:
                left_count, right_count = 0, 0
        
        # Ensure counts are integers
        left_count = int(left_count) if left_count is not None else 0
        right_count = int(right_count) if right_count is not None else 0
        
        self.total_vehicles += (left_count + right_count)
        
        # Update lane statistics
        for i, count in enumerate([left_count, right_count]):
            if i < len(self.lane_stats):
                self.lane_stats[i] += count
        
        # Detect helmet violations if available
        violation_count = 0
        violation_data = None
        
        if ADVANCED_MODE and self.helmet_detector:
            violation_count, violation_data = self._process_helmet_violations(frame, results)
        
        # Decide traffic signal - now includes violation data
        signal_text, signal_color, new_signal = self.traffic_system.decide_traffic_signal(
            (left_count, right_count), violation_data
        )
        
        if new_signal != self.current_signal:
            self.current_signal = new_signal
            self.signal_start_time = time.time()
        
        # Visualize results
        annotated_frame = self._visualize_results(frame, results, left_count, right_count, 
                                                signal_text, signal_color, violation_count)
        
        return annotated_frame, (left_count, right_count), violation_count
    
    def _process_helmet_violations(self, frame, results):
        """Process helmet violations with better integration"""
        if not ADVANCED_MODE or not hasattr(results, 'boxes') or results[0].boxes is None:
            return 0, None
            
        violation_count = 0
        violation_data = None
        
        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0])
            
            # Only process motorcycles (class 3 in COCO dataset)
            if class_id == 3:  # Motorcycle class
                is_violation, confidence, violation_info, evidence_img = self.helmet_detector.detect_helmet_violation(
                    frame, box.xyxy[0], i
                )
                
                if is_violation and confidence > self.helmet_detector.min_confidence:
                    # Determine which lane the violation is in
                    x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                    width = frame.shape[1]
                    violation_lane = "LEFT" if x_center < width / 2 else "RIGHT"
                    
                    # Draw violation indicator on frame
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box for violation
                    cv2.putText(frame, "NO HELMET!", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Prepare violation data for traffic signal control
                    violation_data = {
                        'has_violation': True,
                        'lane': violation_lane,
                        'confidence': confidence,
                        'timestamp': time.time()
                    }
                    
                    # Log violation
                    success, violation_id = self.violation_manager.log_violation(violation_info, evidence_img)
                    if success:
                        violation_count += 1
                        self.total_violations += 1
                        if CONSOLE_LOGGING:
                            print(f"🚨 VIOLATION #{violation_id}: Track {violation_info.get('track_id', 'N/A')} - Confidence: {confidence:.2f}")
        
        return violation_count, violation_data
    
    def _visualize_results(self, frame, results, left_count, right_count, signal_text, 
                          signal_color, violation_count):
        """Visualize detection results on frame"""
        # Draw bounding boxes
        if hasattr(results, 'boxes') and results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = results[0].boxes.conf[i] if i < len(results[0].boxes.conf) else 0.5
                class_id = int(results[0].boxes.cls[i]) if i < len(results[0].boxes.cls) else 2
                
                # Draw box
                color = (0, 255, 0)  # Green for vehicles
                if class_id == 3:  # Motorcycle
                    color = (0, 165, 255)  # Orange for motorcycles
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"Class {class_id}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw lane dividers and counts
        height, width = frame.shape[:2]
        cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Left: {left_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Right: {right_count}", (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw traffic signal
        cv2.putText(frame, signal_text, (width//2 - 120, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, signal_color, 3)
        
        # Draw violation count
        cv2.putText(frame, f"Violations: {violation_count}", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (width - 150, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (width - 150, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _create_test_pattern(self):
        """Create a test pattern when no video source is available"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "NO VIDEO SOURCE", (100, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'Q' to quit", (150, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return frame
    
    def run(self):
        """Main system loop"""
        try:
            print("🚦 Starting AI Traffic Management System...")
            print(f"🔧 Mode: {'Advanced' if ADVANCED_MODE else 'Basic'}")
            
            # Initialize video capture
            width, height = self.initialize_video_capture()
            
            if CONSOLE_LOGGING:
                print("🎮 Controls:")
                print("   - Press 'q' to quit")
                print("   - Press 'p' to pause")
                print("   - Press 'd' to debug helmet detection")
                print("=" * 50)
                print("LEFT | RIGHT | SIGNAL | VIOLATIONS")
                print("=" * 50)
            
            paused = False
            
            while True:
                if not paused:
                    # Read frame
                    ret, frame = self.cap.read()
                    if not ret:
                        if VIDEO_RESTART and not USE_WEBCAM:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            frame = None  # Will use test pattern
                    
                    self.frame_count += 1
                    
                    # Process frame
                    processed_frame, lane_counts, violation_count = self.process_frame(frame)
                    
                    # Display frame
                    display_frame = cv2.resize(processed_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                    cv2.imshow("AI Traffic Management System", display_frame)
                    
                    # Console logging
                    if CONSOLE_LOGGING and self.frame_count % 30 == 0:
                        left_count, right_count = lane_counts if isinstance(lane_counts, (list, tuple)) and len(lane_counts) >= 2 else (0, 0)
                        signal_text = "GREEN: LEFT" if self.current_signal == "LEFT" else "GREEN: RIGHT"
                        print(f"{left_count:4d} | {right_count:6d} | {signal_text:12} | {violation_count:>9}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord(EXIT_KEY):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("⏸️  Paused" if paused else "▶️  Resumed")
                elif key == ord('d') and ADVANCED_MODE and self.helmet_detector:
                    print("🔧 Debugging helmet detection...")
                    self._debug_helmet_detection(frame.copy())
                    
        except Exception as e:
            print(f"❌ System error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.cleanup()
    
    def _debug_helmet_detection(self, frame):
        """Debug method to test helmet detection"""
        print("🧪 Testing helmet detection on current frame...")
        
        if frame is None:
            print("❌ No frame available for debugging")
            return
        
        # Create a test motorcycle ROI (adjust these coordinates as needed)
        height, width = frame.shape[:2]
        test_bbox = [width//4, height//4, 3*width//4, 3*height//4]
        
        is_violation, confidence, violation_data, evidence_img = self.helmet_detector.detect_helmet_violation(
            frame, test_bbox, 999
        )
        
        print(f"🧪 Test result: Violation={is_violation}, Confidence={confidence:.2f}")
        
        # Draw test bounding box
        x1, y1, x2, y2 = map(int, test_bbox)
        color = (0, 0, 255) if is_violation else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"Test: Violation={is_violation}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show debug frame
        debug_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("Helmet Detection Debug", debug_frame)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyWindow("Helmet Detection Debug")
    
    def save_statistics(self):
        """Save system statistics to file"""
        try:
            stats_file = os.path.join(DATA_FOLDER, "statistics.txt")
            with open(stats_file, "w") as f:
                f.write("AI Traffic Management System - Statistics\n")
                f.write("=" * 50 + "\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total runtime: {time.time() - self.start_time:.1f}s\n")
                f.write(f"Frames processed: {self.frame_count}\n")
                if hasattr(self, 'fps'):
                    f.write(f"Average FPS: {self.fps:.1f}\n")
                f.write(f"Total vehicles detected: {self.total_vehicles}\n")
                f.write(f"Total violations: {self.total_violations}\n\n")
                
                f.write("Lane Statistics:\n")
                for i, count in enumerate(self.lane_stats):
                    f.write(f"  Lane {i+1}: {count} vehicles\n")
                
                if ADVANCED_MODE and self.violation_manager:
                    try:
                        violation_stats = self.violation_manager.get_violation_stats()
                        f.write(f"\nViolation Statistics:\n")
                        f.write(f"  Total: {violation_stats['total']}\n")
                        f.write(f"  Pending review: {violation_stats['pending']}\n")
                        f.write(f"  Approved: {violation_stats['approved']}\n")
                        f.write(f"  Rejected: {violation_stats['rejected']}\n")
                    except:
                        f.write(f"\nViolation Statistics: Not available\n")
            
            print(f"✅ Statistics saved to {stats_file}")
            
        except Exception as e:
            print(f"❌ Failed to save statistics: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Clean up helmet detector buffer
        if ADVANCED_MODE and self.helmet_detector:
            self.helmet_detector.cleanup_buffer()
        
        # Save final statistics
        self.save_statistics()
        
        if CONSOLE_LOGGING:
            total_time = time.time() - self.start_time
            print("\n" + "=" * 50)
            print("🛑 System shutdown complete")
            print(f"📊 Total frames processed: {self.frame_count}")
            print(f"⏱️  Total runtime: {total_time:.1f}s")
            print(f"🚗 Total vehicles detected: {self.total_vehicles}")
            print(f"🚨 Total violations: {self.total_violations}")
            print("=" * 50)

def main():
    """Main entry point"""
    try:
        system = TrafficManagementSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()