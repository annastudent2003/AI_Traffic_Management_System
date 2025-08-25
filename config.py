# VIDEO INPUT 
VIDEO_PATH = "no_helmet_biker.webm"    
USE_WEBCAM = False
CAMERA_ID = 0
CAMERA_RESOLUTION = (640, 480)

# DISPLAY 
DISPLAY_WIDTH = 800                 
DISPLAY_HEIGHT = 600                
WINDOW_NAME = "AI Traffic System"   

# CHANGE THESE VALUES:
HELMET_MIN_CONFIDENCE = 0.3  # Lower confidence threshold
HELMET_CONSECUTIVE_FRAMES = 1  # Only need 1 frame for violation
HELMET_DEBUG_MODE = True  # See what's happening
DATA_COLLECTION_MODE = False  # Turn off for now
# YOLO MODEL - OPTIMIZED FOR ACCURACY
MODEL_PATH = "models/yolov8m.pt"  # Medium model for better accuracy          
VEHICLE_CLASSES = [2, 3, 5, 7]      
CONFIDENCE_THRESHOLD = 0.25  # Lower threshold to catch more objects        
DEVICE = "cpu"                     

# LANE PROCESS
LANE_DIVIDER_COLOR = (0, 255, 0)   
LANE_DIVIDER_THICKNESS = 2    
LEFT_LANE_COLOR = (255, 0, 0)      
RIGHT_LANE_COLOR = (0, 0, 255)     
LANE_COUNT = 2

# TRAFFIC SIGNAL
SIGNAL_COLOR_GREEN = (0, 255, 0)  
SIGNAL_COLOR_RED = (0, 0, 255)     
SIGNAL_FONT_SCALE = 0.8            
SIGNAL_FONT_THICKNESS = 2           

# TEXT DISPLAY 
FONT_FACE = 0               
FONT_SCALE = 0.7                   
FONT_THICKNESS = 2                 
TEXT_COLOR_WHITE = (255, 255, 255) 

# SYSTEM 
EXIT_KEY = "q"                     
CONSOLE_LOGGING = True             
VIDEO_RESTART = True
LOG_LEVEL = "INFO"
SKIP_FRAMES = 1  # Process every frame for maximum accuracy

# HELMET DETECTION - OPTIMIZED SETTINGS
HELMET_MODEL_PATH = "models/helmet_detection_best.pt"  # Custom trained model
HELMET_MIN_CONFIDENCE = 0.6  # Higher confidence for trained model
HELMET_CONSECUTIVE_FRAMES = 2  # Faster confirmation
HELMET_DEBUG_MODE = True
DATA_COLLECTION_MODE = True  # Set to True to collect training data
# Make sure these are set correctly:
DATA_COLLECTION_MODE = True
TRAINING_DATA_DIR = "training_data"
# PERFORMANCE OPTIMIZATION FOR ACCURACY
MAX_FRAME_WIDTH = 1920  # Higher resolution for better detection


EVIDENCE_QUALITY = 95
# VIOLATION SETTINGS
VIOLATION_COOLDOWN_TIME = 5

# DIRECTORIES
DATA_FOLDER = "data"
VIOLATIONS_DIR = "violations"
MODELS_DIR = "models"
TRAINING_DATA_DIR = "training_data"

# TRAFFIC LIGHT TIMING
MIN_GREEN_TIME = 5
MAX_GREEN_TIME = 30
YELLOW_TIME = 3

# BRAND RECOGNITION
MOTORCYCLE_BRANDS = ["Honda", "Yamaha", "Suzuki", "Bajaj", "TVS", "Unknown"]