# AI Traffic Management System

This is a deep learning-based traffic management project integrating traffic detection and helmet violation detection. I developed several parts of the system and will provide all relevant code for replication and experimentation.

## Features

### Traffic Detection:

1) Monitors all lanes and counts vehicles.
2) Determines which lane needs a green light based on vehicle density.
3) Dynamically allocates green signals to optimize traffic flow.

### Helmet Detection & Violation Logging:

1) Scans all bikers on the road.
2) Detects riders without helmets and logs their information for authorities.
3) Captures bike brand, color, and number plate, storing data in CSV or database for later review.



## Project Structure:

#### AITrafficManagementSystem/
├── main.py           

├── config.py     

├── database.py            

├── detectors/
│   ├── __init__.py
│   ├── traffic_detector.py # Vehicle detection module
│   └── helmet_detector.py  # Helmet detection module
├── logic/
│   ├── __init__.py
│   ├── traffic_light.py    # Traffic light management logic
│   └── violation_manager.py # Handles violations and database logging
├── utils/
│   ├── __init__.py
│   ├── plate_recognizer.py     # Improved OCR for license plates
│   ├── brand_recognizer.py     # Detect bike brand
│   ├── ml_brand_recognizer.py  # ML-based brand recognition
│   └── perspective_transform.py # Image perspective utilities
├── models/
│   ├── yolov8l-traffic.pt  # Pre-trained YOLOv8 traffic model
│   └── yolov8s-helmet.pt   # YOLOv8 helmet detection model (trainable)
├── data/                    # Database storage
└── violations/              # Stores images of violations

