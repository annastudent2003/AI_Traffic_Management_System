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

### TrafficManagementSystem/ 

├── main.py                  # Main entry point  
├── config.py                # Configuration file  
├── database.py              # SQLite database integration  

├── detectors/  
│   ├── __init__.py          # Package initializer  
│   ├── traffic_detector.py  # Traffic detection logic  
│   └── helmet_detector.py   # Helmet detection logic  

