import os
from datetime import datetime
import cv2

# Import database with fallback
try:
    from database import violation_db
    USE_DATABASE = True
except ImportError:
    USE_DATABASE = False
    print("⚠️  Database not available, using CSV fallback")

try:
    from config import VIOLATIONS_DIR, EVIDENCE_QUALITY
except ImportError:
    VIOLATIONS_DIR = "violations"
    EVIDENCE_QUALITY = 95

class ViolationManager:
    def __init__(self):
        self._ensure_directories_exist()
    
    def _ensure_directories_exist(self):
        os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    
    def log_violation(self, violation_data, evidence_image=None):
        timestamp = datetime.now().isoformat()
        
        # Save evidence image
        evidence_path = ""
        if evidence_image is not None:
            evidence_path = self._save_evidence_image(evidence_image, timestamp)
        
        # Add timestamp and evidence path to violation data
        violation_data['timestamp'] = timestamp
        violation_data['evidence_path'] = evidence_path
        
        if USE_DATABASE:
            # Save to database
            success, result = violation_db.add_violation(violation_data)
            if success:
                print(f"✅ Violation logged to database: {result}")
                return True, f"DB_{result}"
            else:
                print(f"❌ Database error: {result}")
                # Fallback to CSV
                return self._log_violation_csv(violation_data, evidence_path)
        else:
            # Fallback to CSV
            return self._log_violation_csv(violation_data, evidence_path)
    
    def _save_evidence_image(self, image, timestamp):
        try:
            filename = f"violation_{timestamp.replace(':', '-')}.jpg"
            filepath = os.path.join(VIOLATIONS_DIR, filename)
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, EVIDENCE_QUALITY])
            return filepath
        except Exception as e:
            print(f"Failed to save evidence image: {e}")
            return ""
    
    def get_violation_stats(self):
        if USE_DATABASE:
            return violation_db.get_violation_stats()
        else:
            # CSV fallback
            return {'total': 0, 'pending': 0, 'approved': 0, 'rejected': 0}
    
    def _log_violation_csv(self, violation_data, evidence_path):
        """Fallback CSV logging"""
        try:
            import csv
            from config import DATA_FOLDER
            
            csv_file = os.path.join(DATA_FOLDER, "violations.csv")
            os.makedirs(DATA_FOLDER, exist_ok=True)
            
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                if not file_exists:
                    writer.writerow([
                        'Timestamp', 'Track_ID', 'Plate_Number', 'Plate_Confidence',
                        'Bike_Brand', 'Brand_Confidence', 'Bike_Color', 
                        'Helmet_Confidence', 'Evidence_Path', 'Status'
                    ])
                
                writer.writerow([
                    violation_data['timestamp'],
                    violation_data.get('track_id', ''),
                    violation_data.get('plate_number', ''),
                    violation_data.get('plate_confidence', 0.0),
                    violation_data.get('bike_brand', ''),
                    violation_data.get('brand_confidence', 0.0),
                    violation_data.get('bike_color', ''),
                    violation_data.get('helmet_confidence', 0.0),
                    evidence_path,
                    'PENDING'
                ])
            
            print(f"✅ Violation logged to CSV: {violation_data.get('track_id', '')}")
            return True, "CSV_ENTRY"
            
        except Exception as e:
            print(f"❌ CSV logging failed: {e}")
            return False, str(e)

def create_violation_manager():
    return ViolationManager()