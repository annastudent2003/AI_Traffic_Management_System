import csv
import os
from datetime import datetime

class ExcelWriter:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        os.makedirs(data_folder, exist_ok=True)
        self.csv_file = os.path.join(data_folder, "violations_export.csv")
    
    def export_violations(self, violations_data):
        """Export violations data to CSV format"""
        try:
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'Timestamp', 'Plate Number', 'Bike Brand', 
                    'Bike Color', 'Confidence', 'Status'
                ])
                
                for violation in violations_data:
                    writer.writerow([
                        violation.get('timestamp', ''),
                        violation.get('plate_number', 'UNKNOWN'),
                        violation.get('bike_brand', 'UNKNOWN'),
                        violation.get('bike_color', 'UNKNOWN'),
                        violation.get('helmet_confidence', 0.0),
                        violation.get('status', 'PENDING')
                    ])
            
            return True, f"Exported {len(violations_data)} violations to {self.csv_file}"
        except Exception as e:
            return False, f"Export failed: {str(e)}"