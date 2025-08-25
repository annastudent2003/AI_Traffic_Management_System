import sqlite3
import json
from datetime import datetime
import os
from config import DATA_FOLDER

class ViolationDatabase:
    def __init__(self):
        self.db_path = os.path.join(DATA_FOLDER, "violations.db")
        self._init_database()
    
    def _init_database(self):
        """Initialize database with proper schema"""
        os.makedirs(DATA_FOLDER, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create violations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            track_id INTEGER,
            plate_number TEXT,
            plate_confidence REAL,
            bike_brand TEXT,
            brand_confidence REAL,
            bike_color TEXT,
            helmet_confidence REAL,
            evidence_path TEXT,
            status TEXT DEFAULT 'PENDING',
            reviewer_notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create statistics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            total_vehicles INTEGER DEFAULT 0,
            total_violations INTEGER DEFAULT 0,
            avg_processing_time REAL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_violations_status ON violations(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_statistics_date ON statistics(date)')
        
        conn.commit()
        conn.close()
    
    def add_violation(self, violation_data):
        """Add a new violation to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO violations 
            (timestamp, track_id, plate_number, plate_confidence, bike_brand, 
             brand_confidence, bike_color, helmet_confidence, evidence_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                violation_data.get('timestamp', datetime.now().isoformat()),
                violation_data.get('track_id'),
                violation_data.get('plate_number'),
                violation_data.get('plate_confidence'),
                violation_data.get('bike_brand'),
                violation_data.get('brand_confidence'),
                violation_data.get('bike_color'),
                violation_data.get('helmet_confidence'),
                violation_data.get('evidence_path', '')
            ))
            
            violation_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return True, violation_id
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False, str(e)
    
    def update_violation_status(self, violation_id, status, notes=""):
        """Update violation status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE violations 
            SET status = ?, reviewer_notes = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            ''', (status, notes, violation_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
    
    def get_violations(self, status=None, limit=100, offset=0):
        """Get violations with pagination"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if status:
                cursor.execute('''
                SELECT * FROM violations 
                WHERE status = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                ''', (status, limit, offset))
            else:
                cursor.execute('''
                SELECT * FROM violations 
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                ''', (limit, offset))
            
            violations = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return violations
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return []
    
    def get_violation_stats(self):
        """Get comprehensive violation statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get status counts
            cursor.execute('''
            SELECT status, COUNT(*) as count 
            FROM violations 
            GROUP BY status
            ''')
            
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get daily stats
            cursor.execute('''
            SELECT date, total_violations 
            FROM statistics 
            ORDER BY date DESC 
            LIMIT 7
            ''')
            
            daily_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            
            return {
                'total': sum(status_counts.values()),
                'pending': status_counts.get('PENDING', 0),
                'approved': status_counts.get('APPROVED', 0),
                'rejected': status_counts.get('REJECTED', 0),
                'daily': daily_stats
            }
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return {'total': 0, 'pending': 0, 'approved': 0, 'rejected': 0, 'daily': {}}
    
    def add_daily_stats(self, total_vehicles, total_violations, avg_processing_time):
        """Add or update daily statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().strftime('%Y-%m-%d')
            
            cursor.execute('''
            INSERT OR REPLACE INTO statistics (date, total_vehicles, total_violations, avg_processing_time)
            VALUES (?, ?, ?, ?)
            ''', (today, total_vehicles, total_violations, avg_processing_time))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False

# Global database instance
violation_db = ViolationDatabase()