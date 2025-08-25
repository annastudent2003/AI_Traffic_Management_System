import cv2
import numpy as np
import time

try:
    from config import (LANE_DIVIDER_COLOR, LANE_DIVIDER_THICKNESS,
                       LEFT_LANE_COLOR, RIGHT_LANE_COLOR,
                       SIGNAL_COLOR_GREEN, SIGNAL_COLOR_RED,
                       FONT_FACE, FONT_SCALE, FONT_THICKNESS,
                       SIGNAL_FONT_SCALE, SIGNAL_FONT_THICKNESS,
                       MIN_GREEN_TIME, MAX_GREEN_TIME, YELLOW_TIME)
except ImportError:
    LANE_DIVIDER_COLOR = (0, 255, 0)
    LANE_DIVIDER_THICKNESS = 2
    LEFT_LANE_COLOR = (255, 0, 0)
    RIGHT_LANE_COLOR = (0, 0, 255)
    SIGNAL_COLOR_GREEN = (0, 255, 0)
    SIGNAL_COLOR_RED = (0, 0, 255)
    FONT_FACE = 0
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    SIGNAL_FONT_SCALE = 0.8
    SIGNAL_FONT_THICKNESS = 2
    MIN_GREEN_TIME = 5
    MAX_GREEN_TIME = 30
    YELLOW_TIME = 3

class TrafficLightSystem:
    def __init__(self):
        self.current_signal = "LEFT"
        self.signal_start_time = time.time()
        self.violation_detected = False
        self.violation_lane = None
        self.violation_time = 0
    
    def decide_traffic_signal(self, lane_vehicle_counts, violation_data=None):
        """Decide traffic signal based on vehicle counts and violations"""
        # FIX: Handle both list and tuple inputs for lane counts
        if isinstance(lane_vehicle_counts, (list, tuple)) and len(lane_vehicle_counts) >= 2:
            left_count = lane_vehicle_counts[0] if lane_vehicle_counts[0] is not None else 0
            right_count = lane_vehicle_counts[1] if lane_vehicle_counts[1] is not None else 0
        else:
            # Fallback if input is not a list/tuple
            left_count = lane_vehicle_counts if lane_vehicle_counts is not None else 0
            right_count = 0
        
        # Ensure counts are integers
        left_count = int(left_count)
        right_count = int(right_count)
        
        current_time = time.time()
        signal_duration = current_time - self.signal_start_time
        
        # Check if we need to handle a violation
        if violation_data and violation_data.get('has_violation', False):
            self.violation_detected = True
            self.violation_lane = violation_data.get('lane', 'LEFT')
            self.violation_time = current_time
            print(f"🚨 VIOLATION DETECTED in {self.violation_lane} lane!")
        
        # Handle violation priority (stop traffic in violation lane)
        if self.violation_detected and current_time - self.violation_time < 10.0:  # 10-second violation handling
            if self.violation_lane == "LEFT":
                signal_text = "RED: LEFT (VIOLATION)"
                signal_color = SIGNAL_COLOR_RED
                new_signal = "LEFT"
            else:
                signal_text = "RED: RIGHT (VIOLATION)"
                signal_color = SIGNAL_COLOR_RED
                new_signal = "RIGHT"
            
            # Reset violation after handling period
            if current_time - self.violation_time > 8.0:
                self.violation_detected = False
                print("⚠️  Violation handling complete")
        
        # Normal traffic flow decision
        elif left_count > right_count + 2:
            signal_text = "GREEN: LEFT"
            signal_color = SIGNAL_COLOR_GREEN
            new_signal = "LEFT"
        elif right_count > left_count + 2:
            signal_text = "GREEN: RIGHT"
            signal_color = SIGNAL_COLOR_GREEN
            new_signal = "RIGHT"
        else:
            # Maintain current signal if counts are balanced
            new_signal = self.current_signal
            signal_text = f"GREEN: {new_signal}"
            signal_color = SIGNAL_COLOR_GREEN
        
        # Check if signal needs to change
        if new_signal != self.current_signal:
            self.current_signal = new_signal
            self.signal_start_time = current_time
        
        return signal_text, signal_color, new_signal
    
    @staticmethod
    def visualize_traffic_data(frame, left_count, right_count, signal_text, signal_color, frame_width, frame_height):
        # FIX: Handle list inputs for counts
        if isinstance(left_count, (list, tuple)) and len(left_count) > 0:
            left_count = left_count[0]
        if isinstance(right_count, (list, tuple)) and len(right_count) > 0:
            right_count = right_count[0]
        
        cv2.line(frame, (frame_width//2, 0), (frame_width//2, frame_height), 
                LANE_DIVIDER_COLOR, LANE_DIVIDER_THICKNESS)
        
        cv2.putText(frame, f"Left: {left_count}", (10, 30), 
                   FONT_FACE, FONT_SCALE, LEFT_LANE_COLOR, FONT_THICKNESS)
        cv2.putText(frame, f"Right: {right_count}", (frame_width - 150, 30), 
                   FONT_FACE, FONT_SCALE, RIGHT_LANE_COLOR, FONT_THICKNESS)
        
        cv2.putText(frame, signal_text, (frame_width//2 - 120, 60), 
                   FONT_FACE, SIGNAL_FONT_SCALE, signal_color, SIGNAL_FONT_THICKNESS)
        
        return frame
    
    @staticmethod
    def draw_traffic_light_overlay(frame, signal, position=(50, 50)):
        cv2.rectangle(frame, (position[0], position[1]), (position[0] + 60, position[1] + 160), (30, 30, 30), -1)
        
        left_color = SIGNAL_COLOR_GREEN if signal == "LEFT" else (0, 70, 0)
        right_color = SIGNAL_COLOR_GREEN if signal == "RIGHT" else (0, 70, 0)
        
        cv2.circle(frame, (position[0] + 30, position[1] + 30), 15, left_color, -1)
        cv2.circle(frame, (position[0] + 30, position[1] + 90), 15, right_color, -1)
        
        return frame
    
    @staticmethod
    def calculate_traffic_density(left_count, right_count, max_vehicles=20):
        # FIX: Handle list inputs for counts
        if isinstance(left_count, (list, tuple)) and len(left_count) > 0:
            left_count = left_count[0]
        if isinstance(right_count, (list, tuple)) and len(right_count) > 0:
            right_count = right_count[0]
        
        left_density = min(left_count / max_vehicles, 1.0) if left_count is not None else 0
        right_density = min(right_count / max_vehicles, 1.0) if right_count is not None else 0
        return left_density, right_density