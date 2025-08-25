import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self, frame_shape=(720, 1280)):
        self.frame_shape = frame_shape
        self.setup_transformation_points(frame_shape)
    
    def setup_transformation_points(self, frame_shape):
        """Dynamically set transformation points based on frame size"""
        height, width = frame_shape
        
        # Dynamic points based on frame dimensions
        self.src_points = np.float32([
            [width * 0.15, height * 0.95],    # Bottom left
            [width * 0.40, height * 0.65],    # Top left
            [width * 0.60, height * 0.65],    # Top right
            [width * 0.85, height * 0.95]     # Bottom right
        ])
        
        self.dst_points = np.float32([
            [width * 0.25, height],           # Bottom left
            [width * 0.25, 0],                # Top left
            [width * 0.75, 0],                # Top right
            [width * 0.75, height]            # Bottom right
        ])
        
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
    
    def transform(self, frame):
        """Apply perspective transformation with optimized parameters"""
        if frame.shape[:2] != self.frame_shape:
            self.setup_transformation_points(frame.shape[:2])
        
        return cv2.warpPerspective(frame, self.M, (frame.shape[1], frame.shape[0]), 
                                 flags=cv2.INTER_LINEAR)
    
    def inverse_transform(self, frame):
        """Apply inverse perspective transformation"""
        return cv2.warpPerspective(frame, self.M_inv, (frame.shape[1], frame.shape[0]),
                                 flags=cv2.INTER_LINEAR)
    
    def get_lane_regions(self, frame, lane_count=4):
        """Divide transformed frame into lane regions efficiently"""
        transformed = self.transform(frame)
        height, width = transformed.shape[:2]
        lane_width = width // lane_count
        
        lanes = []
        for i in range(lane_count):
            x_start = i * lane_width
            x_end = (i + 1) * lane_width
            lane_region = transformed[:, x_start:x_end]
            lanes.append(lane_region)
        
        return lanes, lane_width
    
    def visualize_points(self, frame):
        """Visualize transformation points for debugging"""
        debug_frame = frame.copy()
        for point in self.src_points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(debug_frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(debug_frame, f"({x},{y})", (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return debug_frame