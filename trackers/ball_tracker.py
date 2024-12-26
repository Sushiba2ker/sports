from collections import deque
import numpy as np
import supervision as sv
from .base import BaseTracker
import time

class BallTracker(BaseTracker):
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)
        self.velocity = np.zeros(2)
        self.last_pos = None
        self.last_time = None
        
    def _init_track(self, box, class_id):
        return {
            'box': box,
            'class_id': class_id,
            'velocity': np.zeros(4),
            'last_update': time.time()
        }
        
    def _predict_next_position(self):
        if len(self.buffer) < 2:
            return None
        current_time = time.time()
        if self.last_time is not None:
            dt = current_time - self.last_time
            return self.last_pos + self.velocity * dt
        return None
        
    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        
        if len(xy) > 0:
            current_time = time.time()
            if self.last_pos is not None and self.last_time is not None:
                dt = current_time - self.last_time
                self.velocity = (xy[0] - self.last_pos) / dt
            
            self.last_pos = xy[0]
            self.last_time = current_time
            
        predicted_pos = self._predict_next_position()
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        # Select best detection using both position and velocity
        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        
        if predicted_pos is not None:
            velocity_factor = np.linalg.norm(
                xy - predicted_pos.reshape(1, -1), axis=1
            )
            distances = distances + 0.5 * velocity_factor
            
        index = np.argmin(distances)
        return detections[[index]]
        
    def reset(self) -> None:
        """Reset tracker state"""
        self.buffer.clear() 