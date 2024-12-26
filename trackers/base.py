from abc import ABC, abstractmethod
import supervision as sv
import numpy as np
import time

class BaseTracker(ABC):
    """Abstract base class for all trackers"""
    
    @abstractmethod
    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """Update tracker with new detections and frame"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state"""
        pass
        
    @abstractmethod
    def _init_track(self, box: np.ndarray, class_id: int) -> dict:
        """Initialize a new track"""
        pass 