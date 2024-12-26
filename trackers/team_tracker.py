import numpy as np
import supervision as sv
from .base import BaseTracker
from sports.common.team import TeamClassifier

class TeamTracker(BaseTracker):
    def __init__(self, device: str = 'cuda'):
        self.team_classifier = TeamClassifier(device=device)
        self.tracked_teams = {}
        self.team_history = {}
        self.confidence_threshold = 0.7
        
    def _init_track(self, box, class_id):
        return {
            'team_id': None,
            'confidence': 0,
            'history': []
        }
        
    def update(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            return np.array([])
            
        crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        team_ids, confidences = self.team_classifier.predict_with_confidence(crops)
        
        # Update tracked teams with confidence
        for track_id, (team_id, conf) in zip(detections.tracker_id, zip(team_ids, confidences)):
            if track_id not in self.tracked_teams:
                self.tracked_teams[track_id] = self._init_track(None, None)
                
            if conf > self.confidence_threshold:
                self.tracked_teams[track_id]['team_id'] = team_id
                self.tracked_teams[track_id]['confidence'] = conf
                self.tracked_teams[track_id]['history'].append(team_id)
                
        return team_ids
        
    def reset(self) -> None:
        """Reset tracker state"""
        self.tracked_teams.clear() 