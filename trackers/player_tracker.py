import numpy as np
import supervision as sv
import cv2
import time
from .base import BaseTracker
from collections import deque

class PlayerTracker(BaseTracker):
    def __init__(self, field_width=1920, field_height=1080):
        self.field_width = field_width
        self.field_height = field_height
        self.tracks = {}
        self.next_id = {
            'player': 1,
            'goalkeeper': 21,
            'referee': 30
        }
        self.max_players = 20
        self.max_distance = 100  # Pixels
        self.lost_tracks = {}
        self.max_lost_frames = 30
        self.appearance_features = {}
        self.position_history = {}
        self.velocity_history = {}
        self.feature_memory = 50
        self.confidence_scores = {}
        self.min_confidence_threshold = 0.3

    def _calculate_distance(self, pos1, pos2):
        """Tính khoảng cách Euclidean giữa hai điểm"""
        return np.sqrt(((pos1[0] - pos2[0]) * self.field_width) ** 2 +
                      ((pos1[1] - pos2[1]) * self.field_height) ** 2)

    def _get_next_id(self, class_id):
        """Lấy ID tiếp theo cho class tương ứng"""
        if class_id == 'goalkeeper':
            return 21 if 21 not in self.tracks and 21 not in self.lost_tracks else 22
        elif class_id == 'referee':
            return self.next_id['referee']
        else:  # player
            for i in range(1, self.max_players + 1):
                if i not in self.tracks and i not in self.lost_tracks:
                    return i
            return self.next_id['player']

    def _extract_features(self, frame, bbox):
        """Trích xuất đặc trưng từ vùng ảnh của đối tượng"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                return None
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        except Exception as e:
            print(f"Lỗi khi trích xuất đặc trưng: {e}")
            return None

    def _calculate_velocity(self, current_pos, last_pos):
        """Tính vận tốc của đối tượng"""
        if last_pos is None:
            return np.zeros(2)
        return np.array(current_pos) - np.array(last_pos)

    def _match_features(self, current_features, track_id):
        """So khớp đặc trưng với track đã lưu"""
        if track_id not in self.appearance_features or current_features is None:
            return 0
        stored_features = self.appearance_features[track_id]
        try:
            similarity = cv2.compareHist(current_features, stored_features, cv2.HISTCMP_CORREL)
            return max(0, similarity)
        except Exception as e:
            print(f"Lỗi khi so khớp đặc trưng: {e}")
            return 0

    def update(self, detections, frame):
        """Cập nhật tracking với các detections mới"""
        current_tracks = {}
        used_detections = set()

        # Bước 1: Cập nhật các track hiện có
        for track_id, track in self.tracks.items():
            best_detection_idx = None
            max_score = float('-inf')

            for i in range(len(detections)):
                if i in used_detections:
                    continue

                bbox = detections.xyxy[i]
                center_x = (bbox[0] + bbox[2]) / 2 / self.field_width
                center_y = (bbox[1] + bbox[3]) / 2 / self.field_height
                current_pos = (center_x, center_y)

                # Tính điểm số tổng hợp
                position_score = -self._calculate_distance(track['position'], current_pos)
                features = self._extract_features(frame, bbox)
                appearance_score = self._match_features(features, track_id)

                # Dự đoán chuyển động
                motion_score = 0
                if track_id in self.velocity_history:
                    predicted_pos = np.array(track['position']) + self.velocity_history[track_id]
                    motion_score = -self._calculate_distance(predicted_pos, current_pos)

                total_score = (
                    position_score * 0.5 +
                    appearance_score * 0.3 +
                    motion_score * 0.2
                )

                if total_score > max_score:
                    max_score = total_score
                    best_detection_idx = i

            # Cập nhật track nếu tìm thấy match tốt
            if best_detection_idx is not None and max_score > -self.max_distance:
                bbox = detections.xyxy[best_detection_idx]
                center_x = (bbox[0] + bbox[2]) / 2 / self.field_width
                center_y = (bbox[1] + bbox[3]) / 2 / self.field_height

                features = self._extract_features(frame, bbox)
                if features is not None:
                    self.appearance_features[track_id] = features

                # Cập nhật vận tốc và vị trí
                if track_id in self.position_history:
                    velocity = self._calculate_velocity(
                        (center_x, center_y),
                        self.position_history[track_id][-1] if self.position_history[track_id] else None
                    )
                    self.velocity_history[track_id] = velocity * 0.7 + \
                        self.velocity_history.get(track_id, np.zeros(2)) * 0.3

                # Cập nhật lịch sử vị trí
                if track_id not in self.position_history:
                    self.position_history[track_id] = deque(maxlen=self.feature_memory)
                self.position_history[track_id].append((center_x, center_y))

                # Cập nhật độ tin cậy
                self.confidence_scores[track_id] = min(1.0,
                    self.confidence_scores.get(track_id, 0.5) + 0.1)

                current_tracks[track_id] = {
                    'bbox': bbox,
                    'position': (center_x, center_y),
                    'class_id': track['class_id'],
                    'confidence': self.confidence_scores[track_id]
                }
                used_detections.add(best_detection_idx)
            else:
                # Chuyển sang lost tracks
                self.lost_tracks[track_id] = {
                    **track,
                    'lost_frames': self.lost_tracks.get(track_id, {}).get('lost_frames', 0) + 1
                }

        # Bước 2: Xử lý các detection chưa được sử dụng
        for i in range(len(detections)):
            if i in used_detections:
                continue

            bbox = detections.xyxy[i]
            class_id = detections.class_id[i]
            center_x = (bbox[0] + bbox[2]) / 2 / self.field_width
            center_y = (bbox[1] + bbox[3]) / 2 / self.field_height

            # Tìm trong lost_tracks
            recovered_id = None
            min_distance = self.max_distance

            for lost_id, lost_track in list(self.lost_tracks.items()):
                if lost_track['class_id'] != class_id:
                    continue

                distance = self._calculate_distance(
                    lost_track['position'],
                    (center_x, center_y)
                )

                if distance < min_distance:
                    features = self._extract_features(frame, bbox)
                    if features is not None:
                        similarity = self._match_features(features, lost_id)
                        if similarity > 0.7:  # Similarity threshold
                            min_distance = distance
                            recovered_id = lost_id

            if recovered_id is not None:
                track_id = recovered_id
                del self.lost_tracks[recovered_id]
            else:
                track_id = self._get_next_id(class_id)

            # Khởi tạo track mới
            features = self._extract_features(frame, bbox)
            if features is not None:
                self.appearance_features[track_id] = features

            current_tracks[track_id] = {
                'bbox': bbox,
                'position': (center_x, center_y),
                'class_id': class_id,
                'confidence': 0.5  # Độ tin cậy ban đầu
            }

        # Cập nhật và dọn dẹp
        self.tracks = {k: v for k, v in current_tracks.items()
                      if v['confidence'] > self.min_confidence_threshold}
        self.lost_tracks = {k: v for k, v in self.lost_tracks.items()
                          if v['lost_frames'] < self.max_lost_frames}

        return self.tracks

    def get_stable_ids(self, detections, frame):
        """Lấy ID ổn định cho các detections"""
        tracks = self.update(detections, frame)
        stable_ids = np.zeros(len(detections), dtype=int)

        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            for track_id, track in tracks.items():
                if np.allclose(track['bbox'], bbox, rtol=1e-5, atol=1e-5):
                    stable_ids[i] = track_id
                    break

        return stable_ids

    def reset(self):
        """Reset trạng thái của tracker"""
        self.tracks.clear()
        self.lost_tracks.clear()
        self.appearance_features.clear()
        self.position_history.clear()
        self.velocity_history.clear()
        self.confidence_scores.clear()