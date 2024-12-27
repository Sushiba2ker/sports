from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
import cv2

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32, n_clusters: int = 2):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
           n_clusters (int): The number of clusters for clustering.
       """
        self.device = device
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        
        # Nâng cấp model lên siglip-large để có feature tốt hơn
        self.features_model = SiglipVisionModel.from_pretrained('google/siglip-large-patch16-224').to(device)
        self.processor = AutoProcessor.from_pretrained('google/siglip-large-patch16-224')
        
        # Ensemble của nhiều UMAP với các parameters khác nhau
        self.reducers = [
            umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine'),
            umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.2, metric='euclidean'),
            umap.UMAP(n_components=3, n_neighbors=20, min_dist=0.15, metric='manhattan')
        ]
        
        # Ensemble của nhiều KMeans
        self.cluster_models = [
            KMeans(n_clusters=n_clusters, n_init='auto', max_iter=500, random_state=42),
            KMeans(n_clusters=n_clusters, n_init='auto', max_iter=500, random_state=43),
            KMeans(n_clusters=n_clusters, n_init='auto', max_iter=500, random_state=44)
        ]

    def _augment_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Thực hiện data augmentation đa dạng"""
        augmented = []
        # Ảnh gốc
        augmented.append(image)
        
        # Flip ngang
        augmented.append(cv2.flip(image, 1))
        
        # Điều chỉnh độ sáng
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
        augmented.extend([bright, dark])
        
        # Rotation nhẹ
        h, w = image.shape[:2]
        center = (w//2, h//2)
        for angle in [-10, 10]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented.append(rotated)
            
        return augmented

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        augmented_crops = []
        for crop in crops:
            augmented_crops.extend(self._augment_image(crop))
            
        crops = [sv.cv2_to_pillow(crop) for crop in augmented_crops]
        
        # Xử lý theo batch với gradient accumulation
        batches = create_batches(crops, self.batch_size)
        data = []
        
        with torch.no_grad():
            for batch in tqdm(batches, desc='Extracting features'):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                
                # Lấy features từ nhiều layer khác nhau
                outputs = self.features_model(**inputs, output_hidden_states=True)
                
                # Kết hợp features từ các layer cuối
                last_hidden_states = outputs.hidden_states[-1]
                second_last_hidden_states = outputs.hidden_states[-2] 
                
                # Weighted average của các layer features
                combined_features = (0.7 * torch.mean(last_hidden_states, dim=1) + 
                                  0.3 * torch.mean(second_last_hidden_states, dim=1))
                
                data.append(combined_features.cpu().numpy())

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        data = self.extract_features(crops)
        
        # Fit với ensemble của reducers và clusterers
        self.fitted_reducers = []
        self.fitted_clusters = []
        
        for reducer in self.reducers:
            projections = reducer.fit_transform(data)
            self.fitted_reducers.append(reducer)
            
            clusters = []
            for clusterer in self.cluster_models:
                clusterer.fit(projections)
                clusters.append(clusterer)
            self.fitted_clusters.append(clusters)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        
        # Ensemble prediction
        all_predictions = []
        
        for reducer, cluster_models in zip(self.fitted_reducers, self.fitted_clusters):
            projections = reducer.transform(data)
            
            for clusterer in cluster_models:
                predictions = clusterer.predict(projections)
                all_predictions.append(predictions)
        
        # Voting để ra kết quả cuối cùng
        ensemble_predictions = np.stack(all_predictions)
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=0, 
            arr=ensemble_predictions
        )
        
        # Xử lý kết quả augmentation bằng cách lấy mode
        if len(final_predictions) > len(crops):
            n_aug = len(final_predictions) // len(crops)
            reshaped_pred = final_predictions.reshape(-1, n_aug)
            final_predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=1,
                arr=reshaped_pred
            )
            
        return final_predictions
