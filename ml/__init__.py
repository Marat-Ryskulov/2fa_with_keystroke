# ml/__init__.py - Очищенная версия
from .feature_extractor import FeatureExtractor
from .model_manager import ModelManager
from .simple_knn_trainer import SimpleKNNTrainer

__all__ = ['FeatureExtractor', 'ModelManager', 'SimpleKNNTrainer']