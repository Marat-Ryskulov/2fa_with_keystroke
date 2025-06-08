# ml/simple_knn_trainer.py - Улучшенная версия без переобучения

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

from ml.feature_extractor import FeatureExtractor
from config import MODELS_DIR, MIN_TRAINING_SAMPLES

class SimpleKNNTrainer:
    """Улучшенная система обучения kNN модели без переобучения"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = {}
        self.training_stats = {}
        
    def prepare_training_data(self, positive_samples: List) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка более реалистичных данных для предотвращения переобучения"""
        
        # Извлечение признаков из положительных образцов
        X_positive = self.feature_extractor.extract_features_from_samples(positive_samples)
        n_positive = len(X_positive)
        
        if n_positive < MIN_TRAINING_SAMPLES:
            raise ValueError(f"Недостаточно образцов: {n_positive}, нужно минимум {MIN_TRAINING_SAMPLES}")
        
        print(f"📊 АНАЛИЗ ВАШИХ ДАННЫХ:")
        print(f"   Количество образцов: {n_positive}")
        
        # Анализируем ваши данные
        mean_positive = np.mean(X_positive, axis=0)
        std_positive = np.std(X_positive, axis=0)
        
        print(f"   Время удержания: {mean_positive[0]*1000:.1f}±{std_positive[0]*1000:.1f} мс")
        print(f"   Время между клавишами: {mean_positive[2]*1000:.1f}±{std_positive[2]*1000:.1f} мс")
        print(f"   Скорость: {mean_positive[4]:.1f}±{std_positive[4]:.1f} кл/сек")
        
        # ✅ ИСПРАВЛЕНИЕ: Генерация более РАЗЛИЧИМЫХ негативов
        X_negative = self._generate_diverse_negatives(X_positive, factor=1.2)
        n_negative = len(X_negative)
        
        # Проверяем разделимость
        self._check_separability(X_positive, X_negative)
        
        # Комбинирование данных
        X = np.vstack([X_positive, X_negative])
        y = np.hstack([np.ones(n_positive), np.zeros(n_negative)])
        
        # Нормализация признаков
        X_normalized = self.scaler.fit_transform(X)
        
        print(f"✅ Данные подготовлены: {n_positive} ваших, {n_negative} негативных")
        return X_normalized, y
    
    def _generate_diverse_negatives(self, X_positive: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """Генерация БОЛЕЕ СЛОЖНЫХ негативных примеров без переобучения"""
        n_samples = len(X_positive)
        n_negatives = int(n_samples * factor)
        
        mean = np.mean(X_positive, axis=0)
        std = np.std(X_positive, axis=0)
        
        # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Увеличиваем минимальную вариативность
        std = np.maximum(std, mean * 0.3)  # Было 0.15, стало 0.3
        
        negatives = []
        
        print(f"🔧 СОЗДАНИЕ БОЛЕЕ СЛОЖНЫХ НЕГАТИВОВ:")
        
        # ✅ ИСПРАВЛЕНИЕ: Стратегия 1 - НЕМНОГО различающиеся (40% вместо 25%)
        similar_count = int(n_negatives * 0.4)
        print(f"   {similar_count} похожих но различающихся...")
        for _ in range(similar_count):
            # Создаем образцы БЛИЗКИЕ к вашим, но с небольшими отличиями
            base_idx = np.random.randint(0, len(X_positive))
            sample = X_positive[base_idx].copy()
            
            # Изменяем только 1-2 признака на небольшую величину
            features_to_change = np.random.choice(6, size=np.random.randint(1, 3), replace=False)
            
            for feat_idx in features_to_change:
                # НЕБОЛЬШИЕ изменения: 80%-120% от исходного
                change_factor = np.random.uniform(0.8, 1.2)
                sample[feat_idx] *= change_factor
            
            # Добавляем небольшой шум
            noise = np.random.normal(0, std * 0.2)  # Уменьшили шум
            sample += noise
            sample = np.maximum(sample, mean * 0.3)
            negatives.append(sample)
        
        # Стратегия 2: Умеренно медленные (20%)
        slow_count = int(n_negatives * 0.2)
        print(f"   {slow_count} умеренно медленных...")
        for _ in range(slow_count):
            sample = mean.copy()
            # Умеренно медленно (не экстремально!)
            sample[0] *= np.random.uniform(1.3, 1.8)    # dwell time
            sample[2] *= np.random.uniform(1.4, 2.2)    # flight time
            sample[4] *= np.random.uniform(0.6, 0.8)    # speed
            sample[5] *= np.random.uniform(1.3, 1.8)    # total time
            
            # Умеренная вариативность
            sample[1] *= np.random.uniform(1.2, 2.0)    # dwell std
            sample[3] *= np.random.uniform(1.2, 2.0)    # flight std
            
            noise = np.random.normal(0, std * 0.3)
            sample += noise
            sample = np.maximum(sample, mean * 0.2)
            negatives.append(sample)
        
        # Стратегия 3: Умеренно быстрые (20%)
        fast_count = int(n_negatives * 0.2)
        print(f"   {fast_count} умеренно быстрых...")
        for _ in range(fast_count):
            sample = mean.copy()
            # Умеренно быстро
            sample[0] *= np.random.uniform(0.6, 0.8)    # dwell time
            sample[2] *= np.random.uniform(0.5, 0.7)    # flight time
            sample[4] *= np.random.uniform(1.2, 1.6)    # speed
            sample[5] *= np.random.uniform(0.6, 0.8)    # total time
            
            # Низкая вариативность
            sample[1] *= np.random.uniform(0.4, 0.8)    # dwell std
            sample[3] *= np.random.uniform(0.4, 0.8)    # flight std
            
            noise = np.random.normal(0, std * 0.3)
            sample += noise
            sample = np.maximum(sample, mean * 0.2)
            negatives.append(sample)
        
        # Стратегия 4: Заполняем оставшееся случайными вариациями
        remaining_count = n_negatives - similar_count - slow_count - fast_count
        print(f"   {remaining_count} случайных вариаций...")
        for _ in range(remaining_count):
            # Берем случайный образец как основу
            base_idx = np.random.randint(0, len(X_positive))
            sample = X_positive[base_idx].copy()
            
            # Применяем случайные, но умеренные изменения
            for i in range(len(sample)):
                change_factor = np.random.uniform(0.7, 1.4)  # Более умеренные изменения
                sample[i] *= change_factor
            
            # Умеренный шум
            noise = np.random.normal(0, std * 0.4)
            sample += noise
            sample = np.maximum(sample, mean * 0.1)
            negatives.append(sample)
        
        negatives_array = np.array(negatives)
        
        print(f"   Создано: {len(negatives_array)} умеренно различающихся негативов")
        return negatives_array
    
    def _check_separability(self, X_positive: np.ndarray, X_negative: np.ndarray):
        """Проверка разделимости классов с предупреждением о переобучении"""
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Расстояния между классами
        distances = euclidean_distances(X_positive, X_negative)
        min_dist = np.min(distances)
        mean_dist = np.mean(distances)
        
        # Внутриклассовые расстояния (ваши данные)
        if len(X_positive) > 1:
            intra_distances = euclidean_distances(X_positive, X_positive)
            intra_distances = intra_distances[intra_distances > 0]
            mean_intra = np.mean(intra_distances)
            
            separation_ratio = mean_dist / mean_intra if mean_intra > 0 else float('inf')
        else:
            separation_ratio = mean_dist
            mean_intra = 0
        
        print(f"🔍 ПРОВЕРКА РАЗДЕЛИМОСТИ:")
        print(f"   Мин. расстояние между классами: {min_dist:.2f}")
        print(f"   Среднее расстояние между классами: {mean_dist:.2f}")
        if mean_intra > 0:
            print(f"   Среднее расстояние внутри класса: {mean_intra:.2f}")
            print(f"   Коэффициент разделимости: {separation_ratio:.2f}")
            
            # ✅ КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ О ПЕРЕОБУЧЕНИИ
            if separation_ratio > 20.0:
                print(f"⚠️  ВЫСОКИЙ РИСК ПЕРЕОБУЧЕНИЯ! Коэффициент > 20")
                print(f"    Негативы слишком далеки от ваших данных")
            elif separation_ratio > 10.0:
                print(f"⚠️  СРЕДНИЙ РИСК ПЕРЕОБУЧЕНИЯ! Коэффициент > 10")  
            elif separation_ratio > 5.0:
                print(f"✅ Приемлемая разделимость (коэффициент 5-10)")
            else:
                print(f"⚠️  Возможно слабая разделимость (коэффициент < 5)")
                
            # Если слишком хорошая разделимость - это плохо!
            if separation_ratio > 15.0:
                print(f"🚨 РЕКОМЕНДАЦИЯ: Уменьшить различия между классами!")
        else:
            print(f"✅ Базовая проверка пройдена")
    
    def train_user_model(self, positive_samples: List) -> Tuple[bool, float, str]:
        """Основное обучение модели с защитой от переобучения"""
        try:
            print(f"🎯 НАЧАЛО УЛУЧШЕННОГО ОБУЧЕНИЯ для пользователя {self.user_id}")
            
            # Подготовка данных
            X, y = self.prepare_training_data(positive_samples)
            
            # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Еще больше данных в тест
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=42, stratify=y  # 50% в тест!
            )

            print(f"📊 Разделение данных:")
            print(f"   Обучение: {len(X_train)} образцов ({np.sum(y_train == 1)} ваших, {np.sum(y_train == 0)} негативных)")
            print(f"   Тест: {len(X_test)} образцов ({np.sum(y_test == 1)} ваших, {np.sum(y_test == 0)} негативных)")
            
            # Поиск оптимальных параметров
            best_params = self._optimize_hyperparameters(X_train, y_train)
            
            # Обучение финальной модели
            self.model = KNeighborsClassifier(**best_params)
            self.model.fit(X_train, y_train)
            
            # ✅ Честная оценка на тестовой выборке
            test_accuracy, roc_data = self._evaluate_model(X_test, y_test)
            
            # ✅ Дополнительная кросс-валидация для честности
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            print(f"📈 КРОСС-ВАЛИДАЦИЯ:")
            print(f"   CV точность: {cv_mean:.3f} ± {cv_std:.3f}")
            print(f"   Разброс: {cv_std:.3f} ({'Низкий' if cv_std < 0.1 else 'Высокий'})")
            
            # Статистика обучения
            self.training_stats = {
                'user_id': self.user_id,
                'training_samples': len(positive_samples),
                'total_samples': len(X),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'best_params': best_params,
                'test_accuracy': test_accuracy,
                'cv_accuracy': cv_mean,
                'cv_std': cv_std,
                'training_date': datetime.now().isoformat(),
                **roc_data  # Добавляем ROC данные
            }
            
            # Сохранение модели
            self._save_model()
            
            print(f"✅ ОБУЧЕНИЕ ЗАВЕРШЕНО:")
            print(f"   Тестовая точность: {test_accuracy:.2%}")
            print(f"   CV точность: {cv_mean:.2%}")
            
            return True, test_accuracy, f"Модель обучена с точностью {test_accuracy:.2%} (CV: {cv_mean:.2%})"
            
        except Exception as e:
            print(f"❌ Ошибка обучения: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0, f"Ошибка обучения: {str(e)}"
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Более консервативная оптимизация гиперпараметров"""
        
        # ✅ ИСПРАВЛЕНИЕ: Более разумные параметры
        param_grid = {
            'n_neighbors': range(3, min(15, len(X_train) // 8)),  # Увеличили минимум
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]  # Для minkowski
        }
        
        # ✅ Стратифицированная кросс-валидация
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"🔧 ОПТИМИЗАЦИЯ ПАРАМЕТРОВ:")
        print(f"   Лучшие параметры: {grid_search.best_params_}")
        print(f"   CV точность: {grid_search.best_score_:.3f}")
        
        self.best_params = grid_search.best_params_
        return grid_search.best_params_
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, Dict]:
        """Честная оценка модели с ROC данными"""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"📊 МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-score: {f1:.3f}")
        
        # ✅ ROC данные для графика
        try:
            if len(np.unique(y_test)) > 1:  # Проверяем, что есть оба класса
                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                roc_data = {
                    'precision': precision,
                    'recall': recall, 
                    'f1_score': f1,
                    'y_test': y_test.tolist(),
                    'y_proba': y_proba.tolist(),
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'roc_auc': roc_auc
                }
                
                print(f"   ROC AUC: {roc_auc:.3f}")
            else:
                print("⚠️  Предупреждение: В тестовой выборке только один класс!")
                roc_data = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'y_test': y_test.tolist(),
                    'y_proba': y_proba.tolist()
                }
        except Exception as e:
            print(f"⚠️  Ошибка расчета ROC: {e}")
            roc_data = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'y_test': y_test.tolist(),
                'y_proba': y_proba.tolist()
            }
        
        return accuracy, roc_data
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """Предсказание с отладкой"""
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        # Нормализация признаков
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Получение вероятности
        proba = self.model.predict_proba(features_scaled)[0]
        confidence = proba[1] if len(proba) > 1 else proba[0]
        
        # ✅ Адаптивный порог в зависимости от качества модели
        base_threshold = 0.75
        if hasattr(self, 'training_stats'):
            test_acc = self.training_stats.get('test_accuracy', 0.85)
            cv_std = self.training_stats.get('cv_std', 0.1)
            
            # Если модель нестабильная, снижаем порог
            if cv_std > 0.15:
                threshold = base_threshold - 0.1
            elif test_acc < 0.8:
                threshold = base_threshold - 0.05
            else:
                threshold = base_threshold
        else:
            threshold = base_threshold
        
        is_legitimate = confidence >= threshold
        
        print(f"🔍 ПРЕДСКАЗАНИЕ:")
        print(f"   Признаки: {features}")
        print(f"   Уверенность: {confidence:.3f}")
        print(f"   Порог: {threshold:.3f}")
        print(f"   Результат: {'ПРИНЯТ' if is_legitimate else 'ОТКЛОНЕН'}")
        
        return is_legitimate, confidence
    
    def _save_model(self):
        """Сохранение модели"""
        model_path = os.path.join(MODELS_DIR, f"user_{self.user_id}_simple_knn.pkl")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'best_params': self.best_params,
            'training_stats': self.training_stats
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Модель сохранена: {model_path}")
    
    @classmethod
    def load_model(cls, user_id: int) -> Optional['SimpleKNNTrainer']:
        """Загрузка модели"""
        model_path = os.path.join(MODELS_DIR, f"user_{user_id}_simple_knn.pkl")
        
        if not os.path.exists(model_path):
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            trainer = cls(user_id)
            trainer.model = model_data['model']
            trainer.scaler = model_data['scaler']
            trainer.best_params = model_data['best_params']
            trainer.training_stats = model_data.get('training_stats', {})
            
            return trainer
            
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return None
    
    def get_model_info(self) -> Dict:
        """Информация о модели"""
        return {
            'is_trained': self.model is not None,
            'best_params': self.best_params,
            'training_stats': self.training_stats
        }