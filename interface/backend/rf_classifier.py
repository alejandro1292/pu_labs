"""
Clasificador Random Forest para detección de keywords.
Usa características acústicas agregadas (estadísticas sobre MFCCs, ZCR, spectral centroid).
"""
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import librosa

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """
    Extractor de características acústicas usando Librosa.
    Extrae MFCCs, ZCR, Spectral Centroid y Energy, luego agrega con estadísticas.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mfcc: int = 13,
                 n_fft: int = 512,
                 hop_length: int = 160):  # 10ms a 16kHz
        """
        Args:
            sample_rate: Frecuencia de muestreo
            n_mfcc: Número de coeficientes MFCC
            n_fft: Tamaño de ventana FFT
            hop_length: Desplazamiento entre frames
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Calcular dimensión del vector de características
        # Por cada feature: mean, std, max, min = 4 estadísticas
        # Features: n_mfcc MFCCs + ZCR + Spectral Centroid + Energy
        n_features = n_mfcc + 3
        self.feature_dim = n_features * 4  # 4 estadísticas por feature
        
        logger.info(f"✓ AudioFeatureExtractor inicializado: {self.feature_dim} features")
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrae características agregadas de un audio.
        
        Args:
            audio: Señal de audio 1D (sample_rate Hz, mono)
        
        Returns:
            Vector de características (feature_dim,) con estadísticas agregadas
        """
        if len(audio) == 0:
            logger.warning("⚠️ Audio vacío")
            return np.zeros(self.feature_dim)
        
        # Validar entrada
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            logger.warning("⚠️ Audio contiene NaN o Inf")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        features_list = []
        
        try:
            # 1. MFCCs (13 coeficientes)
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )  # (n_mfcc, n_frames)
            
            # Agregar estadísticas por cada coeficiente MFCC
            for i in range(self.n_mfcc):
                mfcc_coef = mfccs[i, :]
                features_list.extend([
                    np.mean(mfcc_coef),
                    np.std(mfcc_coef),
                    np.max(mfcc_coef),
                    np.min(mfcc_coef)
                ])
            
            # 2. Zero-Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(
                y=audio,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )[0]  # (n_frames,)
            
            features_list.extend([
                np.mean(zcr),
                np.std(zcr),
                np.max(zcr),
                np.min(zcr)
            ])
            
            # 3. Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]  # (n_frames,)
            
            features_list.extend([
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.max(spectral_centroid),
                np.min(spectral_centroid)
            ])
            
            # 4. Energy (RMS)
            energy = librosa.feature.rms(
                y=audio,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )[0]  # (n_frames,)
            
            features_list.extend([
                np.mean(energy),
                np.std(energy),
                np.max(energy),
                np.min(energy)
            ])
            
            # Convertir a array y validar
            features = np.array(features_list)
            
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.warning("⚠️ Features contienen NaN o Inf, reemplazando con 0")
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo features: {e}")
            return np.zeros(self.feature_dim)
    
    def extract_features_batch(self, audio_list: List[np.ndarray]) -> np.ndarray:
        """
        Extrae features de múltiples audios.
        
        Args:
            audio_list: Lista de arrays de audio
        
        Returns:
            Matriz (n_samples, feature_dim)
        """
        features_matrix = []
        
        for i, audio in enumerate(audio_list):
            features = self.extract_features(audio)
            features_matrix.append(features)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"   Procesados {i+1}/{len(audio_list)} audios")
        
        return np.array(features_matrix)


class RandomForestKeywordClassifier:
    """
    Clasificador Random Forest para keyword spotting.
    Usa características acústicas agregadas para clasificación.
    """
    
    def __init__(self,
                 keywords: List[str],
                 sample_rate: int = 16000,
                 n_estimators: int = 100,
                 max_depth: int = 20,
                 min_samples_split: int = 5,
                 random_state: int = 42):
        """
        Args:
            keywords: Lista de palabras clave a detectar
            sample_rate: Frecuencia de muestreo
            n_estimators: Número de árboles en el bosque
            max_depth: Profundidad máxima de árboles
            min_samples_split: Mínimo de samples para split
            random_state: Semilla aleatoria
        """
        self.keywords = sorted(keywords)  # Orden consistente
        self.sample_rate = sample_rate
        
        # Crear mapeo keyword <-> índice
        self.keyword_to_idx = {kw: i for i, kw in enumerate(self.keywords)}
        self.idx_to_keyword = {i: kw for i, kw in enumerate(self.keywords)}
        
        # Extractor de features
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        
        # Clasificador Random Forest
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,  # Usar todos los cores
            verbose=0
        )
        
        # Estado
        self.is_trained = False
        self.training_stats = {}
        
        logger.info(f"✓ RandomForestKeywordClassifier inicializado")
        logger.info(f"   Keywords: {', '.join(self.keywords)}")
        logger.info(f"   Parámetros RF: n_estimators={n_estimators}, max_depth={max_depth}")
    
    def train(self,
              audio_samples: Dict[str, List[np.ndarray]],
              test_size: float = 0.2,
              cross_validate: bool = True) -> Dict:
        """
        Entrena el clasificador con muestras de audio.
        
        Args:
            audio_samples: Dict {keyword: [audio1, audio2, ...]}
            test_size: Proporción para test set (0.2 = 20%)
            cross_validate: Si hacer validación cruzada
        
        Returns:
            Diccionario con estadísticas de entrenamiento
        """
        logger.info("\n" + "="*60)
        logger.info("🌲 ENTRENAMIENTO RANDOM FOREST")
        logger.info("="*60)
        
        # Validar datos
        for keyword in self.keywords:
            if keyword not in audio_samples:
                raise ValueError(f"Falta keyword '{keyword}' en audio_samples")
            
            n_samples = len(audio_samples[keyword])
            if n_samples < 20:
                logger.warning(f"⚠️ '{keyword}' tiene solo {n_samples} samples (mínimo recomendado: 20)")
        
        # 1. Extraer features de todos los audios
        logger.info("\n1️⃣ EXTRACCIÓN DE CARACTERÍSTICAS:")
        X_list = []
        y_list = []
        
        for keyword in self.keywords:
            audios = audio_samples[keyword]
            label_idx = self.keyword_to_idx[keyword]
            
            logger.info(f"   Procesando '{keyword}': {len(audios)} samples")
            
            # Extraer features
            features = self.feature_extractor.extract_features_batch(audios)
            
            X_list.append(features)
            y_list.extend([label_idx] * len(features))
            
            logger.info(f"   ✓ '{keyword}': shape={features.shape}")
        
        # Concatenar todos los datos
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        logger.info(f"\n   📊 Dataset total: X={X.shape}, y={y.shape}")
        logger.info(f"   Distribución de clases:")
        for keyword in self.keywords:
            idx = self.keyword_to_idx[keyword]
            count = np.sum(y == idx)
            logger.info(f"      {keyword}: {count} samples ({count/len(y)*100:.1f}%)")
        
        # 2. Dividir train/test
        logger.info(f"\n2️⃣ DIVISIÓN TRAIN/TEST ({int((1-test_size)*100)}% / {int(test_size*100)}%):")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y  # Mantener proporción de clases
        )
        
        logger.info(f"   Train: {X_train.shape[0]} samples")
        logger.info(f"   Test:  {X_test.shape[0]} samples")
        
        # 3. Entrenar Random Forest
        logger.info(f"\n3️⃣ ENTRENAMIENTO RANDOM FOREST:")
        logger.info(f"   Entrenando {self.classifier.n_estimators} árboles...")
        
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info(f"   ✓ Entrenamiento completado")
        
        # 4. Evaluación en test set
        logger.info(f"\n4️⃣ EVALUACIÓN EN TEST SET:")
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"   Accuracy: {accuracy:.2%}")
        
        # Classification report
        logger.info(f"\n   📋 REPORTE DE CLASIFICACIÓN:")
        report = classification_report(
            y_test, y_pred,
            target_names=self.keywords,
            digits=3
        )
        logger.info("\n" + report)
        
        # Confusion matrix
        logger.info(f"   🔢 MATRIZ DE CONFUSIÓN:")
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"        Predicho →")
        logger.info(f"   Real ↓    " + "  ".join(f"{kw:>6}" for kw in self.keywords))
        for i, keyword in enumerate(self.keywords):
            row = "  ".join(f"{cm[i,j]:>6}" for j in range(len(self.keywords)))
            logger.info(f"   {keyword:>6}    {row}")
        
        # 5. Validación cruzada (opcional)
        cv_scores = None
        if cross_validate and len(X_train) >= 50:
            logger.info(f"\n5️⃣ VALIDACIÓN CRUZADA (5-fold):")
            cv_scores = cross_val_score(
                self.classifier,
                X_train, y_train,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            logger.info(f"   CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
            logger.info(f"   Scores por fold: {[f'{s:.2%}' for s in cv_scores]}")
        
        # 6. Feature importance
        logger.info(f"\n6️⃣ IMPORTANCIA DE FEATURES:")
        feature_importance = self.classifier.feature_importances_
        
        # Top 10 features más importantes
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        logger.info(f"   Top 10 features más importantes:")
        for rank, idx in enumerate(top_indices, 1):
            importance = feature_importance[idx]
            feature_name = self._get_feature_name(idx)
            logger.info(f"      {rank}. {feature_name}: {importance:.4f}")
        
        # Guardar estadísticas
        self.training_stats = {
            'n_samples_total': len(X),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'n_samples': len(X),
            'test_accuracy': accuracy,
            'test_metrics': {
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            },
            'cross_validation': {
                'scores': cv_scores.tolist() if cv_scores is not None else None,
                'mean': float(cv_scores.mean()) if cv_scores is not None else None,
                'std': float(cv_scores.std()) if cv_scores is not None else None
            } if cv_scores is not None else None,
            'classes': self.keywords,
            'class_distribution': {kw: int(np.sum(y == self.keyword_to_idx[kw])) for kw in self.keywords},
            'feature_importance': feature_importance.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        logger.info("\n" + "="*60)
        logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("="*60 + "\n")
        
        return self.training_stats
    
    def predict(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Predice keyword para un audio.
        
        Args:
            audio: Señal de audio
        
        Returns:
            (keyword, confidence)
        """
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Extraer features
        features = self.feature_extractor.extract_features(audio)
        features = features.reshape(1, -1)  # (1, feature_dim)
        
        # Predecir
        pred_idx = self.classifier.predict(features)[0]
        pred_proba = self.classifier.predict_proba(features)[0]
        
        keyword = self.idx_to_keyword[pred_idx]
        confidence = pred_proba[pred_idx]
        
        return keyword, confidence
    
    def predict_with_details(self, audio: np.ndarray) -> Dict:
        """
        Predice keyword con información detallada.
        
        Args:
            audio: Señal de audio
        
        Returns:
            Dict con predicción y probabilidades de todas las clases
        """
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Extraer features
        features = self.feature_extractor.extract_features(audio)
        features = features.reshape(1, -1)
        
        # Predecir
        pred_idx = self.classifier.predict(features)[0]
        pred_proba = self.classifier.predict_proba(features)[0]
        
        keyword = self.idx_to_keyword[pred_idx]
        confidence = pred_proba[pred_idx]
        
        # Probabilidades de todas las clases
        all_proba = {self.idx_to_keyword[i]: float(p) for i, p in enumerate(pred_proba)}
        
        return {
            'predicted_keyword': keyword,
            'confidence': float(confidence),
            'all_probabilities': all_proba,
            'features': features[0].tolist()
        }
    
    def _get_feature_name(self, idx: int) -> str:
        """Retorna nombre descriptivo de una feature."""
        n_mfcc = self.feature_extractor.n_mfcc
        
        # MFCCs (13 coef × 4 stats = 52 features)
        if idx < n_mfcc * 4:
            mfcc_idx = idx // 4
            stat_idx = idx % 4
            stat_names = ['mean', 'std', 'max', 'min']
            return f"MFCC_{mfcc_idx}_{stat_names[stat_idx]}"
        
        idx -= n_mfcc * 4
        
        # ZCR (4 stats)
        if idx < 4:
            stat_names = ['mean', 'std', 'max', 'min']
            return f"ZCR_{stat_names[idx]}"
        
        idx -= 4
        
        # Spectral Centroid (4 stats)
        if idx < 4:
            stat_names = ['mean', 'std', 'max', 'min']
            return f"SpectralCentroid_{stat_names[idx]}"
        
        idx -= 4
        
        # Energy (4 stats)
        stat_names = ['mean', 'std', 'max', 'min']
        return f"Energy_{stat_names[idx]}"
    
    def save(self, filepath: Path):
        """Guarda el modelo entrenado."""
        if not self.is_trained:
            raise RuntimeError("No hay modelo entrenado para guardar")
        
        model_data = {
            'keywords': self.keywords,
            'keyword_to_idx': self.keyword_to_idx,
            'idx_to_keyword': self.idx_to_keyword,
            'sample_rate': self.sample_rate,
            'classifier': self.classifier,
            'training_stats': self.training_stats,
            'feature_extractor_params': {
                'sample_rate': self.feature_extractor.sample_rate,
                'n_mfcc': self.feature_extractor.n_mfcc,
                'n_fft': self.feature_extractor.n_fft,
                'hop_length': self.feature_extractor.hop_length,
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ Modelo guardado en {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'RandomForestKeywordClassifier':
        """Carga un modelo entrenado."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Crear instancia
        instance = cls(
            keywords=model_data['keywords'],
            sample_rate=model_data['sample_rate']
        )
        
        # Restaurar estado
        instance.keyword_to_idx = model_data['keyword_to_idx']
        instance.idx_to_keyword = model_data['idx_to_keyword']
        instance.classifier = model_data['classifier']
        instance.training_stats = model_data['training_stats']
        instance.is_trained = True
        
        logger.info(f"✓ Modelo cargado desde {filepath}")
        logger.info(f"   Keywords: {', '.join(instance.keywords)}")
        logger.info(f"   Test accuracy: {instance.training_stats.get('test_accuracy', 0):.2%}")
        
        return instance
