"""
Combinación Fourier + Wavelet para detección de keywords.
Concatena características de magnitud STFT (bandas) y coeficientes wavelet
para formar un vector de features más rico. Usa NearestCentroid como clasificador
y mantiene la API compatible con los otros clasificadores (train, predict, save, load, etc.).
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestCentroid

import librosa
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
import io

logger = logging.getLogger(__name__)

# wavelet dependency is optional
try:
    import pywt
except Exception:
    pywt = None
    logger.warning("pywt no está disponible: WaveletFeatureExtractor requerirá 'pywt' para funcionar")


class AudioPreprocessor:
    """
    Clase para preprocesamiento de audio: reducción de ruido, pre-énfasis y normalización.
    """
    def __init__(self, sample_rate: int = 16000, apply_filter: bool = True, apply_preemphasis: bool = True):
        self.sample_rate = sample_rate
        self.apply_filter = apply_filter
        self.apply_preemphasis = apply_preemphasis
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio
            
        # 1. Eliminar DC offset
        audio = audio - np.mean(audio)
        
        # 2. Filtro paso banda (Bandpass) para voz (aprox 80Hz - 7000Hz)
        if self.apply_filter:
            nyquist = 0.5 * self.sample_rate
            low = 80 / nyquist
            high = 7000 / nyquist
            # Asegurar que high < 1.0
            if high >= 1.0: high = 0.99
            
            b, a = scipy.signal.butter(4, [low, high], btype='band')
            audio = scipy.signal.lfilter(b, a, audio)
            
        # 3. Pre-énfasis
        if self.apply_preemphasis:
            audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
            
        # 4. Normalización de amplitud
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
        return audio


class FourierFeatureExtractor:
    """
    Extrae características basadas en la magnitud de la STFT agrupada en bandas.
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 512, hop_length: int = 160, n_bins: int = 64, n_segments: int = 3):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.n_segments = n_segments

        # Por cada segmento: (mean, std, max, min) * n_bins
        # Total features = n_segments * n_bins * 4
        self.feature_dim = n_segments * n_bins * 4

        logger.info(f"✓ FourierFeatureExtractor inicializado: {self.feature_dim} features ({n_bins} bandas x {n_segments} segmentos)")

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            logger.warning("⚠️ Audio vacío")
            return np.zeros(self.feature_dim)

        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            logger.warning("⚠️ Audio contiene NaN o Inf -> reemplazando con 0")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            # STFT Magnitude
            stft = np.abs(librosa.stft(y=audio, n_fft=self.n_fft, hop_length=self.hop_length))
            # stft shape: (n_freq_bins, n_time_frames)
            n_freq, n_time = stft.shape

            # Dividir en segmentos temporales
            segment_len = n_time // self.n_segments
            if segment_len == 0:
                # Audio muy corto, usar todo como un segmento y repetir
                segments = [stft] * self.n_segments
            else:
                segments = []
                for i in range(self.n_segments):
                    start_col = i * segment_len
                    # El último segmento toma el resto
                    end_col = (i + 1) * segment_len if i < self.n_segments - 1 else n_time
                    segments.append(stft[:, start_col:end_col])

            # Definir bordes de bandas de frecuencia
            bin_edges = np.linspace(0, n_freq, self.n_bins + 1, dtype=int)

            all_features = []

            for seg_idx, seg_stft in enumerate(segments):
                # Calcular features por banda para este segmento
                seg_features = []
                for b in range(self.n_bins):
                    start_row = bin_edges[b]
                    end_row = bin_edges[b + 1]
                    
                    if end_row <= start_row:
                        band_vals = np.zeros(1)
                    else:
                        # Extraer banda de frecuencia para este segmento temporal
                        band_matrix = seg_stft[start_row:end_row, :]
                        if band_matrix.size == 0:
                            band_vals = np.zeros(1)
                        else:
                            # Promediar sobre frecuencia para tener serie temporal de la banda
                            band_series = np.mean(band_matrix, axis=0)
                            band_vals = band_series

                    seg_features.extend([
                        float(np.mean(band_vals)),
                        float(np.std(band_vals)),
                        float(np.max(band_vals)),
                        float(np.min(band_vals))
                    ])
                all_features.extend(seg_features)

            features = np.array(all_features, dtype=float)
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.warning("⚠️ Features contienen NaN/Inf -> reemplazando con 0")
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            return features

        except Exception as e:
            logger.error(f"❌ Error extrayendo características Fourier: {e}")
            return np.zeros(self.feature_dim)

    def generate_spectrogram(self, audio: Optional[np.ndarray] = None, title: str = "Spectrogram", magnitude_matrix: Optional[np.ndarray] = None) -> bytes:
        """
        Genera una imagen del espectrograma en bytes (PNG) para visualización.
        Puede recibir el audio crudo o una matriz de magnitudes pre-calculada (para promedios).
        """
        try:
            if magnitude_matrix is not None:
                S = magnitude_matrix
            elif audio is not None:
                S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
            else:
                return b""

            D = librosa.amplitude_to_db(S, ref=np.max)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(D, sr=self.sample_rate, hop_length=self.hop_length, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title(title)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Error generando espectrograma: {e}")
            return b""

    def extract_features_batch(self, audio_list: List[np.ndarray]) -> np.ndarray:
        features = []
        for i, audio in enumerate(audio_list):
            features.append(self.extract_features(audio))
            if (i + 1) % 10 == 0:
                logger.debug(f"   Procesados {i+1}/{len(audio_list)} audios (Fourier)")
        return np.array(features)


class FourierKeywordClassifier:
    """
    Clasificador que usa características Fourier y NearestCentroid.
    API compatible con RandomForestKeywordClassifier.
    """

    def __init__(self, keywords: List[str], sample_rate: int = 16000, n_fft: int = 512, hop_length: int = 160, n_bins: int = 64):
        self.keywords = sorted(keywords)
        self.sample_rate = sample_rate

        self.keyword_to_idx = {kw: i for i, kw in enumerate(self.keywords)}
        self.idx_to_keyword = {i: kw for i, kw in enumerate(self.keywords)}

        self.feature_extractor = FourierFeatureExtractor(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_bins=n_bins)
        self.classifier = NearestCentroid()

        self.is_trained = False
        self.training_stats = {}
        self.centroids = None

        logger.info(f"✓ FourierKeywordClassifier inicializado: {', '.join(self.keywords)}")

    def train(self, audio_samples: Dict[str, List[np.ndarray]], test_size: float = 0.2, cross_validate: bool = False) -> Dict:
        logger.info("\n" + "="*60)
        logger.info("🌊 ENTRENAMIENTO FOURIER")
        logger.info("="*60)

        for keyword in self.keywords:
            if keyword not in audio_samples:
                raise ValueError(f"Falta keyword '{keyword}' en audio_samples")
            n_samples = len(audio_samples[keyword])
            if n_samples < 5:
                logger.warning(f"⚠️ '{keyword}' tiene solo {n_samples} samples (recomendado >=5)")

        X_list = []
        y_list = []
        for keyword in self.keywords:
            audios = audio_samples[keyword]
            label_idx = self.keyword_to_idx[keyword]
            logger.info(f"   Procesando '{keyword}': {len(audios)} samples")
            features = self.feature_extractor.extract_features_batch(audios)
            X_list.append(features)
            y_list.extend([label_idx] * len(features))

        X = np.vstack(X_list)
        y = np.array(y_list)

        logger.info(f"   Dataset total: X={X.shape}, y={y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        try:
            self.centroids = np.vstack([self.classifier.centroids_[i] for i in range(len(self.keywords))])
        except Exception:
            self.centroids = np.array([np.mean(X_train[y_train == i], axis=0) for i in range(len(self.keywords))])

        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred, target_names=self.keywords, digits=3)
        cm = confusion_matrix(y_test, y_pred)

        cv_scores = None
        if cross_validate and len(X_train) >= 20:
            cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

        class_distribution = {kw: int(np.sum(y == self.keyword_to_idx[kw])) for kw in self.keywords}

        intra_scores = []
        for i, kw in enumerate(self.keywords):
            members = X[y == i]
            if members.shape[0] == 0:
                intra_scores.append(0.0)
                continue
            dists = np.linalg.norm(members - self.centroids[i], axis=1)
            intra_scores.append(float(np.mean(dists)))

        mean_intra = float(np.mean(intra_scores)) if intra_scores else 0.0

        self.training_stats = {
            'n_samples_total': len(X),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'test_accuracy': float(accuracy),
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
            'class_distribution': class_distribution,
            'feature_importance': None,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'n_bins': self.feature_extractor.n_bins,
            'mean_intra_score': mean_intra
        }

        logger.info("\n" + "="*60)
        logger.info("✅ ENTRENAMIENTO FOURIER COMPLETADO")
        logger.info("="*60 + "\n")

        return self.training_stats

    def predict(self, audio: np.ndarray) -> Tuple[str, float]:
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Ejecutar train() primero.")

        features = self.feature_extractor.extract_features(audio).reshape(1, -1)
        pred_idx = int(self.classifier.predict(features)[0])

        dists = np.linalg.norm(self.centroids - features, axis=1).flatten()
        exps = np.exp(-dists)
        probs = exps / np.sum(exps)
        confidence = float(probs[pred_idx])

        return self.idx_to_keyword[pred_idx], confidence

    def predict_with_details(self, audio: np.ndarray) -> Dict:
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Ejecutar train() primero.")

        features = self.feature_extractor.extract_features(audio).reshape(1, -1)
        pred_idx = int(self.classifier.predict(features)[0])

        dists = np.linalg.norm(self.centroids - features, axis=1).flatten()
        exps = np.exp(-dists)
        probs = exps / np.sum(exps)

        all_proba = {self.idx_to_keyword[i]: float(probs[i]) for i in range(len(self.keywords))}

        return {
            'predicted_keyword': self.idx_to_keyword[pred_idx],
            'confidence': float(probs[pred_idx]),
            'all_probabilities': all_proba,
            'features': features[0].tolist()
        }

    def save(self, filepath: Path):
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
                'n_fft': self.feature_extractor.n_fft,
                'hop_length': self.feature_extractor.hop_length,
                'n_bins': self.feature_extractor.n_bins
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"✓ Modelo guardado en {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'FourierKeywordClassifier':
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls(keywords=model_data['keywords'], sample_rate=model_data.get('sample_rate', 16000),
                       n_fft=model_data.get('feature_extractor_params', {}).get('n_fft', 512),
                       hop_length=model_data.get('feature_extractor_params', {}).get('hop_length', 160),
                       n_bins=model_data.get('feature_extractor_params', {}).get('n_bins', 64))

        instance.keyword_to_idx = model_data['keyword_to_idx']
        instance.idx_to_keyword = model_data['idx_to_keyword']
        instance.classifier = model_data['classifier']
        instance.training_stats = model_data['training_stats']
        instance.is_trained = True

        logger.info(f"✓ Modelo Fourier cargado desde {filepath}")
        logger.info(f"   Keywords: {', '.join(instance.keywords)}")
        logger.info(f"   Test accuracy: {instance.training_stats.get('test_accuracy', 0):.2%}")

        return instance


class WaveletFeatureExtractor:
    """
    Extrae características a partir de la descomposición wavelet.
    Para cada nivel (aproximación y detalles) calcula mean/std/max/min de los coeficientes.
    """

    def __init__(self, sample_rate: int = 16000, wavelet: str = 'db4', level: int = 5):
        if pywt is None:
            raise ImportError("pywt no está instalado. Instala 'PyWavelets' para usar WaveletFeatureExtractor")

        self.sample_rate = sample_rate
        self.wavelet = wavelet
        self.level = level

        # Coeficientes: cA_n + cD_n + cD_n-1 + ... -> n+1 bandas
        self.n_bands = level + 1
        self.feature_dim = self.n_bands * 4  # mean, std, max, min por banda

        logger.info(f"✓ WaveletFeatureExtractor inicializado: wavelet={wavelet}, level={level}, features={self.feature_dim}")

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            logger.warning("⚠️ Audio vacío")
            return np.zeros(self.feature_dim)

        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            logger.warning("⚠️ Audio contiene NaN/Inf -> reemplazando con 0")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            if pywt is None:
                raise ImportError("pywt no está disponible")
                
            coeffs = pywt.wavedec(data=audio, wavelet=self.wavelet, level=self.level)
            # coeffs: [cA_n, cD_n, cD_n-1, ..., cD1]
            features = []
            for band in coeffs:
                # usar magnitud de coeficientes para evitar cancelaciones
                vals = np.abs(band)
                if vals.size == 0:
                    features.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    features.extend([
                        float(np.mean(vals)),
                        float(np.std(vals)),
                        float(np.max(vals)),
                        float(np.min(vals))
                    ])

            features = np.array(features, dtype=float)

            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.warning("⚠️ Features contienen NaN/Inf -> reemplazando con 0")
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            return features

        except ImportError:
            logger.error("❌ PyWavelets no instalado. Wavelet features serán ceros.")
            return np.zeros(self.feature_dim)
        except Exception as e:
            logger.error(f"❌ Error extrayendo características Wavelet: {e}")
            return np.zeros(self.feature_dim)

    def extract_features_batch(self, audio_list: List[np.ndarray]) -> np.ndarray:
        features = []
        for i, audio in enumerate(audio_list):
            features.append(self.extract_features(audio))
            if (i + 1) % 10 == 0:
                logger.debug(f"   Procesados {i+1}/{len(audio_list)} audios (Wavelet)")
        return np.array(features)


class WaveletKeywordClassifier:
    """
    Clasificador que usa características Wavelet y NearestCentroid.
    API compatible con RandomForestKeywordClassifier.
    """

    def __init__(self, keywords: List[str], sample_rate: int = 16000, wavelet: str = 'db4', level: int = 5):
        if pywt is None:
            raise ImportError("pywt no está instalado. Instala 'PyWavelets' para usar WaveletKeywordClassifier")

        self.keywords = sorted(keywords)
        self.sample_rate = sample_rate

        self.keyword_to_idx = {kw: i for i, kw in enumerate(self.keywords)}
        self.idx_to_keyword = {i: kw for i, kw in enumerate(self.keywords)}

        self.feature_extractor = WaveletFeatureExtractor(sample_rate=sample_rate, wavelet=wavelet, level=level)

        self.classifier = NearestCentroid()
        self.is_trained = False
        self.training_stats = {}
        self.centroids = None

        logger.info(f"✓ WaveletKeywordClassifier inicializado: wavelet={wavelet}, level={level}")

    def train(self, audio_samples: Dict[str, List[np.ndarray]], test_size: float = 0.2, cross_validate: bool = False) -> Dict:
        logger.info("\n" + "="*60)
        logger.info("🌀 ENTRENAMIENTO WAVELET")
        logger.info("="*60)

        for keyword in self.keywords:
            if keyword not in audio_samples:
                raise ValueError(f"Falta keyword '{keyword}' en audio_samples")
            n = len(audio_samples[keyword])
            if n < 5:
                logger.warning(f"⚠️ '{keyword}' tiene solo {n} samples (recomendado >=5)")

        # Extraer features
        X_list = []
        y_list = []
        for keyword in self.keywords:
            audios = audio_samples[keyword]
            label_idx = self.keyword_to_idx[keyword]
            logger.info(f"   Procesando '{keyword}': {len(audios)} samples")
            feats = self.feature_extractor.extract_features_batch(audios)
            X_list.append(feats)
            y_list.extend([label_idx] * len(feats))

        X = np.vstack(X_list)
        y = np.array(y_list)

        logger.info(f"   Dataset total: X={X.shape}, y={y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        # Entrenar
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        # Centroids
        try:
            self.centroids = np.vstack([self.classifier.centroids_[i] for i in range(len(self.keywords))])
        except Exception:
            self.centroids = np.array([np.mean(X_train[y_train == i], axis=0) for i in range(len(self.keywords))])

        # Evaluación
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.keywords, digits=3)
        cm = confusion_matrix(y_test, y_pred)

        cv_scores = None
        if cross_validate and len(X_train) >= 20:
            cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

        class_distribution = {kw: int(np.sum(y == self.keyword_to_idx[kw])) for kw in self.keywords}

        # intra-clase: media de distancia a su centroide
        intra_scores = []
        for i in range(len(self.keywords)):
            members = X[y == i]
            if members.shape[0] == 0:
                intra_scores.append(0.0)
                continue
            d = np.linalg.norm(members - self.centroids[i], axis=1)
            intra_scores.append(float(np.mean(d)))

        mean_intra = float(np.mean(intra_scores)) if intra_scores else 0.0

        self.training_stats = {
            'n_samples_total': len(X),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'test_accuracy': float(accuracy),
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
            'class_distribution': class_distribution,
            'feature_importance': None,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'wavelet': self.feature_extractor.wavelet,
            'level': self.feature_extractor.level,
            'mean_intra_score': mean_intra
        }

        logger.info("\n" + "="*60)
        logger.info("✅ ENTRENAMIENTO WAVELET COMPLETADO")
        logger.info("="*60 + "\n")

        return self.training_stats

logger = logging.getLogger(__name__)


class FWFeatureExtractor:
    """
    Combina FourierFeatureExtractor + WaveletFeatureExtractor.
    Si Wavelet no está disponible (pywt no instalado), funciona con Fourier solamente y
    marca `wavelet_enabled = False`.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 fourier_params: Dict = None,
                 wavelet_params: Dict = None):
        fourier_params = fourier_params or {}
        wavelet_params = wavelet_params or {}

        # Instanciar componente Fourier
        self.fourier = FourierFeatureExtractor(
            sample_rate=sample_rate,
            n_fft=fourier_params.get('n_fft', 512),
            hop_length=fourier_params.get('hop_length', 160),
            n_bins=fourier_params.get('n_bins', 64)
        )

        # Intentar instanciar wavelet; manejar ImportError si pywt no está disponible
        try:
            self.wavelet = WaveletFeatureExtractor(
                sample_rate=sample_rate,
                wavelet=wavelet_params.get('wavelet', 'db4'),
                level=wavelet_params.get('level', 5)
            )
            self.wavelet_enabled = True
        except Exception as e:
            logger.warning(f"⚠️ Wavelet extractor no disponible: {e}. Usando solo Fourier.")
            self.wavelet = None
            self.wavelet_enabled = False

        # Dimensión total
        self.feature_dim = self.fourier.feature_dim + (self.wavelet.feature_dim if self.wavelet_enabled else 0)

        logger.info(f"✓ FWFeatureExtractor inicializado: feature_dim={self.feature_dim} (wavelet_enabled={self.wavelet_enabled})")

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        f_feats = self.fourier.extract_features(audio)
        if self.wavelet_enabled and self.wavelet is not None:
            w_feats = self.wavelet.extract_features(audio)
            return np.concatenate([f_feats, w_feats])
        else:
            return f_feats

    def extract_features_batch(self, audio_list: List[np.ndarray]) -> np.ndarray:
        parts = []
        for i, audio in enumerate(audio_list):
            parts.append(self.extract_features(audio))
            if (i + 1) % 10 == 0:
                logger.debug(f"   Procesados {i+1}/{len(audio_list)} audios (FW)")
        return np.array(parts)


class FWKeywordClassifier:
    """
    Clasificador que usa features combinadas Fourier + Wavelet y NearestCentroid.
    API compatible con RandomForestKeywordClassifier / FourierKeywordClassifier / WaveletKeywordClassifier.
    """

    def __init__(self,
                 keywords: List[str],
                 sample_rate: int = 16000,
                 fourier_params: Dict = None,
                 wavelet_params: Dict = None,
                 debug_mode: bool = False):
        self.keywords = sorted(keywords)
        self.sample_rate = sample_rate
        self.debug_mode = debug_mode

        self.keyword_to_idx = {kw: i for i, kw in enumerate(self.keywords)}
        self.idx_to_keyword = {i: kw for i, kw in enumerate(self.keywords)}

        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.feature_extractor = FWFeatureExtractor(sample_rate=sample_rate, fourier_params=fourier_params, wavelet_params=wavelet_params)

        self.classifier = NearestCentroid()
        self.is_trained = False
        self.training_stats = {}
        self.centroids = None
        self.normalized_centroids = None

        logger.info(f"✓ FWKeywordClassifier inicializado: keywords={len(self.keywords)}, feature_dim={self.feature_extractor.feature_dim}")

    def train(self, audio_samples: Dict[str, List[np.ndarray]], test_size: float = 0.2, cross_validate: bool = False) -> Dict:
        logger.info("\n" + "="*60)
        logger.info("🌊 ENTRENAMIENTO FOURIER+WAVELET (FW)")
        logger.info("="*60)

        for keyword in self.keywords:
            if keyword not in audio_samples:
                raise ValueError(f"Falta keyword '{keyword}' en audio_samples")
            n = len(audio_samples[keyword])
            if n < 5:
                logger.warning(f"⚠️ '{keyword}' tiene solo {n} samples (recomendado >=5)")

        # Extraer features por keyword
        X_list = []
        y_list = []
        for keyword in self.keywords:
            audios = audio_samples[keyword]
            label_idx = self.keyword_to_idx[keyword]
            logger.info(f"   Procesando '{keyword}': {len(audios)} samples")
            
            # Preprocesar audios
            processed_audios = [self.preprocessor.process(a) for a in audios]
            
            if self.debug_mode:
                # Guardar espectrograma del primer audio de cada clase
                if len(processed_audios) > 0:
                    spec_bytes = self.feature_extractor.fourier.generate_spectrogram(processed_audios[0], title=f"Spectrogram: {keyword}")
                    if spec_bytes:
                        try:
                            with open(f"debug_spectrogram_{keyword}.png", "wb") as f:
                                f.write(spec_bytes)
                        except Exception as e:
                            logger.warning(f"No se pudo guardar espectrograma de debug: {e}")
            
            feats = self.feature_extractor.extract_features_batch(processed_audios)
            X_list.append(feats)
            y_list.extend([label_idx] * len(feats))

        X = np.vstack(X_list)
        y = np.array(y_list)

        logger.info(f"   Dataset total: X={X.shape}, y={y.shape}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        # Entrenar
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        # Guardar centroides
        try:
            self.centroids = np.vstack([self.classifier.centroids_[i] for i in range(len(self.keywords))])
        except Exception:
            self.centroids = np.array([np.mean(X_train[y_train == i], axis=0) for i in range(len(self.keywords))])

        # Calcular centroides normalizados para producto interno (similitud de coseno)
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        # Evitar división por cero
        norms[norms == 0] = 1.0
        self.normalized_centroids = self.centroids / norms
        logger.info(f"   Centroides normalizados calculados para {len(self.keywords)} clases")

        # Evaluación
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.keywords, digits=3)
        cm = confusion_matrix(y_test, y_pred)

        cv_scores = None
        if cross_validate and len(X_train) >= 20:
            cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

        class_distribution = {kw: int(np.sum(y == self.keyword_to_idx[kw])) for kw in self.keywords}

        # intra-clase y estadísticas por keyword
        intra_scores = []
        per_keyword_stats = {}
        for i, kw in enumerate(self.keywords):
            members = X[y == i]
            if members.shape[0] == 0:
                intra_scores.append(0.0)
                per_keyword_stats[kw] = {
                    'n_samples': 0,
                    'mean_length': 0.0,
                    'std_length': 0.0,
                    'intra_score': 0.0,
                    'centroid': None
                }
                continue
            
            # Distancias al centroide
            d = np.linalg.norm(members - self.centroids[i], axis=1)
            intra_val = float(np.mean(d))
            intra_scores.append(intra_val)
            
            # Longitudes de audios originales (si están disponibles)
            audios = audio_samples[kw]
            lengths = [len(a) / self.sample_rate for a in audios]
            
            per_keyword_stats[kw] = {
                'n_samples': len(audios),
                'mean_length': float(np.mean(lengths)),
                'std_length': float(np.std(lengths)),
                'intra_score': intra_val,
                'centroid': self.centroids[i].tolist()
            }

        mean_intra = float(np.mean(intra_scores)) if intra_scores else 0.0

        # Guardar estadísticas compatibles
        self.training_stats = {
            'n_samples_total': len(X),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'test_accuracy': float(accuracy),
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
            'class_distribution': class_distribution,
            'per_keyword_stats': per_keyword_stats,
            'feature_importance': None,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'fourier_params': {
                'n_bins': self.feature_extractor.fourier.n_bins,
                'n_fft': self.feature_extractor.fourier.n_fft,
                'hop_length': self.feature_extractor.fourier.hop_length
            },
            'wavelet_enabled': self.feature_extractor.wavelet_enabled,
            'wavelet_params': {
                'wavelet': getattr(self.feature_extractor.wavelet, 'wavelet', None),
                'level': getattr(self.feature_extractor.wavelet, 'level', None)
            } if self.feature_extractor.wavelet_enabled else None,
            'mean_intra_score': mean_intra
        }

        logger.info("\n" + "="*60)
        logger.info("✅ ENTRENAMIENTO FW COMPLETADO")
        logger.info("="*60 + "\n")

        return self.training_stats

    def _predict_dot_product(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Calcula la similitud de coseno mediante producto interno con centroides normalizados.
        """
        if self.normalized_centroids is None:
            return -1, 0.0

        # Normalizar vector de entrada
        feat_norm = np.linalg.norm(features)
        if feat_norm == 0:
            return -1, 0.0
        
        norm_feat = features / feat_norm
        
        # Producto interno (similitud de coseno ya que ambos están normalizados)
        # shape: (n_keywords,)
        similarities = np.dot(self.normalized_centroids, norm_feat.flatten())
        
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        
        return best_idx, best_score

    def predict(self, audio: np.ndarray) -> Tuple[str, float]:
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Ejecutar train() primero.")

        # Preprocesar
        audio = self.preprocessor.process(audio)

        features = self.feature_extractor.extract_features(audio).reshape(1, -1)
        
        # --- FAST PATH: Producto Interno ---
        dot_idx, dot_score = self._predict_dot_product(features)
        if dot_score > 0.96:  # Umbral alto para detección clara
            kw = self.idx_to_keyword[dot_idx]
            logger.info(f"🚀 Fast-path detectado: '{kw}' (score={dot_score:.4f})")
            return kw, dot_score
        # ------------------------------------

        pred_idx = int(self.classifier.predict(features)[0])

        dists = np.linalg.norm(self.centroids - features, axis=1).flatten()
        exps = np.exp(-dists)
        probs = exps / np.sum(exps)
        confidence = float(probs[pred_idx])

        return self.idx_to_keyword[pred_idx], confidence

    def predict_with_details(self, audio: np.ndarray) -> Dict:
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Ejecutar train() primero.")

        # Preprocesar
        audio = self.preprocessor.process(audio)

        features = self.feature_extractor.extract_features(audio).reshape(1, -1)
        
        # --- FAST PATH: Producto Interno ---
        dot_idx, dot_score = self._predict_dot_product(features)
        if dot_score > 0.96:
            kw = self.idx_to_keyword[dot_idx]
            logger.info(f"🚀 Fast-path detectado (details): '{kw}' (score={dot_score:.4f})")
            
            # Re-calcular probabilidades para consistencia en el objeto de retorno
            # aunque sea una aproximación basada en el score de similitud
            all_proba = {self.idx_to_keyword[i]: 0.0 for i in range(len(self.keywords))}
            all_proba[kw] = dot_score
            
            return {
                'predicted_keyword': kw,
                'confidence': dot_score,
                'all_probabilities': all_proba,
                'features': features[0].tolist(),
                'method': 'dot_product'
            }
        # ------------------------------------

        pred_idx = int(self.classifier.predict(features)[0])

        dists = np.linalg.norm(self.centroids - features, axis=1).flatten()
        exps = np.exp(-dists)
        probs = exps / np.sum(exps)
        all_proba = {self.idx_to_keyword[i]: float(probs[i]) for i in range(len(self.keywords))}

        return {
            'predicted_keyword': self.idx_to_keyword[pred_idx],
            'confidence': float(probs[pred_idx]),
            'all_probabilities': all_proba,
            'features': features[0].tolist(),
            'method': 'nearest_centroid'
        }

    def save(self, filepath: Path):
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
                'fourier': {
                    'n_bins': self.feature_extractor.fourier.n_bins,
                    'n_fft': self.feature_extractor.fourier.n_fft,
                    'hop_length': self.feature_extractor.fourier.hop_length
                },
                'wavelet_enabled': self.feature_extractor.wavelet_enabled,
                'wavelet': {
                    'wavelet': getattr(self.feature_extractor.wavelet, 'wavelet', None),
                    'level': getattr(self.feature_extractor.wavelet, 'level', None)
                } if self.feature_extractor.wavelet_enabled else None
            },
            'centroids': self.centroids,
            'normalized_centroids': self.normalized_centroids
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"✓ Modelo guardado en {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'FWKeywordClassifier':
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        params = model_data.get('feature_extractor_params', {})
        fourier_params = params.get('fourier', {})
        wavelet_params = params.get('wavelet', {}) if params.get('wavelet_enabled', False) else {}

        instance = cls(keywords=model_data['keywords'], sample_rate=model_data.get('sample_rate', 16000),
                       fourier_params=fourier_params, wavelet_params=wavelet_params)

        instance.keyword_to_idx = model_data['keyword_to_idx']
        instance.idx_to_keyword = model_data['idx_to_keyword']
        instance.classifier = model_data['classifier']
        instance.training_stats = model_data['training_stats']
        instance.centroids = model_data.get('centroids', None)
        instance.normalized_centroids = model_data.get('normalized_centroids', None)
        instance.is_trained = True

        logger.info(f"✓ Modelo FW cargado desde {filepath}")
        logger.info(f"   Keywords: {', '.join(instance.keywords)}")
        logger.info(f"   Test accuracy: {instance.training_stats.get('test_accuracy', 0):.2%}")

        return instance
