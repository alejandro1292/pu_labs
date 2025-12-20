"""
API endpoints para entrenamiento y uso del clasificador Fourier+Wavelet (FW).
Basado en rf_api.py — mismos endpoints y parámetros, pero usa FWKeywordClassifier.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import json
import logging
from typing import List, Dict, Optional
import wave
import io
from fastapi.responses import Response

from .fw_classifier import FWKeywordClassifier
from . import database as db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/fw", tags=["fw"])

MODELS_DIR = Path("backend/models")
FW_MODEL_FILE = MODELS_DIR / "fw_classifier.pkl"


class TrainRequest(BaseModel):
    """Request para entrenar (compatible con la interfaz RF)."""
    keywords: List[str]
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: int = 20
    cross_validate: bool = True


async def load_audio_samples_from_db(keywords: List[str]) -> Dict[str, List[np.ndarray]]:
    """
    Carga muestras de audio desde la base de datos.
    """
    audio_samples = {}
    for keyword in keywords:
        kw_data = await db.get_keyword_by_name(keyword)
        if not kw_data:
            raise ValueError(f"Keyword '{keyword}' no existe en la base de datos")

        samples = await db.get_samples_by_keyword(kw_data['id'])

        if len(samples) < 20:
            logger.warning(f"⚠️ '{keyword}' tiene solo {len(samples)} samples (mínimo recomendado: 20)")

        audio_list = []
        for sample in samples:
            file_path = Path(sample['file_path'])
            if not file_path.exists():
                logger.warning(f"⚠️ Archivo no encontrado: {file_path}")
                continue

            with wave.open(str(file_path), 'rb') as wav_file:
                n_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(n_frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
                audio_list.append(audio_array)

        audio_samples[keyword] = audio_list
        logger.info(f"✓ Cargadas {len(audio_list)} muestras de '{keyword}'")

    return audio_samples


@router.post("/train")
async def train_fw(request: TrainRequest):
    """
    Entrena un clasificador FW (Fourier + Wavelet) con las muestras de audio almacenadas.
    Mantiene la misma interfaz y requisitos que el endpoint RF.
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"🌊 INICIANDO ENTRENAMIENTO FW (Fourier+Wavelet)")
        logger.info(f"{'='*60}")
        logger.info(f"Keywords: {', '.join(request.keywords)}")
        logger.info(f"Parámetros: test_size={request.test_size}, n_estimators={request.n_estimators} (no aplican a FW)")

        logger.info(f"\n📂 Cargando muestras desde base de datos...")
        audio_samples = await load_audio_samples_from_db(request.keywords)

        for keyword, audios in audio_samples.items():
            if len(audios) < 5:
                raise HTTPException(status_code=400, detail=f"'{keyword}' tiene solo {len(audios)} muestras. Mínimo requerido: 5")

        classifier = FWKeywordClassifier(keywords=request.keywords)

        training_stats = classifier.train(audio_samples=audio_samples, test_size=request.test_size, cross_validate=request.cross_validate)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        classifier.save(FW_MODEL_FILE)

        logger.info(f"\n✅ Entrenamiento completado")
        logger.info(f"   Modelo guardado en: {FW_MODEL_FILE}")
        logger.info(f"   Test Accuracy: {training_stats['test_accuracy']:.2%}")

        from . import main
        try:
            main.fw_classifier = FWKeywordClassifier.load(FW_MODEL_FILE)
            # Si la configuración actual está en 'fw', recargar como activo
            if getattr(main, 'classifier_type', 'rf') == 'fw':
                main.reload_model()
                logger.info(f"✓ Modelo FW recargado en servidor principal (activo)")
            else:
                logger.info(f"✓ Modelo FW guardado en servidor principal (no activo)")
            logger.info(f"   Keywords activos: {', '.join(main.fw_classifier.keywords)}")
        except Exception as e:
            logger.error(f"⚠️  Error recargando modelo en servidor: {e}")

        return {
            "success": True,
            "model_file": str(FW_MODEL_FILE),
            "training_stats": training_stats
        }

    except ValueError as e:
        logger.error(f"❌ Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error durante entrenamiento FW: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def get_model_info():
    """Obtiene información del modelo FW entrenado."""
    if not FW_MODEL_FILE.exists():
        raise HTTPException(status_code=404, detail="No hay modelo FW entrenado")

    try:
        classifier = FWKeywordClassifier.load(FW_MODEL_FILE)
        return {
            "success": True,
            "model_file": str(FW_MODEL_FILE),
            "keywords": classifier.keywords,
            "is_trained": classifier.is_trained,
            "training_stats": classifier.training_stats,
            "feature_dim": classifier.feature_extractor.feature_dim,
            "fourier_params": classifier.training_stats.get('fourier_params'),
            "wavelet_enabled": classifier.training_stats.get('wavelet_enabled')
        }
    except Exception as e:
        logger.error(f"Error cargando modelo FW: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/feature-importance")
async def get_feature_importance(top_n: int = 20):
    """
    Endpoint mantenido por compatibilidad, pero FW no provee importancias de features.
    Retorna lista vacía y la dimensión de features.
    """
    if not FW_MODEL_FILE.exists():
        raise HTTPException(status_code=404, detail="No hay modelo FW entrenado")

    try:
        classifier = FWKeywordClassifier.load(FW_MODEL_FILE)
        feature_dim = classifier.feature_extractor.feature_dim
        return {
            "success": True,
            "top_features": [],
            "total_features": feature_dim,
            "detail": "Feature importance no disponible para FW (NearestCentroid)"
        }
    except Exception as e:
        logger.error(f"Error obteniendo feature importance FW: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/model")
async def delete_model():
    """Elimina el modelo FW entrenado."""
    if not FW_MODEL_FILE.exists():
        raise HTTPException(status_code=404, detail="No hay modelo para eliminar")

    try:
        FW_MODEL_FILE.unlink()
        logger.info(f"✓ Modelo eliminado: {FW_MODEL_FILE}")
        return {"success": True, "message": "Modelo eliminado exitosamente"}
    except Exception as e:
        logger.error(f"Error eliminando modelo FW: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare-approaches")
async def compare_approaches():
    """
    Compara enfoques DTW, RF y FW.
    """
    comparison = {
        "dtw_approach": {
            "method": "Dynamic Time Warping + Statistical Detector",
            "features": "MFCC sequences (variable length)",
            "comparison": "DTW distance to centroid",
            "pros": [
                "No requiere muchas muestras",
                "Maneja bien variaciones temporales",
                "Interpretable (distancia directa)"
            ],
            "cons": [
                "Computacionalmente costoso en streaming",
                "Sensible a ruido y variaciones",
                "Difícil ajustar thresholds"
            ]
        },
        "fourier_wavelet": {
            "method": "Fourier + Wavelet features + Nearest Centroid",
            "features": "STFT bands + Wavelet coeff stats (fixed length)",
            "comparison": "Distance to centroid in combined feature space",
            "pros": [
                "Rápido en inferencia",
                "Captura características tanto espectrales como multiresolución",
                "Requiere relativamente pocas muestras"
            ],
            "cons": [
                "Menos expresivo que modelos basados en secuencias",
                "La calidad depende de parámetros de extracción (bins, niveles)"
            ]
        },
        "recommendation": {
            "for_few_samples": "DTW (< 10 samples por keyword)",
            "for_many_samples": "Random Forest (≥ 20 samples por keyword)",
            "for_balanced_speed_accuracy": "FW (Fourier+Wavelet)"
        }
    }

    return comparison


@router.get("/spectrogram/{keyword}")
async def get_keyword_spectrogram(keyword: str):
    """
    Genera y retorna un espectrograma de una muestra aleatoria del keyword.
    """
    try:
        # 1. Buscar samples en DB
        kw_data = await db.get_keyword_by_name(keyword)
        if not kw_data:
            raise HTTPException(status_code=404, detail=f"Keyword '{keyword}' no encontrado")

        samples = await db.get_samples_by_keyword(kw_data['id'])
        if not samples:
            raise HTTPException(status_code=404, detail=f"No hay muestras para '{keyword}'")

        # 2. Cargar todos los audios y promediar sus magnitudes STFT
        all_audios = []
        for s in samples:
            f_path = Path(s['file_path'])
            if not f_path.exists(): continue
            with wave.open(str(f_path), 'rb') as wav_file:
                n_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(n_frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                all_audios.append(audio_array)

        if not all_audios:
            raise HTTPException(status_code=404, detail="No se pudieron cargar audios")

        # 3. Preparar extractor y promediar
        if FW_MODEL_FILE.exists():
            classifier = FWKeywordClassifier.load(FW_MODEL_FILE)
            extractor = classifier.feature_extractor.fourier
        else:
            from .fw_classifier import FourierFeatureExtractor
            extractor = FourierFeatureExtractor()

        # Encontrar longitud máxima para padding
        max_len = max(len(a) for a in all_audios)
        
        import librosa
        stft_sum = None
        count = 0
        
        for audio in all_audios:
            # Pad/trim a la longitud máxima para que las STFTs tengan el mismo tamaño
            audio_fixed = librosa.util.fix_length(audio, size=max_len)
            stft = np.abs(librosa.stft(audio_fixed, n_fft=extractor.n_fft, hop_length=extractor.hop_length))
            
            if stft_sum is None:
                stft_sum = stft
            else:
                stft_sum += stft
            count += 1
            
        avg_stft = stft_sum / count
        
        # 4. Generar imagen
        img_bytes = extractor.generate_spectrogram(magnitude_matrix=avg_stft, title=f"Average Spectrogram: {keyword} ({count} samples)")
        
        return Response(content=img_bytes, media_type="image/png")

    except Exception as e:
        logger.error(f"Error generando espectrograma para {keyword}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
