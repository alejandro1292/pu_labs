"""
API endpoints para entrenamiento y uso del clasificador Random Forest.
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

from .rf_classifier import RandomForestKeywordClassifier
from . import database as db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rf", tags=["random-forest"])

MODELS_DIR = Path("backend/models")
RF_MODEL_FILE = MODELS_DIR / "rf_classifier.pkl"


class TrainRequest(BaseModel):
    """Request para entrenar Random Forest."""
    keywords: List[str]
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: int = 20
    cross_validate: bool = True


async def load_audio_samples_from_db(keywords: List[str]) -> Dict[str, List[np.ndarray]]:
    """
    Carga muestras de audio desde la base de datos.
    
    Args:
        keywords: Lista de keywords
    
    Returns:
        Dict {keyword: [audio_array1, audio_array2, ...]}
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
            # Leer audio desde archivo
            file_path = Path(sample['file_path'])
            
            if not file_path.exists():
                logger.warning(f"⚠️ Archivo no encontrado: {file_path}")
                continue
            
            # Leer WAV desde archivo
            with wave.open(str(file_path), 'rb') as wav_file:
                n_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(n_frames)
                
                # Convertir a numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Normalizar a [-1, 1]
                audio_array = audio_array.astype(np.float32) / 32768.0
                
                audio_list.append(audio_array)
        
        audio_samples[keyword] = audio_list
        logger.info(f"✓ Cargadas {len(audio_list)} muestras de '{keyword}'")
    
    return audio_samples


@router.post("/train")
async def train_random_forest(request: TrainRequest):
    """
    Entrena un clasificador Random Forest con las muestras de audio almacenadas.
    
    Requiere al menos 20 muestras por keyword.
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"🌲 INICIANDO ENTRENAMIENTO RANDOM FOREST")
        logger.info(f"{'='*60}")
        logger.info(f"Keywords: {', '.join(request.keywords)}")
        logger.info(f"Parámetros: test_size={request.test_size}, n_estimators={request.n_estimators}")
        
        # 1. Cargar muestras de audio desde DB
        logger.info(f"\n📂 Cargando muestras desde base de datos...")
        audio_samples = await load_audio_samples_from_db(request.keywords)
        
        # Validar mínimo de muestras
        for keyword, audios in audio_samples.items():
            if len(audios) < 5:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{keyword}' tiene solo {len(audios)} muestras. Mínimo requerido: 5"
                )
        
        # 2. Crear y entrenar clasificador
        classifier = RandomForestKeywordClassifier(
            keywords=request.keywords,
            n_estimators=request.n_estimators,
            max_depth=request.max_depth
        )
        
        training_stats = classifier.train(
            audio_samples=audio_samples,
            test_size=request.test_size,
            cross_validate=request.cross_validate
        )
        
        # 3. Guardar modelo
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        classifier.save(RF_MODEL_FILE)
        
        logger.info(f"\n✅ Entrenamiento completado")
        logger.info(f"   Modelo guardado en: {RF_MODEL_FILE}")
        logger.info(f"   Test Accuracy: {training_stats['test_accuracy']:.2%}")
        
        # 4. Recargar el modelo en el servidor principal
        from . import main
        try:
            main.rf_classifier = RandomForestKeywordClassifier.load(RF_MODEL_FILE)
            logger.info(f"✓ Modelo recargado en servidor principal")
            logger.info(f"   Keywords activos: {', '.join(main.rf_classifier.keywords)}")
        except Exception as e:
            logger.error(f"⚠️  Error recargando modelo en servidor: {e}")
        
        return {
            "success": True,
            "model_file": str(RF_MODEL_FILE),
            "training_stats": training_stats
        }
        
    except ValueError as e:
        logger.error(f"❌ Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error durante entrenamiento: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def get_model_info():
    """
    Obtiene información del modelo Random Forest entrenado.
    """
    if not RF_MODEL_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="No hay modelo Random Forest entrenado"
        )
    
    try:
        classifier = RandomForestKeywordClassifier.load(RF_MODEL_FILE)
        
        return {
            "success": True,
            "model_file": str(RF_MODEL_FILE),
            "keywords": classifier.keywords,
            "is_trained": classifier.is_trained,
            "training_stats": classifier.training_stats,
            "feature_dim": classifier.feature_extractor.feature_dim,
            "n_estimators": classifier.classifier.n_estimators,
            "max_depth": classifier.classifier.max_depth
        }
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/feature-importance")
async def get_feature_importance(top_n: int = 20):
    """
    Obtiene las features más importantes del modelo.
    """
    if not RF_MODEL_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="No hay modelo Random Forest entrenado"
        )
    
    try:
        classifier = RandomForestKeywordClassifier.load(RF_MODEL_FILE)
        
        feature_importance = np.array(classifier.training_stats['feature_importance'])
        
        # Top N features
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]
        
        top_features = []
        for rank, idx in enumerate(top_indices, 1):
            importance = feature_importance[idx]
            feature_name = classifier._get_feature_name(idx)
            
            top_features.append({
                'rank': rank,
                'feature_name': feature_name,
                'importance': float(importance)
            })
        
        return {
            "success": True,
            "top_features": top_features,
            "total_features": len(feature_importance)
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/model")
async def delete_model():
    """
    Elimina el modelo Random Forest entrenado.
    """
    if not RF_MODEL_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="No hay modelo para eliminar"
        )
    
    try:
        RF_MODEL_FILE.unlink()
        logger.info(f"✓ Modelo eliminado: {RF_MODEL_FILE}")
        
        return {
            "success": True,
            "message": "Modelo eliminado exitosamente"
        }
    except Exception as e:
        logger.error(f"Error eliminando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare-approaches")
async def compare_dtw_vs_rf():
    """
    Compara el enfoque DTW vs Random Forest.
    Útil para análisis y debugging.
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
        "random_forest": {
            "method": "Random Forest Classifier",
            "features": "Aggregated acoustic features (fixed length)",
            "comparison": "Probabilistic classification",
            "pros": [
                "Muy rápido en inferencia",
                "Data-driven (aprende de datos)",
                "Robusto a variaciones",
                "Métricas claras (accuracy, confusion matrix)"
            ],
            "cons": [
                "Requiere mínimo 20 samples por clase",
                "Pierde información temporal detallada",
                "Necesita re-entrenar para nuevos keywords"
            ]
        },
        "recommendation": {
            "for_few_samples": "DTW (< 10 samples por keyword)",
            "for_many_samples": "Random Forest (≥ 20 samples por keyword)",
            "for_production": "Random Forest (más robusto y rápido)",
            "for_prototyping": "DTW (rápido de implementar)"
        }
    }
    
    return comparison
