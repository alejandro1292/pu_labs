"""
API endpoints para gestión de keywords y samples de audio.
El entrenamiento del modelo se realiza con Random Forest (ver rf_api.py y train_rf.py).
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import json
import logging
from typing import List, Dict, Optional
import wave
import io
import asyncio

from . import database as db
from .synthetic_voice import generate_synthetic_samples

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])

MODELS_DIR = Path("backend/models")
RECORDINGS_DIR = MODELS_DIR / "recordings"


class KeywordCreate(BaseModel):
    """Modelo para crear un nuevo keyword."""
    name: str


def ensure_dirs():
    """Asegura que existan los directorios necesarios."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/keywords")
async def create_keyword_endpoint(keyword_data: KeywordCreate):
    """Crea un nuevo keyword en la base de datos."""
    ensure_dirs()
    
    keyword_name = keyword_data.name.lower().strip()
    
    # Validar nombre
    if not keyword_name or not keyword_name.isalpha():
        raise HTTPException(
            status_code=400,
            detail="El nombre del keyword debe contener solo letras"
        )
    
    # Verificar si ya existe
    existing = await db.get_keyword_by_name(keyword_name)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"El keyword '{keyword_name}' ya existe"
        )
    
    # Crear keyword en BD
    kw_data = await db.create_keyword(keyword_name)
    
    # Crear directorio para recordings
    keyword_dir = RECORDINGS_DIR / keyword_name
    keyword_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "success": True,
        "keyword": {
            "id": kw_data['id'],
            "name": kw_data['name'],
            "n_templates": kw_data['n_templates']
        }
    }


@router.get("/keywords")
async def list_keywords():
    """Lista todos los keywords con su información."""
    ensure_dirs()
    
    keywords = await db.get_all_keywords()
    
    result = []
    for keyword in keywords:
        # Contar samples desde DB
        samples = await db.get_samples_by_keyword(keyword['id'])
        
        # Verificar si tiene modelo RF entrenado
        model_path = MODELS_DIR / "rf_classifier.pkl"
        has_trained_model = model_path.exists()
        
        # Un keyword necesita entrenamiento si tiene samples pero no hay modelo
        # o si tiene menos de 20 samples (mínimo recomendado para RF)
        needs_training = len(samples) > 0 and (not has_trained_model or len(samples) < 20)
        
        result.append({
            "name": keyword['name'],
            "n_templates": keyword.get('n_templates', 0),
            "n_samples": len(samples),
            "needs_training": needs_training,
            "created_at": keyword.get('created_at')
        })
    
    return {"keywords": result}


@router.get("/keywords/{keyword}/samples")
async def get_keyword_samples(keyword: str):
    """Lista todas las muestras de un keyword."""
    kw_data = await db.get_keyword_by_name(keyword)
    if not kw_data:
        raise HTTPException(status_code=404, detail=f"Keyword '{keyword}' no encontrado")
    
    samples = await db.get_samples_by_keyword(kw_data['id'])
    
    return {
        "keyword": keyword,
        "samples": [
            {
                "id": s['id'],
                "filename": s['filename'],
                "file_path": s['file_path'],
                "duration": s['duration'],
                "sample_rate": s.get('sample_rate', 16000),
                "created_at": s['created_at']
            }
            for s in samples
        ]
    }


@router.post("/keywords/{keyword}/samples")
async def upload_sample(keyword: str, file: UploadFile = File(...)):
    """
    Sube una muestra de audio para un keyword.
    El archivo debe ser WAV mono 16kHz.
    """
    # Verificar que el keyword existe
    kw_data = await db.get_keyword_by_name(keyword)
    if not kw_data:
        raise HTTPException(status_code=404, detail=f"Keyword '{keyword}' no encontrado")
    
    # Validar formato
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos WAV")
    
    # Leer archivo
    audio_bytes = await file.read()
    
    # Validar que es WAV válido
    try:
        with io.BytesIO(audio_bytes) as wav_io:
            with wave.open(wav_io, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                
                # Validaciones
                if sample_rate != 16000:
                    raise HTTPException(
                        status_code=400,
                        detail=f"El audio debe ser 16kHz (recibido: {sample_rate}Hz)"
                    )
                if n_channels != 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"El audio debe ser mono (recibido: {n_channels} canales)"
                    )
                if sample_width != 2:
                    raise HTTPException(
                        status_code=400,
                        detail=f"El audio debe ser 16-bit (recibido: {sample_width*8}-bit)"
                    )
                
                duration_s = n_frames / sample_rate
                if duration_s < 0.1 or duration_s > 5.0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Duración debe estar entre 0.1s y 5.0s (recibido: {duration_s:.2f}s)"
                    )
    
    except wave.Error as e:
        raise HTTPException(status_code=400, detail=f"Archivo WAV inválido: {e}")
    
    # Guardar archivo
    keyword_dir = RECORDINGS_DIR / keyword
    keyword_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar nombre único
    import time
    timestamp = int(time.time() * 1000)
    filename = f"{keyword}_{timestamp}.wav"
    file_path = keyword_dir / filename
    
    with open(file_path, 'wb') as f:
        f.write(audio_bytes)
    
    # Guardar en base de datos
    sample_data = await db.add_sample(
        keyword_id=kw_data['id'],
        filename=filename,
        duration=duration_s,
        sample_rate=16000,
        file_path=str(file_path)
    )
    
    logger.info(f"✓ Sample guardado: {filename} ({duration_s:.2f}s)")
    
    return {
        "success": True,
        "sample": {
            "id": sample_data['id'],
            "filename": filename,
            "file_path": str(file_path),
            "duration": duration_s
        }
    }


@router.delete("/keywords/{keyword}/samples/{sample_id}")
async def delete_sample_endpoint(keyword: str, sample_id: int):
    """Elimina una muestra de audio."""
    # Verificar que el keyword existe
    kw_data = await db.get_keyword_by_name(keyword)
    if not kw_data:
        raise HTTPException(status_code=404, detail=f"Keyword '{keyword}' no encontrado")
    
    # Eliminar de BD y obtener datos
    sample_data = await db.delete_sample(sample_id)
    
    if not sample_data:
        raise HTTPException(status_code=404, detail="Sample no encontrado")
    
    # Verificar que el sample pertenece al keyword correcto
    if sample_data['keyword_id'] != kw_data['id']:
        raise HTTPException(status_code=404, detail="Sample no pertenece a este keyword")
    
    # Eliminar archivo del filesystem
    file_path = Path(sample_data['file_path'])
    if file_path.exists():
        file_path.unlink()
        logger.info(f"✓ Archivo eliminado: {file_path}")
    
    return {
        "success": True,
        "message": f"Sample {sample_id} eliminado",
        "deleted_file": sample_data['filename']
    }


@router.delete("/keywords/{keyword}")
async def delete_keyword(keyword: str):
    """Elimina completamente un keyword y todas sus muestras."""
    # Obtener keyword de DB
    kw_data = await db.get_keyword_by_name(keyword)
    if not kw_data:
        raise HTTPException(status_code=404, detail=f"Keyword '{keyword}' no encontrado")
    
    # Eliminar de DB (CASCADE borra samples automáticamente)
    await db.delete_keyword(kw_data['id'])
    
    # Eliminar directorio de recordings
    keyword_dir = RECORDINGS_DIR / keyword
    if keyword_dir.exists():
        import shutil
        shutil.rmtree(keyword_dir)
        logger.info(f"✓ Directorio eliminado: {keyword_dir}")
    
    logger.info(f"✓ Keyword '{keyword}' eliminado completamente")
    
    return {
        "success": True,
        "message": f"Keyword '{keyword}' eliminado",
        "keyword": keyword
    }


@router.post("/keywords/{keyword}/generate-synthetic")
async def generate_synthetic_samples_endpoint(keyword: str, n_samples: int = 10):
    """
    Genera muestras sintéticas de voz para un keyword.
    Usa gTTS y audiomentations para crear variaciones realistas.
    """
    # Verificar que el keyword existe
    kw_data = await db.get_keyword_by_name(keyword)
    if not kw_data:
        raise HTTPException(status_code=404, detail=f"Keyword '{keyword}' no encontrado")
    
    if n_samples < 1 or n_samples > 100:
        raise HTTPException(status_code=400, detail="n_samples debe estar entre 1 y 100")
    
    logger.info(f"Generando {n_samples} muestras sintéticas para '{keyword}'...")
    
    try:
        # Generar samples
        keyword_dir = RECORDINGS_DIR / keyword
        generated_files = await generate_synthetic_samples(keyword, n_samples, keyword_dir)
        
        # Guardar en base de datos
        saved_count = 0
        for file_path in generated_files:
            # Obtener duración del audio
            import wave
            with wave.open(str(file_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
            
            # Guardar en BD
            await db.add_sample(
                keyword_id=kw_data['id'],
                filename=file_path.name,
                duration=duration,
                sample_rate=16000,
                file_path=str(file_path)
            )
            saved_count += 1
        
        logger.info(f"✓ {saved_count} muestras sintéticas guardadas para '{keyword}'")
        
        return {
            "success": True,
            "message": f"{saved_count} muestras sintéticas generadas para '{keyword}'",
            "keyword": keyword,
            "n_generated": len(generated_files),
            "generated_samples": saved_count,
            "files": [str(f) for f in generated_files]
        }
    
    except Exception as e:
        logger.error(f"Error generando muestras sintéticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))
