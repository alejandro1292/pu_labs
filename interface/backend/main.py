"""
Backend FastAPI para Keyword Spotting en tiempo real con Random Forest.
Maneja WebSocket para streaming de audio y detección de keywords.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
import logging
from .rf_classifier import RandomForestKeywordClassifier
from .training_api import router as training_router
from .rf_api import router as rf_router
from . import database as db
import soundfile as sf
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(title="Keyword Spotting API", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Inicializar base de datos y cargar modelo Random Forest."""
    global rf_classifier
    
    await db.init_db()
    logger.info("Database initialized")
    
    # Intentar cargar modelo Random Forest
    rf_model_path = Path("backend/models/rf_classifier.pkl")
    if rf_model_path.exists():
        try:
            rf_classifier = RandomForestKeywordClassifier.load(rf_model_path)
            logger.info(f"✓ Modelo Random Forest cargado")
            logger.info(f"   Keywords: {', '.join(rf_classifier.keywords)}")
            logger.info(f"   Test Accuracy: {rf_classifier.training_stats.get('test_accuracy', 0):.2%}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo Random Forest: {e}")
            rf_classifier = None
    else:
        logger.warning("⚠️  No hay modelo Random Forest entrenado")
        logger.info("   Entrenar con: python train_rf.py")
        rf_classifier = None


# Incluir router de training
app.include_router(training_router)
# Incluir router de Random Forest
app.include_router(rf_router)

# Montar directorio de recordings como estático para reproducción
from pathlib import Path
recordings_path = Path("backend/models/recordings")
recordings_path.mkdir(parents=True, exist_ok=True)
app.mount("/recordings", StaticFiles(directory=str(recordings_path)), name="recordings")

# Configuración global
SAMPLE_RATE = 16000

# Modelo Random Forest (se carga en startup)
rf_classifier: Optional[RandomForestKeywordClassifier] = None

# Conexiones activas
active_processors: Dict[int, 'AudioStreamProcessor'] = {}
game_connections: Dict[int, WebSocket] = {}  # Conexiones del juego


async def broadcast_detection(keyword: str, confidence: float):
    """
    Envía una detección a todas las conexiones de juego activas.
    """
    disconnected = []
    
    for client_id, ws in game_connections.items():
        try:
            await ws.send_json({
                "type": "detection",
                "keyword": keyword,
                "confidence": confidence
            })
        except Exception as e:
            logger.warning(f"Error enviando a cliente juego {client_id}: {e}")
            disconnected.append(client_id)
    
    # Limpiar conexiones muertas
    for client_id in disconnected:
        if client_id in game_connections:
            del game_connections[client_id]


def reload_rf_model():
    """
    Recarga el modelo Random Forest desde disco.
    Útil cuando se entrena un nuevo modelo sin reiniciar el servidor.
    """
    global rf_classifier
    
    rf_model_path = Path("backend/models/rf_classifier.pkl")
    if rf_model_path.exists():
        try:
            rf_classifier = RandomForestKeywordClassifier.load(rf_model_path)
            logger.info(f"✓ Modelo Random Forest recargado")
            logger.info(f"   Keywords: {', '.join(rf_classifier.keywords)}")
            return True
        except Exception as e:
            logger.error(f"❌ Error recargando modelo: {e}")
            return False
    else:
        logger.warning("⚠️  No hay modelo para recargar")
        rf_classifier = None
        return False


def save_audio_segment(audio_data: np.ndarray, prefix: str = "segment"):
    """Guarda un segmento de audio para debugging."""
    debug_dir = Path("backend/models/debug_segments")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time() * 1000)
    filename = debug_dir / f"{prefix}_{timestamp}.wav"
    
    sf.write(filename, audio_data, SAMPLE_RATE)
    logger.info(f"💾 Segmento guardado: {filename}")
    return filename


class AudioStreamProcessor:
    """
    Procesador de audio en streaming usando Random Forest.
    """
    
    def __init__(self):
        """Inicializa el procesador."""
        self.min_speech_samples = int(SAMPLE_RATE * 0.15)  # 150ms mínimo
        self.high_energy_threshold = 0.03
        self.last_detection_time = 0
        self.cooldown_ms = 2500  # 2.5 segundos entre detecciones
    
    def process_audio_chunk(self, audio_data: np.ndarray) -> Dict:
        """
        Procesa un segmento de audio usando Random Forest.
        
        Args:
            audio_data: Array numpy con audio PCM float32
        
        Returns:
            Diccionario con resultado de detección
        """
        # Log de entrada
        energy = np.mean(audio_data ** 2)
        duration_s = len(audio_data) / SAMPLE_RATE
        is_high_energy = energy > self.high_energy_threshold
        
        logger.info(f"📥 SEGMENTO recibido: {len(audio_data)} samples ({duration_s:.2f}s), energy:{energy:.6f}{' [ALTA]' if is_high_energy else ''}")
        
        # Guardar segmento para debugging
        save_audio_segment(audio_data, "received")
        
        result = {
            "has_voice": True,
            "keyword": None,
            "confidence": 0.0,
            "action": None
        }
        
        # Verificar que hay modelo cargado
        if rf_classifier is None:
            logger.warning("⚠️  No hay modelo Random Forest cargado")
            return result
        
        # Verificar cooldown
        current_time = time.time() * 1000
        time_since_last = current_time - self.last_detection_time
        if time_since_last < self.cooldown_ms:
            logger.debug(f"🕐 En cooldown ({time_since_last:.0f}ms / {self.cooldown_ms:.0f}ms)")
            return result
        
        # Validar duración mínima
        min_required = self.min_speech_samples * 0.8 if is_high_energy else self.min_speech_samples
        if len(audio_data) < min_required:
            logger.warning(f"⚠️  Segmento muy corto ({len(audio_data)} < {int(min_required)}), ignorando")
            return result
        
        # Predecir con Random Forest
        try:
            logger.info(f"🔍 Clasificando con Random Forest...")
            prediction = rf_classifier.predict_with_details(audio_data)
            
            keyword = prediction['predicted_keyword']
            confidence = prediction['confidence']
            all_proba = prediction['all_probabilities']
            
            logger.info(f"📊 Predicción: '{keyword}' (confianza: {confidence:.2%})")
            logger.debug(f"   Todas las probabilidades: {all_proba}")
            
            # Umbral de confianza (ajustable)
            confidence_threshold = 0.35
            
            if confidence >= confidence_threshold:
                result["keyword"] = keyword
                result["confidence"] = confidence
                result["action"] = "detected"
                result["all_probabilities"] = all_proba
                
                self.last_detection_time = current_time
                
                logger.info(f"✅ ¡DETECTADO! Keyword='{keyword}' Confianza={confidence:.2%}")
            else:
                logger.info(f"❌ Confianza insuficiente ({confidence:.2%} < {confidence_threshold:.0%})")
        
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}", exc_info=True)
        
        return result
    
    def reset(self):
        """Reinicia el estado del procesador."""
        self.last_detection_time = 0
        logger.info("🔄 Procesador reseteado")


@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    keywords = rf_classifier.keywords if rf_classifier else []
    is_ready = rf_classifier is not None and rf_classifier.is_trained
    
    return {
        "name": "Keyword Spotting API - Random Forest",
        "version": "2.0.0",
        "keywords": keywords,
        "sample_rate": SAMPLE_RATE,
        "model_loaded": rf_classifier is not None,
        "model_ready": is_ready,
        "test_accuracy": rf_classifier.training_stats.get('test_accuracy', 0) if rf_classifier else 0
    }


@app.get("/status")
async def status():
    """Status del sistema."""
    keywords = rf_classifier.keywords if rf_classifier else []
    is_ready = rf_classifier is not None and rf_classifier.is_trained
    
    return {
        "status": "online",
        "keywords": keywords,
        "model_loaded": rf_classifier is not None,
        "model_ready": is_ready,
        "active_connections": len(active_processors),
        "classifier_type": "Random Forest"
    }


@app.websocket("/ws")
async def websocket_game_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint simple para el juego.
    Solo envía lista de keywords y detecciones (sin procesar audio).
    """
    await websocket.accept()
    client_id = id(websocket)
    
    # Registrar conexión
    game_connections[client_id] = websocket
    
    logger.info(f"Cliente juego {client_id} conectado (total: {len(game_connections)})")
    
    # Recargar modelo para obtener keywords actualizados
    reload_rf_model()
    
    # Verificar que hay modelo
    keywords = rf_classifier.keywords if rf_classifier else []
    
    # Enviar lista de keywords
    await websocket.send_json({
        "type": "keywords",
        "keywords": keywords
    })
    
    try:
        while True:
            # Mantener conexión abierta
            data = await websocket.receive()
            
            if "text" in data:
                try:
                    message = json.loads(data["text"])
                    
                    # Responder a pings u otros mensajes
                    if message.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong"
                        })
                
                except json.JSONDecodeError:
                    pass
    
    except WebSocketDisconnect:
        logger.info(f"Cliente juego {client_id} desconectado")
    
    except Exception as e:
        logger.error(f"Error en WebSocket juego: {e}")
    
    finally:
        # Limpiar conexión
        if client_id in game_connections:
            del game_connections[client_id]
            logger.info(f"Conexión juego {client_id} eliminada (quedan: {len(game_connections)})")


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint para streaming de audio.
    Recibe chunks de audio PCM y envía detecciones de keywords.
    """
    await websocket.accept()
    client_id = id(websocket)
    
    # Crear procesador para esta conexión
    processor = AudioStreamProcessor()
    active_processors[client_id] = processor
    
    logger.info(f"Cliente {client_id} conectado")
    
    # Recargar modelo para obtener keywords actualizados
    reload_rf_model()
    
    # Verificar que hay modelo
    keywords = rf_classifier.keywords if rf_classifier else []
    model_ready = rf_classifier is not None and rf_classifier.is_trained
    
    # Enviar mensaje de bienvenida
    await websocket.send_json({
        "type": "connected",
        "message": "Conectado al servidor de keyword spotting (Random Forest)",
        "keywords": keywords,
        "sample_rate": SAMPLE_RATE,
        "model_ready": model_ready
    })
    
    try:
        while True:
            # Recibir datos del cliente
            data = await websocket.receive()
            
            if "bytes" in data:
                # Datos binarios: audio PCM
                audio_bytes = data["bytes"]
                
                # Convertir bytes a numpy array float32
                # Asumimos PCM float32 (4 bytes por sample)
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Procesar audio
                result = processor.process_audio_chunk(audio_array)
                
                # Si hay detección, enviar al cliente
                if result["action"] == "detected":
                    # Enviar al cliente que está grabando
                    await websocket.send_json({
                        "type": "detection",
                        "keyword": result["keyword"],
                        "confidence": result["confidence"],
                        "timestamp": len(audio_array)
                    })
                    
                    # Broadcast a conexiones de juego
                    await broadcast_detection(result["keyword"], result["confidence"])
            
            elif "text" in data:
                # Mensaje de texto (puede ser configuración o comando)
                try:
                    message = json.loads(data["text"])
                    
                    if message.get("type") == "reset":
                        processor.reset()
                        await websocket.send_json({
                            "type": "reset_ack",
                            "message": "Procesador reiniciado"
                        })
                    
                    elif message.get("type") == "config":
                        # Actualizar configuración si es necesario
                        await websocket.send_json({
                            "type": "config_ack",
                            "message": "Configuración recibida"
                        })
                
                except json.JSONDecodeError:
                    logger.warning(f"Mensaje de texto inválido: {data['text']}")
    
    except WebSocketDisconnect:
        logger.info(f"Cliente {client_id} desconectado")
    
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
    
    finally:
        # Limpiar procesador
        if client_id in active_processors:
            del active_processors[client_id]
        

# Servir frontend como archivos estáticos
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    logger.info(f"Frontend servido desde: {frontend_path}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
