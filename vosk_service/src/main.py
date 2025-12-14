from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import os
import json
from typing import Optional
from .reconocedor import ReconocedorVoz
from .audio_stream import AudioStreamProcessor
import wave
import numpy as np

app = FastAPI()

# Inicializar reconocedor Vosk de forma perezosa para evitar excepciones
# en el arranque si el volumen /models aún no contiene un modelo.
reconocedor = None

def get_reconocedor():
    """Devuelve una instancia de ReconocedorVoz, intentando crearla si es necesario.
    Si la carga falla (p. ej. no hay modelo en /models) devuelve None y registra el error.
    """
    global reconocedor
    if reconocedor is None:
        try:
            reconocedor = ReconocedorVoz()
        except Exception as e:
            print(f"No se pudo inicializar ReconocedorVoz: {e}")
            reconocedor = None
    return reconocedor

class SolicitudTranscripcion(BaseModel):
    ruta_archivo: str
    idioma: str
    # Optional configuration (kept for compatibility, not all apply to Vosk)
    min_silence_duration_ms: int = 1000
    repetition_penalty: float = 1.2
    compression_ratio_threshold: float = 2.4
    no_speech_threshold: float = 0.6
    temperature: float = 0.0
    beam_size: int = 5

@app.get("/")
def read_root():
    return {"estado": "Servicio Vosk Local Activo"}

@app.get("/health")
def health_check():
    # Indicar si el modelo está cargado para facilitar debugging en docker-compose
    modelo_cargado = get_reconocedor() is not None
    return {"status": "healthy" if modelo_cargado else "degraded",
            "service": "vosk",
            "modelo_cargado": modelo_cargado}

@app.post("/transcribir")
def transcribir_audio(solicitud: SolicitudTranscripcion):
    if not os.path.exists(solicitud.ruta_archivo):
        raise HTTPException(status_code=404, detail="Archivo no encontrado en la ruta especificada.")
    
    try:
        # Pass all optional params as kwargs
        config = {
            "min_silence_duration_ms": solicitud.min_silence_duration_ms,
            "repetition_penalty": solicitud.repetition_penalty,
            "compression_ratio_threshold": solicitud.compression_ratio_threshold,
            "no_speech_threshold": solicitud.no_speech_threshold,
            "temperature": solicitud.temperature,
            "beam_size": solicitud.beam_size
        }
        # Leer WAV a numpy float32 normalizado y pasar a Vosk
        with wave.open(solicitud.ruta_archivo, 'rb') as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            # Asumimos 16-bit PCM
            audio_int16 = np.frombuffer(frames, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32767.0

        r = get_reconocedor()
        if r is None:
            raise HTTPException(status_code=503, detail="Modelo Vosk no disponible. Coloque el modelo en ./models y espere a que se cargue.")

        resultado = r.transcribir_por_chunks(audio_float)
        return resultado
    except Exception as e:
        print(f"Error procesando audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/transcribir")
async def websocket_transcribir(websocket: WebSocket):
    """
    WebSocket endpoint para transcripción en streaming.
    
    El cliente debe enviar:
    1. Primer mensaje (JSON): Configuración
       {
         "idioma": "es",
         "sample_rate": 16000,
         "min_duration": 2.0,
         "beam_size": 5,
         "temperature": 0.0
       }
    2. Siguientes mensajes (bytes): Chunks de audio PCM 16-bit
    3. Mensaje "END" (texto): Para finalizar y procesar audio restante
    
    El servidor responde con JSON:
    {
      "tipo": "transcripcion|info|error",
      "texto": "...",
      "duracion": 2.5,
      "segmentos": [...]
    }
    """
    await websocket.accept()
    print("WebSocket conexión aceptada")
    
    audio_processor: Optional[AudioStreamProcessor] = None
    config = {}
    
    try:
        # Primer mensaje: configuración
        config_data = await websocket.receive_text()
        config = json.loads(config_data)
        print(f"Configuración recibida: {config}")
        
        idioma = config.get("idioma", "es")
        sample_rate = config.get("sample_rate", 16000)
        min_duration = config.get("min_duration", 1.0)
        
        # Inicializar procesador de audio
        audio_processor = AudioStreamProcessor(sample_rate=sample_rate)
        
        await websocket.send_json({
            "tipo": "info",
            "mensaje": "Conexión establecida. Enviando chunks de audio..."
        })
        print("Mensaje de confirmación enviado")
        
        # Recibir chunks de audio
        while True:
            try:
                # Intentar recibir como bytes (audio)
                message = await websocket.receive()
                
                if "text" in message:
                    # Mensaje de texto, puede ser "END" u otro comando
                    text_msg = message["text"]
                    print(f"Mensaje texto recibido: {text_msg}")
                    
                    if text_msg == "END":
                        # Procesar audio restante si hay
                        duration = audio_processor.get_audio_duration()
                        print(f"Recibido END. Audio acumulado: {duration:.2f}s")
                        
                        if duration > 0.5:  # Procesar si hay al menos 0.5s
                            await process_and_send(websocket, audio_processor, idioma, config)
                        else:
                            print(f"Audio muy corto ({duration:.2f}s), no se procesa")
                        
                        await websocket.send_json({
                            "tipo": "info",
                            "mensaje": "Transcripción finalizada"
                        })
                        break
                    
                elif "bytes" in message:
                    # Chunk de audio
                    audio_chunk = message["bytes"]
                    audio_processor.add_chunk(audio_chunk)
                    
                    duration = audio_processor.get_audio_duration()
                    print(f"Chunk recibido: {len(audio_chunk)} bytes, duración total: {duration:.2f}s")
                    
                    # Procesar inmediatamente si supera la duración mínima
                    # (el cliente con VAD ya envía segmentos completos)
                    if duration >= min_duration:
                        print(f"Procesando segmento de {duration:.2f}s")
                        await process_and_send(websocket, audio_processor, idioma, config)
                        audio_processor.clear_buffer()
                    else:
                        print(f"Acumulando audio, duración actual: {duration:.2f}s")
                        
            except WebSocketDisconnect:
                print("Cliente desconectado")
                break
            except Exception as e:
                print(f"Error recibiendo mensaje: {e}")
                import traceback
                traceback.print_exc()
                break
                
    except json.JSONDecodeError as e:
        print(f"Error JSON: {e}")
        try:
            await websocket.send_json({
                "tipo": "error",
                "mensaje": "Error: El primer mensaje debe ser JSON válido con configuración"
            })
        except:
            pass
    except Exception as e:
        print(f"Error en WebSocket: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                "tipo": "error",
                "mensaje": f"Error del servidor: {str(e)}"
            })
        except:
            pass
    finally:
        # Limpiar recursos
        if audio_processor:
            audio_processor.clear_buffer()
        print("Conexión WebSocket cerrada")

async def process_and_send(websocket: WebSocket, audio_processor: AudioStreamProcessor, 
                           idioma: str, config: dict):
    """Procesa el audio acumulado y envía el resultado por WebSocket"""
    try:
        print("Iniciando procesamiento de audio...")
        
        # Guardar audio como archivo temporal
        temp_file = audio_processor.save_to_temp_wav()
        print(f"Audio guardado en: {temp_file}")
        
        # Leer WAV y convertir a float32 normalizado
        with wave.open(temp_file, 'rb') as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio_int16 = np.frombuffer(frames, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32767.0

        print(f"Transcribiendo segmento (sr={sr}) con Vosk")
        r = get_reconocedor()
        if r is None:
            # enviar error al cliente y devolver
            try:
                await websocket.send_json({
                    "tipo": "error",
                    "mensaje": "Modelo Vosk no disponible en el servidor. Coloque el modelo en /models y reinicie el servicio."
                })
            except:
                pass
            return

        resultado = r.transcribir_por_chunks(audio_float)
        print(f"Transcripción completada: {resultado.get('texto', '')[:100]}...")

        # Enviar resultado (Vosk devuelve 'palabras' y 'resultado_completo')
        await websocket.send_json({
            "tipo": "transcripcion",
            "texto": resultado.get("texto", ""),
            "duracion": audio_processor.get_audio_duration(),
            "segmentos": resultado.get("palabras", [])
        })
        print("Resultado enviado al cliente")

        # Limpiar archivo temporal
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    except Exception as e:
        print(f"Error procesando audio: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                "tipo": "error",
                "mensaje": f"Error al transcribir: {str(e)}"
            })
        except:
            pass
