"""
Motor de reconocimiento de voz usando Vosk.

Este módulo implementa la transcripción de audio a texto
utilizando el modelo Vosk pre-entrenado.
"""

import json
import logging
from typing import Optional, Dict
import numpy as np
from vosk import Model, KaldiRecognizer

from .utils import MODELO_PATH, SAMPLE_RATE

logger = logging.getLogger(__name__)


class ReconocedorVoz:
    """
    Motor de reconocimiento de voz basado en Vosk.
    
    Utiliza modelos pre-entrenados de Vosk para transcribir
    audio a texto en español.
    """
    
    def __init__(self, modelo_path: str = MODELO_PATH, sample_rate: int = SAMPLE_RATE):
        """
        Inicializa el reconocedor de voz.
        
        Args:
            modelo_path: Ruta al modelo de Vosk
            sample_rate: Frecuencia de muestreo del audio
        """
        self.modelo_path = modelo_path
        self.sample_rate = sample_rate
        self.modelo = None
        self.reconocedor = None
        
        self._cargar_modelo()
    
    def _cargar_modelo(self) -> None:
        """
        Carga el modelo de Vosk desde disco.
        
        Raises:
            Exception: Si el modelo no puede ser cargado
        """
        try:
            logger.info(f"Cargando modelo de Vosk desde: {self.modelo_path}")
            self.modelo = Model(self.modelo_path)
            self.reconocedor = KaldiRecognizer(self.modelo, self.sample_rate)
            self.reconocedor.SetWords(True)  # Habilitar timestamps de palabras
            logger.info("Modelo de Vosk cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar modelo de Vosk: {e}")
            raise
    
    def transcribir(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Transcribe audio a texto.
        
        Args:
            audio: Array de audio (float32, mono, sample_rate correcto)
            
        Returns:
            Diccionario con:
            - texto: Texto transcrito
            - confianza: Nivel de confianza (si disponible)
            - palabras: Lista de palabras con timestamps
            - resultado_completo: JSON completo de Vosk
        """
        try:
            # Convertir audio a bytes (int16)
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Reiniciar reconocedor para nueva transcripción
            self.reconocedor = KaldiRecognizer(self.modelo, self.sample_rate)
            self.reconocedor.SetWords(True)
            
            # Procesar audio
            logger.info(f"Procesando {len(audio_bytes)} bytes de audio")
            self.reconocedor.AcceptWaveform(audio_bytes)
            
            # Obtener resultado final
            resultado_json = self.reconocedor.FinalResult()
            resultado = json.loads(resultado_json)
            
            # Extraer información
            texto = resultado.get('text', '')
            palabras = resultado.get('result', [])
            
            # Calcular confianza promedio si hay palabras
            confianza = 0.0
            if palabras:
                confianzas = [palabra.get('conf', 0.0) for palabra in palabras]
                confianza = sum(confianzas) / len(confianzas) if confianzas else 0.0
            
            logger.info(f"Transcripción completada: '{texto}' (confianza={confianza:.2f})")
            
            return {
                'texto': texto,
                'confianza': round(confianza, 3),
                'palabras': palabras,
                'resultado_completo': resultado
            }
            
        except Exception as e:
            logger.error(f"Error durante transcripción: {e}")
            raise
    
    def transcribir_por_chunks(self, audio: np.ndarray, chunk_size: int = 4000) -> Dict[str, any]:
        """
        Transcribe audio procesándolo por chunks (útil para audio largo).
        
        Args:
            audio: Array de audio
            chunk_size: Tamaño de cada chunk en muestras
            
        Returns:
            Diccionario con resultado de transcripción
        """
        try:
            # Convertir a int16
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Reiniciar reconocedor
            self.reconocedor = KaldiRecognizer(self.modelo, self.sample_rate)
            self.reconocedor.SetWords(True)
            
            # Procesar por chunks
            num_chunks = len(audio_int16) // chunk_size + 1
            logger.info(f"Procesando audio en {num_chunks} chunks")
            
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i+chunk_size]
                chunk_bytes = chunk.tobytes()
                self.reconocedor.AcceptWaveform(chunk_bytes)
            
            # Obtener resultado final
            resultado_json = self.reconocedor.FinalResult()
            resultado = json.loads(resultado_json)
            
            texto = resultado.get('text', '')
            palabras = resultado.get('result', [])
            
            confianza = 0.0
            if palabras:
                confianzas = [palabra.get('conf', 0.0) for palabra in palabras]
                confianza = sum(confianzas) / len(confianzas) if confianzas else 0.0
            
            logger.info(f"Transcripción por chunks completada: '{texto}'")
            
            return {
                'texto': texto,
                'confianza': round(confianza, 3),
                'palabras': palabras,
                'resultado_completo': resultado
            }
            
        except Exception as e:
            logger.error(f"Error durante transcripción por chunks: {e}")
            raise