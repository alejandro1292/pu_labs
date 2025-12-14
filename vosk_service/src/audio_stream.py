import numpy as np
import io
import wave
import tempfile
import os
from typing import Optional

class AudioStreamProcessor:
    """Procesa chunks de audio recibidos por streaming y los convierte para Vosk"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_buffer = []
        self.temp_file: Optional[str] = None

    def add_chunk(self, audio_data: bytes):
        """Agrega un chunk de audio al buffer"""
        self.audio_buffer.append(audio_data)

    def clear_buffer(self):
        """Limpia el buffer de audio"""
        self.audio_buffer = []
        if self.temp_file and os.path.exists(self.temp_file):
            os.remove(self.temp_file)
            self.temp_file = None

    def get_audio_duration(self) -> float:
        """Calcula la duración del audio en el buffer (segundos)"""
        total_bytes = sum(len(chunk) for chunk in self.audio_buffer)
        # Asumiendo PCM de 16 bits (2 bytes por muestra)
        samples = total_bytes // 2
        duration = samples / self.sample_rate
        return duration

    def save_to_temp_wav(self) -> str:
        """Guarda el buffer actual como archivo WAV temporal"""
        if not self.audio_buffer:
            raise ValueError("Buffer de audio vacío")

        # Crear archivo temporal
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)

        # Combinar todos los chunks
        audio_data = b''.join(self.audio_buffer)

        # Guardar como WAV
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16 bits = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)

        self.temp_file = temp_path
        return temp_path

    def should_process(self, min_duration: float = 2.0) -> bool:
        """Determina si hay suficiente audio para procesar"""
        return self.get_audio_duration() >= min_duration

