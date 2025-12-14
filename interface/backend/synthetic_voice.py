"""
Generador de voces sintéticas para entrenamiento de keywords.
Utiliza gTTS + edge_tts para sintetizar voces, pydub para variaciones y audiomentations para realismo (ruido)
"""
import os
import tempfile
import wave
from pathlib import Path
from typing import List, Dict
import numpy as np
from gtts import gTTS
import asyncio
import edge_tts
from pydub import AudioSegment
from pydub.effects import speedup
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import logging

logger = logging.getLogger(__name__)


class SyntheticVoiceGenerator:
    """
    Genera muestras sintéticas de voz con variaciones de tono y velocidad.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Inicializa el generador de voces sintéticas.
        
        Args:
            sample_rate: Frecuencia de muestreo objetivo (16000 Hz)
        """
        self.sample_rate = sample_rate
        self.temp_dir = Path(tempfile.gettempdir()) / "keyword_synthetic"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar pipeline de augmentación con audiomentations
        # Configuración MÁS CONSERVADORA para evitar distorsión
        self.augmenter = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=0.6),
            TimeStretch(min_rate=0.90, max_rate=1.10, p=0.5),  # Reducido: 90%-110%
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),  # Reducido: -2 a +2
            Shift(min_shift=-0.15, max_shift=0.15, p=0.3),  # Reducido: ±15%
        ])
        
        logger.info("✓ Pipeline de augmentación inicializado")
    
    EDGE_VOICES_ES = [
        "es-ES-AlvaroNeural",
        "es-ES-ElviraNeural",
        "es-MX-DaliaNeural",
        "es-MX-JorgeNeural",
        "es-AR-ElenaNeural",
    ]

    VOICE_VARIATIONS = [
        {"rate": "+0%",  "pitch": "+0Hz"},
        {"rate": "+5%", "pitch": "+1Hz"},
        {"rate": "-5%", "pitch": "-1Hz"},
        {"rate": "+5%",  "pitch": "+2Hz"},
        {"rate": "-5%",  "pitch": "-2Hz"},
    ]

    GTTS_VARIATIONS = [
        {'tld': 'com.mx', 'slow': False, 'pitch_shift': 0},
        {'tld': 'com.mx', 'slow': True, 'pitch_shift': 0},
        {'tld': 'es', 'slow': False, 'pitch_shift': 0},
        {'tld': 'es', 'slow': True, 'pitch_shift': 0},
        {'tld': 'com.ar', 'slow': False, 'pitch_shift': 0},
        {'tld': 'com', 'slow': False, 'pitch_shift': 0},
        {'tld': 'com', 'slow': True, 'pitch_shift': 0},
        {'tld': 'com.mx', 'slow': True, 'pitch_shift': -4},
        {'tld': 'es', 'slow': True, 'pitch_shift': -4},
        {'tld': 'com.ar', 'slow': True, 'pitch_shift': -5},
        {'tld': 'com', 'slow': True, 'pitch_shift': -4},
    ]

    def generate_variations(self, 
                           text: str, 
                           n_samples: int = 10,
                           lang: str = 'es',
                           edge_ratio: float = 0.7) -> List[Path]:
        """
        Genera múltiples variaciones sintéticas de una frase.
        Usa diferentes voces (TLDs) y aplica variaciones de tono/velocidad.
        
        Args:
            text: Texto a sintetizar
            n_samples: Número de variaciones a generar
            lang: Código de idioma ('es' para español)
        
        Returns:
            Lista de paths a archivos WAV generados
        """
        generated_files = []
        
        try:
            logger.info(f"Generando {n_samples} voces sintéticas para: '{text}'")
            n_edge = int(n_samples * edge_ratio)
            n_gtts = n_samples - n_edge
            # --- EDGE-TTS ---
            edge_variants = []
            for voice in self.EDGE_VOICES_ES:
                for var in self.VOICE_VARIATIONS:
                    edge_variants.append({
                        "voice": voice,
                        "rate": var["rate"],
                        "pitch": var["pitch"]
                    })
            # Seleccionar n_edge combinaciones únicas
            import random
            random.shuffle(edge_variants)
            edge_variants = edge_variants[:n_edge]
            for idx, ev in enumerate(edge_variants):
                try:
                    logger.info(f"[edge-tts] {ev['voice']} rate={ev['rate']} pitch={ev['pitch']}")
                    base_audio = self._generate_edge_tts_audio(text, ev["voice"], ev["rate"], ev["pitch"])
                    if base_audio is None:
                        logger.warning(f"No se pudo generar audio con edge-tts: {ev}")
                        continue
                    output_path = self.temp_dir / f"synthetic_{text.replace(' ', '_')}_edge_{idx+1:03d}.wav"
                    # Convertir a numpy y augmentar
                    audio_array = self._audiosegment_to_numpy(base_audio)
                    augmented_array = self.augmenter(samples=audio_array, sample_rate=self.sample_rate)
                    self._save_numpy_as_wav(augmented_array, output_path)
                    generated_files.append(output_path)
                except Exception as e:
                    logger.error(f"Error edge-tts: {e}")
                    continue
            
            random.shuffle(self.GTTS_VARIATIONS)
            self.GTTS_VARIATIONS = self.GTTS_VARIATIONS[:n_gtts]
            for idx, gc in enumerate(self.GTTS_VARIATIONS):
                try:
                    base_pitch_shift = gc.get('pitch_shift', 0)
                    base_audio = self._generate_base_audio(text, lang, gc['tld'], gc['slow'])
                    if base_audio is None:
                        logger.warning(f"No se pudo generar audio con TLD: {gc['tld']}")
                        continue
                    # Variación de volumen y pitch
                    config = self._get_variation_configs(1)[0]
                    varied_audio = self._apply_variations(base_audio, config)
                    audio_array = self._audiosegment_to_numpy(varied_audio)
                    if base_pitch_shift != 0:
                        pitch_shifter = PitchShift(
                            min_semitones=base_pitch_shift,
                            max_semitones=base_pitch_shift,
                            p=1.0
                        )
                        audio_array = pitch_shifter(samples=audio_array, sample_rate=self.sample_rate)
                    augmented_array = self.augmenter(samples=audio_array, sample_rate=self.sample_rate)
                    output_path = self.temp_dir / f"synthetic_{text.replace(' ', '_')}_gtts_{idx+1:03d}.wav"
                    self._save_numpy_as_wav(augmented_array, output_path)
                    generated_files.append(output_path)
                except Exception as e:
                    logger.error(f"Error gTTS: {e}")
                    continue
            logger.info(f"Generadas {len(generated_files)} muestras sintéticas")
        except Exception as e:
            logger.error(f"Error en generate_variations: {e}")
        return generated_files

    def _generate_edge_tts_audio(self, text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> AudioSegment:
        """
        Genera audio base usando edge-tts (Microsoft TTS) y lo retorna como AudioSegment.
        Args:
            text: Texto a sintetizar
            voice: Nombre de la voz edge-tts (ej: 'es-ES-ElviraNeural')
            rate: Variación de velocidad (ej: '+15%')
            pitch: Variación de pitch (ej: '+2Hz')
        Returns:
            AudioSegment con el audio base
        """
        try:
            temp_mp3 = self.temp_dir / f"temp_edge_{voice.replace('-', '_')}.mp3"
            async def synthesize():
                communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
                await communicate.save(str(temp_mp3))
            asyncio.run(synthesize())
            audio = AudioSegment.from_mp3(str(temp_mp3))
            temp_mp3.unlink()
            return audio
        except Exception as e:
            logger.error(f"Error generando audio base con edge-tts {voice}: {e}")
            return None
    
    def _generate_base_audio(self, text: str, lang: str, tld: str = 'com', slow: bool = False) -> AudioSegment:
        """
        Genera audio base usando gTTS con un TLD específico (voz diferente).
        
        Args:
            text: Texto a sintetizar
            lang: Código de idioma
            tld: Top-Level Domain para seleccionar voz/acento
                 'com.mx' = México (femenino)
                 'es' = España (masculino)
                 'com.ar' = Argentina
                 'com' = Global
            slow: Si True, genera audio más lento y claro
        
        Returns:
            AudioSegment con el audio base
        """
        try:
            # Generar con gTTS usando TLD específico y modo slow
            tts = gTTS(text=text, lang=lang, tld=tld, slow=slow)
            
            # Guardar en archivo temporal
            temp_mp3 = self.temp_dir / f"temp_base_{tld.replace('.', '_')}.mp3"
            tts.save(str(temp_mp3))
            
            # Cargar con pydub
            audio = AudioSegment.from_mp3(str(temp_mp3))
            
            # Limpiar temporal
            temp_mp3.unlink()
            
            return audio
            
        except Exception as e:
            logger.error(f"Error generando audio base con TLD {tld}: {e}")
            return None
    
    def _get_voice_name(self, tld: str, slow: bool = False, pitch_shift: int = 0) -> str:
        """Retorna nombre descriptivo de la voz según TLD."""
        voice_names = {
            'com.mx': 'México🇲🇽',
            'es': 'España🇪🇸',
            'com.ar': 'Argentina🇦🇷',
            'com': 'Global🌍'
        }
        name = voice_names.get(tld, tld)
        speed_emoji = '🐌' if slow else '⚡'
        
        # Emoji para voces graves masculinas
        if pitch_shift <= -4:
            return f"{name}{speed_emoji}🎙️"  # Voz grave/masculina
        else:
            return f"{name}{speed_emoji}"
    
    def _get_variation_configs(self, n_samples: int) -> List[Dict]:
        """
        Genera configuraciones de variación distribuidas uniformemente.
        Solo volumen - pitch y speed ahora son manejados por audiomentations.
        
        Args:
            n_samples: Número de variaciones
        
        Returns:
            Lista de diccionarios con parámetros de variación (solo volume_gain)
        """
        configs = []
        
        # Solo variamos volumen con pydub
        # audiomentations maneja pitch/speed con mejores algoritmos
        volume_gains = np.linspace(-3, 3, min(n_samples, 7))  # -3 a +3 dB
        
        # Distribuir variaciones uniformemente
        for i in range(n_samples):
            config = {
                'volume_gain': float(volume_gains[i % len(volume_gains)])
            }
            configs.append(config)
        
        return configs
    
    def _apply_variations(self, audio: AudioSegment, config: Dict) -> AudioSegment:
        """
        Aplica variaciones SUTILES de volumen.
        Pitch y velocidad se manejan con audiomentations para mejor calidad.
        
        Args:
            audio: AudioSegment original
            config: Diccionario con parámetros de variación
        
        Returns:
            AudioSegment modificado
        """
        result = audio
        
        # ELIMINADO: Pitch shift con pydub (causaba sonidos acelerados/agudos)
        # Ahora audiomentations maneja pitch con algoritmos más avanzados
        
        # ELIMINADO: Speed factor con pydub (se suma con TimeStretch de audiomentations)
        
        # Solo aplicar cambio de volumen (seguro y simple)
        volume_gain = config.get('volume_gain', 0)
        if abs(volume_gain) > 0.1:
            result = result + volume_gain
        
        return result
    
    def _audiosegment_to_numpy(self, audio: AudioSegment) -> np.ndarray:
        """
        Convierte AudioSegment a numpy array float32 normalizado.
        
        Args:
            audio: AudioSegment a convertir
        
        Returns:
            numpy array float32 con valores en rango [-1, 1]
        """
        # Convertir a mono 16kHz
        audio_mono = audio.set_channels(1).set_frame_rate(self.sample_rate)
        
        # Obtener datos crudos
        samples = np.array(audio_mono.get_array_of_samples())
        
        # Normalizar a float32 [-1, 1]
        if audio_mono.sample_width == 2:  # 16-bit
            samples = samples.astype(np.float32) / 32768.0
        elif audio_mono.sample_width == 1:  # 8-bit
            samples = (samples.astype(np.float32) - 128) / 128.0
        else:  # 32-bit
            samples = samples.astype(np.float32) / 2147483648.0
        
        return samples
    
    def _save_numpy_as_wav(self, audio_array: np.ndarray, output_path: Path):
        """
        Guarda numpy array como WAV mono 16kHz.
        
        Args:
            audio_array: Array numpy float32 con valores en [-1, 1]
            output_path: Ruta de salida
        """
        # Convertir de float32 a int16
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Guardar como WAV
        with wave.open(str(output_path), 'w') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
    
    def _save_as_wav(self, audio: AudioSegment, output_path: Path):
        """
        Guarda AudioSegment como WAV mono 16kHz.
        
        Args:
            audio: AudioSegment a guardar
            output_path: Ruta de salida
        """
        # Convertir a mono
        audio_mono = audio.set_channels(1)
        
        # Resamplear a 16kHz
        audio_16k = audio_mono.set_frame_rate(self.sample_rate)
        
        # Exportar como WAV
        audio_16k.export(
            str(output_path),
            format='wav',
            parameters=['-ac', '1', '-ar', str(self.sample_rate)]
        )
    
    def cleanup(self):
        """Limpia archivos temporales."""
        try:
            for file in self.temp_dir.glob("temp_*.mp3"):
                file.unlink()
        except Exception as e:
            logger.warning(f"Error limpiando temporales: {e}")


async def generate_synthetic_samples(keyword: str, 
                                     n_samples: int = 10,
                                     output_dir: Path = None) -> List[Path]:
    """
    Función auxiliar para generar muestras sintéticas de un keyword.
    Ejecuta la generación en un thread para no bloquear el event loop.
    
    Args:
        keyword: Palabra clave a sintetizar
        n_samples: Número de variaciones
        output_dir: Directorio de salida (default: backend/models/recordings/{keyword})
    
    Returns:
        Lista de paths a archivos generados
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    if output_dir is None:
        output_dir = Path("backend/models/recordings") / keyword
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate():
        """Función interna que se ejecuta en thread separado."""
        generator = SyntheticVoiceGenerator()
        
        # Generar variaciones temporales
        temp_files = generator.generate_variations(keyword, n_samples=n_samples, lang='es')
        
        # Mover a directorio final
        final_files = []
        for i, temp_file in enumerate(temp_files):
            final_path = output_dir / f"{keyword}_{i+1:03d}_synthetic.wav"
            
            # Copiar archivo
            import shutil
            shutil.copy(str(temp_file), str(final_path))
            final_files.append(final_path)
            
            # Eliminar temporal
            temp_file.unlink()
        
        generator.cleanup()
        
        return final_files
    
    # Ejecutar en thread pool para no bloquear
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        final_files = await loop.run_in_executor(executor, _generate)
    
    return final_files
