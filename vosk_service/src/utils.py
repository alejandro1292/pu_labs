import os

# Ruta por defecto al modelo Vosk. Se puede sobrescribir con la variable
# de entorno VOSK_MODEL_PATH si se desea usar otro modelo.
MODELO_PATH = os.environ.get('VOSK_MODEL_PATH', '/models/vosk-model-small')

# Frecuencia de muestreo por defecto esperada por el servicio (Hz)
SAMPLE_RATE = int(os.environ.get('VOSK_SAMPLE_RATE', 16000))

