# Dockerfile para PuLabs
# Multi-stage build para optimizar el tamaño de la imagen

FROM python:3.11-slim as base

# Variables de entorno para Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema necesarias para audio y compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Crear directorios necesarios para datos persistentes
RUN mkdir -p backend/models/recordings/sube \
    backend/models/recordings/baja \
    backend/models/recordings/fuego \
    backend/models/debug_segments

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/', timeout=5)"

# Comando por defecto para iniciar el servidor
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "info"]
