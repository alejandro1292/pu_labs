FROM python:3.10-slim-bookworm

WORKDIR /app

# Instalar dependencias del sistema necesarias (ffmpeg útil pero opcional)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    ffmpeg \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip, setuptools and wheel to avoid incompatibility with some wheels
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy startup scripts and make executable
COPY scripts /app/scripts
RUN chmod +x /app/scripts/*.sh

EXPOSE 8000

# Use our start script which will download a model into /models if missing
ENTRYPOINT ["/app/scripts/start.sh"]
