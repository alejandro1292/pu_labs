# 🎮 PuLabs

Sistema de juegos controlados por voz usando **Random Forest** para detección de palabras clave en tiempo real. Incluye reconocimiento de voz con Voice Activity Detection (VAD) y múltiples juegos interactivos.

## 🕹️ Juegos Disponibles

### 🚀 Galaxy Voice Commander
Shooter espacial controlado por comandos de voz.
- **Comandos**: "sube", "baja", "fuego"
- **Mecánicas**: Esquiva enemigos, dispara con tu voz
- **Puntuación**: Sistema de vidas y bombas especiales

### 🏃 Voice Jump Platform
Plataformas controladas por gritos.
- **Controles**: Intensidad y duración del grito controlan el salto
- **Mecánicas**: 
  - Salto variable según duración del grito
  - Doble salto en el aire
  - Rebote en el techo (resetea doble salto con chispas)
- **Obstáculos**: Pinchos terrestres y obstáculos colgantes
- **Física juicy**: Squash & stretch, partículas, rotación

### 📚 Keyword Training
Interfaz para entrenar nuevas palabras clave.
- Grabación de muestras
- Generación de voces sintéticas
- Entrenamiento del modelo

### 🎤 Voice Testing
Herramienta de prueba para verificar detección en tiempo real.

## 🚀 Inicio Rápido

### Instalación

```bash
# Clonar repositorio
git clone <repo-url>
cd voice_controlled_games

# Opción 1: Script automático
./install.sh

# Opción 2: Manual
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Iniciar Servidor

```bash
./start.sh
```

El servidor iniciará en `http://localhost:8000`

### Docker (Alternativa)

```bash
# Con Docker Compose
docker-compose up -d

# Acceder a http://localhost:8000
```

Ver [DOCKER.md](DOCKER.md) para más detalles.

## 🎯 Uso Rápido

1. **Entrenar palabras clave** (si es primera vez):
   - Ir a http://localhost:8000/keywords.html
   - Crear keywords: "sube", "baja", "fuego"
   - Grabar 20+ muestras por keyword
   - Entrenar modelo

2. **Jugar**:
   - **Galaxy**: http://localhost:8000/galaxy.html
   - **Platform**: http://localhost:8000/platform.html

3. **Configurar micrófono**:
   - Permitir acceso al micrófono en el navegador
   - Ajustar sensibilidad si es necesario

## 🏗️ Arquitectura

### Backend (FastAPI + WebSocket)

```
Audio Stream (16kHz PCM)
    ↓
Voice Activity Detection (VAD)
    ↓
Feature Extraction (Librosa)
    ↓
Random Forest Classifier
    ↓
WebSocket Event → Frontend
```

**Características extraídas (64 totales):**
- 13 MFCCs (Mel-Frequency Cepstral Coefficients)
- Zero-Crossing Rate (ZCR)
- Spectral Centroid
- Energy (RMS)

Cada característica: `mean`, `std`, `max`, `min`

### Frontend (Vanilla JavaScript)

- **Canvas Games**: Renderizado 60 FPS
- **WebSocket**: Comunicación bidireccional en tiempo real
- **Web Audio API**: Captura y procesamiento de audio
- **VAD Client-side**: Detección de actividad vocal local
- **Circular Visualizer**: Visualización de audio en tiempo real

### Sistema de Temas CSS

Variables temáticas por juego:
- **Galaxy**: Cyan/Purple (#00d9ff, #6c5ce7)
- **Platform**: Red/Yellow (#ff6b6b, #feca57)

Cambio automático mediante clases en `<body>`.

## 📁 Estructura del Proyecto

```
voice_controlled_games/
├── backend/
│   ├── main.py              # FastAPI server + WebSocket
│   ├── rf_classifier.py     # Random Forest classifier
│   ├── rf_api.py            # Training API endpoints
│   ├── training_api.py      # Keyword management
│   ├── database.py          # SQLite database
│   └── models/              # Modelos entrenados
│       └── recordings/      # Grabaciones por keyword
├── frontend/
│   ├── index.html           # Menú principal
│   ├── galaxy.html          # Galaxy game
│   ├── platform.html        # Platform game
│   ├── keywords.html        # Training interface
│   ├── voice.html           # Voice testing
│   ├── css/
│   │   └── style.css        # Estilos unificados con temas
│   └── js/
│       ├── audio.js         # Audio capture + VAD
│       ├── galaxy.js        # Galaxy game logic
│       ├── platform.js      # Platform game logic
│       └── utils.js         # Utilidades compartidas
├── Dockerfile               # Docker image
├── docker-compose.yml       # Docker orchestration
└── start.sh                 # Script de inicio
```

## 🎮 Mecánicas de Juego

### Platform Game - Física Juicy

**Salto continuo:**
- Grita para ascender (velocidad basada en intensidad)
- Suelta para caer con gravedad
- Altura máxima: 250px

**Doble salto:**
- Se activa al despegar del suelo
- Úsalo mientras caes (solo 1 vez)
- Resetea al tocar suelo, plataformas o techo

**Rebote en techo:**
- No causa game over si está vacío
- Resetea doble salto
- Crea chispas doradas
- Efecto squash al impactar

**Efectos visuales:**
- Squash & stretch en saltos/aterrizajes
- Rotación del jugador
- Partículas de estela
- Explosión al morir

### Galaxy Game - Combate Espacial

**Controles por voz:**
- "sube" / "baja": Movimiento vertical
- "fuego": Disparar

**Sistema de puntos:**
- Destruir enemigos: +10 puntos
- 3 vidas iniciales
- 3 bombas especiales (limpian pantalla)

## 🛠️ API Reference

### WebSocket - Audio Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.event === 'keyword_detected') {
    console.log(`${data.keyword}: ${data.confidence}%`);
  }
};

// Enviar audio PCM Float32Array
ws.send(audioData.buffer);
```

### REST API - Training

```bash
# Crear keyword
POST /api/training/keywords
{"name": "saltar"}

# Listar keywords
GET /api/training/keywords

# Subir muestra
POST /api/training/keywords/{id}/samples
Content-Type: multipart/form-data

# Generar voces sintéticas
POST /api/training/keywords/{id}/synthetic
{"count": 10, "voice": "es"}

# Entrenar modelo
POST /api/rf/train
{"keywords": ["sube", "baja", "fuego"]}

# Info del modelo
GET /api/rf/model/info
```

## 🔧 Configuración

### VAD (Voice Activity Detection)

En `frontend/js/audio.js`:

```javascript
const VAD_CONFIG = {
  energyThreshold: 0.002,  // Sensibilidad (más bajo = más sensible)
  silenceChunks: 1,        // Chunks de silencio antes de cortar
  minDuration: 50          // Duración mínima en ms
};
```

### Random Forest

En `backend/rf_classifier.py`:

```python
RandomForestClassifier(
    n_estimators=100,      # Número de árboles
    max_depth=20,          # Profundidad máxima
    min_samples_split=5,   # Mínimo samples para split
    random_state=42
)
```

### Física del Juego

En `frontend/js/platform.js`:

```javascript
const config = {
    gravity: 0.6,
    maxJumpHeight: 250,
    obstacleMinGap: 500,
    obstacleMaxGap: 800,
    ceilingY: 120
};
```

## 📊 Rendimiento del Sistema

### Métricas del Modelo

- **Accuracy**: 92-96% (con 20+ muestras por keyword)
- **Inferencia**: ~2ms por detección
- **Latencia WebSocket**: <50ms
- **FPS Juegos**: 60 FPS constante

### Requisitos de Entrenamiento

| Keyword Quality | Muestras Mínimas | Accuracy Esperada |
|-----------------|------------------|-------------------|
| Buena (clara, sin ruido) | 20 | 92-95% |
| Media (algo de ruido) | 30 | 90-93% |
| Baja (mucho ruido) | 40+ | 85-90% |

**Mejores prácticas:**
- Grabar en ambiente silencioso
- Usar diferentes tonos de voz
- Combinar grabaciones reales + sintéticas
- Mínimo 20 muestras por keyword

## 🔧 Troubleshooting

### El juego no responde a mi voz

**Verificar:**
1. Micrófono permitido en el navegador
2. Visualizador de audio muestra actividad
3. Keywords entrenadas (ver `/api/rf/model/info`)
4. Ajustar sensibilidad VAD (`energyThreshold` en `audio.js`)

**Solución rápida:**
```bash
# Re-entrenar modelo con más muestras
curl -X POST http://localhost:8000/api/rf/train \
  -H "Content-Type: application/json" \
  -d '{"keywords": ["sube", "baja", "fuego"]}'
```

### Falsos positivos frecuentes

**Causa:** Umbral de confianza bajo o keywords similares

**Solución en `backend/main.py`:**
```python
CONFIDENCE_THRESHOLD = 0.75  # Aumentar de 0.60 a 0.75
DETECTION_COOLDOWN = 3000    # Aumentar cooldown a 3 segundos
```

### Audio distorsionado o cortado

**Causa:** VAD muy agresivo

**Solución en `frontend/js/audio.js`:**
```javascript
const VAD_CONFIG = {
  energyThreshold: 0.001,  // Más sensible
  silenceChunks: 2,        // Más tolerancia al silencio
  minDuration: 100         // Duración mínima mayor
};
```

### Docker: Audio no funciona

**Limitación:** Docker no tiene acceso directo al micrófono del host.

**Solución:** Usar instalación nativa con `./start.sh` para desarrollo.

## 📦 Dependencias Principales

```txt
# Backend
fastapi>=0.104.0           # API framework
uvicorn[standard]>=0.24.0  # ASGI server
websockets>=12.0           # WebSocket support
numpy>=1.24.0              # Numerical computing
scipy>=1.11.0              # Scientific computing
audiomentations==0.43.1    # Audio augmentation

# Audio Processing
soundfile>=0.12.1          # Audio I/O
gTTS>=2.5.0               # Text-to-speech
pydub>=0.25.1             # Audio manipulation

# Database
aiosqlite>=0.19.0         # Async SQLite
```

## 🎯 Roadmap

- [ ] Soporte para más idiomas (inglés, francés)
- [ ] Modo multijugador online
- [ ] Leaderboard global
- [ ] Más juegos (endless runner, rhythm game)
- [ ] Reconocimiento de frases completas
- [ ] Mobile support (Progressive Web App)
- [ ] Efectos de sonido dinámicos

## 📄 Licencia

MIT License

## 🤝 Contribución

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -am 'Añade nuevo juego'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

## 📚 Documentación Adicional

- [DOCKER.md](DOCKER.md) - Configuración Docker
- [RF_SYSTEM.md](RF_SYSTEM.md) - Documentación técnica del clasificador
- [Librosa Docs](https://librosa.org/) - Feature extraction
- [FastAPI Docs](https://fastapi.tiangolo.com/) - API framework

## 🎮 Créditos

**Desarrollado con:**
- FastAPI + WebSocket para backend real-time
- Canvas API para renderizado de juegos
- Random Forest (scikit-learn) para clasificación
- Librosa para feature extraction
- Web Audio API para captura de audio

---

**¡Juega con tu voz! 🎤🎮**
