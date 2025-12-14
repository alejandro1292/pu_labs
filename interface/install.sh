#!/bin/bash

# Script de instalación para Keyword Spotting System
# Instala dependencias del sistema y Python

echo "================================================"
echo "📦 Instalando dependencias del sistema..."
echo "================================================"
echo ""

# Detectar sistema operativo
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Sistema: Linux"
    echo ""
    echo "Instalando dependencias del sistema (requiere sudo)..."
    echo "  • python3-distutils"
    echo "  • python3-dev"
    echo "  • portaudio19-dev"
    echo "  • libasound2-dev"
    echo "  • libportaudio2"
    echo ""
    
    sudo apt-get update
    sudo apt-get install -y python3-distutils python3-dev portaudio19-dev libasound2-dev libportaudio2
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Sistema: macOS"
    echo ""
    echo "Instalando dependencias con Homebrew..."
    
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew no está instalado"
        echo "   Instala desde: https://brew.sh"
        exit 1
    fi
    
    brew install portaudio
    
else
    echo "⚠️  Sistema operativo no detectado automáticamente"
    echo "   Por favor instala manualmente:"
    echo "   • Python 3.8+"
    echo "   • PortAudio"
fi

echo ""
echo "================================================"
echo "🐍 Configurando entorno Python..."
echo "================================================"
echo ""

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "Actualizando pip..."
pip install --upgrade pip setuptools wheel

# Instalar dependencias Python
echo ""
echo "Instalando dependencias Python..."
pip install -r requirements.txt


echo "=================================================="
echo "Instalando dependencias para síntesis de voz"
echo "=================================================="

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo "Activando entorno virtual..."
    source venv/bin/activate
fi

# Instalar ffmpeg (requerido por pydub)
echo ""
echo "Instalando ffmpeg..."
sudo apt-get update
sudo apt-get install -y ffmpeg

# Instalar dependencias de Python
echo ""
echo "Instalando paquetes de Python..."
pip install gTTS>=2.5.0
pip install pydub>=0.25.1

echo ""
echo "=================================================="
echo "✓ Instalación completada"
echo "=================================================="
echo ""
echo "Ahora puedes usar la función de generación de voces sintéticas:"
echo "  - En la interfaz web: botón '🤖 Generar Sintéticas'"
echo "  - Genera 10-20 muestras con variaciones de tono y velocidad"
echo "  - Luego entrena el modelo con el botón '🎓 Entrenar'"
echo ""


echo "=================================================="
echo "Instalando dependencias para síntesis de voz"
echo "=================================================="

# Instalar ffmpeg (requerido por pydub)
echo ""
echo "Instalando ffmpeg..."
sudo apt-get update
sudo apt-get install -y ffmpeg

# Instalar dependencias de Python
echo ""
echo "Instalando paquetes de Python..."
pip install gTTS>=2.5.0
pip install pydub>=0.25.1



echo ""
echo "================================================"
echo "✅ Instalación completada"
echo "================================================"
echo ""
echo "Siguiente paso:"
echo "  1. Entrena keywords: python train.py"
echo "  2. Inicia servidor: ./start.sh"
echo ""
