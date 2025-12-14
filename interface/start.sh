#!/bin/bash

# Script de inicio para Keyword Spotting System
# Inicia el servidor backend FastAPI con uvicorn

echo "================================================"
echo "🎤 Keyword Spotting System - Iniciando..."
echo "================================================"
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: No se encuentra backend/main.py"
    echo "   Asegúrate de ejecutar este script desde la raíz del proyecto"
    exit 1
fi

# Verificar que el entorno virtual existe
if [ ! -d "venv" ]; then
    echo "⚠️  No se encontró entorno virtual"
    echo "   Creando entorno virtual..."
    python3 -m venv venv
    
    echo "   Activando entorno virtual..."
    source venv/bin/activate
    
    echo "   Instalando dependencias..."
    pip install -r requirements.txt
else
    echo "✓ Activando entorno virtual..."
    source venv/bin/activate
fi

# Verificar que las dependencias están instaladas
echo ""
echo "Verificando dependencias..."
python -c "import fastapi, uvicorn, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Instalando dependencias faltantes..."
    pip install -r requirements.txt
fi

# Verificar si hay templates entrenados
echo ""
if [ -d "backend/models" ] && [ "$(ls -A backend/models/*.npz 2>/dev/null)" ]; then
    echo "✓ Templates encontrados:"
    ls -1 backend/models/*_templates.npz 2>/dev/null | while read file; do
        basename "$file" | sed 's/_templates.npz//' | sed 's/^/  • /'
    done
else
    echo "⚠️  No hay templates entrenados"
    echo "   Ejecuta: python train.py"
    echo "   para entrenar keywords antes de usar el sistema"
fi

echo ""
echo "================================================"
echo "🚀 Iniciando servidor en http://localhost:8000"
echo "================================================"
echo ""
echo "Frontend disponible en:"
echo "  → http://localhost:8000/app/index.html"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo ""

# Iniciar uvicorn
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
