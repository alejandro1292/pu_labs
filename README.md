# PuLabs

## Descripción del Proyecto

PuLabs es un laboratorio de audio experimental enfocado en la exploración, análisis y creación de experiencias interactivas basadas en sonido. El nombre nace de la combinación de "pu", que en guaraní significa sonido, y "labs", abreviatura de laboratorio, reflejando su espíritu de investigación y prueba constante.

El proyecto utiliza el micrófono como herramienta principal para experimentar y validar conceptos relacionados con el audio en tiempo real. Entre sus líneas de trabajo se encuentra la identificación de patrones sonoros mediante algoritmos de Machine Learning, especialmente Random Forest, entrenando el sistema para reconocer palabras clave (keywords) y distintos comportamientos acústicos.

PuLabs también desarrolla juegos y experiencias interactivas que responden a comandos de voz específicos o a características como la longitud e intensidad del audio capturado por el micrófono, integrando el sonido como mecánica central de juego.

Además, el laboratorio incorpora un transcriptor de voz a texto en tiempo real basado en Vosk, permitiendo transformar el audio capturado en texto de forma eficiente y local, ampliando las posibilidades de análisis, accesibilidad e interacción.

PuLabs es, en esencia, un espacio para experimentar con el sonido, combinar creatividad y tecnología, y explorar nuevas formas de interacción a través del audio.

## Autor
- **Nombre**: Alejandro Ortiz
- **Correo**: alejandro.ortiz1292@gmail.com
- **Fecha**: Diciembre 2025

## Arquitectura del Proyecto

El proyecto está estructurado para facilitar el desarrollo y despliegue utilizando contenedores Docker. El archivo `docker-compose.yml` define los servicios necesarios, incluyendo la interfaz web, el backend de procesamiento y el servicio de reconocimiento de voz.

### Interfaz (Frontend/Backend)
- **Frontend**: Construido con HTML, CSS y JavaScript, proporciona una interfaz web intuitiva para interactuar con las funcionalidades del laboratorio. Incluye páginas para entrenamiento de keywords, pruebas de voz, juegos interactivos y transcripción en tiempo real.
- **Backend**: Desarrollado en Python con FastAPI, maneja las APIs para el procesamiento de audio, entrenamiento de modelos y comunicación en tiempo real.

## Estructura del Proyecto

El proyecto está organizado en los siguientes directorios y archivos principales:

- **`docker-compose.yml`**: Archivo de configuración para orquestar los contenedores Docker del proyecto, incluyendo la interfaz, backend y servicios de audio.

- **`interface/`**: Contiene la aplicación web completa.
  - **`DOCKER.md`**: Documentación específica para el despliegue con Docker.
  - **`Dockerfile`**: Instrucciones para construir la imagen Docker de la interfaz.
  - **`install.sh`** y **`start.sh`**: Scripts para instalación y ejecución local.
  - **`requirements.txt`**: Dependencias Python para el entorno.
  - **`README.md`**: Documentación específica de la interfaz.
  - **`backend/`**: Código del servidor backend.
    - **`main.py`**: Punto de entrada del servidor FastAPI.
    - **`rf_api.py`**: API para el modelo Random Forest.
    - **`rf_classifier.py`**: Implementación del clasificador Random Forest.
    - **`synthetic_voice.py`**: Módulo para generación de voz sintética.
    - **`training_api.py`**: API para entrenamiento de modelos.
    - **`database.py`**: Gestión de base de datos.
    - **`data/`**: Almacenamiento de datos de entrenamiento.
    - **`models/`**: Modelos entrenados y segmentos de debug.
  - **`frontend/`**: Archivos estáticos de la interfaz web.
    - Páginas HTML: `index.html`, `keywords.html`, `voice.html`, `galaxy.html`, `platform.html`, `vosk.html`.
    - **`css/`**: Hojas de estilo.
    - **`js/`**: Scripts JavaScript para la lógica del frontend.

- **`models/`**: Modelos pre-entrenados de Kaldi para reconocimiento de voz.
  - **`am/`**: Modelo acústico.
  - **`conf/`**: Configuraciones.
  - **`graph/`**: Grafo de decodificación.
  - **`ivector/`**: Modelo de vectores i.

- **`vosk_service/`**: Servicio independiente para transcripción con Vosk.
  - **`Dockerfile`**: Imagen Docker para el servicio.
  - **`requirements.txt`**: Dependencias.
  - **`scripts/start.sh`**: Script de inicio.
  - **`src/`**: Código fuente del servicio.
    - **`main.py`**: Servidor WebSocket.
    - **`reconocedor.py`**: Lógica de reconocimiento.
    - **`audio_stream.py`**: Manejo de streams de audio.
    - **`utils.py`**: Utilidades.

## Random Forest
Random Forest es un algoritmo de Machine Learning utilizado para la clasificación de comandos de voz. El sistema entrena modelos basados en características extraídas del audio, como MFCC (Mel-Frequency Cepstral Coefficients), para reconocer palabras clave específicas. Esto permite una identificación precisa de comandos de voz en tiempo real, integrándose en juegos y aplicaciones interactivas.

## Servicio Vosk
Vosk es una biblioteca de reconocimiento de voz offline que permite la transcripción de audio a texto en tiempo real. El servicio Vosk se ejecuta en un contenedor separado, procesando el stream de audio del micrófono y enviando el texto transcrito a través de WebSocket para su uso en la interfaz. [Sitio oficial](https://alphacephei.com/vosk/)

## Voz Sintética (Text-to-Speech)
El proyecto incluye funcionalidades de generación de voz sintética utilizando bibliotecas avanzadas:
- **gTTS (Google Text-to-Speech)**: Biblioteca de Python que utiliza la API de Google para convertir texto en habla. [Sitio oficial](https://pypi.org/project/gTTS/)
- **edge-tts (Microsoft Edge Text-to-Speech)**: Herramienta que aprovecha el motor de síntesis de voz de Microsoft Edge para generar audio de alta calidad. [Sitio oficial](https://github.com/rany2/edge-tts)

Estas herramientas permiten crear respuestas auditivas y mejorar la interactividad del sistema.

## WebSocket
WebSocket se utiliza para la comunicación bidireccional en tiempo real entre el frontend y el backend. Permite el streaming continuo de audio desde el micrófono al servidor, y la transmisión de resultados de transcripción o comandos reconocidos de vuelta al cliente, asegurando una experiencia interactiva fluida.

## Uso del Micrófono
El micrófono es el dispositivo principal de entrada. El proyecto captura audio en tiempo real para:
- Entrenamiento de modelos Random Forest con grabaciones de keywords
- Pruebas de reconocimiento de voz
- Control de juegos mediante comandos vocales
- Transcripción continua con Vosk

El audio se procesa localmente para mantener la privacidad y reducir la latencia.

## Ejemplos de Uso

### Entrenamiento de Keywords
1. Accede a la página "Keywords" desde la interfaz principal.
2. Selecciona una palabra clave a entrenar (ej: "sube", "baja").
3. Graba varias muestras de audio diciendo la palabra.
4. El sistema entrena un modelo Random Forest con las características MFCC extraídas.
5. Una vez entrenado, el modelo puede reconocer la palabra en tiempo real.

### Prueba de Reconocimiento de Voz
1. Ve a la página "Prueba de Voz".
2. Permite el acceso al micrófono.
3. Di comandos de voz entrenados previamente.
4. El sistema clasifica el audio usando Random Forest y muestra el resultado reconocido.

### Juegos Interactivos
- **Galaxy Commander**: Controla una nave espacial diciendo comandos como "sube", "baja", "fuego".
- **Voice Jump**: Salta obstáculos vocalizando sonidos o palabras específicas.
- Los juegos integran el reconocimiento de voz como mecánica central de jugabilidad.

### Transcripción con Vosk
1. Abre la página "Transcripción Vosk".
2. Inicia el streaming de audio desde el micrófono.
3. Habla normalmente; el texto se transcribe en tiempo real y se muestra en la interfaz.
4. Útil para accesibilidad, análisis de audio o integración en aplicaciones.

### Generación de Voz Sintética
- Utiliza la API de backend para convertir texto en audio usando gTTS o edge-tts.
- Ejemplo: Envía un POST a `/api/tts` con texto, y recibe audio generado.

## Instalación y Uso
1. Clona el repositorio
2. Ejecuta `docker-compose up` para iniciar los servicios
3. Accede a la interfaz web en `http://localhost`
4. Comienza experimentando con las diferentes funcionalidades