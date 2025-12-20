/**
 * Audio Visualization Utilities
 * Módulo reutilizable para visualización circular de audio en tiempo real
 */

// =====================================
// WEBSOCKET MANAGER
// =====================================

/**
 * Configuración común de WebSocket
 */
const WEBSOCKET_CONFIG = {
    baseUrl: `ws://${window.location.host}`,
    reconnectDelay: 3000,
    maxReconnectAttempts: 5
};

/**
 * Inicializa conexión WebSocket con manejo de mensajes personalizado
 * @param {string} endpoint - Endpoint del WebSocket (ej: '/ws/audio', '/ws')
 * @param {Function} messageHandler - Función callback para manejar mensajes recibidos
 * @param {Object} options - Opciones adicionales
 * @returns {Promise<WebSocket>} Promesa que resuelve con el WebSocket conectado
 */
function initWebSocket(endpoint, messageHandler, options = {}) {
    return new Promise((resolve, reject) => {
        const wsUrl = `${WEBSOCKET_CONFIG.baseUrl}${endpoint}`;
        let reconnectAttempts = 0;
        let reconnectTimer = null;
        
        function connect() {
            try {
                const ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    console.log(`✓ WebSocket conectado: ${endpoint}`);
                    reconnectAttempts = 0;
                    
                    if (options.onOpen) {
                        options.onOpen(ws);
                    }
                    
                    resolve(ws);
                };
                
                ws.onmessage = (event) => {
                    try {
                        // Intentar parsear como JSON
                        const data = JSON.parse(event.data);
                        messageHandler(data, ws);
                    } catch (error) {
                        // Si no es JSON, pasar el dato raw
                        messageHandler(event.data, ws);
                    }
                };
                
                ws.onerror = (error) => {
                    console.error(`✗ Error en WebSocket (${endpoint}):`, error);
                    
                    if (options.onError) {
                        options.onError(error, ws);
                    }
                };
                
                ws.onclose = () => {
                    console.log(`WebSocket cerrado: ${endpoint}`);
                    
                    if (options.onClose) {
                        options.onClose(ws);
                    }
                    
                    // Auto-reconexión si está habilitado
                    if (options.autoReconnect && reconnectAttempts < WEBSOCKET_CONFIG.maxReconnectAttempts) {
                        reconnectAttempts++;
                        console.log(`Reintentando conexión (${reconnectAttempts}/${WEBSOCKET_CONFIG.maxReconnectAttempts})...`);
                        
                        reconnectTimer = setTimeout(() => {
                            connect();
                        }, WEBSOCKET_CONFIG.reconnectDelay);
                    }
                };
                
                // Guardar timer para limpieza
                ws._reconnectTimer = reconnectTimer;
                
                return ws;
                
            } catch (error) {
                console.error(`Error al crear WebSocket: ${error}`);
                reject(error);
            }
        }
        
        connect();
    });
}

/**
 * Cierra WebSocket de forma limpia
 * @param {WebSocket} ws - WebSocket a cerrar
 */
function closeWebSocket(ws) {
    if (!ws) return;
    
    // Cancelar timer de reconexión si existe
    if (ws._reconnectTimer) {
        clearTimeout(ws._reconnectTimer);
        ws._reconnectTimer = null;
    }
    
    // Cerrar conexión
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
    }
}

// =====================================
// VOICE ACTIVITY DETECTION (VAD)
// =====================================

/**
 * Configuración por defecto del VAD
 */
const VAD_DEFAULT_CONFIG = {
    energyThreshold: 0.003,        // Umbral de energía para detectar voz
    highEnergyThreshold: 0.02,     // Umbral alto: permite speech cortos
    minSpeechDuration: 150,        // ms mínimo de voz antes de enviar
    minHighEnergySpeech: 50,       // ms mínimo si energía alta
    maxSpeechDuration: 3000,       // ms máximo de speech
    silenceChunks: 3,              // Chunks consecutivos de silencio para finalizar
    sampleRate: 16000              // Sample rate del audio
};

/**
 * Clase para Voice Activity Detection (VAD)
 * Detecta cuando hay voz vs silencio y acumula segmentos completos
 */
class VoiceActivityDetector {
    constructor(config = {}) {
        this.config = { ...VAD_DEFAULT_CONFIG, ...config };
        
        // Estado VAD
        this.speechBuffer = [];
        this.isSpeechActive = false;
        this.speechStartTime = 0;
        this.lastSpeechTime = 0;
        this.silenceChunks = 0;
        this.maxEnergyInSegment = 0;
        
        // Callbacks
        this.onSpeechStart = config.onSpeechStart || (() => {});
        this.onSpeechEnd = config.onSpeechEnd || (() => {});
        this.onEnergyUpdate = config.onEnergyUpdate || (() => {});
    }
    
    /**
     * Procesa un chunk de audio
     * @param {Float32Array} audioData - Datos de audio
     * @returns {Object} Resultado con acción y datos
     */
    process(audioData) {
        const now = Date.now();
        
        // Calcular energía del chunk
        let energy = 0;
        for (let i = 0; i < audioData.length; i++) {
            energy += audioData[i] * audioData[i];
        }
        energy = energy / audioData.length;
        
        const hasVoice = energy > this.config.energyThreshold;
        const isHighEnergy = energy > this.config.highEnergyThreshold;
        
        // Notificar cambio de energía
        this.onEnergyUpdate(energy, hasVoice, isHighEnergy);
        
        if (hasVoice) {
            // Voz detectada
            if (!this.isSpeechActive) {
                // Inicio de speech
                this.isSpeechActive = true;
                this.speechStartTime = now;
                this.lastSpeechTime = now;
                this.speechBuffer = [];
                this.maxEnergyInSegment = energy;
                this.silenceChunks = 0;
                
                this.onSpeechStart(energy);
            }
            
            // Trackear energía máxima
            if (energy > this.maxEnergyInSegment) {
                this.maxEnergyInSegment = energy;
            }
            
            // Agregar al buffer
            this.speechBuffer.push(new Float32Array(audioData));
            this.lastSpeechTime = now;
            this.silenceChunks = 0;
            
            const speechDuration = now - this.speechStartTime;
            
            // Si excedemos duración máxima, finalizar y enviar
            if (speechDuration >= this.config.maxSpeechDuration) {
                return this._finalizeSpeechSegment('max_duration');
            }
            
            return { action: 'accumulating' };
            
        } else {
            // Silencio detectado
            if (this.isSpeechActive) {
                this.silenceChunks++;
                
                // Si hay suficientes chunks de silencio consecutivos, finalizar segmento
                if (this.silenceChunks >= this.config.silenceChunks) {
                    // Calcular duración del speech acumulado
                    const totalSamples = this.speechBuffer.reduce((sum, arr) => sum + arr.length, 0);
                    const speechDuration = (totalSamples / this.config.sampleRate) * 1000;
                    
                    // Determinar duración mínima según energía máxima
                    const minDuration = this.maxEnergyInSegment > this.config.highEnergyThreshold 
                        ? this.config.minHighEnergySpeech 
                        : this.config.minSpeechDuration;
                    
                    if (speechDuration >= minDuration) {
                        return this._finalizeSpeechSegment('silence_detected');
                    } else {
                        // Speech muy corto, descartar
                        this._reset();
                        return { action: 'discarded', reason: 'too_short', duration: speechDuration };
                    }
                }
                
                return { action: 'silence_chunk', count: this.silenceChunks };
            }
            
            return { action: 'silence' };
        }
    }
    
    /**
     * Finaliza segmento de speech y retorna los datos
     * @private
     */
    _finalizeSpeechSegment(reason) {
        if (this.speechBuffer.length === 0) {
            this._reset();
            return { action: 'no_data' };
        }
        
        // Concatenar todo el buffer
        const totalLength = this.speechBuffer.reduce((sum, arr) => sum + arr.length, 0);
        const fullSegment = new Float32Array(totalLength);
        
        let offset = 0;
        for (const chunk of this.speechBuffer) {
            fullSegment.set(chunk, offset);
            offset += chunk.length;
        }
        
        const duration = (fullSegment.length / this.config.sampleRate).toFixed(2);
        const maxEnergy = this.maxEnergyInSegment;
        
        // Notificar fin de speech
        this.onSpeechEnd(fullSegment, duration, maxEnergy);
        
        // Limpiar estado
        this._reset();
        
        return {
            action: 'speech_complete',
            reason: reason,
            audioData: fullSegment,
            duration: parseFloat(duration),
            maxEnergy: maxEnergy,
            samples: fullSegment.length
        };
    }
    
    /**
     * Resetea el estado del VAD
     * @private
     */
    _reset() {
        this.speechBuffer = [];
        this.isSpeechActive = false;
        this.maxEnergyInSegment = 0;
        this.silenceChunks = 0;
    }
    
    /**
     * Resetea el VAD (público)
     */
    reset() {
        this._reset();
    }
}

// =====================================
// CIRCULAR AUDIO VISUALIZER
// =====================================

class CircularAudioVisualizer {
    /**
     * @param {HTMLCanvasElement} canvas - Canvas element para dibujar
     * @param {AnalyserNode} analyser - Web Audio API AnalyserNode
     * @param {Object} options - Opciones de configuración
     */
    constructor(canvas, analyser, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.analyser = analyser;
        this.animationId = null;
        this.isRunning = false;
        
        // Opciones configurables
        this.options = {
            radius: options.radius || 80,           // Radio del círculo interno
            barCount: options.barCount || 64,       // Número de barras
            maxBarHeight: options.maxBarHeight || 60, // Altura máxima de barras
            lineWidth: options.lineWidth || 3,      // Grosor de líneas
            colorful: options.colorful !== false,   // Usar colores o monocromo
            baseColor: options.baseColor || [0, 217, 255], // Color base [r, g, b]
            rotation: options.rotation || 0,        // Rotación en radianes
            smoothing: options.smoothing || 0.7     // Suavizado (0-1)
        };
        
        // Buffer para suavizado
        this.previousData = null;
    }
    
    /**
     * Inicia la visualización
     */
    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this.draw();
    }
    
    /**
     * Detiene la visualización
     */
    stop() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.clear();
    }
    
    /**
     * Limpia el canvas
     */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    /**
     * Dibuja la visualización
     */
    draw() {
        if (!this.isRunning || !this.analyser) return;
        
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        this.analyser.getByteFrequencyData(dataArray);
        
        const width = this.canvas.width;
        const height = this.canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        
        this.ctx.clearRect(0, 0, width, height);
        
        // Aplicar suavizado
        const smoothedData = this.smoothData(dataArray);
        
        // Dibujar barras radiales
        for (let i = 0; i < this.options.barCount; i++) {
            const angle = (i / this.options.barCount) * Math.PI * 2 + this.options.rotation;
            const dataIndex = Math.floor((i / this.options.barCount) * bufferLength);
            const value = smoothedData[dataIndex];
            const barHeight = (value / 255) * this.options.maxBarHeight;
            
            // Calcular posición de inicio y fin de la barra
            const x1 = centerX + Math.cos(angle) * this.options.radius;
            const y1 = centerY + Math.sin(angle) * this.options.radius;
            const x2 = centerX + Math.cos(angle) * (this.options.radius + barHeight);
            const y2 = centerY + Math.sin(angle) * (this.options.radius + barHeight);
            
            // Color
            if (this.options.colorful) {
                // Color degradado arco iris
                const hue = (i / this.options.barCount) * 360;
                this.ctx.strokeStyle = `hsl(${hue}, 100%, ${50 + (value / 255) * 30}%)`;
            } else {
                // Color monocromo con intensidad variable
                const [r, g, b] = this.options.baseColor;
                const intensity = 0.3 + (value / 255) * 0.7;
                this.ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${intensity})`;
            }
            
            this.ctx.lineWidth = this.options.lineWidth;
            this.ctx.lineCap = 'round';
            
            this.ctx.beginPath();
            this.ctx.moveTo(x1, y1);
            this.ctx.lineTo(x2, y2);
            this.ctx.stroke();
        }
        
        this.animationId = requestAnimationFrame(() => this.draw());
    }
    
    /**
     * Suaviza los datos de frecuencia
     */
    smoothData(dataArray) {
        if (!this.previousData) {
            this.previousData = new Uint8Array(dataArray);
            return dataArray;
        }
        
        const smoothed = new Uint8Array(dataArray.length);
        const alpha = 1 - this.options.smoothing;
        
        for (let i = 0; i < dataArray.length; i++) {
            smoothed[i] = alpha * dataArray[i] + this.options.smoothing * this.previousData[i];
        }
        
        this.previousData = smoothed;
        return smoothed;
    }
    
    /**
     * Actualiza opciones
     */
    updateOptions(newOptions) {
        this.options = { ...this.options, ...newOptions };
    }
    
    /**
     * Redimensiona el canvas
     */
    resize(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
    }
}

/**
 * Crea un visualizador circular de audio
 * @param {string|HTMLCanvasElement} canvasOrId - Canvas element o ID
 * @param {AudioContext} audioContext - Web Audio API AudioContext
 * @param {MediaStream} mediaStream - Stream de audio
 * @param {Object} options - Opciones de configuración
 * @returns {Object} { visualizer, analyser, source }
 */
function createAudioVisualizer(canvasOrId, audioContext, mediaStream, options = {}) {
    const canvas = typeof canvasOrId === 'string' 
        ? document.getElementById(canvasOrId) 
        : canvasOrId;
    
    if (!canvas) {
        console.error('Canvas no encontrado');
        return null;
    }
    
    // Crear analizador
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = options.fftSize || 256;
    analyser.smoothingTimeConstant = options.smoothingTimeConstant || 0.8;
    
    // Conectar source al analyser
    const source = audioContext.createMediaStreamSource(mediaStream);
    source.connect(analyser);
    
    // Crear visualizador
    const visualizer = new CircularAudioVisualizer(canvas, analyser, options);
    
    return { visualizer, analyser, source };
}

/**
 * Versión simplificada para uso rápido
 */
function quickCircularVisualizer(canvasId, audioContext, mediaStream, options = {}) {
    const result = createAudioVisualizer(canvasId, audioContext, mediaStream, options);
    if (result) {
        result.visualizer.start();
        return result;
    }
    return null;
}

// Exportar para uso en módulos
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CircularAudioVisualizer,
        createAudioVisualizer,
        quickCircularVisualizer
    };
}
