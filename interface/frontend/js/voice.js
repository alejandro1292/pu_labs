/**
 * Keyword Spotting - Frontend Application
 * Maneja captura de audio, WebSocket y visualización
 */

// Configuración
const CONFIG = {
    targetSampleRate: 16000,
    bufferSize: 4096,
    visualizationEnabled: true,
    // VAD en cliente
    vadEnabled: true,
    vadEnergyThreshold: 0.003,     // Umbral de energía para detectar voz (reducido de 0.01)
    vadHighEnergyThreshold: 0.02,  // Umbral alto: permite speech cortos (reducido de 0.05)
    vadMinSpeechDuration: 150,     // ms mínimo de voz antes de enviar (normal)
    vadMinHighEnergySpeech: 50,   // ms mínimo si energía alta
    vadMaxSpeechDuration: 3000,    // ms máximo de speech
    vadSilenceChunks: 3            // Chunks consecutivos de silencio para finalizar (3 chunks = ~384ms @ 4096 samples, ajustado para capturar palabras completas)
};

// Estado global
const state = {
    audioContext: null,
    mediaStream: null,
    audioWorkletNode: null,
    scriptProcessorNode: null,
    websocket: null,
    isRecording: false,
    detectionCount: 0,
    keywords: [],
    lastDetectionTime: 0,
    // VAD
    vad: null,
    // Visualización circular
    circularVisualizer: null,
    // Overlay fade timeout
    overlayTimeout: null
};

// Referencias DOM
const elements = {
    startBtn: document.getElementById('start-btn'),
    stopBtn: document.getElementById('stop-btn'),
    resetBtn: document.getElementById('reset-btn'),
    clearHistoryBtn: document.getElementById('clear-history-btn'),
    micStatus: document.getElementById('mic-status'),
    wsStatus: document.getElementById('ws-status'),
    vadStatus: document.getElementById('vad-status'),
    classifierStatus: document.getElementById('classifier-status'),
    currentDetection: document.getElementById('current-detection'),
    detectionHistory: document.getElementById('detection-history'),
    keywordsList: document.getElementById('keywords-list'),
    liveAudioCanvas: document.getElementById('live-audio-canvas'),
    audioLevel: document.getElementById('audio-level'),
    classifierSelect: document.getElementById('classifier-select')
};

/**
 * Conecta al WebSocket del servidor
 */
async function connectWebSocket() {
    state.websocket = await initWebSocket('/ws/audio', handleWebSocketMessage, {
        onOpen: (ws) => {
            updateStatus('ws', 'connected', 'Conectado');
        },
        onError: (error) => {
            updateStatus('ws', 'error', 'Error');
        },
        onClose: () => {
            updateStatus('ws', 'disconnected', 'Desconectado');
        }
    });
}

/**
 * Maneja mensajes del WebSocket
 */
function handleWebSocketMessage(data, ws) {
    console.log('Mensaje recibido:', data);

    switch (data.type) {
        case 'connected':
            state.keywords = data.keywords || [];
            updateKeywordsList();
            if (!data.templates_available) {
                showNotification('No hay templates entrenados. Usa train.py para crear templates.', 'warning');
            }
            // Also refresh classifier status when connected
            refreshClassifierStatus();
            break;

        case 'detection':
            handleDetection(data.keyword, data.confidence);
            break;

        case 'reset_ack':
            showNotification('Sistema reiniciado', 'info');
            break;
    }
}

/**
 * Maneja una detección de keyword
 */
function handleDetection(keyword, confidence) {
    const now = Date.now();

    // Evitar detecciones duplicadas muy cercanas
    if (now - state.lastDetectionTime < 500) {
        return;
    }

    state.lastDetectionTime = now;

    console.log(`🎯 Resultado: ${keyword} (confianza: ${confidence.toFixed(2)})`);

    // Actualizar display actual (overlay circular)
    updateCurrentDetection(keyword, confidence);

    // Si es una detección válida (no un rechazo "?"), agregar al historial
    if (keyword !== "?") {
        state.detectionCount++;
        addToHistory(keyword, confidence);
        playDetectionSound();
    } else {
        playErrorSound();
    }
}

/**
 * Actualiza la visualización de detección actual
 */
function updateCurrentDetection(keyword, confidence) {
    const confidencePercent = (confidence * 100).toFixed(0);

    const overlay = document.getElementById('audio-overlay');

    // Limpiar timeout anterior si existe
    if (state.overlayTimeout) {
        clearTimeout(state.overlayTimeout);
    }

    // Mostrar overlay (quitar fade-out si estaba)
    overlay.classList.remove('fade-out');
    overlay.classList.add('fade-in');

    elements.currentDetection.innerHTML = `
        <div class="detection-keyword">${keyword.toUpperCase()}</div>
        <div class="detection-confidence">
            ${confidencePercent}%
        </div>
    `;

    // Desvanecer después de 5 segundos
    state.overlayTimeout = setTimeout(() => {
        overlay.classList.remove('fade-in');
        overlay.classList.add('fade-out');
    }, 5000);
}

/**
 * Agrega detección al historial
 */
function addToHistory(keyword, confidence) {
    // Remover placeholder si existe
    const placeholder = elements.detectionHistory.querySelector('.history-placeholder');
    if (placeholder) {
        placeholder.remove();
    }

    const timestamp = new Date().toLocaleTimeString();
    const confidencePercent = (confidence * 100).toFixed(0);

    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `
        <span class="history-number">#${state.detectionCount}</span>
        <span class="history-keyword">${keyword}</span>
        <span class="history-confidence">${confidencePercent}%</span>
        <span class="history-time">${timestamp}</span>
    `;

    // Insertar al inicio
    elements.detectionHistory.insertBefore(item, elements.detectionHistory.firstChild);

    // Limitar a 50 items
    const items = elements.detectionHistory.querySelectorAll('.history-item');
    if (items.length > 50) {
        items[items.length - 1].remove();
    }
}

/**
 * Actualiza lista de keywords disponibles
 */
function updateKeywordsList() {
    if (state.keywords.length === 0) {
        elements.keywordsList.textContent = 'No hay keywords configuradas';
        return;
    }

    elements.keywordsList.innerHTML = state.keywords
        .map(kw => `<span class="keyword-tag">${kw}</span>`)
        .join('');
}

async function refreshClassifierStatus() {
    try {
        const resp = await fetch(`${window.location.protocol}//${window.location.host}/status`);
        if (!resp.ok) return;
        const data = await resp.json();
        const cls = data.classifier_type || data.classifier || 'rf';
        const ready = data.model_ready ? true : false;
        const acc = data.test_accuracy || 0;

        if (elements.classifierStatus) {
            elements.classifierStatus.textContent = cls.toUpperCase();
            elements.classifierStatus.classList.toggle('status-inactive', !ready);
            elements.classifierStatus.classList.toggle('status-active', ready);
            if (ready) {
                elements.classifierStatus.title = `Modelo listo (accuracy: ${(acc * 100).toFixed(1)}%)`;
            } else {
                elements.classifierStatus.title = 'Modelo no cargado o no entrenado';
            }
        }
    } catch (err) {
        console.warn('No se pudo obtener status del servidor:', err);
    }
}

// Refresh classifier status periodically
setInterval(refreshClassifierStatus, 15000);

/**
 * Fetch the currently active classifier from the backend and set the select value
 */
async function getActiveClassifier() {
    try {
        const resp = await fetch(`${window.location.protocol}//${window.location.host}/api/training/classifier`);
        if (!resp.ok) return;
        const data = await resp.json();
        if (data && data.classifier && elements.classifierSelect) {
            elements.classifierSelect.value = data.classifier;
        }
    } catch (err) {
        console.warn('Could not fetch active classifier:', err);
    }
}

/**
 * Post the selected classifier to the backend to set it active
 * @param {string} classifier
 */
async function setActiveClassifier(classifier) {
    if (!classifier) return;
    if (elements.classifierSelect) elements.classifierSelect.disabled = true;
    try {
        const resp = await fetch(`${window.location.protocol}//${window.location.host}/api/training/classifier`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ classifier })
        });
        if (!resp.ok) {
            const txt = await resp.text().catch(() => null);
            throw new Error(txt || `HTTP ${resp.status}`);
        }
        // Refresh status immediately after change
        await refreshClassifierStatus();
    } catch (err) {
        console.error('Failed to set active classifier:', err);
        // attempt to revert select to backend value
        await getActiveClassifier();
    } finally {
        if (elements.classifierSelect) elements.classifierSelect.disabled = false;
    }
}

if (elements.classifierSelect) {
    elements.classifierSelect.addEventListener('change', async () => {
        await setActiveClassifier(elements.classifierSelect.value);
    });
}

// Also fetch once on load
document.addEventListener('DOMContentLoaded', () => {
    refreshClassifierStatus();
    getActiveClassifier();
});

/**
 * Inicializa captura de audio
 */
async function initAudio() {
    try {
        // Solicitar acceso al micrófono
        state.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: CONFIG.targetSampleRate
            }
        });

        // Crear / obtener AudioContext compartido (preferir 16 kHz)
        state.audioContext = (typeof getSharedAudioContext === 'function')
            ? getSharedAudioContext(CONFIG.targetSampleRate)
            : new (window.AudioContext || window.webkitAudioContext)({ sampleRate: CONFIG.targetSampleRate });

        // Intentar crear MediaStreamSource en el contexto seleccionado. Algunos navegadores
        // pueden lanzar si hay otros AudioContexts con distinto sampleRate; en ese caso
        // creamos un contexto dedicado (sin forzar sampleRate) y reintentamos.
        let source;
        try {
            // Log de ayuda para depuración
            const trackSettings = state.mediaStream.getAudioTracks()[0]?.getSettings?.() || {};
            console.log(`Audio track settings: sampleRate=${trackSettings.sampleRate || 'n/a'}`);
            console.log(`Usando AudioContext sampleRate=${state.audioContext.sampleRate}`);

            source = state.audioContext.createMediaStreamSource(state.mediaStream);
        } catch (err) {
            console.warn('createMediaStreamSource falló en el contexto actual (posible conflicto de sample-rate). Intentando contexto dedicado...', err);

            try {
                const Ctx = window.AudioContext || window.webkitAudioContext;
                const dedicatedCtx = new Ctx(); // dejar que el navegador establezca el sampleRate disponible

                // Marcar que usamos un contexto dedicado para cerrarlo al detener
                state._usesDedicatedAudioContext = true;

                state.audioContext = dedicatedCtx;
                console.log(`Contexto dedicado creado con sampleRate=${state.audioContext.sampleRate}`);

                source = state.audioContext.createMediaStreamSource(state.mediaStream);
            } catch (err2) {
                console.error('Error al crear contexto dedicado o MediaStreamSource:', err2);
                throw err; // rethrow original para que el flujo de error sea manejado arriba
            }
        }

        // Crear visualizador circular
        const vizResult = createAudioVisualizer(
            elements.liveAudioCanvas,
            state.audioContext,
            state.mediaStream,
            {
                radius: 60,
                barCount: 48,
                maxBarHeight: 40,
                lineWidth: 2.5,
                colorful: true,
                smoothing: 0.75
            }
        );

        if (vizResult) {
            state.circularVisualizer = vizResult.visualizer;
            state.circularVisualizer.start();
        }

        // Crear VAD (Voice Activity Detector)
        state.vad = new VoiceActivityDetector({
            energyThreshold: CONFIG.vadEnergyThreshold,
            highEnergyThreshold: CONFIG.vadHighEnergyThreshold,
            minSpeechDuration: CONFIG.vadMinSpeechDuration,
            minHighEnergySpeech: CONFIG.vadMinHighEnergySpeech,
            maxSpeechDuration: CONFIG.vadMaxSpeechDuration,
            silenceChunks: CONFIG.vadSilenceChunks,
            sampleRate: CONFIG.targetSampleRate,
            onSpeechStart: (energy) => {
                console.log(`🎤 Inicio de speech detectado - energy: ${(energy * 1000).toFixed(1)}`);
            },
            onSpeechEnd: (audioData, duration, maxEnergy) => {
                console.log(`📤 Enviando segmento completo: ${audioData.length} samples (${duration}s) - energía máx: ${(maxEnergy * 1000).toFixed(1)}`);
                sendAudioData(audioData);
            },
            onEnergyUpdate: (energy, hasVoice, isHighEnergy) => {
                // Actualizar nivel de audio
                const rms = Math.sqrt(energy);
                const level = Math.min(100, Math.round(rms * 1000));
                elements.audioLevel.textContent = `Nivel: ${level}%`;

                // Actualizar indicador VAD
                if (isHighEnergy) {
                    updateStatus('vad', 'active', `VOZ ALTA (${(energy * 1000).toFixed(1)})`);
                } else if (hasVoice) {
                    updateStatus('vad', 'active', `Voz (${(energy * 1000).toFixed(1)})`);
                } else {
                    updateStatus('vad', 'inactive', `Silencio (${(energy * 1000).toFixed(1)})`);
                }
            }
        });

        // Usar ScriptProcessorNode (compatible con más navegadores)
        state.scriptProcessorNode = state.audioContext.createScriptProcessor(CONFIG.bufferSize, 1, 1);

        state.scriptProcessorNode.onaudioprocess = (event) => {
            if (!state.isRecording) return;

            const inputData = event.inputBuffer.getChannelData(0);

            // Resample si el AudioContext no está a 16 kHz
            const ctxSampleRate = state.audioContext ? state.audioContext.sampleRate : CONFIG.targetSampleRate;
            let dataToUse = inputData;
            if (ctxSampleRate !== CONFIG.targetSampleRate && typeof resampleFloat32 === 'function') {
                dataToUse = resampleFloat32(inputData, ctxSampleRate, CONFIG.targetSampleRate);
            }

            // VAD en cliente: solo enviar si hay voz
            if (CONFIG.vadEnabled && state.vad) {
                state.vad.process(dataToUse);
            } else {
                // Modo sin VAD: enviar todo
                sendAudioData(dataToUse);
            }
        };

        // Conectar nodos
        source.connect(state.scriptProcessorNode);
        state.scriptProcessorNode.connect(state.audioContext.destination);

        console.log('✓ Audio inicializado');
        console.log(`  Sample rate: ${state.audioContext.sampleRate} Hz`);
        console.log(`  Buffer size: ${CONFIG.bufferSize} samples`);

        return true;
    } catch (error) {
        console.error('✗ Error al inicializar audio:', error);
        alert('Error al acceder al micrófono. Verifica los permisos.');
        return false;
    }
}

/**
 * Envía datos de audio al servidor
 */
function sendAudioData(audioData) {
    if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
        // Convertir Float32Array a buffer
        const buffer = audioData.buffer.slice(
            audioData.byteOffset,
            audioData.byteOffset + audioData.byteLength
        );

        state.websocket.send(buffer);
    }
}

/**
 * Inicia grabación
 */
async function startRecording() {
    try {
        // Conectar WebSocket
        await connectWebSocket();

        // Inicializar audio
        const audioInitialized = await initAudio();
        if (!audioInitialized) return;

        // Iniciar grabación
        state.isRecording = true;

        // Actualizar UI
        updateStatus('mic', 'active', 'Activo');
        elements.startBtn.disabled = true;
        elements.stopBtn.disabled = false;

        showNotification('Grabación iniciada. Di un comando...', 'success');

    } catch (error) {
        console.error('Error al iniciar grabación:', error);
        showNotification('Error al iniciar grabación', 'error');
    }
}

/**
 * Detiene grabación
 */
function stopRecording() {
    state.isRecording = false;

    // Limpiar timeout del overlay
    if (state.overlayTimeout) {
        clearTimeout(state.overlayTimeout);
        state.overlayTimeout = null;
    }

    // Detener visualizador circular
    if (state.circularVisualizer) {
        state.circularVisualizer.stop();
        state.circularVisualizer = null;
    }

    // Detener audio
    if (state.scriptProcessorNode) {
        state.scriptProcessorNode.disconnect();
        state.scriptProcessorNode = null;
    }

    if (state.mediaStream) {
        state.mediaStream.getTracks().forEach(track => track.stop());
        state.mediaStream = null;
    }

    if (state.audioContext) {
        try {
            // Si es el contexto compartido, no lo cerramos (se reutiliza). Solo cerramos si era un contexto dedicado.
            if (!window._sharedAudioContext || state.audioContext !== window._sharedAudioContext) {
                state.audioContext.close();
            }
        } catch (e) {
            console.warn('Error cerrando AudioContext:', e);
        }
        state.audioContext = null;
    }

    // Cerrar WebSocket
    if (state.websocket) {
        closeWebSocket(state.websocket);
        state.websocket = null;
    }

    // Actualizar UI
    updateStatus('mic', 'inactive', 'Inactivo');
    updateStatus('ws', 'disconnected', 'Desconectado');
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.audioLevel.textContent = 'Nivel: -';

    showNotification('Grabación detenida', 'info');
}

/**
 * Reinicia el sistema
 */
function resetSystem() {
    if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
        state.websocket.send(JSON.stringify({ type: 'reset' }));
    }

    state.detectionCount = 0;

    // Limpiar timeout del overlay
    if (state.overlayTimeout) {
        clearTimeout(state.overlayTimeout);
        state.overlayTimeout = null;
    }

    // Resetear overlay
    const overlay = document.getElementById('audio-overlay');
    overlay.classList.remove('fade-out', 'fade-in');

    elements.currentDetection.innerHTML = `
        <span class="detection-placeholder">Esperando...</span>
    `;
}

/**
 * Limpia el historial
 */
function clearHistory() {
    elements.detectionHistory.innerHTML = `
        <div class="history-placeholder">
            No hay detecciones aún
        </div>
    `;
    state.detectionCount = 0;
}

/**
 * Actualiza indicadores de estado
 */
function updateStatus(type, status, text) {
    let element;

    switch (type) {
        case 'mic':
            element = elements.micStatus;
            break;
        case 'ws':
            element = elements.wsStatus;
            break;
        case 'vad':
            element = elements.vadStatus;
            break;
    }

    if (element) {
        element.className = `status-badge status-${status}`;
        element.textContent = text;
    }
}

/**
 * Muestra notificación temporal
 */
function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    // Aquí podrías agregar un toast/snackbar más sofisticado
}

/**
 * Reproduce sonido de detección
 */
function playDetectionSound() {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);

        oscillator.frequency.value = 800;
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.2);

        oscillator.start(audioCtx.currentTime);
        oscillator.stop(audioCtx.currentTime + 0.2);
    } catch (error) {
        console.error('Error al reproducir sonido de detección:', error);
    }
}

/**
 * Reproduce sonido de error (para no-detecciones)
 */
function playErrorSound() {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);

        // Frecuencia más baja y tipo sierra para sonar a "error"
        oscillator.frequency.value = 150;
        oscillator.type = 'sawtooth';

        gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);

        oscillator.start(audioCtx.currentTime);
        oscillator.stop(audioCtx.currentTime + 0.3);
    } catch (error) {
        console.error('Error al reproducir sonido de error:', error);
    }
}

// Event listeners
elements.startBtn.addEventListener('click', startRecording);
elements.stopBtn.addEventListener('click', stopRecording);
elements.resetBtn.addEventListener('click', resetSystem);
elements.clearHistoryBtn.addEventListener('click', clearHistory);

// Limpiar al cerrar página
window.addEventListener('beforeunload', () => {
    if (state.isRecording) {
        stopRecording();
    }
});

// Inicialización
console.log('🎤 Keyword Spotting Frontend v1.0');
console.log('Configuración:', CONFIG);
