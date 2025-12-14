/**
 * Cliente WebSocket para transcripción en tiempo real con Vosk
 */

class VoskClient {
    constructor() {
        this.ws = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        this.isRecording = false;
        this.audioDuration = 0;
        this.audioChunksCount = 0;
        
        // Buffer para acumular audio
        this.audioBuffer = [];
        this.bufferStartTime = 0;
        this.isSpeechActive = false;
        this.silenceChunks = 0;
        this.maxEnergyInSegment = 0;
        this.segmentDuration = 0;  // Duración del segmento actual
        
        // Configuración VAD
        this.vadConfig = {
            energyThreshold: 0.003,        // Umbral de energía para detectar voz
            highEnergyThreshold: 0.02,     // Umbral alto para speech cortos
            minSpeechDuration: 300,        // ms mínimo de voz antes de enviar
            minHighEnergySpeech: 150,      // ms mínimo si energía alta
            maxSpeechDuration: 5000,       // ms máximo de speech (5s)
            silenceChunks: 4,              // Chunks consecutivos de silencio para finalizar
            sampleRate: 16000
        };
        
        // Canvas para visualización circular
        this.canvas = document.getElementById('audio-canvas');
        this.circularVisualizer = null;
        this.analyser = null;
        
        // Elementos del DOM
        this.elements = {
            startBtn: document.getElementById('start-recording'),
            stopBtn: document.getElementById('stop-recording'),
            clearBtn: document.getElementById('clear-transcription'),
            micStatus: document.getElementById('mic-status'),
            wsStatus: document.getElementById('ws-status'),
            recordingStatus: document.getElementById('recording-status'),
            transcriptionContainer: document.getElementById('transcription-container'),
            segmentsContainer: document.getElementById('segments-container'),
            logContainer: document.getElementById('log-container'),
            languageSelect: document.getElementById('language-select'),
            minDuration: document.getElementById('min-duration'),
            wsUrl: document.getElementById('ws-url'),
            audioDuration: document.getElementById('audio-duration'),
            audioLevel: document.getElementById('audio-level'),
            currentTranscription: document.getElementById('current-transcription')
        };
        
        this.initEventListeners();
        this.log('Sistema inicializado', 'info');
    }
    
    initEventListeners() {
        this.elements.startBtn.addEventListener('click', () => this.startRecording());
        this.elements.stopBtn.addEventListener('click', () => this.stopRecording());
        this.elements.clearBtn.addEventListener('click', () => this.clearTranscription());
    }
    
    async startRecording() {
        try {
            this.log('Iniciando grabación...', 'info');
            
            // 1. Conectar WebSocket
            await this.connectWebSocket();
            
            // 2. Iniciar captura de audio
            await this.initAudio();
            
            // 3. Enviar configuración
            await this.sendConfiguration();
            
            // 4. Iniciar envío de audio
            this.startAudioStreaming();
            
            this.isRecording = true;
            this.updateUIRecording(true);
            this.log('Grabación iniciada correctamente', 'success');
            
        } catch (error) {
            this.log(`Error al iniciar grabación: ${error.message}`, 'error');
            this.stopRecording();
        }
    }
    
    async stopRecording() {
        this.log('Deteniendo grabación...', 'info');
        this.isRecording = false;
        // Esperar un breve margen para que el último onaudioprocess empuje
        // cualquier chunk final al buffer antes de enviarlo.
        await new Promise(resolve => setTimeout(resolve, 200));

        // Enviar audio acumulado si existe
        if (this.audioBuffer.length > 0) {
            this.log('Enviando audio restante en buffer...', 'info');
            this.sendBufferedAudio('stop_recording');
        }

        // Enviar señal de finalización
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send('END');
            this.log('Señal END enviada', 'info');
        }

        // Detener audio y visualizador (hacerlo después del retardo)
        if (this.circularVisualizer) {
            this.circularVisualizer.stop();
            this.circularVisualizer = null;
        }

        if (this.processor) {
            try { this.processor.disconnect(); } catch(e) {}
            this.processor = null;
        }
        if (this.mediaStream) {
            try { this.mediaStream.getTracks().forEach(track => track.stop()); } catch(e) {}
            this.mediaStream = null;
        }
        if (this.audioContext) {
            try { await this.audioContext.close(); } catch(e) {}
            this.audioContext = null;
        }
        
        // Resetear estado VAD
        this.resetVADState();
        
        // Cerrar WebSocket después de un breve delay
        setTimeout(() => {
            if (this.ws) {
                this.ws.close();
            }
        }, 500);
        
        this.updateUIRecording(false);
        this.log('Grabación detenida', 'info');
    }
    
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const wsUrl = this.elements.wsUrl.value;
            this.log(`Conectando a ${wsUrl}...`, 'info');
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.updateStatus('ws-status', 'Conectado', 'status-connected');
                this.log('WebSocket conectado', 'success');
                resolve();
            };
            
            this.ws.onerror = (error) => {
                this.updateStatus('ws-status', 'Error', 'status-error');
                this.log('Error en WebSocket', 'error');
                reject(new Error('Error al conectar WebSocket'));
            };
            
            this.ws.onclose = () => {
                this.updateStatus('ws-status', 'Desconectado', 'status-disconnected');
                this.log('WebSocket desconectado', 'info');
            };
            
            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };
            
            // Timeout de 5 segundos
            setTimeout(() => {
                if (this.ws.readyState !== WebSocket.OPEN) {
                    reject(new Error('Timeout al conectar WebSocket'));
                }
            }, 5000);
        });
    }
    
    async initAudio() {
        this.log('Solicitando acceso al micrófono...', 'info');
        
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        
        this.audioContext = new AudioContext({ sampleRate: 16000 });
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        // Analizador para visualización
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.analyser.smoothingTimeConstant = 0.8;
        source.connect(this.analyser);
        
        // Crear visualizador circular
        const vizResult = createAudioVisualizer(
            this.canvas,
            this.audioContext,
            this.mediaStream,
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
            this.circularVisualizer = vizResult.visualizer;
            this.circularVisualizer.start();
        }
        
        // Procesador para capturar audio
        this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        source.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
        
        this.updateStatus('mic-status', 'Activo', 'status-active');
        this.log('Micrófono iniciado correctamente', 'success');
    }
    
    async sendConfiguration() {
        const config = {
            idioma: this.elements.languageSelect.value,
            sample_rate: 16000,
            min_duration: parseFloat(this.elements.minDuration.value),
            beam_size: 5,
            temperature: 0.0,
            repetition_penalty: 1.2,
            no_speech_threshold: 0.6,
            compression_ratio_threshold: 2.4,
            min_silence_duration_ms: 1000
        };
        
        this.ws.send(JSON.stringify(config));
        this.log(`Configuración enviada: ${config.idioma}, min_duration: ${config.min_duration}s`, 'info');
        
        // Esperar confirmación
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    startAudioStreaming() {
        this.audioDuration = 0;
        this.audioChunksCount = 0;
        this.segmentDuration = 0;
        this.resetVADState();
        
        this.processor.onaudioprocess = (e) => {
            if (!this.isRecording) return;
            
            const float32Data = e.inputBuffer.getChannelData(0);
            
            // Calcular energía del chunk
            let energy = 0;
            for (let i = 0; i < float32Data.length; i++) {
                energy += float32Data[i] * float32Data[i];
            }
            energy = energy / float32Data.length;
            
            // Actualizar nivel de audio para UI
            const rms = Math.sqrt(energy);
            const level = Math.min(100, Math.floor(rms * 1000));
            this.elements.audioLevel.textContent = `Nivel: ${level}%`;
            
            // Detectar si hay voz
            const hasVoice = energy > this.vadConfig.energyThreshold;
            const isHighEnergy = energy > this.vadConfig.highEnergyThreshold;
            
            const now = Date.now();
            
            if (hasVoice) {
                // Voz detectada
                if (!this.isSpeechActive) {
                    // Inicio de speech
                    this.isSpeechActive = true;
                    this.bufferStartTime = now;
                    this.audioBuffer = [];
                    this.maxEnergyInSegment = energy;
                    this.silenceChunks = 0;
                    this.segmentDuration = 0;
                    
                    this.log(`🎤 Inicio de speech - energía: ${(energy * 1000).toFixed(1)}`, 'info');
                    this.updateStatus('recording-status', 'Detectando voz...', 'status-recording');
                }
                
                // Trackear energía máxima
                if (energy > this.maxEnergyInSegment) {
                    this.maxEnergyInSegment = energy;
                }
                
                // Agregar al buffer (guardar copia)
                this.audioBuffer.push(new Float32Array(float32Data));
                this.silenceChunks = 0;
                
                // Actualizar duración del segmento actual
                this.segmentDuration += float32Data.length / this.vadConfig.sampleRate;
                this.elements.audioDuration.textContent = `Duración: ${this.segmentDuration.toFixed(1)}s`;
                
                const speechDuration = now - this.bufferStartTime;
                
                // Si excedemos duración máxima, enviar segmento
                if (speechDuration >= this.vadConfig.maxSpeechDuration) {
                    this.log(`⏱️ Duración máxima alcanzada (${speechDuration}ms)`, 'info');
                    this.sendBufferedAudio('max_duration');
                }
                
            } else {
                // Silencio detectado
                if (this.isSpeechActive) {
                    this.silenceChunks++;
                    
                    // Si hay suficientes chunks de silencio, finalizar segmento
                    if (this.silenceChunks >= this.vadConfig.silenceChunks) {
                        const totalSamples = this.audioBuffer.reduce((sum, arr) => sum + arr.length, 0);
                        const speechDuration = (totalSamples / this.vadConfig.sampleRate) * 1000;
                        
                        // Determinar duración mínima según energía máxima
                        const minDuration = this.maxEnergyInSegment > this.vadConfig.highEnergyThreshold 
                            ? this.vadConfig.minHighEnergySpeech 
                            : this.vadConfig.minSpeechDuration;
                        
                        if (speechDuration >= minDuration) {
                            this.log(`🔇 Silencio detectado - enviando segmento (${speechDuration.toFixed(0)}ms)`, 'info');
                            this.sendBufferedAudio('silence_detected');
                        } else {
                            this.log(`⏭️ Speech muy corto descartado (${speechDuration.toFixed(0)}ms < ${minDuration}ms)`, 'info');
                            this.resetVADState();
                            this.updateStatus('recording-status', 'Esperando...', 'status-info');
                            // Resetear duración al descartar
                            this.segmentDuration = 0;
                            this.elements.audioDuration.textContent = 'Duración: 0.0s';
                        }
                    }
                } else {
                    // Silencio sin speech activo - mantener duración en 0
                    this.segmentDuration = 0;
                    this.elements.audioDuration.textContent = 'Duración: 0.0s';
                }
            }
        };
    }
    
    /**
     * Envía el audio acumulado en el buffer
     */
    sendBufferedAudio(reason) {
        if (this.audioBuffer.length === 0) {
            this.resetVADState();
            return;
        }
        
        // Concatenar todo el buffer
        const totalLength = this.audioBuffer.reduce((sum, arr) => sum + arr.length, 0);
        const fullSegment = new Float32Array(totalLength);
        
        let offset = 0;
        for (const chunk of this.audioBuffer) {
            fullSegment.set(chunk, offset);
            offset += chunk.length;
        }
        
        const duration = (fullSegment.length / this.vadConfig.sampleRate).toFixed(2);
        
        // Convertir a PCM 16-bit
        const int16Data = new Int16Array(fullSegment.length);
        for (let i = 0; i < fullSegment.length; i++) {
            const s = Math.max(-1, Math.min(1, fullSegment[i]));
            int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        
        // Enviar por WebSocket
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(int16Data.buffer);
            this.audioChunksCount++;
            
            this.log(`📤 Segmento enviado: ${fullSegment.length} samples (${duration}s) - razón: ${reason}`, 'success');
            this.updateStatus('recording-status', 'Procesando...', 'status-info');
        }
        
        // Resetear estado
        this.resetVADState();
    }
    
    /**
     * Resetea el estado del VAD
     */
    resetVADState() {
        this.audioBuffer = [];
        this.isSpeechActive = false;
        this.maxEnergyInSegment = 0;
        this.silenceChunks = 0;
        this.segmentDuration = 0;
        this.elements.audioDuration.textContent = 'Duración: 0.0s';
    }
    
    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            if (data.tipo === 'transcripcion') {
                this.displayTranscription(data);
                this.log(`Transcripción recibida: ${data.texto.substring(0, 50)}...`, 'success');
            } else if (data.tipo === 'info') {
                this.log(data.mensaje, 'info');
                this.updateStatus('recording-status', data.mensaje, 'status-info');
            } else if (data.tipo === 'error') {
                this.log(data.mensaje, 'error');
                this.updateStatus('recording-status', 'Error', 'status-error');
            }
        } catch (error) {
            this.log(`Error al parsear mensaje: ${error.message}`, 'error');
        }
    }
    
    displayTranscription(data) {
        // Actualizar overlay central
        if (this.elements.currentTranscription) {
            const overlay = document.getElementById('audio-overlay');
            overlay.classList.remove('fade-out');
            overlay.classList.add('fade-in');
            
            this.elements.currentTranscription.innerHTML = `
                <div class="detection-keyword" style="font-size: 0.9rem; max-height: 140px; overflow-y: auto;">
                    ${data.texto}
                </div>
            `;
            
            // Desvanecer después de 8 segundos
            setTimeout(() => {
                overlay.classList.remove('fade-in');
                overlay.classList.add('fade-out');
            }, 8000);
        }
        
        // Limpiar placeholder si existe
        const placeholder = this.elements.transcriptionContainer.querySelector('.placeholder-text');
        if (placeholder) {
            placeholder.remove();
        }
        
        // Agregar transcripción al historial
        const transcriptionDiv = document.createElement('div');
        transcriptionDiv.className = 'transcription-item';
        transcriptionDiv.innerHTML = `
            <div class="transcription-header">
                <span class="transcription-time">${new Date().toLocaleTimeString()}</span>
                <span class="transcription-duration">${data.duracion.toFixed(1)}s</span>
            </div>
            <div class="transcription-text">${data.texto}</div>
        `;
        this.elements.transcriptionContainer.appendChild(transcriptionDiv);
        this.elements.transcriptionContainer.scrollTop = this.elements.transcriptionContainer.scrollHeight;
        
        // Mostrar segmentos si existen
        if (data.segmentos && data.segmentos.length > 0) {
            this.displaySegments(data.segmentos);
        }
    }
    
    displaySegments(segmentos) {
        // Si no existe el contenedor (interfaz simplificada), salir silenciosamente
        if (!this.elements.segmentsContainer) return;

        // Limpiar placeholder si existe
        const placeholder = this.elements.segmentsContainer.querySelector('.placeholder-text');
        if (placeholder) {
            placeholder.remove();
        }
        
        segmentos.forEach(seg => {
            const segmentDiv = document.createElement('div');
            segmentDiv.className = 'segment-item';
            segmentDiv.innerHTML = `
                <span class="segment-time">[${seg.inicio.toFixed(2)}s - ${seg.fin.toFixed(2)}s]</span>
                <span class="segment-text">${seg.texto}</span>
            `;
            this.elements.segmentsContainer.appendChild(segmentDiv);
        });
        
        this.elements.segmentsContainer.scrollTop = this.elements.segmentsContainer.scrollHeight;
    }
    
    clearTranscription() {
        this.elements.transcriptionContainer.innerHTML = '<p class="placeholder-text">La transcripción aparecerá aquí en tiempo real...</p>';
        if (this.elements.segmentsContainer) {
            this.elements.segmentsContainer.innerHTML = '<p class="placeholder-text">Los segmentos de audio aparecerán aquí...</p>';
        }
        if (this.elements.logContainer) {
            this.elements.logContainer.innerHTML = '';
        }
        this.audioDuration = 0;
        this.audioChunksCount = 0;
        this.elements.audioDuration.textContent = 'Duración: 0.0s';
        this.elements.audioLevel.textContent = 'Nivel: 0%';
        this.log('Interfaz limpiada', 'info');
    }
    
    updateUIRecording(recording) {
        this.elements.startBtn.disabled = recording;
        this.elements.stopBtn.disabled = !recording;
        
        if (recording) {
            this.updateStatus('recording-status', 'Grabando...', 'status-recording');
        } else {
            this.updateStatus('recording-status', 'Detenido', 'status-inactive');
            this.updateStatus('mic-status', 'Inactivo', 'status-inactive');
        }
    }
    
    updateStatus(elementId, text, className) {
        const element = this.elements[elementId.replace('-', '')];
        if (!element) return;
        
        element.textContent = text;
        element.className = 'status-badge ' + className;
    }
    
    log(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`;
        logEntry.innerHTML = `<span class="log-time">[${timestamp}]</span> <span class="log-message">${message}</span>`;
        // Si no existe el contenedor de log (interfaz simplificada), sólo imprimir en consola
        if (!this.elements.logContainer) {
            console.log(`[Vosk ${type.toUpperCase()}] ${message}`);
            return;
        }

        this.elements.logContainer.appendChild(logEntry);
        this.elements.logContainer.scrollTop = this.elements.logContainer.scrollHeight;

        // Limitar a 100 entradas
        while (this.elements.logContainer.children.length > 100) {
            this.elements.logContainer.removeChild(this.elements.logContainer.firstChild);
        }
        console.log(`[Vosk ${type.toUpperCase()}] ${message}`);
    }
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    window.voskClient = new VoskClient();

    const openBtn = document.getElementById('open-config');
    const modal = document.getElementById('config-modal');
    const backdrop = document.getElementById('config-backdrop');
    const closeBtn = document.getElementById('config-close');
    function open(){ modal.setAttribute('aria-hidden','false'); closeBtn && closeBtn.focus(); document.body.style.overflow='hidden'; }
    function close(){ modal.setAttribute('aria-hidden','true'); document.body.style.overflow=''; openBtn && openBtn.focus(); }
    openBtn && openBtn.addEventListener('click', open);
    closeBtn && closeBtn.addEventListener('click', close);
    backdrop && backdrop.addEventListener('click', close);
    document.addEventListener('keydown', (e)=>{ if(e.key==='Escape' && modal.getAttribute('aria-hidden')==='false'){ close(); } });
});
