/**
 * Training Interface - JavaScript
 * Maneja grabación de muestras y entrenamiento de keywords
 */

// Configuración
const API_BASE = `http://${window.location.hostname}:8000/api/training`;
const RECORDING_DURATION = 2000; // 2 segundos
const COUNTDOWN_DURATION = 3000; // 3 segundos

// Estado
const state = {
    keywords: [],
    selectedKeyword: null,
    isRecording: false,
    audioContext: null,
    mediaStream: null,
    recordedChunks: [],
    recordedSamples: 0,
    circularVisualizer: null
};

// Referencias DOM
const elements = {
    // Keywords
    keywordsContainer: document.getElementById('keywords-container'),
    newKeywordInput: document.getElementById('new-keyword-input'),
    createKeywordBtn: document.getElementById('create-keyword-btn'),
    
    // Recording modal
    recordingModal: document.getElementById('recording-modal'),
    modalKeywordName: document.getElementById('modal-keyword-name'),
    modalCountdown: document.getElementById('modal-countdown'),
    modalStatus: document.getElementById('modal-status'),
    modalInstructions: document.getElementById('modal-instructions'),
    modalStartBtn: document.getElementById('modal-start-btn'),
    modalStopBtn: document.getElementById('modal-stop-btn'),
    modalSamplesCount: document.getElementById('modal-samples-count'),
    closeModalBtn: document.getElementById('close-modal-btn'),
    waveformCanvas: document.getElementById('recording-waveform-canvas'),
    progressRingCircle: document.getElementById('progress-ring-circle'),
    recordingTimeLeft: document.getElementById('recording-time-left'),
    
    // Model visualization modal
    modelVizModal: document.getElementById('model-viz-modal'),
    modelKeywordName: document.getElementById('model-keyword-name'),
    modelNTemplates: document.getElementById('model-n-templates'),
    modelMeanLength: document.getElementById('model-mean-length'),
    modelStdLength: document.getElementById('model-std-length'),
    modelIntraScore: document.getElementById('model-intra-score'),
    centroidCanvas: document.getElementById('centroid-canvas'),
    thresholdMinVal: document.getElementById('threshold-min-val'),
    thresholdMaxVal: document.getElementById('threshold-max-val'),
    thresholdFill: document.getElementById('threshold-fill'),
    thresholdMinMarker: document.getElementById('threshold-min-marker'),
    thresholdMaxMarker: document.getElementById('threshold-max-marker'),
    centroidInfo: document.getElementById('centroid-info'),
    closeModelModalBtn: document.getElementById('close-model-modal-btn'),
    
};

// ============================================
// API Functions
// ============================================

async function fetchKeywords() {
    try {
        const response = await fetch(`${API_BASE}/keywords`);
        const data = await response.json();
        state.keywords = data.keywords;
        return data.keywords;
    } catch (error) {
        console.error('Error fetching keywords:', error);
        showNotification('Error al cargar keywords', 'error');
        return [];
    }
}

async function fetchKeywordSamples(keyword) {
    try {
        const response = await fetch(`${API_BASE}/keywords/${keyword}/samples`);
        const data = await response.json();
        return data.samples;
    } catch (error) {
        console.error('Error fetching samples:', error);
        return [];
    }
}

async function deleteSample(keyword, sampleId) {
    try {
        const response = await fetch(`${API_BASE}/keywords/${keyword}/samples/${sampleId}`, {
            method: 'DELETE'
        });
        return await response.json();
    } catch (error) {
        console.error('Error deleting sample:', error);
        throw error;
    }
}

async function trainAllKeywords(keywords) {
    try {
        const response = await fetch(`http://${window.location.hostname}:8000/api/rf/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                keywords: keywords,
                test_size: 0.2,
                n_estimators: 100,
                max_depth: 20,
                cross_validate: true
            })
        });
        
        const data = await response.json();
        
        // Si el servidor devuelve error, lanzar excepción con el detalle
        if (!response.ok) {
            throw new Error(data.detail || `Error ${response.status}: ${response.statusText}`);
        }
        
        return data;
    } catch (error) {
        console.error('Error training Random Forest:', error);
        throw error;
    }
}

async function deleteKeyword(keyword) {
    try {
        const response = await fetch(`${API_BASE}/keywords/${keyword}`, {
            method: 'DELETE'
        });
        return await response.json();
    } catch (error) {
        console.error('Error deleting keyword:', error);
        throw error;
    }
}

async function uploadSample(keyword, audioBlob) {
    try {
        const formData = new FormData();
        formData.append('file', audioBlob, `${keyword}_sample.wav`);
        
        const response = await fetch(`${API_BASE}/keywords/${keyword}/samples`, {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    } catch (error) {
        console.error('Error uploading sample:', error);
        throw error;
    }
}

async function generateSyntheticVoices(keyword, nSamples = 10) {
    try {
        const response = await fetch(`${API_BASE}/keywords/${keyword}/generate-synthetic?n_samples=${nSamples}`, {
            method: 'POST'
        });
        return await response.json();
    } catch (error) {
        console.error('Error generating synthetic voices:', error);
        throw error;
    }
}

async function createKeyword(keyword) {
    try {
        const response = await fetch(`${API_BASE}/keywords`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name: keyword })
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Error creating keyword:', error);
        throw error;
    }
}

// ============================================
// UI Rendering
// ============================================

async function renderKeywords() {
    const keywords = await fetchKeywords();
    
    if (keywords && keywords.length === 0) {
        elements.keywordsContainer.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-secondary);">
                <p style="font-size: 1.2rem;">No hay keywords creados</p>
                <p>Crea uno nuevo usando el campo de arriba</p>
            </div>
        `;
        return;
    }
    
    elements.keywordsContainer.innerHTML = '';
    
    for (const keyword of keywords) {
        const samples = await fetchKeywordSamples(keyword.name);
        const card = createKeywordCard(keyword, samples);
        elements.keywordsContainer.appendChild(card);
    }
}

function createKeywordCard(keyword, samples) {
    const card = document.createElement('div');
    card.className = 'keyword-card';
    if (state.selectedKeyword === keyword.name) {
        card.classList.add('selected');
    }
    
    card.innerHTML = `
        <div class="keyword-header">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div class="keyword-name">${keyword.name}</div>
                ${keyword.needs_training ? 
                    '<span class="training-status-badge needs-training" title="Necesita reentrenamiento">⚠️ Entrenar</span>' : 
                    '<span class="training-status-badge trained" title="Modelo entrenado">✅ OK</span>'
                }
            </div>
            <button class="btn btn-small btn-icon-only delete-keyword-btn" data-keyword="${keyword.name}" title="Eliminar keyword">
                🗑️
            </button>
        </div>
        
        <div class="keyword-stats">
            <div class="stat-item">
                <span>Templates:</span>
                <span class="stat-value">${keyword.n_templates}</span>
            </div>
            <div class="stat-item">
                <span>Muestras:</span>
                <span class="stat-value">${samples.length}</span>
            </div>
        </div>
        
        ${samples.length > 0 ? `
            <div class="samples-header">
                <span class="samples-title">Muestras</span>
                <span class="samples-badge">${samples.length}</span>
            </div>
        ` : '<div style="margin: 15px 0; padding: 20px; text-align: center; color: var(--text-secondary); font-size: 0.9rem; background: var(--bg-dark); border-radius: 8px; border: 2px dashed var(--bg-medium);">Sin muestras aún</div>'}
        
        <div class="sample-list" id="samples-${keyword.name}">
        </div>
        
        <div style="display: flex; flex-direction: column; gap: 8px; margin-top: auto; padding-top: 15px;">
            <button class="btn btn-primary select-keyword-btn" data-keyword="${keyword.name}" style="width: 100%;">
                🔴 Grabar Muestra
            </button>
            <button class="btn btn-secondary generate-synthetic-btn" data-keyword="${keyword.name}" title="Generar voces sintéticas con variaciones" style="width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                ⚡ Generar Voces Sintéticas
            </button>
        </div>
    `;
    
    // Agregar muestras
    if (samples.length > 0) {
        const samplesList = card.querySelector(`#samples-${keyword.name}`);
        samples.forEach(sample => {
            const sampleItem = document.createElement('div');
            sampleItem.className = 'sample-item';
            const duration = sample.duration ? sample.duration.toFixed(2) : '?.??';
            sampleItem.innerHTML = `
                <div class="sample-info">
                    <div class="sample-name">🎧 ${sample.filename}</div>
                    <div class="sample-duration">⏱️ ${duration}s</div>
                </div>
                <div class="sample-actions">
                    <button class="btn btn-small btn-icon-only play-sample-btn" data-keyword="${keyword.name}" data-sample="${sample.id}" data-filename="${sample.filename}" title="Reproducir">
                        ▶️
                    </button>
                    <button class="btn btn-small btn-icon-only delete-sample-btn" data-keyword="${keyword.name}" data-sample="${sample.id}" title="Eliminar">
                        ✖
                    </button>
                </div>
            `;
            samplesList.appendChild(sampleItem);
        });
    }
    
    return card;
}

// ============================================
// Recording Functions
// ============================================

async function initAudio() {
    try {
        state.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                sampleRate: 16000
            }
        });
        
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });
        
        return true;
    } catch (error) {
        console.error('Error initializing audio:', error);
        showNotification('Error al acceder al micrófono. Verifica los permisos.', 'error');
        return false;
    }
}

async function startRecording() {
    if (!state.selectedKeyword) {
        showNotification('Selecciona un keyword primero', 'warning');
        return;
    }
    
    // Inicializar audio si es necesario
    if (!state.audioContext) {
        const success = await initAudio();
        if (!success) return;
    }
    
    // Ajustar tamaño del canvas
    const canvas = elements.waveformCanvas;
    canvas.width = 280;
    canvas.height = 280;
    
    // Deshabilitar botón
    elements.modalStartBtn.disabled = true;
    
    // Countdown
    await showCountdown();
    
    // Iniciar grabación
    state.isRecording = true;
    state.recordedChunks = [];
    
    elements.modalStatus.textContent = '🔴 Grabando...';
    elements.modalInstructions.textContent = `Di "${state.selectedKeyword}" ahora!`;
    elements.modalStopBtn.disabled = false;
    
    // Usar AudioContext para capturar audio de manera más robusta
    const source = state.audioContext.createMediaStreamSource(state.mediaStream);
    const processor = state.audioContext.createScriptProcessor(4096, 1, 1);
    
    // Crear visualizador circular
    const vizResult = createAudioVisualizer(
        canvas,
        state.audioContext,
        state.mediaStream,
        {
            radius: 100,
            barCount: 64,
            maxBarHeight: 60,
            lineWidth: 3,
            colorful: true,
            smoothing: 0.7
        }
    );
    
    if (vizResult) {
        state.circularVisualizer = vizResult.visualizer;
        state.circularVisualizer.start();
    }
    
    const audioData = [];
    
    processor.onaudioprocess = (e) => {
        if (state.isRecording) {
            const inputData = e.inputBuffer.getChannelData(0);
            audioData.push(new Float32Array(inputData));
        }
    };
    
    source.connect(processor);
    processor.connect(state.audioContext.destination);
    
    // Actualizar progreso circular y tiempo
    const circumference = 2 * Math.PI * 130; // Radio = 130
    let elapsed = 0;
    const interval = setInterval(() => {
        elapsed += 50;
        const progress = elapsed / RECORDING_DURATION;
        const offset = circumference * (1 - progress);
        elements.progressRingCircle.style.strokeDashoffset = offset;
        
        const timeLeft = ((RECORDING_DURATION - elapsed) / 1000).toFixed(1);
        elements.recordingTimeLeft.textContent = `${timeLeft}s`;
        
        if (elapsed >= RECORDING_DURATION) {
            clearInterval(interval);
        }
    }, 50);
    
    // Detener automáticamente después de RECORDING_DURATION
    setTimeout(async () => {
        if (state.isRecording) {
            state.isRecording = false;
            processor.disconnect();
            source.disconnect();
            
            // Detener visualización
            if (state.circularVisualizer) {
                state.circularVisualizer.stop();
            }
            
            // Concatenar todos los chunks de audio
            const totalLength = audioData.reduce((acc, arr) => acc + arr.length, 0);
            const fullAudio = new Float32Array(totalLength);
            let offset = 0;
            for (const chunk of audioData) {
                fullAudio.set(chunk, offset);
                offset += chunk.length;
            }
            
            // Convertir a WAV
            const wavBlob = audioBufferToWav(fullAudio, state.audioContext.sampleRate);
            
            stopRecording();
            await uploadAndRefresh(wavBlob);
        }
    }, RECORDING_DURATION);
}

function stopRecording() {
    state.isRecording = false;
    
    // Detener visualizador circular
    if (state.circularVisualizer) {
        state.circularVisualizer.stop();
        state.circularVisualizer = null;
    }
    
    elements.modalStatus.textContent = 'Procesando...';
    elements.modalInstructions.textContent = 'Guardando muestra...';
    elements.modalStopBtn.disabled = true;
    
    // Resetear progreso circular
    elements.progressRingCircle.style.strokeDashoffset = 817;
    elements.recordingTimeLeft.textContent = '2.0s';
}

async function uploadAndRefresh(audioBlob) {
    try {
        await uploadSample(state.selectedKeyword, audioBlob);
        
        state.recordedSamples++;
        elements.modalSamplesCount.textContent = `Muestras grabadas: ${state.recordedSamples}`;
        
        // Refrescar la tarjeta del keyword (sin cerrar modal)
        await renderKeywords();
        
        // Resetear UI del modal
        elements.progressRingCircle.style.strokeDashoffset = 817;
        elements.recordingTimeLeft.textContent = '2.0s';
        elements.modalStatus.textContent = '✅ Muestra guardada';
        elements.modalInstructions.textContent = 'Listo para grabar otra muestra';
        elements.modalStartBtn.disabled = false;
        
        showNotification(`Muestra de "${state.selectedKeyword}" guardada`, 'success');
    } catch (error) {
        console.error('Error uploading sample:', error);
        showNotification('Error al guardar muestra', 'error');
        elements.modalStartBtn.disabled = false;
    }
}

async function showCountdown() {
    for (let i = 3; i > 0; i--) {
        elements.modalCountdown.textContent = i;
        await sleep(1000);
    }
    elements.modalCountdown.textContent = '¡AHORA!';
    await sleep(500);
    elements.modalCountdown.textContent = '';
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================
// Audio Utilities
// ============================================

function audioBufferToWav(audioData, sampleRate) {
    /**
     * Convierte Float32Array a WAV Blob
     */
    const numChannels = 1;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const blockAlign = numChannels * bytesPerSample;
    
    const dataLength = audioData.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);
    
    // WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true); // byte rate
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataLength, true);
    
    // Audio data
    let offset = 44;
    for (let i = 0; i < audioData.length; i++) {
        const sample = Math.max(-1, Math.min(1, audioData[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// ============================================
// Event Handlers
// ============================================

elements.createKeywordBtn.addEventListener('click', async () => {
    const keyword = elements.newKeywordInput.value.trim().toLowerCase();
    
    if (!keyword) {
        showNotification('Ingresa un nombre de keyword', 'warning');
        return;
    }
    
    if (!/^[a-z]+$/.test(keyword)) {
        showNotification('El keyword solo debe contener letras minúsculas', 'warning');
        return;
    }
    
    // Verificar si ya existe
    if (state.keywords.some(kw => kw.name === keyword)) {
        showNotification(`El keyword "${keyword}" ya existe`, 'warning');
        return;
    }
    
    try {
        // Deshabilitar botón mientras se crea
        elements.createKeywordBtn.disabled = true;
        elements.createKeywordBtn.textContent = '⏳ Creando...';
        
        // Crear keyword en BD
        await createKeyword(keyword);
        
        // Limpiar input
        elements.newKeywordInput.value = '';
        
        // Refrescar lista completa desde servidor
        await fetchKeywords();
        await renderKeywords();
        
        showNotification(`Keyword "${keyword}" creado exitosamente`, 'success');
        
        // Abrir modal para empezar a grabar
        selectKeyword(keyword);
    } catch (error) {
        console.error('Error creating keyword:', error);
        showNotification(`Error al crear keyword: ${error.message}`, 'error');
    } finally {
        // Rehabilitar botón
        elements.createKeywordBtn.disabled = false;
        elements.createKeywordBtn.textContent = '✅ Crear Keyword';
    }
});

// Modal event listeners
elements.modalStartBtn.addEventListener('click', startRecording);
elements.modalStopBtn.addEventListener('click', stopRecording);
elements.closeModalBtn.addEventListener('click', closeRecordingModal);
elements.closeModelModalBtn.addEventListener('click', closeModelVisualizationModal);

// Cerrar modal al hacer click fuera
elements.recordingModal.addEventListener('click', (e) => {
    if (e.target === elements.recordingModal) {
        closeRecordingModal();
    }
});

elements.modelVizModal.addEventListener('click', (e) => {
    if (e.target === elements.modelVizModal) {
        closeModelVisualizationModal();
    }
});

// Event delegation para botones dinámicos
elements.keywordsContainer.addEventListener('click', async (e) => {
    const target = e.target;
    
    if (target.classList.contains('select-keyword-btn')) {
        const keyword = target.dataset.keyword;
        selectKeyword(keyword);
    }
    
    else if (target.classList.contains('generate-synthetic-btn')) {
        const keyword = target.dataset.keyword;
        await handleGenerateSynthetic(keyword);
    }
    
    else if (target.classList.contains('delete-keyword-btn')) {
        const keyword = target.dataset.keyword;
        await handleDeleteKeyword(keyword);
    }
    
    else if (target.classList.contains('delete-sample-btn')) {
        const keyword = target.dataset.keyword;
        const sampleId = target.dataset.sample;
        await handleDeleteSample(keyword, sampleId);
    }
    
    else if (target.classList.contains('play-sample-btn')) {
        const keyword = target.dataset.keyword;
        const filename = target.dataset.filename;
        await handlePlaySample(keyword, filename);
    }
});

function selectKeyword(keyword) {
    state.selectedKeyword = keyword;
    state.recordedSamples = 0;
    
    // Abrir modal de grabación
    openRecordingModal(keyword);
}

function openRecordingModal(keyword) {
    elements.modalKeywordName.textContent = keyword.toUpperCase();
    elements.modalStatus.textContent = 'Listo para grabar';
    elements.modalInstructions.textContent = 'Presiona "Iniciar" y di la palabra cuando se indique';
    // Resetear progreso circular
    elements.progressRingCircle.style.strokeDashoffset = 817;
    elements.recordingTimeLeft.textContent = '2.0s';
    elements.modalStartBtn.disabled = false;
    elements.modalStopBtn.disabled = true;
    elements.modalSamplesCount.textContent = `Muestras grabadas: ${state.recordedSamples}`;
    elements.recordingModal.style.display = 'flex';
}

function closeRecordingModal() {
    elements.recordingModal.style.display = 'none';
    state.selectedKeyword = null;
    
    // Refrescar keywords para mostrar las nuevas muestras
    renderKeywords();
}

async function handleTrainKeyword() {
    // Obtener todos los keywords para entrenar el modelo completo
    const allKeywords = state.keywords.map(kw => kw.name);
    
    const confirmed = await showConfirm(
        '¿Entrenar modelo Random Forest?',
        `Se entrenará con todos los keywords disponibles: ${allKeywords.join(', ')}\n\nNota: Cada keyword necesita al menos 5 muestras (recomendado 20+)`
    );
    
    if (!confirmed) return;
    
    try {
        showLoader(`Entrenando Random Forest...`, `Keywords: ${allKeywords.join(', ')}`);
        
        const result = await trainAllKeywords(allKeywords);
        
        hideLoader();
        
        // Verificar si el entrenamiento fue exitoso
        if (!result || !result.success) {
            showNotification(`Error: ${result?.detail || 'Entrenamiento fallido'}`, 'error');
            return;
        }
        
        showNotification(`Modelo entrenado exitosamente`, 'success');
        
        // Mostrar resultados si existen
        if (result.training_stats) {
            const stats = result.training_stats;
            let message = `🎓 Modelo Random Forest Entrenado\n\n`;
            message += `📊 Métricas:\n`;
            message += `  - Accuracy: ${(stats.test_accuracy * 100).toFixed(1)}%\n`;
            
            if (stats.test_metrics) {
                message += `  - Precision: ${(stats.test_metrics.precision * 100).toFixed(1)}%\n`;
                message += `  - Recall: ${(stats.test_metrics.recall * 100).toFixed(1)}%\n`;
                message += `  - F1-Score: ${(stats.test_metrics.f1_score * 100).toFixed(1)}%\n`;
            }
            
            if (stats.cross_validation) {
                message += `\n🔄 Cross-Validation:\n`;
                message += `  - Mean: ${(stats.cross_validation.mean * 100).toFixed(1)}%\n`;
                message += `  - Std: ±${(stats.cross_validation.std * 100).toFixed(1)}%\n`;
            }
            
            message += `\n📁 Modelo guardado en: ${result.model_file}`;
            
            // Mostrar modal con gráficos
            showTrainingResultsModal(result);
        } else {
            await showAlert('✅ Entrenamiento Completado', 'El modelo se entrenó correctamente.');
        }
        
        await renderKeywords();
    } catch (error) {
        hideLoader();
        showNotification(`Error al entrenar modelo: ${error.message}`, 'error');
        console.error(error);
    }
}

async function handleGenerateSynthetic(keyword) {
    const nSamples = await showPrompt(
        'Generar Voces Sintéticas',
        `¿Cuántas muestras sintéticas generar para "${keyword}"?\n(Recomendado: 10-20)`,
        '15'
    );
    
    if (!nSamples) return;
    
    const n = parseInt(nSamples);
    if (isNaN(n) || n < 1 || n > 50) {
        showNotification('Número inválido. Debe estar entre 1 y 50.', 'warning');
        return;
    }
    
    try {
        showLoader(`Generando ${n} voces sintéticas...`, `Keyword: ${keyword}`);
        
        const result = await generateSyntheticVoices(keyword, n);
        
        hideLoader();
        showNotification(result.message, 'success');
        
        await showAlert(
            'Muestras Sintéticas Generadas',
            `Muestras sintéticas generadas!\n\nKeyword: ${keyword}\nMuestras: ${result.generated_samples}\n\nAhora puedes entrenar el modelo.`
        );
        
        await renderKeywords();
    } catch (error) {
        hideLoader();
        showNotification(`Error generando voces sintéticas`, 'error');
        console.error(error);
    }
}

async function handleDeleteKeyword(keyword) {
    const confirmed = await showConfirm(
        'Eliminar Keyword',
        `¿Eliminar keyword "${keyword}" y todas sus muestras?`
    );
    
    if (!confirmed) return;
    
    try {
        showLoader(`Eliminando "${keyword}"...`, 'Borrando muestras y modelos');
        
        await deleteKeyword(keyword);
        
        hideLoader();
        showNotification(`Keyword "${keyword}" eliminado`, 'success');
        
        if (state.selectedKeyword === keyword) {
            state.selectedKeyword = null;
        }
        
        await renderKeywords();
    } catch (error) {
        hideLoader();
        showNotification(`Error al eliminar "${keyword}"`, 'error');
    }
}

async function handleDeleteSample(keyword, sampleId) {
    try {
        showLoader('Eliminando muestra...', '');
        await deleteSample(keyword, sampleId);
        hideLoader();
        showNotification('Muestra eliminada', 'success');
        await renderKeywords();
    } catch (error) {
        hideLoader();
        showNotification('Error al eliminar muestra', 'error');
    }
}

async function handlePlaySample(keyword, filename) {
    try {
        const audioUrl = `http://${window.location.hostname}:8000/recordings/${keyword}/${filename}`;
        const audio = new Audio(audioUrl);
        
        audio.onerror = () => {
            showNotification('Error al cargar audio', 'error');
        };
        
        audio.play().catch(err => {
            console.error('Error playing audio:', err);
            showNotification('Error al reproducir audio', 'error');
        });
        
        //showNotification(`Reproduciendo ${filename}`, 'info');
    } catch (error) {
        showNotification('Error al reproducir muestra', 'error');
    }
}

function showModelVisualization(keyword, modelStats, thresholds) {
    // Actualizar nombre
    elements.modelKeywordName.textContent = keyword.toUpperCase();
    
    // Actualizar estadísticas
    elements.modelNTemplates.textContent = modelStats.n_templates;
    elements.modelMeanLength.textContent = modelStats.mean_length.toFixed(1);
    elements.modelStdLength.textContent = modelStats.std_length.toFixed(1);
    elements.modelIntraScore.textContent = modelStats.mean_intra_score.toFixed(3);
    
    // Actualizar info del centroide
    elements.centroidInfo.textContent = `${modelStats.centroid_shape[0]} frames × ${modelStats.centroid_shape[1]} MFCC`;
    
    // Actualizar umbrales
    elements.thresholdMinVal.textContent = thresholds.threshold_min.toFixed(3);
    elements.thresholdMaxVal.textContent = thresholds.threshold_max.toFixed(3);
    
    // Posicionar marcadores de umbrales (normalizado 0-1)
    const minPercent = (thresholds.threshold_min / 2.0) * 100;
    const maxPercent = (thresholds.threshold_max / 2.0) * 100;
    const rangePercent = maxPercent - minPercent;
    
    elements.thresholdMinMarker.style.left = `${minPercent}%`;
    elements.thresholdMaxMarker.style.left = `${maxPercent}%`;
    elements.thresholdFill.style.left = `${minPercent}%`;
    elements.thresholdFill.style.width = `${rangePercent}%`;
    
    // Mostrar modal primero
    elements.modelVizModal.style.display = 'flex';
    
    // Dibujar centroide después de que el modal sea visible
    requestAnimationFrame(() => {
        drawCentroid(modelStats.centroid_data);
    });
}

function drawCentroid(centroidData) {
    const canvas = elements.centroidCanvas;
    if (!canvas) {
        console.error('Canvas no encontrado');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    
    // Forzar tamaño del canvas basado en el contenedor
    const container = canvas.parentElement;
    const containerWidth = container.offsetWidth - 40; // padding
    
    canvas.width = containerWidth;
    canvas.height = 300;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // Limpiar
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);
    
    if (!centroidData || centroidData.length === 0) {
        ctx.fillStyle = '#888';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No hay datos de centroide', width / 2, height / 2);
        return;
    }
    
    const nFrames = centroidData.length;
    const nMfcc = centroidData[0].length;
    
    console.log(`Dibujando centroide: ${nFrames} frames × ${nMfcc} MFCC`);
    
    // Encontrar min/max para normalización
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = 0; i < nFrames; i++) {
        for (let j = 0; j < nMfcc; j++) {
            minVal = Math.min(minVal, centroidData[i][j]);
            maxVal = Math.max(maxVal, centroidData[i][j]);
        }
    }
    
    // Dibujar heatmap
    const cellWidth = width / nFrames;
    const cellHeight = height / nMfcc;
    
    for (let i = 0; i < nFrames; i++) {
        for (let j = 0; j < nMfcc; j++) {
            const value = centroidData[i][j];
            const normalized = (value - minVal) / (maxVal - minVal);
            
            // Gradiente de color: azul -> cyan -> amarillo -> rojo
            let r, g, b;
            if (normalized < 0.33) {
                // Azul a cyan
                const t = normalized / 0.33;
                r = Math.floor(0 * (1 - t) + 0 * t);
                g = Math.floor(100 * (1 - t) + 217 * t);
                b = Math.floor(200 * (1 - t) + 255 * t);
            } else if (normalized < 0.66) {
                // Cyan a amarillo
                const t = (normalized - 0.33) / 0.33;
                r = Math.floor(0 * (1 - t) + 255 * t);
                g = Math.floor(217 * (1 - t) + 255 * t);
                b = Math.floor(255 * (1 - t) + 0 * t);
            } else {
                // Amarillo a rojo
                const t = (normalized - 0.66) / 0.34;
                r = 255;
                g = Math.floor(255 * (1 - t) + 100 * t);
                b = Math.floor(0 * (1 - t) + 0 * t);
            }
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(i * cellWidth, j * cellHeight, cellWidth + 1, cellHeight + 1);
        }
    }
    
    // Etiquetas
    ctx.fillStyle = '#fff';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('MFCC →', 10, 20);
    ctx.save();
    ctx.translate(10, height - 10);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('← Tiempo (frames)', 0, 0);
    ctx.restore();
}

function closeModelVisualizationModal() {
    elements.modelVizModal.style.display = 'none';
}

// ============================================
// Training Results Modal
// ============================================

function showTrainingResultsModal(result) {
    const stats = result.training_stats;
    
    // Actualizar métricas
    document.getElementById('tr-accuracy').textContent = (stats.test_accuracy * 100).toFixed(1);
    
    if (stats.test_metrics) {
        document.getElementById('tr-precision').textContent = (stats.test_metrics.precision * 100).toFixed(1);
        document.getElementById('tr-recall').textContent = (stats.test_metrics.recall * 100).toFixed(1);
        document.getElementById('tr-f1').textContent = (stats.test_metrics.f1_score * 100).toFixed(1);
    }
    
    if (stats.cross_validation) {
        document.getElementById('tr-cv-mean').textContent = (stats.cross_validation.mean * 100).toFixed(1);
        document.getElementById('tr-cv-std').textContent = (stats.cross_validation.std * 100).toFixed(1);
    }
    
    // Información del modelo
    document.getElementById('tr-model-path').textContent = result.model_file || '-';
    document.getElementById('tr-keywords').textContent = stats.classes?.join(', ') || '-';
    document.getElementById('tr-total-samples').textContent = stats.n_samples || '-';
    
    // Dibujar matriz de confusión
    if (stats.confusion_matrix) {
        drawConfusionMatrix(stats.confusion_matrix, stats.classes);
    }
    
    // Dibujar distribución de muestras
    if (stats.class_distribution) {
        drawSamplesDistribution(stats.class_distribution, stats.classes);
    }
    
    // Mostrar modal
    document.getElementById('training-results-modal').style.display = 'flex';
}

function drawConfusionMatrix(matrix, classes) {
    const canvas = document.getElementById('confusion-matrix-canvas');
    const ctx = canvas.getContext('2d');
    
    const n = matrix.length;
    const cellSize = 80;
    const padding = 60;
    
    canvas.width = n * cellSize + padding * 2;
    canvas.height = n * cellSize + padding * 2;
    
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Encontrar valor máximo para normalización
    let maxVal = 0;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            maxVal = Math.max(maxVal, matrix[i][j]);
        }
    }
    
    // Dibujar celdas
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const value = matrix[i][j];
            const x = padding + j * cellSize;
            const y = padding + i * cellSize;
            
            // Color basado en valor (diagonal = verde, resto = rojo)
            const intensity = value / maxVal;
            if (i === j) {
                // Diagonal (aciertos) - verde
                ctx.fillStyle = `rgba(0, 255, 100, ${0.3 + intensity * 0.7})`;
            } else {
                // Fuera de diagonal (errores) - rojo
                ctx.fillStyle = `rgba(255, 100, 100, ${intensity * 0.8})`;
            }
            
            ctx.fillRect(x, y, cellSize - 2, cellSize - 2);
            
            // Texto del valor
            ctx.fillStyle = intensity > 0.5 ? '#000' : '#fff';
            ctx.font = 'bold 20px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(value, x + cellSize / 2, y + cellSize / 2);
        }
    }
    
    // Etiquetas
    ctx.fillStyle = '#fff';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    
    // Etiquetas superiores (Predicted)
    for (let j = 0; j < n; j++) {
        const x = padding + j * cellSize + cellSize / 2;
        ctx.fillText(classes[j], x, padding - 20);
    }
    
    // Etiquetas izquierdas (Actual)
    ctx.textAlign = 'right';
    for (let i = 0; i < n; i++) {
        const y = padding + i * cellSize + cellSize / 2;
        ctx.fillText(classes[i], padding - 10, y);
    }
    
    // Títulos
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Predicted', canvas.width / 2, 20);
    
    ctx.save();
    ctx.translate(20, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Actual', 0, 0);
    ctx.restore();
}

function drawSamplesDistribution(distribution, classes) {
    const canvas = document.getElementById('samples-distribution-canvas');
    const ctx = canvas.getContext('2d');
    
    const n = classes.length;
    const barWidth = 80;
    const maxBarHeight = 200;
    const padding = 60;
    
    canvas.width = n * (barWidth + 20) + padding * 2;
    canvas.height = maxBarHeight + padding * 2;
    
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Encontrar valor máximo
    const maxSamples = Math.max(...Object.values(distribution));
    
    // Dibujar barras
    classes.forEach((cls, i) => {
        const samples = distribution[cls] || 0;
        const barHeight = (samples / maxSamples) * maxBarHeight;
        const x = padding + i * (barWidth + 20);
        const y = canvas.height - padding - barHeight;
        
        // Color gradiente
        const hue = (i / n) * 360;
        ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
        ctx.fillRect(x, y, barWidth, barHeight);
        
        // Borde
        ctx.strokeStyle = `hsl(${hue}, 70%, 40%)`;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, barWidth, barHeight);
        
        // Valor encima de la barra
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(samples, x + barWidth / 2, y - 10);
        
        // Etiqueta del keyword
        ctx.font = '14px Arial';
        ctx.fillText(cls, x + barWidth / 2, canvas.height - padding + 25);
    });
    
    // Título
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Muestras por Keyword', canvas.width / 2, 30);
}

// Event listeners para el modal de resultados
document.addEventListener('DOMContentLoaded', () => {
    const closeBtn = document.getElementById('close-training-results-btn');
    const okBtn = document.getElementById('training-results-ok-btn');
    const modal = document.getElementById('training-results-modal');
    
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.style.display = 'none';
        });
    }
    
    if (okBtn) {
        okBtn.addEventListener('click', () => {
            modal.style.display = 'none';
        });
    }
    
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    }
});

// Inicialización
document.addEventListener('DOMContentLoaded', () => {
    renderKeywords();
    console.log('🎓 Training Interface Loaded');
});
