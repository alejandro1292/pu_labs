/**
 * Training Interface - JavaScript
 * Maneja grabación de muestras y entrenamiento de keywords
 */

// Configuración
const API_BASE = `http://${window.location.host}/api/training`;
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
    circularVisualizer: null,
    uploadTargetKeyword: null
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
    centroidInfo: document.getElementById('centroid-info'),
    spectrogramImg: document.getElementById('spectrogram-img'),
    spectrogramPlaceholder: document.getElementById('spectrogram-placeholder'),
    closeModelModalBtn: document.getElementById('close-model-modal-btn'),
    // Input de archivos para subir múltiples muestras
    uploadFilesInput: document.getElementById('upload-files-input'),
    // Controles de clasificador
    classifierSelect: document.getElementById('classifier-select'),
    trainKeywordsBtn: document.getElementById('train-keywords-btn'),
    compareModelsBtn: document.getElementById('compare-models-btn'),

};

// Update train button label when classifier changes to keep UI coherent
function updateTrainButtonLabel() {
    if (!elements.classifierSelect || !elements.trainKeywordsBtn) return;
    const algo = elements.classifierSelect.value || 'rf';
    elements.trainKeywordsBtn.textContent = `🤖 Entrenar Keywords (${algo.toUpperCase()})`;
}

if (elements.classifierSelect) {
    elements.classifierSelect.addEventListener('change', async () => {
        // Update label immediately
        updateTrainButtonLabel();
        // Set classifier as active on the backend
        try {
            await setActiveClassifier(elements.classifierSelect.value);
        } catch (err) {
            console.error('Error setting classifier active:', err);
        }
    });
}

// Initialize label on load
updateTrainButtonLabel();

/**
 * Fetch the currently active classifier from the backend and set the select value
 */
async function getActiveClassifier() {
    try {
        const resp = await fetch(`${API_BASE}/classifier`);
        if (!resp.ok) return;
        const data = await resp.json();
        if (data && data.classifier && elements.classifierSelect) {
            elements.classifierSelect.value = data.classifier;
            updateTrainButtonLabel();
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
        const resp = await fetch(`${API_BASE}/classifier`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ classifier })
        });
        if (!resp.ok) {
            const txt = await resp.text().catch(() => null);
            throw new Error(txt || `HTTP ${resp.status}`);
        }
        showNotification(`Clasificador ${classifier.toUpperCase()} establecido como activo`, 'success');
        // trigger an update to the train button label (already set) and optionally notify other UIs
        updateTrainButtonLabel();
    } catch (err) {
        console.error('Failed to set active classifier:', err);
        showNotification('Error al cambiar clasificador', 'error');
        // attempt to revert select to backend value
        await getActiveClassifier();
        throw err;
    } finally {
        if (elements.classifierSelect) elements.classifierSelect.disabled = false;
    }
}

// Load current active classifier when page loads
getActiveClassifier();

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
    // Determinar algoritmo seleccionado (por defecto rf)
    const algo = elements.classifierSelect ? elements.classifierSelect.value : 'rf';
    const apiPath = algo === 'fw' ? '/api/fw/train' : '/api/rf/train';

    try {
        // Ensure we call the backend port (server runs on :8000)
        const response = await fetch(`http://${window.location.host}${apiPath}`, {
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

        // Network errors will throw; non-ok responses should be parsed too
        const contentType = response.headers.get('content-type') || '';
        const data = contentType.includes('application/json') ? await response.json() : { detail: await response.text() };

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
            
            <button class="btn btn-small btn-icon-only view-model-btn" data-keyword="${keyword.name}" title="Ver Modelo">
                📊
            </button>
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
            <button class="btn btn-secondary upload-training-btn" data-keyword="${keyword.name}" title="Subir audios de entrenamiento" style="width: 100%;">
                📁 Cargar Audios
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

/**
 * Sube múltiples archivos para un keyword (llama a uploadSample por cada archivo)
 * @param {string} keyword
 * @param {File[]} files
 */
async function uploadMultipleSamples(keyword, files) {
    const failed = [];
    let successCount = 0;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        try {
            // Append progressive loader text
            showLoader(`Subiendo ${i + 1}/${files.length}...`, `${file.name}`);
            await uploadSample(keyword, file);
            successCount++;
        } catch (err) {
            console.error(`Error subiendo ${file.name}:`, err);
            failed.push(file.name || `file_${i}`);
        }
    }

    return { successCount, failed };
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

// Classifier controls
if (elements.classifierSelect) {
    // Inicializar selector con el estado del servidor
    async function refreshClassifierInfo() {
        try {
            const resp = await fetch(`http://${window.location.host}/api/training/classifier`);
            const data = await resp.json();
            if (data && data.success) {
                if (elements.classifierSelect) {
                    elements.classifierSelect.value = data.classifier || 'rf';
                }
                updateTrainButtonLabel();
            }
        } catch (err) {
            console.warn('No se pudo obtener info de classifier:', err);
        }
    }

    function updateTrainButtonLabel() {
        const val = elements.classifierSelect.value;
        if (elements.trainKeywordsBtn) {
            elements.trainKeywordsBtn.textContent = val === 'fw' ? '🤖 Entrenar Keywords (FW)' : '🤖 Entrenar Keywords (RF)';
        }
    }

    elements.classifierSelect.addEventListener('change', () => {
        updateTrainButtonLabel();
    });

    if (elements.setClassifierBtn) {
        elements.setClassifierBtn.addEventListener('click', async () => {
            const desired = elements.classifierSelect.value;
            const confirmed = await showConfirm('Cambiar Clasificador Activo', `¿Establecer '${desired}' como clasificador activo?`);
            if (!confirmed) return;

            try {
                showLoader('Cambiando clasificador...', `${desired}`);
                const resp = await fetch(`http://${window.location.host}/api/training/classifier`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ classifier: desired })
                });
                const data = await resp.json();
                hideLoader();
                if (resp.ok && data.success) {
                    showNotification(data.message || 'Clasificador actualizado', 'success');
                } else {
                    showNotification(data.detail || data.message || 'Error cambiando clasificador', 'error');
                }
            } catch (err) {
                hideLoader();
                console.error('Error cambiando clasificador:', err);
                showNotification('Error cambiando clasificador', 'error');
            }
        });
    }

    if (elements.compareModelsBtn) {
        elements.compareModelsBtn.addEventListener('click', async () => {
            try {
                showLoader('Comparando modelos...', 'Obteniendo info');
                const [rfResp, fwResp] = await Promise.all([
                    fetch(`http://${window.location.host}/api/rf/model/info`).then(r => r.json()).catch(() => null),
                    fetch(`http://${window.location.host}/api/fw/model/info`).then(r => r.json()).catch(() => null)
                ]);
                hideLoader();

                const rfAcc = rfResp && rfResp.training_stats ? (rfResp.training_stats.test_accuracy * 100).toFixed(1) + '%' : 'N/A';
                const fwAcc = fwResp && fwResp.training_stats ? (fwResp.training_stats.test_accuracy * 100).toFixed(1) + '%' : 'N/A';

                await showAlert('Comparación de Modelos', `Random Forest: ${rfAcc}\nFourier+Wavelet: ${fwAcc}`);
            } catch (err) {
                hideLoader();
                console.error('Error comparando modelos:', err);
                showNotification('Error comparando modelos', 'error');
            }
        });
    }

    if (elements.trainKeywordsBtn) {
        elements.trainKeywordsBtn.addEventListener('click', async () => {
            // Proxy to main handler (keeps confirmation text clear)
            const allKeywords = state.keywords.map(kw => kw.name);
            const selected = elements.classifierSelect.value;
            const label = selected === 'fw' ? 'Fourier+Wavelet (FW)' : 'Random Forest (RF)';

            const confirmed = await showConfirm(
                `¿Entrenar modelo ${label}?`,
                `Se entrenará con todos los keywords disponibles: ${allKeywords.join(', ')}\n\nNota: Cada keyword necesita al menos 5 muestras (recomendado 20+)`);

            if (!confirmed) return;

            try {
                showLoader(`Entrenando ${label}...`, `Keywords: ${allKeywords.join(', ')}`);
                const result = await trainAllKeywords(allKeywords);
                hideLoader();

                if (!result || !result.success) {
                    showNotification(`Error: ${result?.detail || 'Entrenamiento fallido'}`, 'error');
                    return;
                }

                showNotification(`Modelo entrenado exitosamente`, 'success');
                if (result.training_stats) {
                    showTrainingResultsModal(result);
                } else {
                    await showAlert('✅ Entrenamiento Completado', 'El modelo se entrenó correctamente.');
                }

                // Refresh classifier info (models existence)
                await refreshClassifierInfo();
                await renderKeywords();
            } catch (error) {
                hideLoader();
                showNotification(`Error al entrenar modelo: ${error.message}`, 'error');
                console.error(error);
            }
        });
    }

    // Inicializar al cargar la página
    document.addEventListener('DOMContentLoaded', refreshClassifierInfo);
}

// Modal event listeners
elements.modalStartBtn.addEventListener('click', startRecording);
elements.modalStopBtn.addEventListener('click', stopRecording);
elements.closeModalBtn.addEventListener('click', closeRecordingModal);
elements.closeModelModalBtn.addEventListener('click', closeModelVisualizationModal);
// File input change: upload selected files for the current keyword
if (elements.uploadFilesInput) {
    elements.uploadFilesInput.addEventListener('change', async (e) => {
        const files = Array.from(e.target.files || []);
        // Determine target keyword from state (set when user clicked upload button)
        const keyword = state.uploadTargetKeyword;
        if (!keyword) return;
        if (!files.length) return;

        try {
            showLoader(`Subiendo ${files.length} archivos...`, `Keyword: ${keyword}`);
            const result = await uploadMultipleSamples(keyword, files);
            hideLoader();

            if (result.failed.length === 0) {
                showNotification(`${result.successCount} archivos subidos para "${keyword}"`, 'success');
            } else {
                showNotification(`${result.successCount} subidos, ${result.failed.length} fallaron`, 'warning');
                console.error('Fallaron al subir:', result.failed);
            }

            // Limpiar selección y estado
            elements.uploadFilesInput.value = '';
            state.uploadTargetKeyword = null;

            await renderKeywords();
        } catch (error) {
            hideLoader();
            console.error('Error subiendo archivos:', error);
            showNotification('Error subiendo archivos', 'error');
            elements.uploadFilesInput.value = '';
            state.uploadTargetKeyword = null;
        }
    });
}

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

    else if (target.classList.contains('view-model-btn')) {
        const keyword = target.dataset.keyword;
        await handleViewModel(keyword);
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
    else if (target.classList.contains('upload-training-btn')) {
        const keyword = target.dataset.keyword;
        // Set upload target and open file dialog
        state.uploadTargetKeyword = keyword;
        if (elements.uploadFilesInput) {
            elements.uploadFilesInput.click();
        } else {
            showNotification('Input de archivos no disponible', 'error');
        }
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
        const audioUrl = `http://${window.location.host}/recordings/${keyword}/${filename}`;
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

async function handleViewModel(keyword) {
    try {
        showLoader('Cargando modelo...', keyword);

        // Intentar obtener info del modelo FW
        const resp = await fetch(`http://${window.location.host}/api/fw/model/info`);
        if (resp.ok) {
            const data = await resp.json();
            const stats = data.training_stats;
            const kwStats = stats?.per_keyword_stats ? stats.per_keyword_stats[keyword] : null;

            if (kwStats) {
                const modelStats = {
                    n_templates: kwStats.n_samples || "N/A",
                    mean_length: kwStats.mean_length || 0,
                    std_length: kwStats.std_length || 0,
                    mean_intra_score: kwStats.intra_score || 0,
                    centroid_shape: [1, kwStats.centroid ? kwStats.centroid.length : 0],
                    centroid_data: kwStats.centroid || []
                };

                // Umbrales sugeridos basados en intra_score (distancia Euclidiana)
                // Para visualización: Min = intra_score, Max = intra_score * 1.5
                const thresholds = {
                    threshold_min: kwStats.intra_score || 0.5,
                    threshold_max: (kwStats.intra_score || 0.5) * 1.5
                };

                showModelVisualization(keyword, modelStats, thresholds);
            } else {
                // Fallback si no hay stats por keyword
                const dummyStats = {
                    n_templates: "N/A",
                    mean_length: 0,
                    std_length: 0,
                    mean_intra_score: stats?.mean_intra_score || 0,
                    centroid_shape: [0, 0],
                    centroid_data: []
                };
                const dummyThresholds = { threshold_min: 0, threshold_max: 1 };
                showModelVisualization(keyword, dummyStats, dummyThresholds);
            }
        } else {
            showNotification('No hay modelo FW cargado', 'warning');
        }

        hideLoader();
    } catch (error) {
        hideLoader();
        console.error(error);
        showNotification('Error cargando info del modelo', 'error');
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
    if (modelStats.centroid_shape[0] === 1) {
        elements.centroidInfo.textContent = `${modelStats.centroid_shape[1]} Features (Fourier/Wavelet)`;
    } else {
        elements.centroidInfo.textContent = `${modelStats.centroid_shape[0]} frames × ${modelStats.centroid_shape[1]} MFCC`;
    }

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

    // Cargar espectrograma
    loadSpectrogram(keyword);
}

function loadSpectrogram(keyword) {
    if (!elements.spectrogramImg) return;

    elements.spectrogramImg.style.display = 'none';
    elements.spectrogramPlaceholder.style.display = 'block';
    elements.spectrogramPlaceholder.textContent = 'Cargando espectrograma...';

    const img = new Image();
    img.onload = () => {
        elements.spectrogramImg.src = img.src;
        elements.spectrogramImg.style.display = 'block';
        elements.spectrogramPlaceholder.style.display = 'none';
    };
    img.onerror = () => {
        elements.spectrogramPlaceholder.textContent = 'No se pudo cargar el espectrograma (¿quizás no hay muestras?)';
    };
    // Timestamp para evitar caché
    img.src = `${API_BASE.replace('/training', '/fw')}/spectrogram/${keyword}?t=${Date.now()}`;
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

    let nFrames, nMfcc;
    let data2D = [];

    if (Array.isArray(centroidData[0])) {
        // Es 2D (formato original RF/DTW)
        nFrames = centroidData.length;
        nMfcc = centroidData[0].length;
        data2D = centroidData;
    } else {
        // Es 1D (formato FW)
        const total = centroidData.length;
        if (total === 768) {
            // 3 segmentos x 64 bandas x 4 stats (mean, std, max, min)
            // Visualizamos solo la media (primer stat de cada 4)
            nFrames = 3;
            nMfcc = 64;
            for (let i = 0; i < nFrames; i++) {
                const segmentData = [];
                for (let j = 0; j < nMfcc; j++) {
                    const baseIdx = (i * nMfcc * 4) + (j * 4);
                    segmentData.push(centroidData[baseIdx]); // Tomar 'mean'
                }
                data2D.push(segmentData);
            }
        } else if (total === 192) {
            // 3 segmentos x 64 bandas (solo 1 stat)
            nFrames = 3;
            nMfcc = 64;
            for (let i = 0; i < nFrames; i++) {
                data2D.push(centroidData.slice(i * nMfcc, (i + 1) * nMfcc));
            }
        } else {
            // Fallback: 1 fila con todos los features
            nFrames = 1;
            nMfcc = total;
            data2D = [centroidData];
        }
    }

    console.log(`Dibujando centroide: ${nFrames} × ${nMfcc}`);

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
            const value = data2D[i][j];
            const normalized = (value - minVal) / (maxVal - minVal || 1);

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
