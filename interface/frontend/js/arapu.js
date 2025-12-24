// =====================================
// GALAXY VOICE COMMANDER - GAME LOGIC
// =====================================
// Autor: Alejandro Ortiz
// Fecha: Diciembre 2025

// Audio Streaming State
const audioState = {
    audioContext: null,
    mediaStream: null,
    scriptProcessorNode: null,
    circularVisualizer: null,
    isStreaming: false,
    keywordTimeout: null, // Timeout para fade del keyword
    vad: null, // Voice Activity Detector
    websocket: null // WebSocket connection
};

// Game Configuration
const config = {
    canvas: null,
    ctx: null,
    width: 800,
    height: 600,
    playerSpeed: 8,
    bulletSpeed: 30,
    enemyBaseSpeed: 2,
    enemySpawnRate: 1500, // ms
    bombCooldown: 5000 // ms
};

// Game State
const state = {
    isRunning: false,
    isTestMode: false,
    isPaused: false,
    score: 0,
    lives: 5,
    bombs: 5, // Bombas disponibles
    level: 1,
    enemiesKilled: 0,
    lastBombTime: 0,
    hasSpeedPowerup: false, // Powerup de velocidad de disparo activo

    // Keywords Configuration
    keywords: {
        up: null,
        down: null,
        bomb: null
    },

    // Game Entities
    player: null,
    bullets: [],
    enemies: [],
    explosions: [],
    shockwaves: [], // Ondas expansivas de bombas
    powerups: [], // Premios (vidas y bombas)

    // Timers
    enemySpawnTimer: null,
    gameLoopId: null
};

// Player Ship Class
class Player {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.targetY = y; // Posición objetivo para movimiento suave
        this.width = 40;
        this.height = 30;
        this.speed = config.playerSpeed;
        this.moveDistance = 120; // Distancia de movimiento
        this.smoothing = 0.15; // Factor de suavizado (0-1, menor = más suave)
        this.color = '#00d9ff';
        this.shootCooldown = 0;
        this.normalShootCooldown = 15; // Cooldown normal (~250ms at 60fps)
        this.fastShootCooldown = 7; // Cooldown rápido (~117ms at 60fps)
    }

    draw(ctx) {
        ctx.save();

        // Efecto de powerup de velocidad (brillo amarillo eléctrico)
        if (state.hasSpeedPowerup) {
            const pulseTime = Date.now() * 0.008;
            const pulseSize = 1 + Math.sin(pulseTime) * 0.2;
            const glowGradient = ctx.createRadialGradient(this.x + this.width / 2, this.y, 0,
                this.x + this.width / 2, this.y, 40 * pulseSize);
            glowGradient.addColorStop(0, 'rgba(255, 255, 0, 0.6)');
            glowGradient.addColorStop(0.5, 'rgba(255, 220, 0, 0.3)');
            glowGradient.addColorStop(1, 'rgba(255, 255, 0, 0)');
            ctx.fillStyle = glowGradient;
            ctx.beginPath();
            ctx.arc(this.x + this.width / 2, this.y, 45 * pulseSize, 0, Math.PI * 2);
            ctx.fill();
        }

        // Nave principal (triángulo apuntando a la derecha)
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.moveTo(this.x + this.width, this.y); // Punta derecha
        ctx.lineTo(this.x, this.y - this.height / 2); // Arriba izquierda
        ctx.lineTo(this.x, this.y + this.height / 2); // Abajo izquierda
        ctx.closePath();
        ctx.fill();

        // Brillo
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Alas (rectángulos)
        ctx.fillStyle = '#6c5ce7';
        ctx.fillRect(this.x - 5, this.y - this.height / 2 - 5, 15, 10);
        ctx.fillRect(this.x - 5, this.y + this.height / 2 - 5, 15, 10);

        // Motor (círculo con brillo)
        const gradient = ctx.createRadialGradient(this.x - 10, this.y, 5, this.x - 10, this.y, 15);
        gradient.addColorStop(0, '#ff6b6b');
        gradient.addColorStop(0.5, '#ff8c42');
        gradient.addColorStop(1, 'rgba(255, 107, 107, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(this.x - 10, this.y, 12, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }

    moveUp() {
        this.targetY = Math.max(this.height / 2 + 20, this.targetY - this.moveDistance);
    }

    moveDown() {
        this.targetY = Math.min(config.height - this.height / 2 - 20, this.targetY + this.moveDistance);
    }

    update() {
        // Movimiento suave hacia targetY con interpolación
        const diff = this.targetY - this.y;
        if (Math.abs(diff) > 0.5) {
            this.y += diff * this.smoothing;
        } else {
            this.y = this.targetY;
        }

        if (this.shootCooldown > 0) this.shootCooldown--;
    }

    shoot() {
        if (this.shootCooldown <= 0) {
            state.bullets.push(new Bullet(this.x + this.width, this.y));
            // Usar cooldown según powerup activo
            this.shootCooldown = state.hasSpeedPowerup ? this.fastShootCooldown : this.normalShootCooldown;
        }
    }
}

// Bullet Class
class Bullet {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.width = 15;
        this.height = 5;
        this.speed = config.bulletSpeed;
        this.color = '#00ff88';
    }

    draw(ctx) {
        ctx.save();

        // Proyectil (rombo alargado)
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.moveTo(this.x + this.width, this.y);
        ctx.lineTo(this.x + this.width / 2, this.y - this.height);
        ctx.lineTo(this.x, this.y);
        ctx.lineTo(this.x + this.width / 2, this.y + this.height);
        ctx.closePath();
        ctx.fill();

        // Estela brillante
        const gradient = ctx.createLinearGradient(this.x, this.y, this.x - 20, this.y);
        gradient.addColorStop(0, 'rgba(0, 255, 136, 0.8)');
        gradient.addColorStop(1, 'rgba(0, 255, 136, 0)');
        ctx.fillStyle = gradient;
        ctx.fillRect(this.x - 20, this.y - 2, 20, 4);

        ctx.restore();
    }

    update() {
        this.x += this.speed;
    }

    isOffScreen() {
        return this.x > config.width + 50;
    }
}

// Enemy Class
class Enemy {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.baseY = y;

        // Determinar tipo de enemigo (resistencia)
        const rand = Math.random();
        if (rand < 0.5) {
            this.maxHp = 1;
        } else if (rand < 0.75) {
            this.maxHp = 2;
        } else if (rand < 0.9) {
            this.maxHp = 3;
        } else {
            this.maxHp = 5;
        }

        this.hp = this.maxHp;

        // Tamaño proporcional a la resistencia
        const baseSize = 25;
        this.baseWidth = baseSize + (this.maxHp - 1) * 10;
        this.baseHeight = baseSize + (this.maxHp - 1) * 10;
        this.width = this.baseWidth;
        this.height = this.baseHeight;

        // Velocidad variable (más rápido = menos HP generalmente)
        const speedVariation = (Math.random() - 0.5) * 1.5;
        this.speed = config.enemyBaseSpeed + (state.level - 1) * 0.5 + speedVariation - (this.maxHp - 1) * 0.3;
        this.speed = Math.max(1, this.speed); // Velocidad mínima

        this.amplitude = 50 + Math.random() * 50; // Amplitud de la onda
        this.frequency = 0.02 + Math.random() * 0.02; // Frecuencia de la onda
        this.phase = Math.random() * Math.PI * 2; // Fase inicial
        this.color = this.getColorByHp();
        this.shape = Math.floor(Math.random() * 4); // 0: círculo, 1: cuadrado, 2: triángulo, 3: rombo
    }

    getColorByHp() {
        // Color según HP: más oscuro/rojo = más resistente
        const colors = {
            1: ['#ffd93d', '#ff8c42'], // Amarillo/naranja claro
            2: ['#ff8c42', '#ff6b6b'], // Naranja/rojo claro
            3: ['#ff6b6b', '#e74c3c'], // Rojo
            5: ['#c0392b', '#8e44ad']  // Rojo oscuro/púrpura
        };
        const colorSet = colors[this.maxHp] || colors[1];
        return colorSet[Math.floor(Math.random() * colorSet.length)];
    }

    takeDamage() {
        this.hp--;
        if (this.hp > 0) {
            // Reducir tamaño proporcionalmente
            const sizeRatio = this.hp / this.maxHp;
            this.width = this.baseWidth * sizeRatio;
            this.height = this.baseHeight * sizeRatio;
            // Actualizar color
            this.color = this.getColorByHp();
            // Efecto visual de daño
            return false; // No destruido
        }
        return true; // Destruido
    }

    draw(ctx) {
        ctx.save();
        ctx.translate(this.x, this.y);

        ctx.fillStyle = this.color;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;

        switch (this.shape) {
            case 0: // Círculo
                ctx.beginPath();
                ctx.arc(0, 0, this.width / 2, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
                break;

            case 1: // Cuadrado rotado
                ctx.rotate(Math.PI / 4);
                ctx.fillRect(-this.width / 2, -this.height / 2, this.width, this.height);
                ctx.strokeRect(-this.width / 2, -this.height / 2, this.width, this.height);
                break;

            case 2: // Triángulo
                ctx.beginPath();
                ctx.moveTo(0, -this.height / 2);
                ctx.lineTo(this.width / 2, this.height / 2);
                ctx.lineTo(-this.width / 2, this.height / 2);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
                break;

            case 3: // Rombo
                ctx.beginPath();
                ctx.moveTo(0, -this.height / 2);
                ctx.lineTo(this.width / 2, 0);
                ctx.lineTo(0, this.height / 2);
                ctx.lineTo(-this.width / 2, 0);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
                break;
        }

        ctx.restore();
    }

    update() {
        this.x -= this.speed;

        // Movimiento ondulatorio (sinusoidal)
        const distanceTraveled = config.width - this.x;
        this.y = this.baseY + Math.sin(distanceTraveled * this.frequency + this.phase) * this.amplitude;

        // Mantener dentro de los límites
        this.y = Math.max(this.height / 2 + 20, Math.min(config.height - this.height / 2 - 20, this.y));
    }

    isOffScreen() {
        return this.x < -this.width;
    }

    collidesWith(bullet) {
        const dx = this.x - bullet.x;
        const dy = this.y - bullet.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        return distance < (this.width / 2 + bullet.width / 2);
    }

    collidesWithPlayer(player) {
        const dx = this.x - player.x;
        const dy = this.y - player.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        return distance < (this.width / 2 + player.width / 2);
    }
}

// Explosion Class
class Explosion {
    constructor(x, y, color) {
        this.x = x;
        this.y = y;
        this.color = color;
        this.radius = 5;
        this.maxRadius = 40;
        this.speed = 3;
        this.opacity = 1;
    }

    draw(ctx) {
        ctx.save();
        ctx.globalAlpha = this.opacity;

        const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius);
        gradient.addColorStop(0, '#ffffff');
        gradient.addColorStop(0.3, this.color);
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }

    update() {
        this.radius += this.speed;
        this.opacity -= 0.02;
    }

    isDone() {
        return this.opacity <= 0 || this.radius >= this.maxRadius;
    }
}

// Bomb Shockwave Class
class BombShockwave {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.radius = 0;
        this.maxRadius = Math.max(config.width, config.height) * 1.5;
        this.speed = 25;
        this.opacity = 1;
        this.waves = [
            { offset: 0, color: '#ff6b6b' },
            { offset: 20, color: '#ff8c42' },
            { offset: 40, color: '#ffd93d' }
        ];
    }

    draw(ctx) {
        ctx.save();

        this.waves.forEach(wave => {
            const waveRadius = this.radius + wave.offset;
            if (waveRadius > 0 && waveRadius < this.maxRadius) {
                ctx.globalAlpha = this.opacity * (1 - waveRadius / this.maxRadius);
                ctx.strokeStyle = wave.color;
                ctx.lineWidth = 8;
                ctx.shadowBlur = 20;
                ctx.shadowColor = wave.color;

                ctx.beginPath();
                ctx.arc(this.x, this.y, waveRadius, 0, Math.PI * 2);
                ctx.stroke();
            }
        });

        // Flash central
        if (this.radius < 100) {
            ctx.globalAlpha = this.opacity;
            const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius + 50);
            gradient.addColorStop(0, 'rgba(255, 255, 255, 0.8)');
            gradient.addColorStop(0.3, 'rgba(255, 107, 107, 0.5)');
            gradient.addColorStop(1, 'rgba(255, 107, 107, 0)');

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius + 50, 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.restore();
    }

    update() {
        this.radius += this.speed;
        this.opacity -= 0.01;
    }

    isDone() {
        return this.radius >= this.maxRadius || this.opacity <= 0;
    }
}

// PowerUp Class (Vidas y Bombas)
class PowerUp {
    constructor(x, y, type) {
        this.x = x;
        this.y = y;
        this.type = type; // 'life', 'bomb' o 'speed'
        this.width = 30;
        this.height = 30;
        this.speed = 2;
        this.collected = false;
        this.pulse = 0; // Para efecto de pulso
        this.rotation = 0; // Rotación del icono
    }

    draw(ctx) {
        ctx.save();

        this.pulse += 0.1;
        const pulseScale = 1 + Math.sin(this.pulse) * 0.15;
        const size = this.width * pulseScale;

        ctx.translate(this.x, this.y);
        this.rotation += 0.05;
        ctx.rotate(this.rotation);

        // Fondo brillante
        const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, size);
        if (this.type === 'life') {
            gradient.addColorStop(0, 'rgba(255, 100, 100, 0.8)');
            gradient.addColorStop(0.5, 'rgba(255, 100, 100, 0.4)');
            gradient.addColorStop(1, 'rgba(255, 100, 100, 0)');
        } else if (this.type === 'bomb') {
            gradient.addColorStop(0, 'rgba(255, 200, 0, 0.8)');
            gradient.addColorStop(0.5, 'rgba(255, 200, 0, 0.4)');
            gradient.addColorStop(1, 'rgba(255, 200, 0, 0)');
        } else if (this.type === 'speed') {
            gradient.addColorStop(0, 'rgba(255, 255, 0, 0.9)');
            gradient.addColorStop(0.5, 'rgba(255, 220, 0, 0.5)');
            gradient.addColorStop(1, 'rgba(255, 255, 0, 0)');
        }

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(0, 0, size * 1.5, 0, Math.PI * 2);
        ctx.fill();

        // Icono
        ctx.font = `${size * 1.2}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (this.type === 'life') {
            ctx.fillText('❤️', 0, 0);
        } else if (this.type === 'bomb') {
            ctx.fillText('💣', 0, 0);
        } else if (this.type === 'speed') {
            ctx.fillText('⚡', 0, 0);
        }

        ctx.restore();
    }

    update() {
        this.x -= this.speed;
    }

    isOffScreen() {
        return this.x < -this.width * 2;
    }

    collidesWithPlayer(player) {
        const dx = this.x - player.x;
        const dy = this.y - player.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        return distance < (this.width + player.width) / 2;
    }
}

/**
 * Fetch the currently active classifier from the backend and set the select value
 */
async function getActiveClassifier() {
    try {
        const resp = await fetch(`${window.location.protocol}//${window.location.host}/api/training/classifier`);
        if (!resp.ok) return;
        const data = await resp.json();
        const select = document.getElementById('classifier-select');
        if (data && data.classifier && select) {
            select.value = data.classifier;
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
    const select = document.getElementById('classifier-select');
    if (select) select.disabled = true;
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

        // After changing classifier, we need to refresh keywords
        // If websocket is open, we can send a message or just wait for it to reconnect/refresh
        // For simplicity, let's just trigger a reload of keywords if possible
        if (audioState.websocket && audioState.websocket.readyState === WebSocket.OPEN) {
            // The backend reload_model() is called on /api/training/classifier POST
            // We might need to re-fetch keywords
            const kwResp = await fetch(`${window.location.protocol}//${window.location.host}/api/training/keywords`);
            if (kwResp.ok) {
                const kwData = await kwResp.json();
                loadKeywordsToSelects(kwData.keywords.map(k => k.name));
            }
        }

        showNotification(`Clasificador ${classifier.toUpperCase()} activado`, 'success');
    } catch (err) {
        console.error('Failed to set active classifier:', err);
        showNotification('Error al cambiar clasificador', 'error');
        await getActiveClassifier();
    } finally {
        if (select) select.disabled = false;
    }
}

// =====================================
// WEBSOCKET FUNCTIONS
// =====================================

async function connectWebSocket() {
    if (audioState.websocket && audioState.websocket.readyState === WebSocket.OPEN) {
        return;
    }

    try {
        audioState.websocket = await initWebSocket('/ws/audio', handleWebSocketMessage, {
            autoReconnect: true,
            onOpen: () => {
                console.log('✓ WebSocket conectado');
                updateConnectionStatus('Conectado', true);
                showNotification('Conectado al servidor de voz', 'success');
            },
            onError: () => {
                updateConnectionStatus('Error', false);
            },
            onClose: () => {
                updateConnectionStatus('Desconectado', false);
            }
        });
    } catch (error) {
        console.error('Error al conectar WebSocket:', error);
        updateConnectionStatus('Error', false);
    }
}

function updateConnectionStatus(status, isActive) {
    // Actualizar estado de audio
    const statusElement = document.getElementById('audio-status');
    const badge = document.getElementById('mic-status-badge');

    if (statusElement) {
        statusElement.textContent = `🎤 ${status}`;
        statusElement.style.color = isActive ? 'var(--success-color)' : 'var(--text-secondary)';
    }

    if (badge) {
        badge.style.background = isActive ? 'var(--success-color)' : 'var(--error-color)';
    }

    // Actualizar indicador WebSocket
    const indicator = document.getElementById('ws-indicator');
    const statusText = document.getElementById('ws-status-text');

    if (indicator && statusText) {
        if (isActive) {
            indicator.classList.add('connected');
            statusText.textContent = 'Conectado';
        } else {
            indicator.classList.remove('connected');
            statusText.textContent = 'Desconectado';
        }
    }
}

function handleWebSocketMessage(data, ws) {
    console.log('Mensaje recibido:', data);

    switch (data.type) {
        case 'connected':
            if (!data.keywords || data.keywords.length === 0) {
                showNotification('Debe configurar keywords para poder jugar', 'error');
                return;
            }
            loadKeywordsToSelects(data.keywords);
            break;

        case 'detection':
            if (state.isRunning) {
                handleVoiceCommand(data.keyword);
            }
            break;

        default:
            console.log('Tipo de mensaje desconocido:', data.type);
    }
}

function loadKeywordsToSelects(keywords) {
    const upSelect = document.getElementById('up-keyword');
    const downSelect = document.getElementById('down-keyword');
    const bombSelect = document.getElementById('bomb-keyword');

    [upSelect, downSelect, bombSelect].forEach(select => {
        // Guardar selección actual
        const currentValue = select.value;

        // Limpiar y repoblar
        select.innerHTML = '<option value="">Selecciona keyword...</option>';
        keywords.forEach(kw => {
            const option = document.createElement('option');
            option.value = kw;
            option.textContent = kw;
            select.appendChild(option);
        });

        // Restaurar selección si existe
        if (currentValue && keywords.includes(currentValue)) {
            select.value = currentValue;
        }
    });

    // Autoseleccionar keywords predefinidos si existen
    const keywordMap = {
        'up-keyword': ['sube', 'arriba', 'subir', 'up'],
        'down-keyword': ['baja', 'abajo', 'bajar', 'down'],
        'bomb-keyword': ['fuego', 'bomba', 'disparo', 'go', 'wow']
    };

    Object.entries(keywordMap).forEach(([selectId, possibleKeywords]) => {
        const select = document.getElementById(selectId);
        if (select && !select.value) {
            // Buscar si alguna palabra clave está disponible
            const found = possibleKeywords.find(kw =>
                keywords.some(k => k.toLowerCase() === kw.toLowerCase())
            );
            if (found) {
                const actualKeyword = keywords.find(k => k.toLowerCase() === found.toLowerCase());
                select.value = actualKeyword;
            }
        }
    });
}

function handleVoiceCommand(keyword) {
    if (!state.isRunning || !state.player) return;

    const keywordLower = keyword.toLowerCase();

    // Mostrar keyword en overlay de audio
    showKeywordInAudioOverlay(keyword);

    // Resaltar control activo
    if (state.keywords.up && keywordLower === state.keywords.up.toLowerCase()) {
        highlightControl('up');
        state.player.moveUp();
    } else if (state.keywords.down && keywordLower === state.keywords.down.toLowerCase()) {
        highlightControl('down');
        state.player.moveDown();
    } else if (state.keywords.bomb && keywordLower === state.keywords.bomb.toLowerCase()) {
        highlightControl('bomb');
        useBomb();
    }
}

function showKeywordInAudioOverlay(keyword) {
    const overlay = document.getElementById('audio-keyword-overlay');
    const text = document.getElementById('audio-keyword-text');

    if (!overlay || !text) return;

    // Limpiar timeout anterior
    if (audioState.keywordTimeout) {
        clearTimeout(audioState.keywordTimeout);
    }

    // Mostrar keyword
    text.textContent = keyword.toUpperCase();
    overlay.style.opacity = '1';
    overlay.classList.remove('fade-out');
    overlay.classList.add('fade-in');

    // Fade out después de 5 segundos
    audioState.keywordTimeout = setTimeout(() => {
        overlay.classList.remove('fade-in');
        overlay.classList.add('fade-out');
        overlay.style.opacity = '0';
    }, 5000);
}

function highlightControl(controlType) {
    const element = document.getElementById(`control-${controlType}`);
    element.classList.add('active');
    setTimeout(() => element.classList.remove('active'), 300);
}

function sendAudioData(audioData) {
    if (audioState.websocket && audioState.websocket.readyState === WebSocket.OPEN) {
        // Convertir Float32Array a buffer
        const buffer = audioData.buffer.slice(
            audioData.byteOffset,
            audioData.byteOffset + audioData.byteLength
        );

        audioState.websocket.send(buffer);
    }
}

async function initAudioStream() {
    try {
        // Solicitar acceso al micrófono
        audioState.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000
            }
        });

        // Crear AudioContext
        audioState.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });

        const source = audioState.audioContext.createMediaStreamSource(audioState.mediaStream);

        // Crear visualizador circular
        const canvas = document.getElementById('audio-visualizer');
        const vizResult = createAudioVisualizer(
            canvas,
            audioState.audioContext,
            audioState.mediaStream,
            {
                radius: 35,
                barCount: 32,
                maxBarHeight: 25,
                lineWidth: 2,
                colorful: true,
                smoothing: 0.7
            }
        );

        if (vizResult) {
            audioState.circularVisualizer = vizResult.visualizer;
            audioState.circularVisualizer.start();
        }

        // Crear VAD (Voice Activity Detector)
        audioState.vad = new VoiceActivityDetector({
            energyThreshold: 0.003,
            highEnergyThreshold: 0.02,
            minSpeechDuration: 150,
            minHighEnergySpeech: 50,
            maxSpeechDuration: 3000,
            silenceChunks: 3,
            sampleRate: 16000,
            onSpeechStart: (energy) => {
                console.log(`🎤 Inicio de speech - energy: ${(energy * 1000).toFixed(1)}`);
            },
            onSpeechEnd: (audioData, duration, maxEnergy) => {
                console.log(`📤 Enviando segmento: ${audioData.length} samples (${duration}s)`);
                sendAudioData(audioData);
            },
            onEnergyUpdate: (energy, hasVoice, isHighEnergy) => {
                // Opcional: actualizar UI con nivel de energía
            }
        });

        // Crear ScriptProcessorNode para enviar audio
        audioState.scriptProcessorNode = audioState.audioContext.createScriptProcessor(4096, 1, 1);

        audioState.scriptProcessorNode.onaudioprocess = (event) => {
            if (!audioState.isStreaming || !audioState.websocket || audioState.websocket.readyState !== WebSocket.OPEN) return;

            const inputData = event.inputBuffer.getChannelData(0);

            // Resample si el context no está a 16 kHz
            const ctxSampleRate = audioState.audioContext ? audioState.audioContext.sampleRate : 16000;
            const dataToUse = (ctxSampleRate !== 16000 && typeof resampleFloat32 === 'function') ? resampleFloat32(inputData, ctxSampleRate, 16000) : inputData;

            // Procesar con VAD
            if (audioState.vad) {
                audioState.vad.process(dataToUse);
            }
        };

        // Conectar nodos
        source.connect(audioState.scriptProcessorNode);
        audioState.scriptProcessorNode.connect(audioState.audioContext.destination);

        audioState.isStreaming = true;

        console.log('✓ Audio stream inicializado');
        updateConnectionStatus('Grabando', true);

        return true;
    } catch (error) {
        console.error('✗ Error al inicializar audio:', error);
        showToast('Error al acceder al micrófono', 'error');
        updateConnectionStatus('Error', false);
        return false;
    }
}

function stopAudioStream() {
    audioState.isStreaming = false;

    // Resetear VAD
    if (audioState.vad) {
        audioState.vad.reset();
        audioState.vad = null;
    }

    // Limpiar timeout del keyword overlay
    if (audioState.keywordTimeout) {
        clearTimeout(audioState.keywordTimeout);
        audioState.keywordTimeout = null;
    }

    // Limpiar overlay
    const overlay = document.getElementById('audio-keyword-overlay');
    const text = document.getElementById('audio-keyword-text');
    if (overlay && text) {
        text.textContent = '';
        overlay.style.opacity = '0';
        overlay.classList.remove('fade-in', 'fade-out');
    }

    // Detener visualizador
    if (audioState.circularVisualizer) {
        audioState.circularVisualizer.stop();
        audioState.circularVisualizer = null;
    }

    // Detener audio
    if (audioState.scriptProcessorNode) {
        audioState.scriptProcessorNode.disconnect();
        audioState.scriptProcessorNode = null;
    }

    if (audioState.mediaStream) {
        audioState.mediaStream.getTracks().forEach(track => track.stop());
        audioState.mediaStream = null;
    }

    if (audioState.audioContext) {
        try {
            if (!window._sharedAudioContext || audioState.audioContext !== window._sharedAudioContext) {
                audioState.audioContext.close();
            }
        } catch (e) {
            console.warn('Error cerrando AudioContext:', e);
        }
        audioState.audioContext = null;
    }

    updateConnectionStatus('Detenido', false);
}

// =====================================
// GAME LOGIC
// =====================================

function initGame() {
    config.canvas = document.getElementById('game-canvas');
    config.ctx = config.canvas.getContext('2d');

    // Cargar keywords del localStorage si existen
    const savedKeywords = localStorage.getItem('galaxy_keywords');
    if (savedKeywords) {
        state.keywords = JSON.parse(savedKeywords);
        updateControlsDisplay();
    }

    connectWebSocket();

    // Event Listeners
    document.getElementById('test-btn').addEventListener('click', startTestMode);
    document.getElementById('start-btn').addEventListener('click', startGame);
    document.getElementById('restart-btn').addEventListener('click', restartGame);
    document.getElementById('pause-btn').addEventListener('click', togglePause);

    // Keyboard controls para pruebas
    document.addEventListener('keydown', (e) => {
        // Pausa con P o Escape
        if ((e.key === 'p' || e.key === 'P' || e.key === 'Escape') && state.isRunning) {
            togglePause();
            return;
        }

        if (!state.isRunning || !state.player || state.isPaused) return;

        if (!keyboardModeEnabled) return;

        if (e.key === 'ArrowUp' || e.key === 'w') {
            state.player.moveUp();
            // Agrega el texto del keyword al overlay
            showKeywordInAudioOverlay(state.keywords.up);
        } else if (e.key === 'ArrowDown' || e.key === 's') {
            state.player.moveDown();
            // Agrga el texto del keyword al overlay
            showKeywordInAudioOverlay(state.keywords.down);
        } else if (e.key === ' ') {
            useBomb();
            // Agrega el texto del keyword al overlay
            showKeywordInAudioOverlay(state.keywords.bomb);
        }
    });
}

async function startTestMode() {
    if (!validateKeywords()) return;

    saveKeywords();
    updateControlsDisplay();

    // Iniciar audio streaming
    await connectWebSocket();
    const audioInitialized = await initAudioStream();
    if (!audioInitialized) {
        showToast('Error al inicializar audio. Verifica los permisos del micrófono.', 'error');
        return;
    }

    state.isTestMode = true;

    resetGameState();
    startGameLoop();

    // Mostrar botón de pausa
    document.getElementById('pause-btn').style.display = 'block';
    document.getElementById('test-btn').style.display = 'none';
    document.getElementById('start-btn').style.display = 'none';

    showNotification('Modo prueba iniciado - Usa tu voz o teclado', 'info');
}

async function startGame() {
    if (!validateKeywords()) return;

    saveKeywords();
    updateControlsDisplay();

    // Iniciar audio streaming
    await connectWebSocket();
    const audioInitialized = await initAudioStream();
    if (!audioInitialized) {
        showToast('Error al inicializar audio. Verifica los permisos del micrófono.', 'error');
        return;
    }

    state.isTestMode = false;

    resetGameState();
    startGameLoop();
    startEnemySpawner();

    // Mostrar botón de pausa
    document.getElementById('pause-btn').style.display = 'block';
    document.getElementById('test-btn').style.display = 'none';
    document.getElementById('start-btn').style.display = 'none';

    showNotification('¡Partida iniciada! ¡Buena suerte!', 'success');
}

function validateKeywords() {
    const up = document.getElementById('up-keyword').value;
    const down = document.getElementById('down-keyword').value;
    const bomb = document.getElementById('bomb-keyword').value;

    if (!up || !down || !bomb) {
        showNotification('Por favor selecciona todos los keywords', 'error');
        return false;
    }

    if (up === down || up === bomb || down === bomb) {
        showNotification('Los keywords deben ser diferentes', 'error');
        return false;
    }

    state.keywords = { up, down, bomb };
    return true;
}

function saveKeywords() {
    localStorage.setItem('galaxy_keywords', JSON.stringify(state.keywords));
}

function updateControlsDisplay() {
    // Actualizar los selectores con los valores guardados
    const upSelect = document.getElementById('up-keyword');
    const downSelect = document.getElementById('down-keyword');
    const bombSelect = document.getElementById('bomb-keyword');

    if (upSelect && state.keywords.up) upSelect.value = state.keywords.up;
    if (downSelect && state.keywords.down) downSelect.value = state.keywords.down;
    if (bombSelect && state.keywords.bomb) bombSelect.value = state.keywords.bomb;
}

function resetGameState() {
    state.isRunning = true;
    state.score = 0;
    state.lives = 5;
    state.bombs = 5;
    state.level = 1;
    state.enemiesKilled = 0;
    state.lastBombTime = 0;

    state.player = new Player(80, config.height / 2);
    state.bullets = [];
    state.enemies = [];
    state.explosions = [];
    state.shockwaves = [];
    state.powerups = [];

    updateUI();
}

function startGameLoop() {
    if (state.gameLoopId) {
        cancelAnimationFrame(state.gameLoopId);
    }

    gameLoop();
}

function startEnemySpawner() {
    if (state.enemySpawnTimer) {
        clearInterval(state.enemySpawnTimer);
    }

    state.enemySpawnTimer = setInterval(() => {
        if (state.isRunning && !state.isTestMode) {
            spawnEnemy();
        }
    }, config.enemySpawnRate);
}

function spawnEnemy() {
    const y = 50 + Math.random() * (config.height - 100);
    state.enemies.push(new Enemy(config.width + 50, y));
}

function gameLoop() {
    if (!state.isRunning) return;

    // Clear canvas
    config.ctx.fillStyle = '#0a0a1e';
    config.ctx.fillRect(0, 0, config.width, config.height);

    // Dibujar estrellas de fondo
    drawStars();

    // Si está en pausa, mostrar overlay y no actualizar
    if (state.isPaused) {
        // Dibujar todo congelado
        if (state.player) {
            state.player.draw(config.ctx);
        }
        state.bullets.forEach(bullet => bullet.draw(config.ctx));
        state.enemies.forEach(enemy => enemy.draw(config.ctx));
        state.explosions.forEach(explosion => explosion.draw(config.ctx));

        drawPauseOverlay();
        state.gameLoopId = requestAnimationFrame(gameLoop);
        return;
    }

    // Update & Draw Player
    if (state.player) {
        state.player.update();
        state.player.draw(config.ctx);

        // Auto-shoot
        if (!state.isTestMode) {
            state.player.shoot();
        }
    }

    // Update & Draw Bullets
    state.bullets = state.bullets.filter(bullet => {
        bullet.update();
        bullet.draw(config.ctx);
        return !bullet.isOffScreen();
    });

    // Update & Draw Enemies
    state.enemies = state.enemies.filter(enemy => {
        enemy.update();
        enemy.draw(config.ctx);

        // Colisión con jugador
        if (!state.isTestMode && enemy.collidesWithPlayer(state.player)) {
            createExplosion(enemy.x, enemy.y, enemy.color);
            loseLife();
            return false;
        }

        return !enemy.isOffScreen();
    });

    // Colisiones bala-enemigo
    if (!state.isTestMode) {
        checkBulletCollisions();
    }

    // Update & Draw Explosions
    state.explosions = state.explosions.filter(explosion => {
        explosion.update();
        explosion.draw(config.ctx);
        return !explosion.isDone();
    });

    // Update & Draw Shockwaves (ondas expansivas de bombas)
    state.shockwaves = state.shockwaves.filter(shockwave => {
        shockwave.update();
        shockwave.draw(config.ctx);
        return !shockwave.isDone();
    });

    // Update & Draw PowerUps
    state.powerups = state.powerups.filter(powerup => {
        powerup.update();
        powerup.draw(config.ctx);

        // Colisión con jugador
        if (state.player && powerup.collidesWithPlayer(state.player)) {
            collectPowerup(powerup);
            return false;
        }

        return !powerup.isOffScreen();
    });

    state.gameLoopId = requestAnimationFrame(gameLoop);
}

function drawStars() {
    config.ctx.fillStyle = '#ffffff';
    for (let i = 0; i < 50; i++) {
        const x = (i * 17 + Date.now() * 0.05) % config.width;
        const y = (i * 23) % config.height;
        const size = (i % 3) + 1;
        config.ctx.fillRect(x, y, size, size);
    }
}

function checkBulletCollisions() {
    for (let i = state.bullets.length - 1; i >= 0; i--) {
        const bullet = state.bullets[i];

        for (let j = state.enemies.length - 1; j >= 0; j--) {
            const enemy = state.enemies[j];

            if (enemy.collidesWith(bullet)) {
                // Eliminar bala
                state.bullets.splice(i, 1);

                // Aplicar daño al enemigo
                const destroyed = enemy.takeDamage();
                if (destroyed) {
                    // Eliminar enemigo si fue destruido
                    state.enemies.splice(j, 1);

                    // Explosión y puntos (más puntos por enemigos más resistentes)
                    createExplosion(enemy.x, enemy.y, enemy.color);
                    state.enemiesKilled++;
                    const scoreBonus = enemy.maxHp * 50; // 50, 100, 150, 250 puntos
                    addScore((100 + scoreBonus) * state.level);

                    // Subir nivel cada 10 enemigos
                    if (state.enemiesKilled % 10 === 0) {
                        levelUp();
                    }
                } else {
                    // Efecto de impacto pero no destruido
                    createSmallExplosion(enemy.x, enemy.y, enemy.color);
                    addScore(25 * state.level); // Puntos por impacto
                }

                break;
            }
        }
    }
}

function createExplosion(x, y, color) {
    state.explosions.push(new Explosion(x, y, color));
}

function createSmallExplosion(x, y, color) {
    // Explosión más pequeña para impactos sin destrucción
    const explosion = new Explosion(x, y, color);
    explosion.maxRadius = 20; // Radio más pequeño
    explosion.speed = 2; // Más lenta
    state.explosions.push(explosion);
}

function togglePause() {
    state.isPaused = !state.isPaused;
    const pauseBtn = document.getElementById('pause-btn');

    if (state.isPaused) {
        pauseBtn.textContent = '▶️ Reanudar';
        showNotification('⏸️ Juego pausado (P o ESC para continuar)', 'info');
    } else {
        pauseBtn.textContent = '⏸️ Pausa';
        showNotification('▶️ Juego reanudado', 'success');
    }
}

function drawPauseOverlay() {
    config.ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    config.ctx.fillRect(0, 0, config.width, config.height);

    config.ctx.fillStyle = '#00d9ff';
    config.ctx.font = 'bold 48px Arial';
    config.ctx.textAlign = 'center';
    config.ctx.textBaseline = 'middle';
    config.ctx.fillText('⏸️ PAUSA', config.width / 2, config.height / 2 - 30);

    config.ctx.fillStyle = '#ffffff';
    config.ctx.font = '20px Arial';
    config.ctx.fillText('Presiona P o ESC para continuar', config.width / 2, config.height / 2 + 30);
}

function addScore(points) {
    state.score += points;
    updateUI();
}

function loseLife() {
    state.lives--;

    // Desactivar powerup de velocidad al perder vida
    if (state.hasSpeedPowerup) {
        state.hasSpeedPowerup = false;
        showNotification('⚡ Disparo rápido perdido', 'warning');
    }

    updateUI();

    if (state.lives <= 0) {
        gameOver();
    } else {
        showNotification(`¡Impacto! Vidas restantes: ${state.lives}`, 'error');
    }
}

function levelUp() {
    state.level++;
    updateUI();

    // Spawear horda de enemigos según el nivel
    const hordeSize = state.level * 3; // 3, 6, 9, 12... enemigos
    spawnHorde(hordeSize);

    // Cada 2 niveles, spawear powerup detrás de la horda (rotación: vida, bomba, speed)
    if (state.level % 2 === 0) {
        const powerupCycle = Math.floor(state.level / 2) % 3;
        let powerupType;
        if (powerupCycle === 0) {
            powerupType = 'life';
        } else if (powerupCycle === 1) {
            powerupType = 'bomb';
        } else {
            powerupType = 'speed';
        }
        spawnPowerupBehindHorde(powerupType, hordeSize);
    }

    showNotification(`¡Nivel ${state.level}! Horda de ${hordeSize} enemigos entrantes`, 'warning');
}

function spawnHorde(count) {
    // Spawear enemigos en formación
    const formations = [
        'wave', // Onda
        'line', // Línea
        'v-shape', // Forma de V
        'scattered' // Dispersos
    ];

    const formation = formations[Math.floor(Math.random() * formations.length)];

    for (let i = 0; i < count; i++) {
        let x, y;

        switch (formation) {
            case 'wave':
                // Formación en onda sinusoidal
                x = config.width + 100 + (i * 60);
                y = config.height / 2 + Math.sin(i * 0.5) * 150;
                break;

            case 'line':
                // Línea horizontal
                x = config.width + 100 + (i * 50);
                y = 100 + (config.height - 200) * (i / count);
                break;

            case 'v-shape':
                // Forma de V
                x = config.width + 100 + (i * 50);
                const mid = count / 2;
                y = config.height / 2 + Math.abs(i - mid) * 30;
                break;

            case 'scattered':
            default:
                // Dispersos
                x = config.width + 100 + Math.random() * 200;
                y = 50 + Math.random() * (config.height - 100);
                break;
        }

        // Delay escalonado para que no aparezcan todos a la vez
        setTimeout(() => {
            if (state.isRunning) {
                state.enemies.push(new Enemy(x, y));
            }
        }, i * 150);
    }
}

function spawnPowerupBehindHorde(type, hordeSize) {
    // Calcular posición detrás de la horda (más a la derecha)
    const delay = hordeSize * 150 + 1000; // Aparece 1 segundo después del último enemigo
    const x = config.width + 300; // Más lejos a la derecha
    const y = config.height / 2; // Centro vertical

    setTimeout(() => {
        if (state.isRunning) {
            state.powerups.push(new PowerUp(x, y, type));
            let icon, name;
            if (type === 'life') {
                icon = '❤️';
                name = 'Vida extra';
            } else if (type === 'bomb') {
                icon = '💣';
                name = 'Bomba extra';
            } else if (type === 'speed') {
                icon = '⚡';
                name = 'Disparo rápido';
            }
            showNotification(`${icon} ¡${name} disponible!`, 'success');
        }
    }, delay);
}

function collectPowerup(powerup) {
    if (powerup.type === 'life') {
        state.lives++;
        showNotification('❤️ ¡Vida extra obtenida!', 'success');
        createExplosion(powerup.x, powerup.y, '#ff6464');
    } else if (powerup.type === 'bomb') {
        state.bombs++;
        showNotification('💣 ¡Bomba extra obtenida!', 'success');
        createExplosion(powerup.x, powerup.y, '#ffc800');
    } else if (powerup.type === 'speed') {
        state.hasSpeedPowerup = true;
        showNotification('⚡ ¡Disparo rápido activado! (Se pierde al recibir daño)', 'success');
        createExplosion(powerup.x, powerup.y, '#ffff00');
    }
    updateUI();
}

function useBomb() {
    if (state.bombs <= 0) {
        showNotification('🚨 No quedan bombas disponibles', 'warning');
        return;
    }

    const now = Date.now();
    if (now - state.lastBombTime < config.bombCooldown) {
        const remaining = Math.ceil((config.bombCooldown - (now - state.lastBombTime)) / 1000);
        showNotification(`Bomba en cooldown (${remaining}s)`, 'warning');
        return;
    }

    state.bombs--;
    state.lastBombTime = now;

    // Crear onda expansiva desde el centro de la pantalla
    const centerX = config.width / 2;
    const centerY = config.height / 2;
    state.shockwaves.push(new BombShockwave(centerX, centerY));

    // Eliminar todos los enemigos con pequeño delay para efecto
    const enemiesDestroyed = state.enemies.length;
    let totalScore = 0;

    // Eliminar enemigos de forma progresiva según distancia
    state.enemies.forEach((enemy, index) => {
        setTimeout(() => {
            if (state.enemies.includes(enemy)) {
                createExplosion(enemy.x, enemy.y, enemy.color);
            }
        }, index * 20); // 20ms entre cada explosión

        state.enemiesKilled++;
        totalScore += (100 + enemy.maxHp * 50) * state.level;
    });

    state.enemies = [];

    addScore(totalScore);
    updateUI(); // Actualizar bombas en UI
    showNotification(`💣 ¡BOMBA! ${enemiesDestroyed} enemigos eliminados`, 'success');
}

function gameOver() {
    state.isRunning = false;

    // Detener audio streaming
    stopAudioStream();

    // Cerrar WebSocket
    if (audioState.websocket) {
        closeWebSocket(audioState.websocket);
        audioState.websocket = null;
    }

    if (state.gameLoopId) {
        cancelAnimationFrame(state.gameLoopId);
        state.gameLoopId = null;
    }

    if (state.enemySpawnTimer) {
        clearInterval(state.enemySpawnTimer);
        state.enemySpawnTimer = null;
    }

    document.getElementById('final-score').textContent = state.score;
    document.getElementById('gameover-overlay').classList.remove('hidden');
}

function restartGame() {
    document.getElementById('gameover-overlay').classList.add('hidden');
    // Los controles ya están visibles en el panel izquierdo

    // Mostrar botones de inicio y ocultar pausa
    document.getElementById('pause-btn').style.display = 'none';
    document.getElementById('test-btn').style.display = 'block';
    document.getElementById('start-btn').style.display = 'block';
}

function updateUI() {
    document.getElementById('score-display').textContent = state.score;
    document.getElementById('lives-display').textContent = state.lives;
    document.getElementById('level-display').textContent = state.level;
    document.getElementById('enemies-display').textContent = state.enemiesKilled;

    // Actualizar corazones
    const heartsContainer = document.getElementById('lives-hearts');
    heartsContainer.innerHTML = '';
    for (let i = 0; i < state.lives; i++) {
        const heart = document.createElement('span');
        heart.className = 'life-heart';
        heart.textContent = '❤️';
        heartsContainer.appendChild(heart);
    }

    // Actualizar bombas
    const bombsContainer = document.getElementById('bombs-display');
    if (bombsContainer) {
        const bombIcons = bombsContainer.querySelectorAll('.bomb-icon');
        bombIcons.forEach((icon, index) => {
            if (index < state.bombs) {
                icon.classList.remove('used');
            } else {
                icon.classList.add('used');
            }
        });
    }
}

let spaceKeyPressed = false;
let keyboardModeEnabled = false;
let lastClickTime = 0;
const DOUBLE_CLICK_DELAY = 300; // ms

// Doble click para activar modo teclado
document.addEventListener('click', (e) => {
    const currentTime = Date.now();
    const timeSinceLastClick = currentTime - lastClickTime;

    if (timeSinceLastClick < DOUBLE_CLICK_DELAY && timeSinceLastClick > 0) {
        // Doble click detectado
        keyboardModeEnabled = !keyboardModeEnabled;

        if (keyboardModeEnabled) {
            showNotification('⌨️ Modo teclado activado, usa espacio para saltar', 'info');
        } else {
            showNotification('🎤 Modo teclado desactivado', 'info');
        }
    }

    lastClickTime = currentTime;
});

// Inicialización al cargar la página
window.addEventListener('load', () => {
    initGame();
    getActiveClassifier();

    const classifierSelect = document.getElementById('classifier-select');
    if (classifierSelect) {
        classifierSelect.addEventListener('change', () => {
            setActiveClassifier(classifierSelect.value);
        });
    }
});

// Limpiar al cerrar página
window.addEventListener('beforeunload', () => {
    if (audioState.websocket) {
        closeWebSocket(audioState.websocket);
    }
});

// =====================================
// INITIALIZATION
// =====================================

document.addEventListener('DOMContentLoaded', initGame);
