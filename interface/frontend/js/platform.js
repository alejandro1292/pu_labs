// =====================================
// VOICE JUMP PLATFORM
// =====================================
// Autor: Alejandro Ortiz
// Fecha: Diciembre 2025

// Configuración del juego
const config = {
    canvas: null,
    ctx: null,
    width: 800,
    height: 600,
    
    // Física
    gravity: 0.6,  // Gravedad más suave para saltos más floaty
    minJumpForce: -12,
    maxJumpForce: -18,
    maxJumpHeight: 250, // Mayor altura máxima
    jumpAcceleration: 0.4, // Aceleración al subir
    
    // Velocidad del mundo
    baseScrollSpeed: 5,
    scrollSpeed: 5,
    speedIncrement: 0.0005,
    maxSpeed: 12,
    
    // Obstáculos
    obstacleMinGap: 500,
    obstacleMaxGap: 800,
    obstacleMinWidth: 30,
    obstacleMaxWidth: 100,
    
    // Plataformas
    platformHeight: 100,
    groundY: 500,
    ceilingY: 120
};

// Estado del juego
const state = {
    isRunning: false,
    isPaused: false,
    distance: 0,
    bestDistance: 0,
    attempts: 0,
    
    // Configuración de palabras clave
    keywords: {
        jump: null
    },
    
    // Entidades del juego
    player: null,
    obstacles: [],
    platforms: [],
    particles: [],
    
    // Timers
    gameLoopId: null,
    
    // Caché de elementos DOM
    dom: {
        canvas: null,
        audioVisualizer: null,
        startBtn: null,
        pauseBtn: null,
        distanceValue: null,
        bestDistanceValue: null,
        speedValue: null,
        attemptsValue: null,
        currentKeyword: null,
        audioKeywordOverlay: null
    },

    // Relativos al Audio
    audio: {
        audioContext: null,
        mediaStream: null,
        scriptProcessorNode: null,
        circularVisualizer: null,
        isStreaming: false,
        displayTimeout: null,
        overlayTimeout: null,
        vad: null
    }
};

// Clase Jugador
class Player {
    constructor(x, y) {
        // Tamaño inicial
        this.width = 30;
        this.height = 30;
        this.color = '#ff6b6b';
        this.endColor = '#feca57';
        this.borderColor = '#fff';
        this.borderWidth = 2;

        // Posición inicial
        this.x = x;
        this.y = y;
        this.startY = y;

        // Estados
        this.isGrounded = false;
        this.isJumping = false;
        this.isShouting = false;
        this.jumpStartY = y;

        // Movimiento
        this.velocityY = 0;
        this.rotation = 0;
        this.rotationSpeed = 0;
        this.trailParticles = [];
        this.maxJumpHeight = 250;
        
        // Doble salto
        this.canDoubleJump = false;
        this.hasUsedDoubleJump = false;
        
        // Deformación (squash/stretch)
        this.scaleX = 1;
        this.scaleY = 1;
        this.targetScaleX = 1;
        this.targetScaleY = 1;
        
        // Intensidad del grito
        this.shoutIntensity = 0;
    }

    draw(ctx) {
        ctx.save();
        
        // Dibujar estela de partículas
        this.trailParticles.forEach((particle, index) => {
            ctx.globalAlpha = particle.life;
            ctx.fillStyle = particle.color;
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            ctx.fill();
        });
        
        ctx.globalAlpha = 1;
        
        // Dibujar jugador (cubo que rota)
        ctx.translate(this.x + this.width / 2, this.y + this.height / 2);
        ctx.rotate(this.rotation);
        ctx.scale(this.scaleX, this.scaleY);
        
        // Cubo principal
        const gradient = ctx.createLinearGradient(-this.width / 2, -this.height / 2, this.width / 2, this.height / 2);
        gradient.addColorStop(0, this.color);
        gradient.addColorStop(1, this.endColor);
        ctx.fillStyle = gradient;
        ctx.fillRect(-this.width / 2, -this.height / 2, this.width, this.height);
        
        // Borde brillante
        ctx.strokeStyle = this.borderColor;
        ctx.lineWidth = this.borderWidth;
        ctx.strokeRect(-this.width / 2, -this.height / 2, this.width, this.height);
        
        // Detalle interior
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.strokeRect(-this.width / 3, -this.height / 3, this.width * 2 / 3, this.height * 2 / 3);
        
        ctx.restore();
    }

    update() {
        // Si está gritando, ascender continuamente
        if (this.isShouting) {
            const heightGained = this.jumpStartY - this.y;
            
            if (heightGained < this.maxJumpHeight) {
                // Ascenso más suave con aceleración
                this.velocityY = -10 - (this.shoutIntensity * 0.03); // Velocidad base + intensidad
                
                // Squash al subir (comprimido verticalmente, estirado horizontalmente)
                this.targetScaleX = 0.8;
                this.targetScaleY = 1.2;
            } else {
                this.isShouting = false;
                this.velocityY = 0;
            }
        } else {
            // Aplicar gravedad cuando no está gritando
            this.velocityY += config.gravity;
            
            // Stretch al caer (estirado verticalmente)
            if (this.velocityY > 2) {
                this.targetScaleX = 0.85;
                this.targetScaleY = 1.15;
            }
        }
        
        this.y += this.velocityY;
        
        // Comprobar colisión con el techo
        if (this.y <= config.ceilingY) {
            this.y = config.ceilingY;
            this.velocityY = 0;
            this.isShouting = false;
            //this.canDoubleJump = true;  // Resetear doble salto al tocar el techo
            //this.hasUsedDoubleJump = false;
            
            // Crear chispas al tocar el techo
            createCeilingSparkles(this.x + this.width / 2, this.y);
            
            // Efecto "squash" al golpear el techo
            this.targetScaleX = 1.3;
            this.targetScaleY = 0.7;
        }
        
        // Comprobar colisión con el suelo
        if (this.y + this.height >= config.groundY) {
            this.y = config.groundY - this.height;
            this.velocityY = 0;
            this.isGrounded = true;
            this.isJumping = false;
            this.isShouting = false;
            this.canDoubleJump = false;
            this.hasUsedDoubleJump = false;
            this.rotationSpeed = config.scrollSpeed * 0.05;
            
            // Crear más partículas al aterrizar
            if (Math.abs(this.velocityY) > 5) {
                createJumpParticles(this.x + this.width / 2, this.y + this.height, 15);
            }
        } else {
            this.isGrounded = false;
        }
        
        // Comprobar colisión con plataformas
        state.platforms.forEach(platform => {
            if (this.velocityY >= 0 && // Cayendo
                this.x + this.width > platform.x &&
                this.x < platform.x + platform.width &&
                this.y + this.height >= platform.y &&
                this.y + this.height <= platform.y + 20) {
                
                this.y = platform.y - this.height;
                this.velocityY = 0;
                this.isGrounded = true;
                this.isJumping = false;
                this.canDoubleJump = false;
                this.hasUsedDoubleJump = false;
                this.rotationSpeed = config.scrollSpeed * 0.05;
            }
        });
        
        // Interpolación suave de squash/stretch
        this.scaleX += (this.targetScaleX - this.scaleX) * 0.2;
        this.scaleY += (this.targetScaleY - this.scaleY) * 0.2;
        
        // Rotación del cubo
        if (this.isGrounded) {
            this.rotation += this.rotationSpeed;
            
            // En el suelo siempre es cuadrado perfecto
            this.scaleX = 1;
            this.scaleY = 1;
            this.targetScaleX = 1;
            this.targetScaleY = 1;
        } else {
            this.rotation += 0.1;
        }
        
        // Actualizar estela de partículas
        this.trailParticles = this.trailParticles.filter(p => p.life > 0);
        this.trailParticles.forEach(particle => {
            particle.life -= 0.02;
            particle.x -= config.scrollSpeed * 0.5;
            particle.y += Math.random() * 2 - 1;
        });
        
        // Agregar nuevas partículas de estela
        if (Math.random() < 0.3) {
            this.trailParticles.push({
                x: this.x + this.width / 2,
                y: this.y + this.height / 2,
                size: Math.random() * 3 + 2,
                life: 1,
                color: Math.random() > 0.5 ? '#ff6b6b' : '#feca57'
            });
        }
    }

    startJump() {
        // Salto desde el suelo
        if (this.isGrounded && !this.isJumping) {
            this.isJumping = true;
            this.isShouting = true;
            this.isGrounded = false;
            this.jumpStartY = this.y;
            this.velocityY = -8;
            this.canDoubleJump = true; // Habilitar doble salto al despegar
            
            createJumpParticles(this.x + this.width / 2, this.y + this.height);
        }
        // Doble salto en el aire
        else if (this.canDoubleJump && !this.hasUsedDoubleJump && this.velocityY > 0) {
            this.hasUsedDoubleJump = true;
            this.canDoubleJump = false;
            this.isShouting = true;
            this.jumpStartY = this.y;
            this.velocityY = -8;
            
            // Partículas especiales para doble salto
            createJumpParticles(this.x + this.width / 2, this.y + this.height / 2, 20, '#feca57');
        }
    }
    
    stopJump() {
        // Detener el grito, comenzar a caer
        if (this.isShouting) {
            this.isShouting = false;
            const heightGained = this.jumpStartY - this.y;
        }
    }
}

// Clase Obstáculo
class Obstacle {
    constructor(x, y, width, height, type = 'ground') {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.type = type; // 'ground', 'spike', 'ceiling'
        this.passed = false;
    }

    draw(ctx) {
        ctx.save();
        
        if (this.type === 'spike') {
            // Dibujar pinchos triangulares
            const spikeCount = Math.floor(this.width / 20);
            ctx.fillStyle = '#e74c3c';
            
            for (let i = 0; i < spikeCount; i++) {
                const spikeX = this.x + i * (this.width / spikeCount);
                const spikeWidth = this.width / spikeCount;
                
                ctx.beginPath();
                ctx.moveTo(spikeX, this.y + this.height);
                ctx.lineTo(spikeX + spikeWidth / 2, this.y);
                ctx.lineTo(spikeX + spikeWidth, this.y + this.height);
                ctx.closePath();
                ctx.fill();
                
                // Brillo en el pincho
                ctx.strokeStyle = '#ff6b6b';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        } else if (this.type === 'ceiling') {
            // Obstáculo colgante
            const gradient = ctx.createLinearGradient(this.x, this.y, this.x, this.y + this.height);
            gradient.addColorStop(0, '#e74c3c');
            gradient.addColorStop(1, '#c0392b');
            ctx.fillStyle = gradient;
            ctx.fillRect(this.x, this.y, this.width, this.height);
            
            // Borde
            ctx.strokeStyle = '#ff6b6b';
            ctx.lineWidth = 2;
            ctx.strokeRect(this.x, this.y, this.width, this.height);
            
            // Patrón de peligro
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 3;
            for (let i = 0; i < this.width; i += 20) {
                ctx.beginPath();
                ctx.moveTo(this.x + i, this.y);
                ctx.lineTo(this.x + i + 10, this.y + this.height);
                ctx.stroke();
            }
        } else {
            // Obstáculo sólido desde el suelo
            const gradient = ctx.createLinearGradient(this.x, this.y, this.x, this.y + this.height);
            gradient.addColorStop(0, '#e74c3c');
            gradient.addColorStop(1, '#c0392b');
            ctx.fillStyle = gradient;
            ctx.fillRect(this.x, this.y, this.width, this.height);
            
            // Borde
            ctx.strokeStyle = '#ff6b6b';
            ctx.lineWidth = 3;
            ctx.strokeRect(this.x, this.y, this.width, this.height);
        }
        
        ctx.restore();
    }

    update() {
        this.x -= config.scrollSpeed;
    }

    isOffScreen() {
        return this.x + this.width < 0;
    }

    collidesWith(player) {
        return (
            player.x + player.width > this.x &&
            player.x < this.x + this.width &&
            player.y + player.height > this.y &&
            player.y < this.y + this.height
        );
    }
}

// Clase Plataforma
class Platform {
    constructor(x, y, width) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = 20;
    }

    draw(ctx) {
        ctx.save();
        
        // Plataforma con gradiente
        const gradient = ctx.createLinearGradient(this.x, this.y, this.x, this.y + this.height);
        gradient.addColorStop(0, '#48dbfb');
        gradient.addColorStop(1, '#0abde3');
        ctx.fillStyle = gradient;
        ctx.fillRect(this.x, this.y, this.width, this.height);
        
        // Borde superior brillante
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(this.x + this.width, this.y);
        ctx.stroke();
        
        ctx.restore();
    }

    update() {
        this.x -= config.scrollSpeed;
    }

    isOffScreen() {
        return this.x + this.width < 0;
    }
}

// Particle Class
class Particle {
    constructor(x, y, vx, vy, color, size) {
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.color = color;
        this.size = size;
        this.life = 1;
        this.decay = 0.02;
    }

    draw(ctx) {
        ctx.save();
        ctx.globalAlpha = this.life;
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.vy += 0.3; // Gravedad en partículas
        this.life -= this.decay;
    }

    isDead() {
        return this.life <= 0;
    }
}

// =====================================
// FUNCIONES DEL JUEGO
// =====================================

function createJumpParticles(x, y, count = 10, particleColor = null) {
    for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI;
        const speed = Math.random() * 3 + 2;
        const vx = Math.cos(angle) * speed;
        const vy = Math.sin(angle) * speed + 2;
        const color = particleColor || (Math.random() > 0.5 ? '#ff6b6b' : '#feca57');
        const size = Math.random() * 3 + 2;
        
        state.particles.push(new Particle(x, y, vx, vy, color, size));
    }
}

function createCeilingSparkles(x, y, count = 20) {
    for (let i = 0; i < count; i++) {
        const angle = Math.PI + (Math.random() * Math.PI); // Hacia abajo desde el techo
        const speed = Math.random() * 4 + 2;
        const vx = Math.cos(angle) * speed;
        const vy = Math.abs(Math.sin(angle) * speed); // Forzar hacia abajo
        const color = ['#feca57', '#ffff00', '#fff9a3', '#ffd700'][Math.floor(Math.random() * 4)];
        const size = Math.random() * 2 + 1;
        
        state.particles.push(new Particle(x, y, vx, vy, color, size));
    }
}

function createExplosion(x, y) {
    for (let i = 0; i < 30; i++) {
        const angle = (Math.PI * 2 * i) / 30;
        const speed = Math.random() * 5 + 3;
        const vx = Math.cos(angle) * speed;
        const vy = Math.sin(angle) * speed;
        const color = ['#e74c3c', '#ff6b6b', '#feca57', '#ff9ff3'][Math.floor(Math.random() * 4)];
        const size = Math.random() * 4 + 2;
        
        state.particles.push(new Particle(x, y, vx, vy, color, size));
    }
}

function spawnObstacle() {
    // Solo spawear si no hay obstáculos o si el último está lo suficientemente lejos
    if (state.obstacles.length > 0) {
        const lastObstacle = state.obstacles[state.obstacles.length - 1];
        const distanceFromRight = config.width - lastObstacle.x;
        
        // Si el último obstáculo todavía está cerca del borde derecho, no spawear
        if (distanceFromRight < 500) {
            return;
        }
    }
    
    // Gap aleatorio entre obstáculos (500-800px) - más espaciado
    const minGap = 500 + Math.random() * 300;
    
    // Verificar si es momento de spawear basado en el gap
    if (state.obstacles.length > 0) {
        const lastObstacle = state.obstacles[state.obstacles.length - 1];
        if (lastObstacle.x > config.width - minGap) {
            return;
        }
    }
    
    const x = config.width + 50; // Spawear fuera de la pantalla a la derecha
    const obstacleType = Math.random();
    
    if (obstacleType < 0.6) {
        // Obstáculo desde el suelo (más común)
        const width = 30 + Math.random() * 40; // Ancho entre 30-70px
        const height = 40 + Math.random() * 60; // Altura entre 40-100px
        const y = config.groundY - height;
        state.obstacles.push(new Obstacle(x, y, width, height, 'ground'));
    } else if (obstacleType < 0.85) {
        // Pinchos en el suelo
        const width = 40 + Math.random() * 40; // Ancho entre 40-80px
        const height = 30;
        const y = config.groundY - height;
        state.obstacles.push(new Obstacle(x, y, width, height, 'spike'));
    } else {
        // Obstáculo colgante del techo (menos común y más bajo)
        const width = 30 + Math.random() * 40;
        const height = 80 + Math.random() * 100; // Más largos para llegar más abajo
        state.obstacles.push(new Obstacle(x, config.ceilingY, width, height, 'ceiling'));
    }
    
    // Reducir plataformas flotantes (solo 20% de probabilidad)
    if (Math.random() < 0.20) {
        const platformX = x + 150 + Math.random() * 100;
        const platformY = config.groundY - 150 - Math.random() * 80;
        const platformWidth = 80 + Math.random() * 80;
        state.platforms.push(new Platform(platformX, platformY, platformWidth));
    }
}

function drawBackground(ctx) {
    // Fondo con gradiente
    const gradient = ctx.createLinearGradient(0, 0, 0, config.height);
    gradient.addColorStop(0, '#0a0a1e');
    gradient.addColorStop(1, '#1a1a2e');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, config.width, config.height);
    
    // Estrellas en movimiento
    const starOffset = (state.distance * 0.1) % 50;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    for (let i = 0; i < 50; i++) {
        const x = ((i * 157) % config.width - starOffset) % config.width;
        const y = (i * 211) % (config.groundY - 100);
        const size = (i % 3) + 1;
        ctx.fillRect(x, y, size, size);
    }
    
    // Dibujar techo
    const ceilingGradient = ctx.createLinearGradient(0, 0, 0, config.ceilingY);
    ceilingGradient.addColorStop(0, '#1a1a1e');
    ceilingGradient.addColorStop(1, '#2d3436');
    ctx.fillStyle = ceilingGradient;
    ctx.fillRect(0, 0, config.width, config.ceilingY);
    
    // Línea del techo
    ctx.strokeStyle = '#ff6b6b';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, config.ceilingY);
    ctx.lineTo(config.width, config.ceilingY);
    ctx.stroke();
    
    // Dibujar suelo
    const groundGradient = ctx.createLinearGradient(0, config.groundY, 0, config.height);
    groundGradient.addColorStop(0, '#2d3436');
    groundGradient.addColorStop(1, '#1a1a1e');
    ctx.fillStyle = groundGradient;
    ctx.fillRect(0, config.groundY, config.width, config.height - config.groundY);
    
    // Línea del suelo
    ctx.strokeStyle = '#ff6b6b';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, config.groundY);
    ctx.lineTo(config.width, config.groundY);
    ctx.stroke();
    
    // Patrón del suelo
    const patternOffset = (state.distance * 0.5) % 40;
    ctx.strokeStyle = 'rgba(255, 107, 107, 0.2)';
    ctx.lineWidth = 2;
    for (let x = -patternOffset; x < config.width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, config.groundY);
        ctx.lineTo(x + 20, config.height);
        ctx.stroke();
    }
}

function checkCollisions() {
    for (const obstacle of state.obstacles) {
        if (obstacle.collidesWith(state.player)) {
            return true;
        }
    }
    
    // Comprobar si el jugador cae fuera de la pantalla
    if (state.player.y > config.height) {
        return true;
    }
    
    return false;
}

function updateGame() {
    if (!state.isRunning || state.isPaused) return;
    
    // Actualizar jugador
    state.player.update();
    
    // Incrementar velocidad gradualmente
    if (config.scrollSpeed < config.maxSpeed) {
        config.scrollSpeed += config.speedIncrement;
    }
    
    // Actualizar distancia
    state.distance += config.scrollSpeed * 0.1;
    
    // Actualizar obstáculos
    state.obstacles = state.obstacles.filter(obstacle => {
        obstacle.update();
        
        // Marcar como pasado si está detrás del jugador
        if (!obstacle.passed && obstacle.x + obstacle.width < state.player.x) {
            obstacle.passed = true;
        }
        
        return !obstacle.isOffScreen();
    });
    
    // Actualizar plataformas
    state.platforms = state.platforms.filter(platform => {
        platform.update();
        return !platform.isOffScreen();
    });
    
    // Actualizar partículas
    state.particles = state.particles.filter(particle => {
        particle.update();
        return !particle.isDead();
    });
    
    // Generar nuevos obstáculos
    spawnObstacle();
    
    // Comprobar colisiones
    if (checkCollisions()) {
        gameOver();
        return;
    }
    
    // Actualizar UI
    updateUI();
}

function drawGame() {
    const ctx = config.ctx;
    
    // Limpiar canvas
    ctx.clearRect(0, 0, config.width, config.height);
    
    // Dibujar fondo
    drawBackground(ctx);
    
    // Dibujar plataformas
    state.platforms.forEach(platform => platform.draw(ctx));
    
    // Dibujar obstáculos
    state.obstacles.forEach(obstacle => obstacle.draw(ctx));
    
    // Dibujar partículas
    state.particles.forEach(particle => particle.draw(ctx));
    
    // Dibujar jugador
    if (state.player) {
        state.player.draw(ctx);
    }
    
    // Dibujar overlay de pausa
    if (state.isPaused) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(0, 0, config.width, config.height);
        
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 48px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('PAUSA', config.width / 2, config.height / 2);
        
        ctx.font = '24px Arial';
        ctx.fillText('Presiona P o ESC para continuar', config.width / 2, config.height / 2 + 50);
    }
}

function gameLoop() {
    updateGame();
    drawGame();
    state.gameLoopId = requestAnimationFrame(gameLoop);
}

function startGame() {

    // Reiniciar estado
    state.isRunning = true;
    state.isPaused = false;
    state.distance = 0;
    state.attempts++;
    
    // Reiniciar velocidad
    config.scrollSpeed = config.baseScrollSpeed;
    
    // Crear jugador
    state.player = new Player(150, config.groundY - 30);
    
    // Limpiar entidades
    state.obstacles = [];
    state.platforms = [];
    state.particles = [];
    
    // Actualizar UI
    updateUI();
    state.dom.pauseBtn.disabled = false;
    
    // Iniciar game loop
    if (state.gameLoopId) {
        cancelAnimationFrame(state.gameLoopId);
    }
    gameLoop();
    
    showNotification('🎮 ¡Juego iniciado! ¡Salta para esquivar!', 'success');
}

function togglePause() {
    if (!state.isRunning) return;
    
    state.isPaused = !state.isPaused;
    
    if (state.isPaused) {
        showNotification('⏸️ Juego pausado', 'warning');
    } else {
        showNotification('▶️ Juego reanudado', 'success');
    }
}

function gameOver() {
    state.isRunning = false;
    
    // Actualizar mejor distancia
    if (state.distance > state.bestDistance) {
        state.bestDistance = state.distance;
        showNotification(`🏆 ¡Nuevo récord! ${Math.floor(state.distance)}m`, 'success');
    } else {
        showNotification(`💥 Game Over - Distancia: ${Math.floor(state.distance)}m`, 'error');
    }
    
    // Crear explosión en la posición del jugador
    createExplosion(state.player.x + state.player.width / 2, state.player.y + state.player.height / 2);
    
    updateUI();
    state.dom.pauseBtn.disabled = true;
    
    // Continuar mostrando la animación de muerte por un momento
    setTimeout(() => {
        if (state.gameLoopId) {
            cancelAnimationFrame(state.gameLoopId);
        }
    }, 1000);
}

function updateUI() {
    state.dom.distanceValue.textContent = Math.floor(state.distance) + 'm';
    state.dom.bestDistanceValue.textContent = Math.floor(state.bestDistance) + 'm';
    state.dom.speedValue.textContent = (config.scrollSpeed / config.baseScrollSpeed).toFixed(1) + 'x';
    state.dom.attemptsValue.textContent = state.attempts;
}

async function initAudioStream() {
    try {
        // Solicitar acceso al micrófono
        state.audio.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000
            }
        });
        
        // Crear AudioContext
        state.audio.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });
        
        const source = state.audio.audioContext.createMediaStreamSource(state.audio.mediaStream);
        
        // Crear visualizador circular usando audio.js
        if (state.dom.audioVisualizer) {
            const vizResult = createAudioVisualizer(
                state.dom.audioVisualizer,
                state.audio.audioContext,
                state.audio.mediaStream,
                {
                    radius: 50,
                    barCount: 48,
                    maxBarHeight: 40,
                    lineWidth: 3,
                    colorful: true,
                    smoothing: 0.7
                }
            );
            
            if (vizResult) {
                state.audio.circularVisualizer = vizResult.visualizer;
                state.audio.circularVisualizer.start();
            }
        }
        
        // Crear VAD (Voice Activity Detector) - sensible a sonidos prolongados
        state.audio.vad = new VoiceActivityDetector({
            energyThreshold: 0.002,   // Umbral más bajo para mayor sensibilidad
            highEnergyThreshold: 0.01, // Menos estricto con volumen alto
            minSpeechDuration: 80,     // Respuesta más rápida
            minHighEnergySpeech: 40,   // No requiere tanto volumen alto
            maxSpeechDuration: 5000,   // Permitir gritos más largos
            silenceChunks: 1,          // Respuesta inmediata al silencio
            sampleRate: 16000,
            onSpeechStart: (energy) => {
                console.log(`🎤 Grito iniciado - energy: ${(energy * 1000).toFixed(1)}`);
                
                // Calcular intensidad del 1 al 100
                const intensity = Math.min(100, Math.max(1, Math.round(energy * 10000)));
                
                // Iniciar salto cuando comienza el grito
                if (state.isRunning && !state.isPaused && state.player) {
                    state.player.startJump();
                    state.player.shoutIntensity = intensity;
                    
                    // Actualizar UI
                    if (state.dom.currentKeyword) {
                        state.dom.currentKeyword.textContent = 'Salto!';
                        state.dom.currentKeyword.classList.add('active');
                    }
                    
                    // Mostrar número en el overlay
                    if (state.dom.audioKeywordOverlay) {
                        state.dom.audioKeywordOverlay.textContent = intensity;
                        state.dom.audioKeywordOverlay.classList.add('active');
                        
                        // Limpiar timeout anterior si existe
                        if (state.audio.overlayTimeout) {
                            clearTimeout(state.audio.overlayTimeout);
                        }
                    }
                }
            },
            onSpeechEnd: (audioData, duration, maxEnergy) => {
                console.log(`📤 Grito finalizado: ${duration}s (energy: ${(maxEnergy * 1000).toFixed(1)})`);
                
                // Detener salto cuando termina el grito
                if (state.isRunning && !state.isPaused && state.player) {
                    state.player.stopJump();
                    
                    // Actualizar UI
                    if (state.dom.currentKeyword) {
                        if (state.audio.displayTimeout) {
                            clearTimeout(state.audio.displayTimeout);
                        }
                        
                        state.audio.displayTimeout = setTimeout(() => {
                            state.dom.currentKeyword.classList.remove('active');
                            setTimeout(() => {
                                state.dom.currentKeyword.textContent = 'Esperando...';
                            }, 300);
                        }, 1000);
                    }
                    
                    // Ocultar número después de 2 segundos
                    if (state.dom.audioKeywordOverlay) {
                        state.audio.overlayTimeout = setTimeout(() => {
                            state.dom.audioKeywordOverlay.classList.remove('active');
                        }, 2000);
                    }
                }
            },
            onEnergyUpdate: (energy, hasVoice, isHighEnergy) => {
                // Opcional: actualizar UI con nivel de energía
            }
        });
        
        // Crear ScriptProcessorNode para procesar audio
        state.audio.scriptProcessorNode = state.audio.audioContext.createScriptProcessor(4096, 1, 1);
        
        state.audio.scriptProcessorNode.onaudioprocess = (event) => {
            if (!state.audio.isStreaming) return;
            
            const inputData = event.inputBuffer.getChannelData(0);
            
            // Procesar con VAD
            if (state.audio.vad) {
                state.audio.vad.process(inputData);
            }
        };
        
        // Conectar nodos
        source.connect(state.audio.scriptProcessorNode);
        state.audio.scriptProcessorNode.connect(state.audio.audioContext.destination);
        
        state.audio.isStreaming = true;
        
        console.log('✓ Audio stream inicializado');
        
        return true;
    } catch (error) {
        console.error('✗ Error al inicializar audio:', error);
        showNotification('Error al acceder al micrófono', 'error');
        return false;
    }
}

function stopAudioStream() {
    state.audio.isStreaming = false;
    
    // Resetear VAD
    if (state.audio.vad) {
        state.audio.vad.reset();
        state.audio.vad = null;
    }
    
    // Limpiar timeout del display
    if (state.audio.displayTimeout) {
        clearTimeout(state.audio.displayTimeout);
        state.audio.displayTimeout = null;
    }
    
    // Limpiar timeout del overlay
    if (state.audio.overlayTimeout) {
        clearTimeout(state.audio.overlayTimeout);
        state.audio.overlayTimeout = null;
    }
    
    // Detener visualizador
    if (state.audio.circularVisualizer) {
        state.audio.circularVisualizer.stop();
        state.audio.circularVisualizer = null;
    }
    
    // Detener audio
    if (state.audio.scriptProcessorNode) {
        state.audio.scriptProcessorNode.disconnect();
        state.audio.scriptProcessorNode = null;
    }
    
    if (state.audio.mediaStream) {
        state.audio.mediaStream.getTracks().forEach(track => track.stop());
        state.audio.mediaStream = null;
    }
    
    if (state.audio.audioContext) {
        state.audio.audioContext.close();
        state.audio.audioContext = null;
    }
}

// =====================================
// MANEJADOR DE EVENTOS
// =====================================

function setupEventListeners() {
    state.dom.startBtn.addEventListener('click', async function() {
    // Iniciar audio si no está activo
    if (!state.audio.isStreaming) {
        try {
            const audioInitialized = await initAudioStream();
            if (!audioInitialized) {
                showNotification('Error al inicializar audio. Verifica los permisos del micrófono.', 'error');
                return;
            }
            showNotification('🎤 Audio iniciado', 'success');
            // Pequeña pausa para que el usuario vea el mensaje
            await new Promise(resolve => setTimeout(resolve, 500));
        } catch (error) {
            console.error('Error:', error);
            showNotification('Error al iniciar audio', 'error');
            return;
        }
    }
    
    // Iniciar juego
    startGame();
    });
    
    state.dom.pauseBtn.addEventListener('click', togglePause);

    // Controles de teclado (modo oculto activado por doble click)
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

    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && keyboardModeEnabled && state.isRunning && !state.isPaused && !spaceKeyPressed) {
            e.preventDefault();
            spaceKeyPressed = true;
            if (state.player) {
                state.player.startJump();
            }
        } else if ((e.code === 'KeyP' || e.code === 'Escape') && state.isRunning) {
            e.preventDefault();
            togglePause();
        }
    });

    document.addEventListener('keyup', (e) => {
        if (e.code === 'Space' && keyboardModeEnabled && state.isRunning && !state.isPaused && spaceKeyPressed) {
            e.preventDefault();
            spaceKeyPressed = false;
            if (state.player) {
                state.player.stopJump();
            }
        }
    });
}

// =====================================
// INICIO DEL JUEGO
// =====================================

function initGame() {
    // Cacheamos elementos del DOM
    state.dom.canvas = document.getElementById('game-canvas');
    state.dom.audioVisualizer = document.getElementById('audio-visualizer');
    state.dom.startBtn = document.getElementById('start-btn');
    state.dom.pauseBtn = document.getElementById('pause-button');
    state.dom.distanceValue = document.getElementById('distance-value');
    state.dom.bestDistanceValue = document.getElementById('best-distance-value');
    state.dom.speedValue = document.getElementById('speed-value');
    state.dom.attemptsValue = document.getElementById('attempts-value');
    state.dom.currentKeyword = document.getElementById('current-keyword');
    state.dom.audioKeywordOverlay = document.getElementById('audio-keyword-overlay');
    
    config.canvas = state.dom.canvas;
    config.ctx = config.canvas.getContext('2d');
    
    // Inciamos listeners
    setupEventListeners();
    
    console.log('Voice Jump Platform inicializado - Usa gritos cortos/largos para saltar');
    updateUI();
}

document.addEventListener('DOMContentLoaded', initGame);
