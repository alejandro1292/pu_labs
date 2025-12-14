/**
 * Utilidades UI - Modales y Notificaciones
 * Funciones reutilizables para toasts, modales y loaders
 */

// Referencias a elementos del DOM (se inicializan cuando el DOM está listo)
let uiElements = null;

/**
 * Inicializa las referencias a los elementos del DOM
 * Debe llamarse después de que el DOM esté cargado
 */
function initUIElements() {
    if (uiElements) return; // Ya inicializado
    
    uiElements = {
        // Toast
        toastContainer: document.getElementById('toast-container'),
        
        // Confirm Modal
        confirmModal: document.getElementById('confirm-modal'),
        confirmTitle: document.getElementById('confirm-title'),
        confirmMessage: document.getElementById('confirm-message'),
        confirmOkBtn: document.getElementById('confirm-ok-btn'),
        confirmCancelBtn: document.getElementById('confirm-cancel-btn'),
        
        // Prompt Modal
        promptModal: document.getElementById('prompt-modal'),
        promptTitle: document.getElementById('prompt-title'),
        promptMessage: document.getElementById('prompt-message'),
        promptInput: document.getElementById('prompt-input'),
        promptOkBtn: document.getElementById('prompt-ok-btn'),
        promptCancelBtn: document.getElementById('prompt-cancel-btn'),
        
        // Alert Modal
        alertModal: document.getElementById('alert-modal'),
        alertTitle: document.getElementById('alert-title'),
        alertMessage: document.getElementById('alert-message'),
        alertOkBtn: document.getElementById('alert-ok-btn'),
        
        // Loader
        globalLoader: document.getElementById('global-loader'),
        loaderText: document.getElementById('loader-text'),
        loaderSubtext: document.getElementById('loader-subtext')
    };
    
    // Configurar event listeners para cerrar modales al hacer clic fuera
    setupModalClickOutside();
}

/**
 * Configura los event listeners para cerrar modales al hacer clic fuera
 */
function setupModalClickOutside() {
    if (!uiElements) return;
    
    // Confirm Modal
    uiElements.confirmModal?.addEventListener('click', (e) => {
        if (e.target === uiElements.confirmModal) {
            uiElements.confirmCancelBtn?.click();
        }
    });
    
    // Prompt Modal
    uiElements.promptModal?.addEventListener('click', (e) => {
        if (e.target === uiElements.promptModal) {
            uiElements.promptCancelBtn?.click();
        }
    });
    
    // Alert Modal
    uiElements.alertModal?.addEventListener('click', (e) => {
        if (e.target === uiElements.alertModal) {
            uiElements.alertOkBtn?.click();
        }
    });
}

/**
 * Muestra una notificación toast
 * @param {string} message - Mensaje a mostrar
 * @param {string} type - Tipo de notificación: 'success', 'error', 'warning', 'info'
 */
function showNotification(message, type = 'info') {
    if (!uiElements) initUIElements();
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.style.cssText = `
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : type === 'warning' ? '#f59e0b' : '#3b82f6'};
        color: white;
        padding: 16px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        font-size: 0.95rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: 300px;
        max-width: 400px;
        animation: slideIn 0.3s ease-out;
        pointer-events: auto;
    `;
    
    const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️';
    toast.innerHTML = `<span style="font-size: 1.2rem;">${icon}</span><span>${message}</span>`;
    
    if (uiElements.toastContainer) {
        uiElements.toastContainer.appendChild(toast);
        
        // Auto-remover después de 4 segundos
        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    } else {
        console.warn('Toast container no encontrado');
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
}

/**
 * Muestra un modal de confirmación
 * @param {string} title - Título del modal
 * @param {string} message - Mensaje a mostrar
 * @returns {Promise<boolean>} - true si el usuario acepta, false si cancela
 */
function showConfirm(title, message) {
    if (!uiElements) initUIElements();
    
    return new Promise((resolve) => {
        if (!uiElements.confirmModal) {
            // Fallback a confirm nativo si no hay modal
            resolve(confirm(`${title}\n\n${message}`));
            return;
        }
        
        uiElements.confirmTitle.textContent = title;
        uiElements.confirmMessage.textContent = message;
        uiElements.confirmModal.style.display = 'flex';
        
        const handleOk = () => {
            cleanup();
            resolve(true);
        };
        
        const handleCancel = () => {
            cleanup();
            resolve(false);
        };
        
        const cleanup = () => {
            uiElements.confirmModal.style.display = 'none';
            uiElements.confirmOkBtn.removeEventListener('click', handleOk);
            uiElements.confirmCancelBtn.removeEventListener('click', handleCancel);
            document.removeEventListener('keydown', handleEsc);
        };
        
        const handleEsc = (e) => {
            if (e.key === 'Escape') {
                handleCancel();
            }
        };
        
        uiElements.confirmOkBtn.addEventListener('click', handleOk);
        uiElements.confirmCancelBtn.addEventListener('click', handleCancel);
        document.addEventListener('keydown', handleEsc);
    });
}

/**
 * Muestra un modal de entrada de texto
 * @param {string} title - Título del modal
 * @param {string} message - Mensaje a mostrar
 * @param {string} defaultValue - Valor por defecto del input
 * @returns {Promise<string|null>} - Valor ingresado o null si se cancela
 */
function showPrompt(title, message, defaultValue = '') {
    if (!uiElements) initUIElements();
    
    return new Promise((resolve) => {
        if (!uiElements.promptModal) {
            // Fallback a prompt nativo si no hay modal
            resolve(prompt(`${title}\n\n${message}`, defaultValue));
            return;
        }
        
        uiElements.promptTitle.textContent = title;
        uiElements.promptMessage.textContent = message;
        uiElements.promptInput.value = defaultValue;
        uiElements.promptModal.style.display = 'flex';
        uiElements.promptInput.focus();
        uiElements.promptInput.select();
        
        const handleOk = () => {
            const value = uiElements.promptInput.value.trim();
            cleanup();
            resolve(value || null);
        };
        
        const handleCancel = () => {
            cleanup();
            resolve(null);
        };
        
        const handleEnter = (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleOk();
            }
        };
        
        const handleEsc = (e) => {
            if (e.key === 'Escape') {
                handleCancel();
            }
        };
        
        const cleanup = () => {
            uiElements.promptModal.style.display = 'none';
            uiElements.promptOkBtn.removeEventListener('click', handleOk);
            uiElements.promptCancelBtn.removeEventListener('click', handleCancel);
            uiElements.promptInput.removeEventListener('keydown', handleEnter);
            document.removeEventListener('keydown', handleEsc);
        };
        
        uiElements.promptOkBtn.addEventListener('click', handleOk);
        uiElements.promptCancelBtn.addEventListener('click', handleCancel);
        uiElements.promptInput.addEventListener('keydown', handleEnter);
        document.addEventListener('keydown', handleEsc);
    });
}

/**
 * Muestra un modal de alerta
 * @param {string} title - Título del modal
 * @param {string} message - Mensaje a mostrar
 * @returns {Promise<void>}
 */
function showAlert(title, message) {
    if (!uiElements) initUIElements();
    
    return new Promise((resolve) => {
        if (!uiElements.alertModal) {
            // Fallback a alert nativo si no hay modal
            alert(`${title}\n\n${message}`);
            resolve();
            return;
        }
        
        uiElements.alertTitle.textContent = title;
        uiElements.alertMessage.textContent = message;
        uiElements.alertModal.style.display = 'flex';
        
        const handleOk = () => {
            cleanup();
            resolve();
        };
        
        const handleKey = (e) => {
            if (e.key === 'Escape' || e.key === 'Enter') {
                handleOk();
            }
        };
        
        const cleanup = () => {
            uiElements.alertModal.style.display = 'none';
            uiElements.alertOkBtn.removeEventListener('click', handleOk);
            document.removeEventListener('keydown', handleKey);
        };
        
        uiElements.alertOkBtn.addEventListener('click', handleOk);
        document.addEventListener('keydown', handleKey);
    });
}

/**
 * Muestra el loader global
 * @param {string} text - Texto principal
 * @param {string} subtext - Subtexto opcional
 */
function showLoader(text = 'Procesando...', subtext = '') {
    if (!uiElements) initUIElements();
    
    if (uiElements.globalLoader) {
        uiElements.loaderText.textContent = text;
        uiElements.loaderSubtext.textContent = subtext;
        uiElements.globalLoader.style.display = 'flex';
    }
}

/**
 * Oculta el loader global
 */
function hideLoader() {
    if (!uiElements) initUIElements();
    
    if (uiElements.globalLoader) {
        uiElements.globalLoader.style.display = 'none';
    }
}

// Auto-inicializar cuando el DOM esté listo
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initUIElements);
} else {
    initUIElements();
}
