/**
 * Floor Tile Visualizer - Main Application Script
 * ================================================
 */

// Application State
const state = {
    roomImage: null,
    roomFilename: null,
    selectedTile: null,
    polygonPoints: [],
    isPolygonClosed: false,
    zoom: 1,
    offset: { x: 0, y: 0 },
    imageSize: { width: 0, height: 0 },
    canvasSize: { width: 0, height: 0 },
    isDragging: false,
    dragStart: { x: 0, y: 0 },
    lastResultFile: null
};

// DOM Elements
const elements = {
    // Canvas
    canvasContainer: document.getElementById('canvasContainer'),
    canvasPlaceholder: document.getElementById('canvasPlaceholder'),
    mainCanvas: document.getElementById('mainCanvas'),
    overlayCanvas: document.getElementById('overlayCanvas'),
    mainCtx: null,
    overlayCtx: null,
    
    // Upload
    roomUploadZone: document.getElementById('roomUploadZone'),
    roomInput: document.getElementById('roomInput'),
    tileInput: document.getElementById('tileInput'),
    uploadTileBtn: document.getElementById('uploadTileBtn'),
    
    // Tools
    polygonTool: document.getElementById('polygonTool'),
    autoDetectBtn: document.getElementById('autoDetectBtn'),
    clearSelectionBtn: document.getElementById('clearSelectionBtn'),
    undoPointBtn: document.getElementById('undoPointBtn'),
    toolInstructions: document.getElementById('toolInstructions'),
    
    // Settings
    tileSizeSlider: document.getElementById('tileSizeSlider'),
    tileSizeValue: document.getElementById('tileSizeValue'),
    groutWidthSlider: document.getElementById('groutWidthSlider'),
    groutWidthValue: document.getElementById('groutWidthValue'),
    
    // Tiles
    tilesGrid: document.getElementById('tilesGrid'),
    
    // Actions
    applyTileBtn: document.getElementById('applyTileBtn'),
    
    // Zoom
    zoomInBtn: document.getElementById('zoomInBtn'),
    zoomOutBtn: document.getElementById('zoomOutBtn'),
    zoomFitBtn: document.getElementById('zoomFitBtn'),
    zoomLevel: document.getElementById('zoomLevel'),
    imageInfo: document.getElementById('imageInfo'),
    
    // Modals
    resultModal: document.getElementById('resultModal'),
    closeResultModal: document.getElementById('closeResultModal'),
    resultImage: document.getElementById('resultImage'),
    downloadResultBtn: document.getElementById('downloadResultBtn'),
    tryAgainBtn: document.getElementById('tryAgainBtn'),
    
    helpModal: document.getElementById('helpModal'),
    helpBtn: document.getElementById('helpBtn'),
    closeHelpModal: document.getElementById('closeHelpModal'),
    
    // Loading
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    
    // Toast
    toastContainer: document.getElementById('toastContainer')
};

// Initialize contexts
elements.mainCtx = elements.mainCanvas.getContext('2d');
elements.overlayCtx = elements.overlayCanvas.getContext('2d');

// ============================================================================
// Utility Functions
// ============================================================================

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' ? 'check-circle' : 
                 type === 'error' ? 'exclamation-circle' : 'info-circle';
    
    toast.innerHTML = `
        <i class="fas fa-${icon}"></i>
        <span class="toast-message">${message}</span>
        <button class="toast-close"><i class="fas fa-times"></i></button>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
    
    // Manual close
    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    });
}

function showLoading(text = 'Processing...') {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.classList.add('active');
}

function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
}

function showModal(modal) {
    modal.classList.add('active');
}

function hideModal(modal) {
    modal.classList.remove('active');
}

// ============================================================================
// Canvas Functions
// ============================================================================

function initCanvas() {
    const rect = elements.canvasContainer.getBoundingClientRect();
    state.canvasSize.width = rect.width;
    state.canvasSize.height = rect.height;
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

async function displayRoomImage(imageSrc) {
    try {
        state.roomImage = await loadImage(imageSrc);
        state.imageSize.width = state.roomImage.width;
        state.imageSize.height = state.roomImage.height;
        
        // Hide placeholder, show canvases
        elements.canvasPlaceholder.style.display = 'none';
        elements.mainCanvas.style.display = 'block';
        elements.overlayCanvas.style.display = 'block';
        
        // Fit to view
        fitToView();
        
        // Update info
        elements.imageInfo.textContent = `${state.imageSize.width} × ${state.imageSize.height} px`;
        
        // Clear previous selection
        clearSelection();
        
    } catch (error) {
        showToast('Failed to load image', 'error');
        console.error(error);
    }
}

function fitToView() {
    const containerRect = elements.canvasContainer.getBoundingClientRect();
    const containerWidth = containerRect.width - 40;
    const containerHeight = containerRect.height - 40;
    
    const scaleX = containerWidth / state.imageSize.width;
    const scaleY = containerHeight / state.imageSize.height;
    
    state.zoom = Math.min(scaleX, scaleY, 1);
    state.offset = { x: 0, y: 0 };
    
    updateCanvasTransform();
    redrawCanvas();
}

function updateCanvasTransform() {
    const width = Math.round(state.imageSize.width * state.zoom);
    const height = Math.round(state.imageSize.height * state.zoom);
    
    elements.mainCanvas.width = width;
    elements.mainCanvas.height = height;
    elements.overlayCanvas.width = width;
    elements.overlayCanvas.height = height;
    
    elements.mainCanvas.style.width = `${width}px`;
    elements.mainCanvas.style.height = `${height}px`;
    elements.overlayCanvas.style.width = `${width}px`;
    elements.overlayCanvas.style.height = `${height}px`;
    
    elements.zoomLevel.textContent = `${Math.round(state.zoom * 100)}%`;
}

function redrawCanvas() {
    if (!state.roomImage) return;
    
    const width = elements.mainCanvas.width;
    const height = elements.mainCanvas.height;
    
    // Clear and draw room image
    elements.mainCtx.clearRect(0, 0, width, height);
    elements.mainCtx.drawImage(state.roomImage, 0, 0, width, height);
    
    // Redraw overlay
    redrawOverlay();
}

function redrawOverlay() {
    const width = elements.overlayCanvas.width;
    const height = elements.overlayCanvas.height;
    
    elements.overlayCtx.clearRect(0, 0, width, height);
    
    if (state.polygonPoints.length === 0) return;
    
    const ctx = elements.overlayCtx;
    
    // Draw filled polygon if closed
    if (state.isPolygonClosed) {
        ctx.beginPath();
        ctx.moveTo(
            state.polygonPoints[0][0] * state.zoom,
            state.polygonPoints[0][1] * state.zoom
        );
        for (let i = 1; i < state.polygonPoints.length; i++) {
            ctx.lineTo(
                state.polygonPoints[i][0] * state.zoom,
                state.polygonPoints[i][1] * state.zoom
            );
        }
        ctx.closePath();
        ctx.fillStyle = 'rgba(79, 70, 229, 0.3)';
        ctx.fill();
    }
    
    // Draw lines
    ctx.beginPath();
    ctx.strokeStyle = '#4f46e5';
    ctx.lineWidth = 2;
    ctx.setLineDash(state.isPolygonClosed ? [] : [5, 5]);
    
    ctx.moveTo(
        state.polygonPoints[0][0] * state.zoom,
        state.polygonPoints[0][1] * state.zoom
    );
    
    for (let i = 1; i < state.polygonPoints.length; i++) {
        ctx.lineTo(
            state.polygonPoints[i][0] * state.zoom,
            state.polygonPoints[i][1] * state.zoom
        );
    }
    
    if (state.isPolygonClosed) {
        ctx.closePath();
    }
    
    ctx.stroke();
    
    // Draw points
    state.polygonPoints.forEach((point, index) => {
        const x = point[0] * state.zoom;
        const y = point[1] * state.zoom;
        
        // Outer circle
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fillStyle = index === 0 ? '#22c55e' : '#4f46e5';
        ctx.fill();
        
        // Inner circle
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = 'white';
        ctx.fill();
        
        // Point number
        ctx.fillStyle = 'white';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText((index + 1).toString(), x, y - 14);
    });
}

// ============================================================================
// Selection Functions
// ============================================================================

function getCanvasCoordinates(event) {
    const rect = elements.overlayCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / state.zoom;
    const y = (event.clientY - rect.top) / state.zoom;
    return [Math.round(x), Math.round(y)];
}

function addPolygonPoint(x, y) {
    // Check if near first point to close polygon
    if (state.polygonPoints.length >= 3) {
        const firstPoint = state.polygonPoints[0];
        const distance = Math.sqrt(
            Math.pow(x - firstPoint[0], 2) + Math.pow(y - firstPoint[1], 2)
        );
        
        if (distance < 20 / state.zoom) {
            closePolygon();
            return;
        }
    }
    
    state.polygonPoints.push([x, y]);
    updateSelectionButtons();
    redrawOverlay();
}

function closePolygon() {
    if (state.polygonPoints.length >= 3) {
        state.isPolygonClosed = true;
        updateSelectionButtons();
        updateApplyButton();
        redrawOverlay();
        showToast('Floor area selected!', 'success');
    }
}

function clearSelection() {
    state.polygonPoints = [];
    state.isPolygonClosed = false;
    updateSelectionButtons();
    updateApplyButton();
    redrawOverlay();
}

function undoLastPoint() {
    if (state.polygonPoints.length > 0) {
        state.polygonPoints.pop();
        state.isPolygonClosed = false;
        updateSelectionButtons();
        updateApplyButton();
        redrawOverlay();
    }
}

function updateSelectionButtons() {
    elements.clearSelectionBtn.disabled = state.polygonPoints.length === 0;
    elements.undoPointBtn.disabled = state.polygonPoints.length === 0 || state.isPolygonClosed;
}

function updateApplyButton() {
    const canApply = state.roomFilename && 
                     state.selectedTile && 
                     state.isPolygonClosed &&
                     state.polygonPoints.length >= 3;
    
    elements.applyTileBtn.disabled = !canApply;
}

// ============================================================================
// API Functions
// ============================================================================

async function uploadRoomImage(file) {
    showLoading('Uploading image...');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        state.roomFilename = data.filename;
        
        // Display the image
        await displayRoomImage(data.path);
        
        // Update upload zone
        elements.roomUploadZone.classList.add('has-image');
        elements.roomUploadZone.querySelector('p').textContent = 'Image loaded!';
        
        showToast('Room image uploaded successfully!', 'success');
        
    } catch (error) {
        showToast(error.message || 'Failed to upload image', 'error');
    } finally {
        hideLoading();
    }
}

async function uploadTileImage(file) {
    showLoading('Uploading tile...');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/upload-tile', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Add to tiles grid
        addTileToGrid(data);
        
        // Select the new tile
        selectTile(data.name);
        
        showToast('Custom tile uploaded!', 'success');
        
    } catch (error) {
        showToast(error.message || 'Failed to upload tile', 'error');
    } finally {
        hideLoading();
    }
}

function addTileToGrid(tile) {
    const tileItem = document.createElement('div');
    tileItem.className = 'tile-item';
    tileItem.dataset.tile = tile.name;
    tileItem.innerHTML = `
        <img src="${tile.thumbnail}" alt="${tile.name}">
        <span class="tile-name">${tile.name}</span>
    `;
    
    tileItem.addEventListener('click', () => selectTile(tile.name));
    
    elements.tilesGrid.insertBefore(tileItem, elements.tilesGrid.firstChild);
}

async function autoDetectFloor() {
    if (!state.roomFilename) {
        showToast('Please upload a room image first', 'error');
        return;
    }
    
    showLoading('Detecting floor area...');
    
    try {
        const response = await fetch('/api/auto-detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: state.roomFilename })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Set the detected polygon
        state.polygonPoints = data.points;
        state.isPolygonClosed = true;
        
        updateSelectionButtons();
        updateApplyButton();
        redrawOverlay();
        
        showToast(`Floor detected! Coverage: ${data.coverage.toFixed(1)}%`, 'success');
        
    } catch (error) {
        showToast(error.message || 'Auto-detection failed', 'error');
    } finally {
        hideLoading();
    }
}

async function applyTile() {
    if (!state.roomFilename || !state.selectedTile || !state.isPolygonClosed) {
        showToast('Please complete all steps first', 'error');
        return;
    }
    
    showLoading('Applying tile texture...\nThis may take a moment...');
    
    try {
        const response = await fetch('/api/visualize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                room_image: state.roomFilename,
                tile_image: state.selectedTile,
                polygon_points: state.polygonPoints,
                tile_size_cm: parseInt(elements.tileSizeSlider.value),
                grout_width_mm: parseFloat(elements.groutWidthSlider.value)
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Show result
        state.lastResultFile = data.filename;
        elements.resultImage.src = data.preview_path;
        showModal(elements.resultModal);
        
        showToast('Tile applied successfully!', 'success');
        
    } catch (error) {
        showToast(error.message || 'Failed to apply tile', 'error');
    } finally {
        hideLoading();
    }
}

function downloadResult() {
    if (state.lastResultFile) {
        window.open(`/api/download/${state.lastResultFile}`, '_blank');
    }
}

// ============================================================================
// Tile Selection
// ============================================================================

function selectTile(tileName) {
    state.selectedTile = tileName;
    
    // Update UI
    document.querySelectorAll('.tile-item').forEach(item => {
        item.classList.toggle('selected', item.dataset.tile === tileName);
    });
    
    updateApplyButton();
}

// ============================================================================
// Event Handlers
// ============================================================================

function setupEventListeners() {
    // Room upload
    elements.roomUploadZone.addEventListener('click', () => {
        elements.roomInput.click();
    });
    
    elements.roomUploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.roomUploadZone.classList.add('drag-over');
    });
    
    elements.roomUploadZone.addEventListener('dragleave', () => {
        elements.roomUploadZone.classList.remove('drag-over');
    });
    
    elements.roomUploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.roomUploadZone.classList.remove('drag-over');
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            uploadRoomImage(file);
        }
    });
    
    elements.roomInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            uploadRoomImage(file);
        }
    });
    
    // Tile upload
    elements.uploadTileBtn.addEventListener('click', () => {
        elements.tileInput.click();
    });
    
    elements.tileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            uploadTileImage(file);
        }
    });
    
    // Canvas click for polygon
    elements.overlayCanvas.addEventListener('click', (e) => {
        if (!state.roomImage) return;
        if (state.isPolygonClosed) return;
        
        const [x, y] = getCanvasCoordinates(e);
        addPolygonPoint(x, y);
    });
    
    // Tools
    elements.autoDetectBtn.addEventListener('click', autoDetectFloor);
    elements.clearSelectionBtn.addEventListener('click', clearSelection);
    elements.undoPointBtn.addEventListener('click', undoLastPoint);
    
    // Settings sliders
    elements.tileSizeSlider.addEventListener('input', (e) => {
        elements.tileSizeValue.textContent = e.target.value;
    });
    
    elements.groutWidthSlider.addEventListener('input', (e) => {
        elements.groutWidthValue.textContent = e.target.value;
    });
    
    // Tile selection
    document.querySelectorAll('.tile-item').forEach(item => {
        item.addEventListener('click', () => {
            selectTile(item.dataset.tile);
        });
    });
    
    // Apply button
    elements.applyTileBtn.addEventListener('click', applyTile);
    
    // Zoom controls
    elements.zoomInBtn.addEventListener('click', () => {
        state.zoom = Math.min(state.zoom * 1.2, 3);
        updateCanvasTransform();
        redrawCanvas();
    });
    
    elements.zoomOutBtn.addEventListener('click', () => {
        state.zoom = Math.max(state.zoom / 1.2, 0.1);
        updateCanvasTransform();
        redrawCanvas();
    });
    
    elements.zoomFitBtn.addEventListener('click', fitToView);
    
    // Result modal
    elements.closeResultModal.addEventListener('click', () => {
        hideModal(elements.resultModal);
    });
    
    elements.downloadResultBtn.addEventListener('click', downloadResult);
    
    elements.tryAgainBtn.addEventListener('click', () => {
        hideModal(elements.resultModal);
    });
    
    // Help modal
    elements.helpBtn.addEventListener('click', () => {
        showModal(elements.helpModal);
    });
    
    elements.closeHelpModal.addEventListener('click', () => {
        hideModal(elements.helpModal);
    });
    
    // Close modals on backdrop click
    [elements.resultModal, elements.helpModal].forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                hideModal(modal);
            }
        });
    });
    
    // Window resize
    window.addEventListener('resize', () => {
        initCanvas();
        if (state.roomImage) {
            fitToView();
        }
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            hideModal(elements.resultModal);
            hideModal(elements.helpModal);
        }
        
        if (e.ctrlKey && e.key === 'z' && !state.isPolygonClosed) {
            e.preventDefault();
            undoLastPoint();
        }
    });
}

// ============================================================================
// Initialize
// ============================================================================

function init() {
    initCanvas();
    setupEventListeners();
    
    // Select first tile by default if available
    const firstTile = document.querySelector('.tile-item');
    if (firstTile) {
        selectTile(firstTile.dataset.tile);
    }
    
    console.log('Floor Tile Visualizer initialized');
}

// Start the application
document.addEventListener('DOMContentLoaded', init);
