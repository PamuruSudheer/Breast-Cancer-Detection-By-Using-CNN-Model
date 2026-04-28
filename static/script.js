// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const selectBtn = document.getElementById('selectBtn');
const clearBtn = document.getElementById('clearBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const resultsSection = document.getElementById('resultsSection');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsContent = document.getElementById('resultsContent');
const errorMessage = document.getElementById('errorMessage');
const newPredictionBtn = document.getElementById('newPredictionBtn');

// State
let selectedFile = null;

// Event Listeners
selectBtn.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', handleImageSelect);
clearBtn.addEventListener('click', clearSelection);
newPredictionBtn.addEventListener('click', resetForm);

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleImageSelect();
    }
});

// Handle Image Selection
function handleImageSelect() {
    const file = imageInput.files[0];
    if (!file) return;

    // Validate file type
    if (!['image/png', 'image/jpeg', 'image/gif', 'image/bmp'].includes(file.type)) {
        showError('Invalid file type. Please select an image (PNG, JPG, JPEG, GIF, BMP)');
        return;
    }

    // Validate file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
        showError('File size exceeds 50MB limit');
        return;
    }

    selectedFile = file;
    displayPreview(file);
}

// Display Image Preview
function displayPreview(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        fileName.textContent = `Selected: ${file.name}`;
        previewSection.style.display = 'block';
        selectBtn.style.display = 'none';
        clearBtn.style.display = 'inline-block';
        
        // Auto-predict after preview
        setTimeout(() => {
            makePrediction();
        }, 300);
    };
    
    reader.onerror = () => {
        showError('Error reading file');
    };
    
    reader.readAsDataURL(file);
}

// Make Prediction
async function makePrediction() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    resultsSection.style.display = 'block';
    loadingSpinner.style.display = 'block';
    resultsContent.style.display = 'none';
    errorMessage.style.display = 'none';

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
            loadingSpinner.style.display = 'none';
            resultsContent.style.display = 'block';
        } else {
            showError(data.error || 'Prediction failed');
            loadingSpinner.style.display = 'none';
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Network error: ' + error.message);
        loadingSpinner.style.display = 'none';
    }
}

// Display Results
function displayResults(data) {
    const { prediction, confidence, probabilities } = data;

    // Set prediction label and color
    const predictionLabel = document.getElementById('predictionLabel');
    predictionLabel.textContent = prediction;
    
    // Color code based on prediction
    const resultsCard = document.querySelector('.prediction-card');
    if (prediction === 'Malignant') {
        resultsCard.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
    } else if (prediction === 'Benign') {
        resultsCard.style.background = 'linear-gradient(135deg, #10b981, #059669)';
    } else {
        resultsCard.style.background = 'linear-gradient(135deg, #3b82f6, #1d4ed8)';
    }

    // Set confidence score
    document.getElementById('confidenceScore').textContent = confidence.toFixed(2) + '%';

    // Update probability bars
    updateProbabilityBar('benign', probabilities.benign);
    updateProbabilityBar('malignant', probabilities.malignant);
    updateProbabilityBar('normal', probabilities.normal);
}

// Update Probability Bar
function updateProbabilityBar(className, percentage) {
    const bar = document.getElementById(className + 'Bar');
    const percent = document.getElementById(className + 'Percent');
    
    // Animate bar width
    bar.style.width = '0%';
    percent.textContent = '0%';
    
    setTimeout(() => {
        bar.style.width = percentage + '%';
        percent.textContent = percentage.toFixed(2) + '%';
    }, 100);
}

// Clear Selection
function clearSelection() {
    selectedFile = null;
    imageInput.value = '';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    selectBtn.style.display = 'inline-block';
    clearBtn.style.display = 'none';
    errorMessage.style.display = 'none';
}

// Reset Form
function resetForm() {
    clearSelection();
    uploadArea.scrollIntoView({ behavior: 'smooth' });
}

// Show Error
function showError(message) {
    errorMessage.textContent = '⚠️ ' + message;
    errorMessage.style.display = 'block';
    resultsContent.style.display = 'none';
    loadingSpinner.style.display = 'none';
}

// Check Server Health on Load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        const statusBadge = document.getElementById('server-status');
        if (data.status === 'healthy' && data.model === 'loaded') {
            statusBadge.textContent = '● Online';
            statusBadge.className = 'status-badge online';
        } else {
            statusBadge.textContent = '● Model Not Loaded';
            statusBadge.className = 'status-badge offline';
            showError('Model not loaded. Please train the model first using: python train_model.py');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        const statusBadge = document.getElementById('server-status');
        statusBadge.textContent = '● Offline';
        statusBadge.className = 'status-badge offline';
    }
});

// Prevent default drag behavior on the whole page
document.addEventListener('dragover', (e) => {
    e.preventDefault();
});

document.addEventListener('drop', (e) => {
    e.preventDefault();
});
