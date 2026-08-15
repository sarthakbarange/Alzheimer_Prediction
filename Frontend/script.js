class AlzheimerDetectionApp {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.currentFile = null;
        this.currentImageData = null;
        this.currentResults = null;
    }

    initializeElements() {
        this.uploadPage = document.getElementById('uploadPage');
        this.resultsPage = document.getElementById('resultsPage');
        this.dropZone = document.getElementById('dropZone');
        this.imageUpload = document.getElementById('imageUpload');
        this.preview = document.getElementById('preview');
        this.uploadContent = document.getElementById('uploadContent');
        this.previewContainer = document.getElementById('previewContainer');
        this.fileName = document.getElementById('fileName');
        this.removeImageBtn = document.getElementById('removeImage');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.progressContainer = document.getElementById('progressContainer');
        this.progressBar = document.getElementById('progressBar');
        this.progressText = document.getElementById('progressText');
        this.resultContent = document.getElementById('resultContent');
    }

    setupEventListeners() {
        // File input change
        this.imageUpload.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop events
        this.dropZone.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.dropZone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.dropZone.addEventListener('drop', (e) => this.handleDrop(e));

        // Button events
        this.removeImageBtn.addEventListener('click', () => this.removeImage());
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());

        // Header upload button
        document.querySelector('header button').addEventListener('click', () => {
            this.imageUpload.click();
        });

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleDragOver(e) {
        this.dropZone.classList.add('border-blue-400', 'bg-blue-50');
    }

    handleDragLeave(e) {
        this.dropZone.classList.remove('border-blue-400', 'bg-blue-50');
    }

    handleDrop(e) {
        this.dropZone.classList.remove('border-blue-400', 'bg-blue-50');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    validateFile(file) {
        const maxSize = 20 * 1024 * 1024; // 20MB
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/dicom'];

        if (file.size > maxSize) {
            this.showNotification('File size must be less than 20MB', 'error');
            return false;
        }

        if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.dcm') && !file.name.toLowerCase().endsWith('.nii')) {
            this.showNotification('Please upload a valid image file (DICOM, NIfTI, JPG, or PNG)', 'error');
            return false;
        }

        return true;
    }

    processFile(file) {
        if (!this.validateFile(file)) {
            return;
        }

        this.currentFile = file;
        const reader = new FileReader();

        reader.onload = (e) => {
            this.currentImageData = e.target.result;
            this.preview.src = this.currentImageData;
            this.fileName.textContent = file.name;
            this.showPreview();
        };

        reader.readAsDataURL(file);
    }

    showPreview() {
        this.uploadContent.classList.add('hidden');
        this.previewContainer.classList.remove('hidden');
        this.dropZone.classList.add('border-green-400', 'bg-green-50');
        this.analyzeBtn.disabled = false;
    }

    removeImage() {
        this.currentFile = null;
        this.currentImageData = null;
        this.preview.src = '';
        this.imageUpload.value = '';
        
        this.uploadContent.classList.remove('hidden');
        this.previewContainer.classList.add('hidden');
        this.dropZone.classList.remove('border-green-400', 'bg-green-50');
        this.analyzeBtn.disabled = true;
    }

    async analyzeImage() {
        if (!this.currentFile) {
            this.showNotification('Please upload an MRI image first', 'error');
            return;
        }

        this.showProgress();

        const formData = new FormData();
        formData.append('file', this.currentFile);

        try {
            this.updateProgress(20);

            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            this.updateProgress(60);

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();
            this.updateProgress(100);

            setTimeout(() => {
                this.hideProgress();
                this.displayResults(data);
            }, 500);

        } catch (error) {
            this.hideProgress();
            this.showNotification('Analysis failed. Please try again.', 'error');
            console.error('Error:', error);
        }
    }

    showProgress() {
        this.progressContainer.classList.remove('hidden');
        this.analyzeBtn.disabled = true;
        this.analyzeBtn.classList.add('opacity-50', 'cursor-not-allowed');
        this.updateProgress(0);
    }

    hideProgress() {
        this.progressContainer.classList.add('hidden');
        this.analyzeBtn.disabled = false;
        this.analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }

    updateProgress(percentage) {
        this.progressBar.style.width = `${percentage}%`;
        this.progressText.textContent = `${percentage}%`;
    }

    displayResults(data) {
        this.currentResults = data;
        
        // Switch to results page
        this.uploadPage.classList.add('hidden');
        this.resultsPage.classList.remove('hidden');

        const prediction = data.prediction || 'Unknown';
        const confidence = Math.round((data.confidence || 0) * 100);
        const stageColor = this.getStageColor(prediction);
        const confidenceColor = confidence > 80 ? 'clinical-blue' : confidence > 60 ? 'calm-blue' : 'medical-blue';

        let gradcamSection = '';
        if (data.images && data.images.original && data.images.overlay) {
            gradcamSection = `
                <!-- Attention Heatmap Block -->
                <div class="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-clinical-blue/10">
                    <div class="flex items-center mb-6">
                        <div class="w-12 h-12 bg-gradient-to-br from-calm-blue/20 to-light-blue/20 rounded-lg mr-4 flex items-center justify-center">
                            <i class="fas fa-search-location text-white text-lg"></i>
                        </div>
                        <div>
                            <h3 class="text-2xl font-bold text-medical-blue">Attention Heatmap</h3>
                            <p class="text-gray-600">Visual focus regions for understanding</p>
                        </div>
                    </div>
                    
                    <div class="grid lg:grid-cols-2 gap-6">
                        <div class="text-center">
                            <h4 class="text-lg font-semibold text-medical-blue mb-3">Original MRI</h4>
                            <div class="bg-gradient-to-br from-light-blue/20 to-white/40 rounded-xl p-4 border border-clinical-blue/10">
                                <img src="data:image/png;base64,${data.images.original}" class="w-full h-64 object-contain rounded-lg">
                            </div>
                            <p class="text-sm text-gray-500 mt-2">Your original scan</p>
                        </div>
                        
                        <div class="text-center">
                            <h4 class="text-lg font-semibold text-medical-blue mb-3">Attention Overlay</h4>
                            <div class="bg-gradient-to-br from-soft-blue/20 to-white/40 rounded-xl p-4 border border-clinical-blue/10">
                                <img src="data:image/png;base64,${data.images.overlay}" class="w-full h-64 object-contain rounded-lg">
                            </div>
                            <p class="text-sm text-gray-500 mt-2">Combined visualization</p>
                        </div>
                    </div>
                    
                    <div class="mt-6 p-4 bg-gradient-to-r from-light-blue/30 to-soft-blue/30 rounded-xl border border-clinical-blue/20">
                        <p class="text-gray-700 leading-relaxed">
                            <strong>Understanding Your Results:</strong> The visualization shows which brain regions received focused attention during analysis. 
                            Warmer colors highlight areas that were most significant in the assessment, providing you with clear, 
                            understandable insights about your neurological health.
                        </p>
                    </div>
                </div>
            `;
        }

        this.resultContent.innerHTML = `
            <!-- Alzheimer's Stage Detection Results -->
            <div class="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-clinical-blue/10">
                <div class="flex items-center mb-6">
                    <div class="w-12 h-12 bg-gradient-to-br from-${confidenceColor}/20 to-${confidenceColor}/30 rounded-lg mr-4 flex items-center justify-center">
                        <i class="fas fa-brain text-white text-xl"></i>
                    </div>
                    <div>
                        <h3 class="text-2xl font-bold text-medical-blue">Alzheimer's Stage Detection</h3>
                        <p class="text-gray-600">MRI scan analysis results</p>
                    </div>
                </div>
                
                <div class="bg-gradient-to-br from-light-blue/30 to-soft-blue/30 rounded-xl p-6 border border-clinical-blue/10 mb-6">
                    <h4 class="text-xl font-semibold text-${stageColor} mb-3">Detected Stage: ${prediction}</h4>
                    <p class="text-lg text-gray-700 leading-relaxed mb-4">AI model confidence: ${confidence}%</p>
                </div>
                
                ${data.explanation ? `
                <div class="bg-white/50 rounded-xl p-6 border border-clinical-blue/10 mb-6">
                    <h5 class="text-lg font-semibold text-medical-blue mb-3">Medical Analysis:</h5>
                    <p class="text-gray-700 leading-relaxed mb-3">${data.explanation}</p>
                    ${data.attention_region ? `
                        <div class="bg-gradient-to-r from-light-blue/30 to-soft-blue/30 rounded-lg p-4 border border-clinical-blue/20">
                            <p class="text-gray-800 font-medium">
                                <strong>Region:</strong> ${data.attention_region}
                            </p>
                        </div>
                    ` : ''}
                </div>
                ` : ''}
                
                <div class="bg-white/50 rounded-xl p-6 border border-clinical-blue/10 mb-6">
                    <h5 class="text-lg font-semibold text-medical-blue mb-3">Class Probabilities:</h5>
                    <div class="space-y-2">
                        ${data.class_probabilities ? Object.entries(data.class_probabilities).map(([className, prob]) => 
                            `<div class="flex justify-between items-center">
                                <span class="text-gray-700">${className}:</span>
                                <span class="font-semibold">${Math.round(prob * 100)}%</span>
                            </div>`
                        ).join('') : 'No probability data available'}
                    </div>
                </div>
                
                <div class="mt-6 grid grid-cols-2 gap-4">
                    <div class="text-center p-4 bg-gradient-to-br from-light-blue/40 to-white/60 rounded-xl border border-clinical-blue/10">
                        <div class="text-2xl font-bold text-${stageColor}">${prediction}</div>
                        <div class="text-sm text-gray-600">Detected Stage</div>
                    </div>
                    <div class="text-center p-4 bg-gradient-to-br from-soft-blue/40 to-white/60 rounded-xl border border-clinical-blue/10">
                        <div class="text-2xl font-bold text-${confidenceColor}">${confidence}%</div>
                        <div class="text-sm text-gray-600">Confidence Score</div>
                    </div>
                </div>
            </div>

            ${gradcamSection}

            <!-- Action Buttons -->
            <div class="flex gap-4 justify-center">
                <button onclick="app.newAnalysis()" class="bg-gradient-to-r from-clinical-blue to-calm-blue hover:from-clinical-blue/90 hover:to-calm-blue/90 text-white px-8 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg">
                    <i class="fas fa-redo mr-2"></i>New Analysis
                </button>
                <button onclick="app.downloadReport()" class="bg-gradient-to-r from-clinical-blue to-light-blue hover:from-clinical-blue/90 hover:to-light-blue/90 text-white px-8 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg">
                    <i class="fas fa-download mr-2"></i>Download Report
                </button>
                <button onclick="app.shareResults()" class="bg-gradient-to-r from-calm-blue to-soft-blue hover:from-calm-blue/90 hover:to-soft-blue/90 text-white px-8 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg">
                    <i class="fas fa-share-alt mr-2"></i>Share Results
                </button>
            </div>
        `;
    }

    getStageColor(stage) {
        const colors = {
            'Non Demented': 'black',
            'Very Mild Demented': 'black',
            'Mild Demented': 'black',
            'Moderate Demented': 'black'
        };
        return colors[stage] || 'black';
    }

    newAnalysis() {
        // Reset and go back to upload page
        this.removeImage();
        this.resultsPage.classList.add('hidden');
        this.uploadPage.classList.remove('hidden');
        this.showNotification('Ready for new analysis', 'success');
    }

    downloadReport() {
        // Create a comprehensive report
        const reportData = {
            timestamp: new Date().toISOString(),
            imageData: this.currentImageData,
            results: this.currentResults
        };
        
        const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `alzheimer-analysis-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.showNotification('Report downloaded successfully', 'success');
    }

    shareResults() {
        // Create shareable link (in a real app, this would upload to a server)
        const shareData = {
            prediction: this.currentResults?.prediction,
            confidence: Math.round((this.currentResults?.confidence || 0) * 100),
            timestamp: new Date().toISOString()
        };
        
        if (navigator.share) {
            navigator.share({
                title: 'Alzheimer MRI Analysis Results',
                text: `Analysis Result: ${shareData.prediction} with ${shareData.confidence}% confidence`,
                url: window.location.href
            });
        } else {
            // Fallback: copy to clipboard
            navigator.clipboard.writeText(`Alzheimer Analysis: ${shareData.prediction} (${shareData.confidence}% confidence)`);
            this.showNotification('Results copied to clipboard', 'success');
        }
    }

    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.classList.add('fixed', 'top-4', 'right-4', 'bg-white', 'p-4', 'rounded-lg', 'shadow-lg', 'border', 'border-gray-200');
        notification.innerHTML = `
            <i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'check-circle'} mr-2"></i>
            ${message}
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// Initialize the app
const app = new AlzheimerDetectionApp();

// Initially disable analyze button
document.getElementById('analyzeBtn').disabled = true;
