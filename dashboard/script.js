const API_URL = "http://127.0.0.1:8000";

let currentChart = null;

document.addEventListener('DOMContentLoaded', () => {
    initChart();
    setupUploader();
});

function initChart(chartData = { labels: [], datasets: [] }) {
    const ctx = document.getElementById('probabilityChart').getContext('2d');
    if (currentChart) {
        currentChart.destroy();
    }
    
    currentChart = new Chart(ctx, {
        type: 'bar',
        data: chartData,
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#8ba0b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#8ba0b8' }
                }
            }
        }
    });
}

function setupUploader() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const imageWrapper = document.getElementById('image-wrapper');
    const previewImage = document.getElementById('preview-image');
    const heatmapImage = document.getElementById('heatmap-image');
    const toggleHeatmapBtn = document.getElementById('toggle-heatmap');
    const resetBtn = document.getElementById('reset-upload');
    const spinner = document.getElementById('loading-spinner');

    uploadPlaceholder.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.remove('dragover'), false);
    });

    uploadZone.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    resetBtn.addEventListener('click', () => {
        uploadPlaceholder.classList.remove('hidden');
        imageWrapper.classList.add('hidden');
        heatmapImage.classList.add('hidden');
        toggleHeatmapBtn.classList.add('hidden');
        heatmapImage.src = "";
        previewImage.src = "";
        fileInput.value = "";
        resetUI();
    });

    let heatmapVisible = false;
    toggleHeatmapBtn.addEventListener('click', () => {
        heatmapVisible = !heatmapVisible;
        if(heatmapVisible) {
            heatmapImage.classList.remove('hidden');
            toggleHeatmapBtn.innerText = "Hide Heatmap";
        } else {
            heatmapImage.classList.add('hidden');
            toggleHeatmapBtn.innerText = "Show Heatmap";
        }
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    uploadPlaceholder.classList.add('hidden');
                    imageWrapper.classList.remove('hidden');
                    heatmapImage.classList.add('hidden');
                    heatmapVisible = false;
                    toggleHeatmapBtn.classList.add('hidden');
                    toggleHeatmapBtn.innerText = "Show Heatmap";
                    
                    analyzeImage(file);
                }
            }
        }
    }

    async function analyzeImage(file) {
        spinner.classList.remove('hidden');
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(`${API_URL}/predict-xray-multidisease`, {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error("API Error");

            const data = await res.json();
            
            updateDashboard(data);
            
            if (data.heatmap) {
                heatmapImage.src = "data:image/jpeg;base64," + data.heatmap;
                toggleHeatmapBtn.classList.remove('hidden');
                
                // Show heatmap automatically for better UX
                heatmapVisible = true;
                heatmapImage.classList.remove('hidden');
                toggleHeatmapBtn.innerText = "Hide Heatmap";
            }

        } catch (error) {
            console.error("Analysis Failed:", error);
            alert("Failed to analyze image. Ensure backend is running.");
        } finally {
            spinner.classList.add('hidden');
        }
    }
}

function resetUI() {
    document.querySelector('.confidence-list').innerHTML = "";
    document.querySelector('.findings-content').innerHTML = `
        <div class="finding-block">
            <p>Awaiting upload...</p>
        </div>
    `;
    initChart();
}

function updateDashboard(data) {
    const listContainer = document.querySelector('.confidence-list');
    listContainer.innerHTML = "";
    
    // Sort predictions
    const predictions = Object.entries(data.predictions).sort((a, b) => b[1] - a[1]);
    
    let chartLabels = [];
    let chartDataPoints = [];
    let chartBackgrounds = [];
    
    const colors = ['#00f0ff', '#a855f7', '#ef4444', '#3b82f6', '#f59e0b', '#10b981', '#6366f1', '#64748b'];

    let highestDis = null;

    predictions.forEach(([disease, prob], idx) => {
        const percentage = (prob * 100).toFixed(1);
        let level = "Low";
        if (prob > 0.7) level = "High";
        else if (prob > 0.4) level = "Med";
        
        if(idx === 0) highestDis = { disease, prob, level };

        // Only show top 4 in the list for space
        if (idx < 4) {
            const color = colors[idx % colors.length];
            
            listContainer.innerHTML += `
                <div class="confidence-item">
                    <div class="conf-head">
                        <span>${disease.toUpperCase()}</span>
                        <span class="conf-val" style="color: ${color}">${percentage}% <span class="badge">[${level}]</span></span>
                    </div>
                    <div class="progress-track"><div class="progress-fill" style="width: ${percentage}%; background-color: ${color}; box-shadow: 0 0 10px ${color}"></div></div>
                </div>
            `;
        }
        
        chartLabels.push(disease);
        chartDataPoints.push(percentage);
        chartBackgrounds.push(colors[idx % colors.length]);
    });

    initChart({
        labels: chartLabels,
        datasets: [{
            data: chartDataPoints,
            backgroundColor: chartBackgrounds,
            borderRadius: 4
        }]
    });
    
    // Update findings
    const findingsDiv = document.querySelector('.findings-content');
    findingsDiv.innerHTML = `
        <div class="node-icon">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--cyan)" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M12 2v7M12 15v7"></path></svg>
        </div>
        <div class="finding-block">
            <h4>Primary Automated Detection</h4>
            <p>Highest confidence: <strong>${highestDis.disease}</strong> at ${(highestDis.prob*100).toFixed(1)}%</p>
        </div>
        <div class="finding-block">
            <h4>AI Risk Level</h4>
            <p style="color: ${highestDis.prob > 0.5 ? 'var(--red)' : 'var(--cyan)'}">
                ${highestDis.prob > 0.5 ? 'Action Required. Elevated probability of pathological findings.' : 'Normal/Low probability of severe abnormalities.'}
            </p>
        </div>
    `;
}
