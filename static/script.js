/**
 * AI-Based Water Quality Detection
 * Frontend Dashboard Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // Application State
    let currentSelectedFile = null;
    let phTrendChartInstance = null;
    let distributionChartInstance = null;

    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const dropPrompt = document.getElementById('drop-zone-prompt');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const previewInfo = document.getElementById('preview-info');
    const btnRemovePreview = document.getElementById('btn-remove-preview');
    const btnAnalyze = document.getElementById('btn-analyze');
    const loadingState = document.getElementById('loading-state');
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultContent = document.getElementById('result-content');

    // Result DOM Elements
    const phCircle = document.getElementById('ph-circle');
    const resultPh = document.getElementById('result-ph');
    const resultBadge = document.getElementById('result-badge');
    const resultConfidence = document.getElementById('result-confidence');
    const resultConfidenceFill = document.getElementById('result-confidence-fill');
    const featH = document.getElementById('feat-h');
    const featS = document.getElementById('feat-s');
    const featV = document.getElementById('feat-v');
    const recCard = document.getElementById('recommendation-card');
    const recTitle = document.getElementById('rec-title');
    const recDesc = document.getElementById('rec-desc');
    const resultTime = document.getElementById('result-time');

    // Stats Elements
    const statTotal = document.getElementById('stat-total');
    const statLatest = document.getElementById('stat-latest');
    const statLatestBadge = document.getElementById('stat-latest-badge');
    const statStatus = document.getElementById('stat-status');
    const statAvg = document.getElementById('stat-avg');

    // History Table & Buttons
    const historyTableBody = document.getElementById('history-table-body');
    const refreshTableBtn = document.getElementById('refresh-table-btn');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const refreshAllBtn = document.getElementById('refresh-all-btn');
    const sampleButtons = document.querySelectorAll('.btn-sample');

    // ---------------------------------------------------------
    // 1. FILE UPLOAD & DRAG/DROP HANDLING
    // ---------------------------------------------------------
    
    // Trigger file input when clicking drop zone (unless clicking remove button)
    dropZone.addEventListener('click', (e) => {
        if (!e.target.closest('#btn-remove-preview')) {
            fileInput.click();
        }
    });

    // Drag and Drop Events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    btnRemovePreview.addEventListener('click', (e) => {
        e.stopPropagation();
        resetFileSelection();
    });

    function handleSelectedFile(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp', 'image/bmp'];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp|bmp)$/i)) {
            showToast('Please select a valid image file (JPG, JPEG, or PNG).', 'error');
            return;
        }

        // Validate size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            showToast('File size must be less than 10MB.', 'error');
            return;
        }

        currentSelectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            const sizeKb = Math.round(file.size / 1024);
            previewInfo.textContent = `${file.name} (${sizeKb} KB)`;
            dropPrompt.style.display = 'none';
            previewContainer.style.display = 'flex';
            btnAnalyze.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetFileSelection() {
        currentSelectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        previewContainer.style.display = 'none';
        dropPrompt.style.display = 'block';
        btnAnalyze.disabled = true;
    }

    // ---------------------------------------------------------
    // 2. QUICK DEMO SAMPLES LOADER
    // ---------------------------------------------------------
    sampleButtons.forEach(btn => {
        btn.addEventListener('click', async () => {
            const sampleFile = btn.getAttribute('data-sample');
            const sampleName = btn.getAttribute('data-name');
            try {
                showToast(`Loading demo sample: ${sampleName}...`, 'info');
                const response = await fetch(`/samples/${sampleFile}`);
                if (!response.ok) throw new Error('Sample not found');
                const blob = await response.blob();
                const file = new File([blob], sampleFile, { type: 'image/png' });
                handleSelectedFile(file);
            } catch (err) {
                showToast(`Could not load sample: ${err.message}`, 'error');
            }
        });
    });

    // ---------------------------------------------------------
    // 3. ANALYZE WATER QUALITY (API CALL)
    // ---------------------------------------------------------
    btnAnalyze.addEventListener('click', async () => {
        if (!currentSelectedFile) {
            showToast('Please select or drop an image first.', 'error');
            return;
        }

        // Set Loading State
        btnAnalyze.disabled = true;
        loadingState.style.display = 'block';
        resultPlaceholder.style.display = 'none';
        resultContent.style.display = 'none';

        const formData = new FormData();
        formData.append('file', currentSelectedFile);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.detail || 'Analysis failed. Please check the image and model.');
            }

            // Display Results
            displayAnalysisResult(data);
            showToast(`Analysis complete! Predicted pH: ${data.predicted_ph}`, 'success');

            // Refresh Dashboard & History
            fetchStats();
            fetchHistory();
        } catch (error) {
            showToast(`Error: ${error.message}`, 'error');
            resultPlaceholder.style.display = 'block';
        } finally {
            loadingState.style.display = 'none';
            btnAnalyze.disabled = false;
        }
    });

    function displayAnalysisResult(data) {
        const ph = data.predicted_ph;
        const classification = data.classification;
        const confidence = data.confidence;

        resultPh.textContent = ph.toFixed(2);
        resultBadge.textContent = classification;
        resultConfidence.textContent = `${confidence.toFixed(1)}%`;
        resultConfidenceFill.style.width = `${Math.min(confidence, 100)}%`;
        resultTime.textContent = `Analyzed: ${data.created_at || 'Just now'}`;

        featH.textContent = data.mean_h.toFixed(1);
        featS.textContent = data.mean_s.toFixed(1);
        featV.textContent = data.mean_v.toFixed(1);

        // Reset classes
        phCircle.className = 'ph-circle-display';
        resultBadge.className = 'classification-badge';
        recCard.className = 'recommendation-card';

        // Styling based on water status
        if (classification.toLowerCase().includes('acidic')) {
            phCircle.classList.add('status-acidic');
            resultBadge.classList.add('badge-acidic');
            recCard.classList.add('rec-acidic');
            recTitle.textContent = 'Warning: Acidic Water';
            recDesc.textContent = `pH ${ph.toFixed(2)} is below the safe threshold (< 6.5). Acidic water may corrode plumbing, leach heavy metals, and cause a sour/metallic taste. Neutralization treatment is recommended.`;
        } else if (classification.toLowerCase().includes('safe') || classification.toLowerCase().includes('neutral')) {
            phCircle.classList.add('status-safe');
            resultBadge.classList.add('badge-safe');
            recTitle.textContent = 'Safe & Potable Water';
            recDesc.textContent = `pH ${ph.toFixed(2)} is within the ideal drinking water range (6.5 to 8.5) as defined by EPA/WHO standards. It is safe for domestic use and human consumption.`;
        } else {
            phCircle.classList.add('status-alkaline');
            resultBadge.classList.add('badge-alkaline');
            recCard.classList.add('rec-alkaline');
            recTitle.textContent = 'Caution: Alkaline Water';
            recDesc.textContent = `pH ${ph.toFixed(2)} exceeds the standard neutral threshold (> 8.5). High alkalinity may cause scale buildup in pipes, reduce disinfection efficacy, and yield a bitter taste.`;
        }

        resultPlaceholder.style.display = 'none';
        resultContent.style.display = 'flex';
    }

    // ---------------------------------------------------------
    // 4. FETCH DASHBOARD STATS
    // ---------------------------------------------------------
    async function fetchStats() {
        try {
            const response = await fetch('/stats');
            const data = await response.json();
            if (data.success) {
                statTotal.textContent = data.total_analyses;
                statAvg.textContent = data.average_ph ? data.average_ph.toFixed(2) : '--';
                
                if (data.latest_ph !== null && data.latest_ph !== undefined) {
                    statLatest.textContent = data.latest_ph.toFixed(2);
                    statLatestBadge.textContent = data.latest_classification;
                    statLatestBadge.className = 'stat-badge';
                    
                    if (data.latest_classification.includes('Acidic')) {
                        statLatestBadge.classList.add('badge-acidic-sm');
                        statStatus.textContent = 'Acidic';
                    } else if (data.latest_classification.includes('Safe')) {
                        statLatestBadge.classList.add('badge-safe-sm');
                        statStatus.textContent = 'Safe / Neutral';
                    } else {
                        statLatestBadge.classList.add('badge-alkaline-sm');
                        statStatus.textContent = 'Alkaline';
                    }
                } else {
                    statLatest.textContent = '--';
                    statLatestBadge.textContent = 'No tests yet';
                    statStatus.textContent = '--';
                }

                // Update charts with stats
                updateDistributionChart(data.acidic_count, data.safe_count, data.alkaline_count);
            }
        } catch (err) {
            console.error('Failed to fetch stats:', err);
        }
    }

    // ---------------------------------------------------------
    // 5. FETCH HISTORY & UPDATE TABLE & LINE CHART
    // ---------------------------------------------------------
    async function fetchHistory() {
        try {
            const response = await fetch('/history?limit=30');
            const data = await response.json();
            
            if (data.success) {
                renderHistoryTable(data.records);
                updateTrendChart(data.records);
            }
        } catch (err) {
            console.error('Failed to fetch history:', err);
            historyTableBody.innerHTML = `<tr><td colspan="8" class="text-center empty-msg">Error loading history.</td></tr>`;
        }
    }

    function renderHistoryTable(records) {
        if (!records || records.length === 0) {
            historyTableBody.innerHTML = `<tr><td colspan="8" class="text-center empty-msg">No analysis records found. Upload an image to get started!</td></tr>`;
            return;
        }

        historyTableBody.innerHTML = '';
        records.forEach(r => {
            const tr = document.createElement('tr');
            
            let badgeClass = 'badge-safe-sm';
            if (r.classification.toLowerCase().includes('acidic')) badgeClass = 'badge-acidic-sm';
            else if (r.classification.toLowerCase().includes('alkaline')) badgeClass = 'badge-alkaline-sm';

            tr.innerHTML = `
                <td><strong>#${r.id}</strong></td>
                <td>${r.created_at}</td>
                <td>
                    <a href="${r.file_url}" target="_blank" title="View uploaded image">
                        <img src="${r.file_url}" alt="Strip" class="thumbnail-img">
                    </a>
                </td>
                <td><strong style="font-family: var(--font-mono); font-size: 14px;">${r.predicted_ph.toFixed(2)}</strong></td>
                <td><span class="badge-sm ${badgeClass}">${r.classification}</span></td>
                <td>${r.confidence.toFixed(1)}%</td>
                <td style="font-family: var(--font-mono); font-size: 12px; color: var(--text-secondary);">
                    H:${r.mean_h} S:${r.mean_s} V:${r.mean_v}
                </td>
                <td>
                    <button class="btn-delete-row" data-id="${r.id}" title="Delete record">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </td>
            `;

            // Delete action
            tr.querySelector('.btn-delete-row').addEventListener('click', async (e) => {
                const id = e.currentTarget.getAttribute('data-id');
                if (confirm(`Delete record #${id}?`)) {
                    await deleteRecord(id);
                }
            });

            historyTableBody.appendChild(tr);
        });
    }

    async function deleteRecord(id) {
        try {
            const res = await fetch(`/history/${id}`, { method: 'DELETE' });
            const data = await res.json();
            if (data.success) {
                showToast(`Record #${id} deleted.`, 'success');
                fetchStats();
                fetchHistory();
            }
        } catch (err) {
            showToast(`Could not delete record: ${err.message}`, 'error');
        }
    }

    clearHistoryBtn.addEventListener('click', async () => {
        if (confirm('Are you sure you want to clear all analysis records from the SQLite database?')) {
            try {
                const res = await fetch('/history/clear', { method: 'DELETE' });
                const data = await res.json();
                if (data.success) {
                    showToast('All history cleared.', 'success');
                    fetchStats();
                    fetchHistory();
                }
            } catch (err) {
                showToast(`Error clearing history: ${err.message}`, 'error');
            }
        }
    });

    refreshTableBtn.addEventListener('click', () => {
        fetchHistory();
        showToast('History refreshed.', 'info');
    });

    refreshAllBtn.addEventListener('click', () => {
        fetchStats();
        fetchHistory();
        showToast('Dashboard data refreshed.', 'info');
    });

    // ---------------------------------------------------------
    // 6. CHART.JS VISUALIZATIONS
    // ---------------------------------------------------------
    function initCharts() {
        // pH Timeline Trend Chart
        const ctxTrend = document.getElementById('phTrendChart').getContext('2d');
        phTrendChartInstance = new Chart(ctxTrend, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Predicted pH',
                        data: [],
                        borderColor: '#00adb5',
                        backgroundColor: 'rgba(0, 173, 181, 0.15)',
                        borderWidth: 3,
                        pointBackgroundColor: '#0b192c',
                        pointBorderColor: '#00adb5',
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: 'Safe Max (8.5)',
                        data: [],
                        borderColor: '#10b981',
                        borderDash: [5, 5],
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'Safe Min (6.5)',
                        data: [],
                        borderColor: '#10b981',
                        borderDash: [5, 5],
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: 0,
                        max: 14,
                        title: { display: true, text: 'pH Value (0 - 14)' },
                        grid: { color: '#e2e8f0' }
                    },
                    x: {
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { boxWidth: 12, font: { family: 'Plus Jakarta Sans', size: 11 } }
                    }
                }
            }
        });

        // Distribution Chart
        const ctxDist = document.getElementById('distributionChart').getContext('2d');
        distributionChartInstance = new Chart(ctxDist, {
            type: 'doughnut',
            data: {
                labels: ['Acidic (<6.5)', 'Safe / Neutral (6.5-8.5)', 'Alkaline (>8.5)'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#ef4444', '#10b981', '#3b82f6'],
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { boxWidth: 12, font: { family: 'Plus Jakarta Sans', size: 11 } }
                    }
                },
                cutout: '65%'
            }
        });
    }

    function updateTrendChart(records) {
        if (!phTrendChartInstance) return;

        // Clone and reverse so oldest is on left, newest on right
        const chronRecords = [...records].reverse();
        const labels = chronRecords.map(r => r.created_at.split(' ')[1] || r.created_at);
        const dataValues = chronRecords.map(r => r.predicted_ph);
        const maxSafe = new Array(labels.length).fill(8.5);
        const minSafe = new Array(labels.length).fill(6.5);

        phTrendChartInstance.data.labels = labels;
        phTrendChartInstance.data.datasets[0].data = dataValues;
        phTrendChartInstance.data.datasets[1].data = maxSafe;
        phTrendChartInstance.data.datasets[2].data = minSafe;
        phTrendChartInstance.update();
    }

    function updateDistributionChart(acidic, safe, alkaline) {
        if (!distributionChartInstance) return;
        distributionChartInstance.data.datasets[0].data = [acidic || 0, safe || 0, alkaline || 0];
        distributionChartInstance.update();
    }

    // ---------------------------------------------------------
    // 7. TOAST NOTIFICATIONS & NAVIGATION
    // ---------------------------------------------------------
    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        let icon = '<i class="fa-solid fa-circle-info"></i>';
        if (type === 'success') icon = '<i class="fa-solid fa-circle-check"></i>';
        if (type === 'error') icon = '<i class="fa-solid fa-triangle-exclamation"></i>';

        toast.innerHTML = `${icon} <span>${message}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = '0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    // Smooth Navigation Highlight
    const navLinks = document.querySelectorAll('.nav-link');
    window.addEventListener('scroll', () => {
        let currentSection = '';
        document.querySelectorAll('.content-section').forEach(section => {
            const sectionTop = section.offsetTop - 120;
            if (window.scrollY >= sectionTop) {
                currentSection = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('data-section') === currentSection) {
                link.classList.add('active');
            }
        });
    });

    // ---------------------------------------------------------
    // 8. INITIALIZATION
    // ---------------------------------------------------------
    initCharts();
    fetchStats();
    fetchHistory();
});
