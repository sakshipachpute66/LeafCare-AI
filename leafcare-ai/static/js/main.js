/**
 * LeafCare AI - Frontend Interactive Logic
 * Handles image upload, live previews, model inference AJAX,
 * history management, and disease wiki exploration.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- Global State ---
    let currentSelectedFile = null;
    let allDiseasesData = {};

    // --- DOM Elements ---
    const dropzone = document.getElementById('dropzone');
    const leafFileInput = document.getElementById('leafFileInput');
    const dropzoneIdle = document.getElementById('dropzoneIdle');
    const dropzonePreview = document.getElementById('dropzonePreview');
    const previewImage = document.getElementById('previewImage');
    const previewFileName = document.getElementById('previewFileName');
    const previewFileSize = document.getElementById('previewFileSize');
    const btnBrowseFile = document.getElementById('btnBrowseFile');
    const btnResetUpload = document.getElementById('btn-reset-upload');
    const btnPredict = document.getElementById('btn-predict');
    const uploadErrorAlert = document.getElementById('uploadErrorAlert');
    const uploadErrorMessage = document.getElementById('uploadErrorMessage');

    const resultsBlank = document.getElementById('resultsBlank');
    const resultsLoading = document.getElementById('resultsLoading');
    const resultsActive = document.getElementById('resultsActive');
    const resultStatusBadge = document.getElementById('result-status-badge');

    // Result fields
    const resPlantName = document.getElementById('res-plant-name');
    const resDiseaseName = document.getElementById('res-disease-name');
    const resConfidenceText = document.getElementById('res-confidence-text');
    const resConfidenceBar = document.getElementById('res-confidence-bar');
    const resPathogen = document.getElementById('res-pathogen');
    const resSeverity = document.getElementById('res-severity');
    const resSymptomsList = document.getElementById('res-symptoms-list');
    const resOrganicList = document.getElementById('res-organic-list');
    const resChemicalList = document.getElementById('res-chemical-list');
    const resPreventionList = document.getElementById('res-prevention-list');

    // History elements
    const historyTableBody = document.getElementById('historyTableBody');
    const histTotalCount = document.getElementById('hist-total-count');
    const histHealthyCount = document.getElementById('hist-healthy-count');
    const histInfectedCount = document.getElementById('hist-infected-count');
    const btnRefreshHistory = document.getElementById('btn-refresh-history');
    const btnClearHistory = document.getElementById('btn-clear-history');

    // Wiki elements
    const diseasesGrid = document.getElementById('diseasesGrid');
    const diseaseSearchInput = document.getElementById('diseaseSearchInput');
    const cropFilterButtons = document.getElementById('cropFilterButtons');
    const diseaseModal = document.getElementById('diseaseModal');
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    const modalCropBadge = document.getElementById('modalCropBadge');
    const modalDiseaseTitle = document.getElementById('modalDiseaseTitle');
    const modalBodyContent = document.getElementById('modalBodyContent');

    // Model status
    const modelStatusBadge = document.getElementById('model-status-badge');

    // =========================================================================
    // 1. Navigation & Mobile Menu
    // =========================================================================
    const menuToggle = document.getElementById('menuToggle');
    const navLinks = document.getElementById('navLinks');
    if (menuToggle && navLinks) {
        menuToggle.addEventListener('click', () => {
            navLinks.classList.toggle('open');
        });
    }

    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            if (navLinks.classList.contains('open')) {
                navLinks.classList.remove('open');
            }
        });
    });

    // =========================================================================
    // 2. Model Status Check
    // =========================================================================
    async function checkModelStatus() {
        try {
            const res = await fetch('/api/model-status');
            const data = await res.json();
            if (data.status === 'ready') {
                modelStatusBadge.className = 'badge-status ready';
                modelStatusBadge.innerHTML = `<i class="fa-solid fa-circle-check"></i> Model Online (${data.model_name})`;
            } else {
                modelStatusBadge.className = 'badge-status warning';
                modelStatusBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Model Pending Training`;
            }
        } catch (err) {
            modelStatusBadge.className = 'badge-status warning';
            modelStatusBadge.innerHTML = `<i class="fa-solid fa-server"></i> API Connecting...`;
        }
    }

    // =========================================================================
    // 3. File Upload, Drag & Drop, Validation & Preview
    // =========================================================================
    const btnBrowse = document.getElementById('btn-browse-file');
    if (btnBrowse) {
        btnBrowse.addEventListener('click', (e) => {
            e.stopPropagation();
            leafFileInput.click();
        });
    }

    dropzone.addEventListener('click', () => {
        if (!currentSelectedFile) {
            leafFileInput.click();
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    leafFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    function showError(message) {
        uploadErrorMessage.textContent = message;
        uploadErrorAlert.style.display = 'flex';
    }

    function clearError() {
        uploadErrorAlert.style.display = 'none';
        uploadErrorMessage.textContent = '';
    }

    function handleSelectedFile(file) {
        clearError();
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];

        if (!validTypes.includes(file.type.toLowerCase()) && !file.name.match(/\.(jpe?g|png|webp)$/i)) {
            showError('Invalid file format. Please upload a JPEG, PNG, or WEBP image.');
            return;
        }

        const maxSize = 10 * 1024 * 1024; // 10 MB
        if (file.size > maxSize) {
            showError('File size exceeds the 10 MB limit. Please select a smaller photo.');
            return;
        }

        currentSelectedFile = file;

        // Display preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewFileName.textContent = file.name;
            previewFileSize.textContent = formatBytes(file.size);

            dropzoneIdle.style.display = 'none';
            dropzonePreview.style.display = 'block';
            btnResetUpload.style.display = 'inline-flex';
            btnPredict.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetUploadState() {
        currentSelectedFile = null;
        leafFileInput.value = '';
        previewImage.src = '';
        dropzonePreview.style.display = 'none';
        dropzoneIdle.style.display = 'block';
        btnResetUpload.style.display = 'none';
        btnPredict.disabled = true;
        clearError();

        // Reset Results Panel
        resultsBlank.style.display = 'block';
        resultsLoading.style.display = 'none';
        resultsActive.style.display = 'none';
        resultStatusBadge.textContent = 'Awaiting Input';
        resultStatusBadge.className = 'badge';
    }

    btnResetUpload.addEventListener('click', resetUploadState);

    // =========================================================================
    // 4. Reference Sample Specimens Click Handling
    // =========================================================================
    document.querySelectorAll('.sample-pill').forEach(pill => {
        pill.addEventListener('click', async () => {
            const sampleKey = pill.getAttribute('data-sample');
            try {
                const res = await fetch(`/static/images/samples/${sampleKey}.jpg`);
                if (!res.ok) {
                    showError(`Sample image ${sampleKey} is not available.`);
                    return;
                }
                const blob = await res.blob();
                const file = new File([blob], `${sampleKey}.jpg`, { type: 'image/jpeg' });
                handleSelectedFile(file);
            } catch (err) {
                showError('Could not load sample leaf image.');
            }
        });
    });

    // =========================================================================
    // 5. Run Prediction & Diagnosis (Real Neural Inference)
    // =========================================================================
    btnPredict.addEventListener('click', async () => {
        if (!currentSelectedFile) {
            showError('Please upload or select a leaf image first.');
            return;
        }

        clearError();
        resultsBlank.style.display = 'none';
        resultsActive.style.display = 'none';
        resultsLoading.style.display = 'block';
        btnPredict.disabled = true;

        const formData = new FormData();
        formData.append('file', currentSelectedFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Diagnostic evaluation failed.');
            }

            // Populate Results UI
            displayDiagnosisResults(data);

            // Refresh historical logs & stats
            fetchPredictionHistory();
            fetchDashboardStats();
        } catch (error) {
            resultsLoading.style.display = 'none';
            resultsBlank.style.display = 'block';
            showError(error.message || 'An error occurred during diagnosis.');
        } finally {
            btnPredict.disabled = false;
        }
    });

    function displayDiagnosisResults(data) {
        resultsLoading.style.display = 'none';
        resultsActive.style.display = 'block';

        resPlantName.textContent = data.plant;
        resDiseaseName.textContent = data.disease;

        const confScore = parseFloat(data.confidence).toFixed(1);
        resConfidenceText.textContent = `${confScore}%`;
        resConfidenceBar.style.width = `${Math.min(100, Math.max(5, confScore))}%`;

        resPathogen.textContent = data.pathogen || 'Unknown / Not specified';
        resSeverity.textContent = data.severity || 'Normal';

        // Badge Status
        if (data.status === 'Healthy') {
            resultStatusBadge.textContent = 'Healthy Foliage';
            resultStatusBadge.className = 'badge badge-healthy';
            resDiseaseName.className = 'condition-highlight text-success';
        } else {
            resultStatusBadge.textContent = 'Disease Detected';
            resultStatusBadge.className = 'badge badge-diseased';
            resDiseaseName.className = 'condition-highlight text-danger';
        }

        // Populate tabs
        populateList(resSymptomsList, data.symptoms);
        populateList(resOrganicList, data.organic_treatment);
        populateList(resChemicalList, data.chemical_treatment);
        populateList(resPreventionList, data.prevention);
    }

    function populateList(element, items) {
        element.innerHTML = '';
        if (!items || items.length === 0) {
            element.innerHTML = '<li>No specific items documented.</li>';
            return;
        }
        items.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            element.appendChild(li);
        });
    }

    // Result tabs switching
    document.querySelectorAll('.diag-tab').forEach(tabBtn => {
        tabBtn.addEventListener('click', () => {
            document.querySelectorAll('.diag-tab').forEach(b => b.classList.remove('active'));
            tabBtn.classList.add('active');

            const tabId = tabBtn.getAttribute('data-tab');
            document.querySelectorAll('.diag-tab-content').forEach(content => {
                content.style.display = 'none';
                content.classList.remove('active');
            });

            const activeContent = document.getElementById(`tab-${tabId}`);
            if (activeContent) {
                activeContent.style.display = 'block';
                activeContent.classList.add('active');
            }
        });
    });

    // Print Diagnosis
    const btnPrintReport = document.getElementById('btn-print-report');
    if (btnPrintReport) {
        btnPrintReport.addEventListener('click', () => {
            window.print();
        });
    }

    // =========================================================================
    // 6. Disease Information Encyclopedia
    // =========================================================================
    async function loadDiseaseWiki() {
        try {
            const res = await fetch('/api/diseases');
            allDiseasesData = await res.json();
            renderDiseaseCards(allDiseasesData);
        } catch (err) {
            diseasesGrid.innerHTML = '<p class="text-center">Unable to load disease repository at this time.</p>';
        }
    }

    function renderDiseaseCards(diseases) {
        diseasesGrid.innerHTML = '';
        const keys = Object.keys(diseases);

        if (keys.length === 0) {
            diseasesGrid.innerHTML = '<p class="text-center py-4">No matching disease records found.</p>';
            return;
        }

        keys.forEach(key => {
            const item = diseases[key];
            const isHealthy = item.status === 'Healthy';
            const badgeClass = isHealthy ? 'badge-healthy' : 'badge-diseased';

            const card = document.createElement('div');
            card.className = 'disease-card card';
            card.innerHTML = `
                <div>
                    <div class="disease-card-header">
                        <div>
                            <span class="badge ${badgeClass}">${item.plant}</span>
                            <h3>${item.disease}</h3>
                        </div>
                    </div>
                    <div class="disease-card-body">
                        <p>${item.description || 'No description available.'}</p>
                        <div class="disease-card-symptoms">
                            <h5>Key Symptoms</h5>
                            <ul>
                                ${(item.symptoms || []).slice(0, 2).map(s => `<li>${s}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="disease-card-footer">
                    <small><strong>Pathogen:</strong> ${item.pathogen || 'N/A'}</small>
                    <button class="btn btn-outline btn-sm btn-view-disease" data-key="${key}">
                        <i class="fa-solid fa-circle-info"></i> Full Profile
                    </button>
                </div>
            `;
            diseasesGrid.appendChild(card);
        });

        // Attach modal listeners to cards
        document.querySelectorAll('.btn-view-disease').forEach(btn => {
            btn.addEventListener('click', () => {
                const diseaseKey = btn.getAttribute('data-key');
                openDiseaseModal(allDiseasesData[diseaseKey]);
            });
        });
    }

    // Disease Search Filter
    diseaseSearchInput.addEventListener('input', (e) => {
        filterDiseases(e.target.value.toLowerCase(), getActiveFilter());
    });

    // Crop Filter Pills
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            filterDiseases(diseaseSearchInput.value.toLowerCase(), btn.getAttribute('data-filter'));
        });
    });

    function getActiveFilter() {
        const activeBtn = document.querySelector('.filter-btn.active');
        return activeBtn ? activeBtn.getAttribute('data-filter') : 'all';
    }

    function filterDiseases(query, cropFilter) {
        const filtered = {};
        Object.keys(allDiseasesData).forEach(key => {
            const item = allDiseasesData[key];
            const matchesQuery = !query ||
                item.plant.toLowerCase().includes(query) ||
                item.disease.toLowerCase().includes(query) ||
                (item.description && item.description.toLowerCase().includes(query)) ||
                (item.symptoms && item.symptoms.some(s => s.toLowerCase().includes(query)));

            let matchesCrop = true;
            if (cropFilter !== 'all') {
                if (cropFilter === 'Healthy') {
                    matchesCrop = item.status === 'Healthy';
                } else {
                    matchesCrop = item.plant.toLowerCase() === cropFilter.toLowerCase();
                }
            }

            if (matchesQuery && matchesCrop) {
                filtered[key] = item;
            }
        });
        renderDiseaseCards(filtered);
    }

    function openDiseaseModal(disease) {
        modalCropBadge.textContent = disease.plant;
        modalDiseaseTitle.textContent = disease.disease;

        modalBodyContent.innerHTML = `
            <p class="mb-3"><strong>Overview:</strong> ${disease.description}</p>
            <div class="spec-pills mb-3">
                <div class="spec-pill">
                    <i class="fa-solid fa-biohazard"></i>
                    <div>
                        <small>Pathogen</small>
                        <strong>${disease.pathogen || 'N/A'}</strong>
                    </div>
                </div>
                <div class="spec-pill">
                    <i class="fa-solid fa-gauge-high"></i>
                    <div>
                        <small>Severity</small>
                        <strong>${disease.severity || 'Normal'}</strong>
                    </div>
                </div>
            </div>

            <h4 class="mt-4 mb-2"><i class="fa-solid fa-eye text-warning"></i> Key Symptoms</h4>
            <ul class="remedy-list">
                ${(disease.symptoms || []).map(s => `<li>${s}</li>`).join('')}
            </ul>

            <h4 class="mt-4 mb-2"><i class="fa-solid fa-seedling text-success"></i> Organic & Cultural Management</h4>
            <ul class="remedy-list">
                ${(disease.organic_treatment || []).map(s => `<li>${s}</li>`).join('')}
            </ul>

            <h4 class="mt-4 mb-2"><i class="fa-solid fa-flask-vial text-primary"></i> Chemical Treatment</h4>
            <ul class="remedy-list">
                ${(disease.chemical_treatment || []).map(s => `<li>${s}</li>`).join('')}
            </ul>

            <h4 class="mt-4 mb-2"><i class="fa-solid fa-shield-halved text-success"></i> Long-term Prevention</h4>
            <ul class="remedy-list">
                ${(disease.prevention || []).map(s => `<li>${s}</li>`).join('')}
            </ul>
        `;

        diseaseModal.style.display = 'flex';
    }

    modalCloseBtn.addEventListener('click', () => {
        diseaseModal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === diseaseModal) {
            diseaseModal.style.display = 'none';
        }
    });

    // =========================================================================
    // 7. Prediction History & SQLite Management
    // =========================================================================
    async function fetchPredictionHistory() {
        try {
            const res = await fetch('/api/history');
            const data = await res.json();
            renderHistoryTable(data.history || []);
            updateHistoryMetrics(data.history || []);
        } catch (err) {
            historyTableBody.innerHTML = `
                <tr><td colspan="8" class="text-center py-4">Unable to retrieve history logs.</td></tr>
            `;
        }
    }

    function renderHistoryTable(records) {
        historyTableBody.innerHTML = '';
        if (records.length === 0) {
            historyTableBody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center py-4" style="color: var(--slate-500);">
                        <i class="fa-solid fa-inbox"></i> No leaf predictions recorded yet. Scan a leaf above to log history.
                    </td>
                </tr>
            `;
            return;
        }

        records.forEach(rec => {
            const tr = document.createElement('tr');
            const isHealthy = rec.status === 'Healthy';
            const statusBadge = isHealthy 
                ? '<span class="badge badge-healthy">Healthy</span>' 
                : '<span class="badge badge-diseased">Diseased</span>';

            const confBadge = `<span style="font-weight:700; color:var(--forest-800);">${parseFloat(rec.confidence).toFixed(1)}%</span>`;

            tr.innerHTML = `
                <td><strong>#${rec.id}</strong></td>
                <td>
                    <img src="${rec.image_url}" class="thumb-preview" alt="Thumbnail" onerror="this.src='/static/images/placeholder-leaf.png'">
                </td>
                <td><strong>${rec.plant_name}</strong></td>
                <td>${rec.disease_name}</td>
                <td>${confBadge}</td>
                <td>${statusBadge}</td>
                <td><small style="color:var(--slate-500);">${rec.created_at}</small></td>
                <td>
                    <button class="btn btn-outline btn-sm btn-delete-history" data-id="${rec.id}" title="Delete Record">
                        <i class="fa-solid fa-trash-can text-danger"></i>
                    </button>
                </td>
            `;
            historyTableBody.appendChild(tr);
        });

        // Attach delete actions
        document.querySelectorAll('.btn-delete-history').forEach(btn => {
            btn.addEventListener('click', async () => {
                const id = btn.getAttribute('data-id');
                if (confirm(`Delete prediction record #${id}?`)) {
                    await deleteHistoryItem(id);
                }
            });
        });
    }

    function updateHistoryMetrics(records) {
        histTotalCount.textContent = records.length;
        const healthy = records.filter(r => r.status === 'Healthy').length;
        const infected = records.length - healthy;
        histHealthyCount.textContent = healthy;
        histInfectedCount.textContent = infected;
    }

    async function deleteHistoryItem(id) {
        try {
            const res = await fetch(`/api/history/${id}`, { method: 'DELETE' });
            if (res.ok) {
                fetchPredictionHistory();
                fetchDashboardStats();
            }
        } catch (err) {
            alert('Failed to delete history item.');
        }
    }

    btnRefreshHistory.addEventListener('click', fetchPredictionHistory);

    btnClearHistory.addEventListener('click', async () => {
        if (confirm('Are you sure you want to clear all prediction history?')) {
            try {
                const res = await fetch('/api/history/clear', { method: 'POST' });
                if (res.ok) {
                    fetchPredictionHistory();
                    fetchDashboardStats();
                }
            } catch (err) {
                alert('Could not clear history.');
            }
        }
    });

    // =========================================================================
    // 8. Dashboard Statistics
    // =========================================================================
    async function fetchDashboardStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            document.getElementById('stat-total-scans').textContent = data.total_scans || 0;
        } catch (e) {
            // silent fail
        }
    }

    // --- Initial Bootstrapping ---
    checkModelStatus();
    loadDiseaseWiki();
    fetchPredictionHistory();
    fetchDashboardStats();
});
