/* ═══════════════════════════════════════════════════════════════════════
   NeuroScan AI – app.js
   Brain Tumor Detection System – Core Application Logic

   HOW THE AI MODEL WORKS (currently SIMULATED):
   ─────────────────────────────────────────────
   Currently, this app uses a SIMULATED AI prediction engine.
   When you have a real trained model, you replace the `simulateAIAnalysis()`
   function (or the `callRealModelAPI()` function stub) with your actual
   model call – either:

   A) TensorFlow.js (run model locally in browser)
      ► Load a .h5 / SavedModel converted to TF.js format
      ► See: model_integration.md for instructions

   B) Python Backend API (Flask / FastAPI)
      ► Send image to your Python server
      ► Server loads model, runs prediction, returns JSON result

   C) Cloud API (Google Vertex AI, AWS SageMaker, Azure ML)
      ► Send base64 image to your cloud endpoint
      ► Get prediction response

   See the `callRealModelAPI()` function below for the integration point.
═══════════════════════════════════════════════════════════════════════ */

'use strict';

// ──────────────────────────────────────────────────────────
// CONFIGURATION
// ──────────────────────────────────────────────────────────
const CONFIG = {
  // Set to true when you have a real backend/model API
  USE_REAL_MODEL: false,

  // URL of your Python/Flask backend or cloud endpoint
  // Only used when USE_REAL_MODEL = true
  MODEL_API_URL: 'http://localhost:5000/predict',

  // Analysis simulation delay (ms) — only for demo mode
  SIM_DELAY: 3000,
};

// ──────────────────────────────────────────────────────────
// TUMOR DATABASE
// Known tumor classes with medical details
// ──────────────────────────────────────────────────────────
const TUMOR_DATABASE = {
  glioma: {
    name: 'Glioma',
    fullName: 'Glioma (Glial Cell Tumor)',
    levels: ['Grade I (Benign)', 'Grade II (Low-Grade)', 'Grade III (Anaplastic)', 'Grade IV (Glioblastoma)'],
    levelKeys: ['low', 'moderate', 'high', 'critical'],
    description: 'Gliomas originate from glial cells in the brain or spine. They range from slow-growing (Grade I/II) to aggressive (Grade III/IV — Glioblastoma Multiforme).',
    observations: [
      'Irregular, infiltrative mass with ill-defined margins detected.',
      'Hyperintense signal noted on T2/FLAIR sequences.',
      'Significant surrounding edema observed.',
      'Mass effect on adjacent brain structures present.',
    ],
    recommendations: [
      'Immediate neurosurgical consultation recommended.',
      'MRI spectroscopy for metabolite profiling.',
      'Stereotactic biopsy for histological confirmation.',
      'Genetic testing (IDH mutation, MGMT methylation) advised.',
    ],
  },
  meningioma: {
    name: 'Meningioma',
    fullName: 'Meningioma (Meningeal Tumor)',
    levels: ['Grade I (Benign)', 'Grade II (Atypical)', 'Grade III (Anaplastic)'],
    levelKeys: ['low', 'moderate', 'high'],
    description: 'Meningiomas arise from the meninges (brain/spinal cord covering membranes). Most are benign (Grade I) and grow slowly; higher grades are rare but aggressive.',
    observations: [
      'Extra-axial, well-circumscribed, dural-based mass identified.',
      'Homogeneous contrast enhancement with dural tail sign.',
      'No significant surrounding edema noted.',
      'Possible calcification within the mass.',
    ],
    recommendations: [
      'Neurosurgical evaluation for resection candidacy.',
      'Serial MRI imaging every 6 months for Grade I.',
      'Radiosurgery (Gamma Knife) may be considered.',
      'Annual follow-up imaging post-resection.',
    ],
  },
  pituitary: {
    name: 'Pituitary Adenoma',
    fullName: 'Pituitary Adenoma (Pituitary Tumor)',
    levels: ['Microadenoma (<10mm)', 'Macroadenoma (≥10mm)', 'Giant Adenoma (>40mm)'],
    levelKeys: ['low', 'moderate', 'high'],
    description: 'Pituitary adenomas are benign tumors of the pituitary gland. They may cause hormonal imbalances (functioning) or grow to compress surrounding structures (non-functioning).',
    observations: [
      'Sellar and suprasellar mass identified.',
      'Optic chiasm compression possible at this size.',
      'Heterogeneous signal intensity noted within the mass.',
      'Cavernous sinus involvement requires evaluation.',
    ],
    recommendations: [
      'Endocrinology referral for hormonal assessment.',
      'Visual field testing (perimetry) recommended.',
      'Transsphenoidal surgical resection evaluation.',
      'Hormone replacement therapy may be needed post-op.',
    ],
  },
  no_tumor: {
    name: 'No Tumor Detected',
    fullName: 'No Tumor Detected — Normal Scan',
    levels: ['Normal'],
    levelKeys: ['low'],
    description: 'AI analysis did not detect any significant brain tumor or mass lesion in the uploaded MRI scan. Routine monitoring may still be advised based on clinical symptoms.',
    observations: [
      'No intracranial mass lesion identified.',
      'Normal grey and white matter differentiation.',
      'No significant midline shift or mass effect.',
      'Ventricular system appears symmetric and normal.',
    ],
    recommendations: [
      'Clinical correlation with presenting symptoms advised.',
      'Routine annual MRI if neurological symptoms persist.',
      'EEG may be considered for seizure evaluation.',
      'Continue routine neurological follow-up.',
    ],
  },
};

// ──────────────────────────────────────────────────────────
// STATE
// ──────────────────────────────────────────────────────────
const state = {
  mriFile: null,
  mriDataURL: null,
  analysisResult: null,
  patientSaved: false,
};

// ──────────────────────────────────────────────────────────
// DOM ELEMENTS
// ──────────────────────────────────────────────────────────
const uploadZone     = document.getElementById('uploadZone');
const mriInput       = document.getElementById('mriInput');
const uploadContent  = document.getElementById('uploadContent');
const uploadPreview  = document.getElementById('uploadPreview');
const previewImg     = document.getElementById('previewImg');
const previewName    = document.getElementById('previewName');
const changeFile     = document.getElementById('changeFile');
const analyzeBtn     = document.getElementById('analyzeBtn');
const analysisPanel  = document.getElementById('analysisPanel');
const analysisStatus = document.getElementById('analysisStatus');
const progressBar    = document.getElementById('progressBar');
const resultGrid     = document.getElementById('resultGrid');
const tumorName      = document.getElementById('tumorName');
const tumorLevel     = document.getElementById('tumorLevel');
const confidenceScore = document.getElementById('confidenceScore');
const aiObservations = document.getElementById('aiObservations');
const severityBar    = document.getElementById('severityBar');

const patientForm    = document.getElementById('patientForm');
const formSaved      = document.getElementById('formSaved');

const generateReportBtn = document.getElementById('generateReportBtn');
const chkScanIcon    = document.getElementById('chk-scan-icon');
const chkPatientIcon = document.getElementById('chk-patient-icon');
const chkScan        = document.getElementById('chk-scan');
const chkPatient     = document.getElementById('chk-patient');

const reportModal    = document.getElementById('reportModal');
const reportPreview  = document.getElementById('reportPreview');
const closeModal     = document.getElementById('closeModal');
const closeModalBtn  = document.getElementById('closeModalBtn');
const downloadReportBtn = document.getElementById('downloadReportBtn');
const printReportBtn = document.getElementById('printReportBtn');
const toast          = document.getElementById('toast');

// ──────────────────────────────────────────────────────────
// UPLOAD HANDLING
// ──────────────────────────────────────────────────────────
uploadZone.addEventListener('click', () => {
  if (!uploadPreview.style.display || uploadPreview.style.display === 'none') {
    mriInput.click();
  }
});

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('dragover');
});
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    handleFileSelect(file);
  } else {
    showToast('Please upload a valid image file.', 'error');
  }
});

mriInput.addEventListener('change', () => {
  if (mriInput.files[0]) handleFileSelect(mriInput.files[0]);
});

changeFile.addEventListener('click', (e) => {
  e.stopPropagation();
  mriInput.click();
});

function handleFileSelect(file) {
  if (file.size > 20 * 1024 * 1024) {
    showToast('File size exceeds 20MB limit.', 'error');
    return;
  }
  state.mriFile = file;
  state.analysisResult = null;

  const reader = new FileReader();
  reader.onload = (e) => {
    state.mriDataURL = e.target.result;
    previewImg.src = state.mriDataURL;
    previewName.textContent = file.name;
    uploadContent.style.display = 'none';
    uploadPreview.style.display = 'flex';
    analysisPanel.style.display = 'none';
    resultGrid.style.display = 'none';
    analyzeBtn.disabled = false;
    updateChecklist();
    showToast('MRI image loaded successfully!', 'success');
  };
  reader.readAsDataURL(file);
}

// ──────────────────────────────────────────────────────────
// AI ANALYSIS
// ──────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!state.mriFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span> Analyzing…';
  analysisPanel.style.display = 'block';
  resultGrid.style.display = 'none';
  analysisStatus.textContent = 'Initializing AI model…';
  progressBar.style.width = '0%';

  try {
    let result;
    if (CONFIG.USE_REAL_MODEL) {
      result = await callRealModelAPI(state.mriFile);
    } else {
      result = await simulateAIAnalysis();
    }
    state.analysisResult = result;
    displayResult(result);
    updateChecklist();
    showToast('Analysis complete!', 'success');
  } catch (err) {
    analysisStatus.textContent = 'Analysis failed. Please try again.';
    progressBar.style.width = '0%';
    showToast('Analysis failed: ' + err.message, 'error');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<span class="btn-icon">🔄</span> Re-Analyze';
  }
});

// ══════════════════════════════════════════════════════════
// ★ REAL MODEL INTEGRATION POINT ★
//
// Replace this function with your actual model API call.
// Expected return format:
// {
//   tumorType: 'glioma' | 'meningioma' | 'pituitary' | 'no_tumor',
//   levelIndex: 0 | 1 | 2 | 3,   // index into tumor.levels array
//   confidence: 0.0 – 1.0,        // model confidence score
// }
// ══════════════════════════════════════════════════════════
async function callRealModelAPI(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);

  const response = await fetch(CONFIG.MODEL_API_URL, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }

  const data = await response.json();

  // Map your API response to the expected format.
  // Adjust keys below based on your model's actual response structure:
  return {
    tumorType: data.tumor_type,       // e.g. 'glioma'
    levelIndex: data.grade_index,     // e.g. 2
    confidence: data.confidence,      // e.g. 0.947
  };
}

// ──────────────────────────────────────────────────────────
// SIMULATED AI (Demo Mode — Replace with real model)
// ──────────────────────────────────────────────────────────
async function simulateAIAnalysis() {
  const steps = [
    { msg: 'Preprocessing MRI image…', pct: 15, delay: 500 },
    { msg: 'Applying convolutional neural network…', pct: 35, delay: 700 },
    { msg: 'Running feature extraction…', pct: 55, delay: 600 },
    { msg: 'Classifying tumor type…', pct: 75, delay: 500 },
    { msg: 'Computing confidence scores…', pct: 90, delay: 400 },
    { msg: 'Generating AI observations…', pct: 100, delay: 300 },
  ];

  for (const step of steps) {
    analysisStatus.textContent = step.msg;
    progressBar.style.width = step.pct + '%';
    await delay(step.delay);
  }

  // Random prediction for demo
  const types = Object.keys(TUMOR_DATABASE);
  const tumorType = types[Math.floor(Math.random() * types.length)];
  const tumor = TUMOR_DATABASE[tumorType];
  const levelIndex = Math.floor(Math.random() * tumor.levels.length);
  const confidence = 0.82 + Math.random() * 0.16; // 82–98%

  return { tumorType, levelIndex, confidence };
}

function delay(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

// ──────────────────────────────────────────────────────────
// DISPLAY RESULT
// ──────────────────────────────────────────────────────────
function displayResult(result) {
  const tumor = TUMOR_DATABASE[result.tumorType];
  if (!tumor) return;

  const level = tumor.levels[result.levelIndex];
  const levelKey = tumor.levelKeys[result.levelIndex];
  const confidencePct = Math.round(result.confidence * 100);
  const obs = tumor.observations.join(' ');

  tumorName.textContent = tumor.name;
  tumorLevel.textContent = level;
  confidenceScore.textContent = confidencePct + '%';
  aiObservations.textContent = obs;

  // Severity bar color
  const sevColors = {
    low: '#10b981',
    moderate: '#f59e0b',
    high: '#f97316',
    critical: '#ef4444',
  };
  const sevWidths = { low: '25%', moderate: '55%', high: '78%', critical: '100%' };
  severityBar.style.background = sevColors[levelKey] || '#10b981';
  severityBar.style.width = sevWidths[levelKey] || '25%';

  analysisStatus.textContent = 'Analysis complete ✓';

  resultGrid.style.display = 'grid';

  setTimeout(() => {
    resultGrid.style.opacity = '1';
  }, 50);
}

// ──────────────────────────────────────────────────────────
// PATIENT FORM
// ──────────────────────────────────────────────────────────
patientForm.addEventListener('submit', (e) => {
  e.preventDefault();
  state.patientSaved = true;
  formSaved.style.display = 'flex';
  updateChecklist();
  showToast('Patient details saved!', 'success');
  setTimeout(() => {
    formSaved.style.display = 'none';
  }, 4000);
});

// ──────────────────────────────────────────────────────────
// CHECKLIST & REPORT ENABLE
// ──────────────────────────────────────────────────────────
function updateChecklist() {
  const scanDone = !!state.analysisResult;
  const patientDone = state.patientSaved;

  chkScanIcon.textContent = scanDone ? '✅' : '○';
  chkScan.classList.toggle('done', scanDone);

  chkPatientIcon.textContent = patientDone ? '✅' : '○';
  chkPatient.classList.toggle('done', patientDone);

  generateReportBtn.disabled = !(scanDone && patientDone);
}

// ──────────────────────────────────────────────────────────
// REPORT GENERATION
// ──────────────────────────────────────────────────────────
generateReportBtn.addEventListener('click', () => {
  const html = buildReportHTML();
  reportPreview.innerHTML = html;
  reportModal.style.display = 'flex';
  document.body.style.overflow = 'hidden';
});

closeModal.addEventListener('click', closeReportModal);
closeModalBtn.addEventListener('click', closeReportModal);
reportModal.addEventListener('click', (e) => {
  if (e.target === reportModal) closeReportModal();
});

function closeReportModal() {
  reportModal.style.display = 'none';
  document.body.style.overflow = '';
}

printReportBtn.addEventListener('click', () => {
  const printContent = reportPreview.innerHTML;
  const printWindow = window.open('', '_blank', 'width=900,height=700');
  printWindow.document.write(`
    <!DOCTYPE html><html><head>
    <title>NeuroScan AI – Medical Report</title>
    <style>
      body { font-family: Inter, sans-serif; margin: 0; padding: 24px; background: #fff; }
      * { box-sizing: border-box; }
      .rep-header { background: #0f172a !important; color: #fff; padding: 32px; }
      .rep-logo-name { color: #67e8f9 !important; font-size: 20px; font-weight: 800; }
      .rep-meta { display: flex; gap: 32px; margin-top: 16px; }
      .rep-meta-label { color: #94a3b8; font-size: 12px; }
      .rep-meta-val { color: #e2e8f0; font-weight: 600; font-size: 12px; }
      .rep-body { padding: 28px; }
      .rep-section { margin-bottom: 24px; }
      .rep-section-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .7px; color: #0891b2; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 14px; }
      .rep-row { display: flex; gap: 0; margin-bottom: 8px; font-size: 13px; }
      .rep-key { width: 180px; color: #64748b; font-weight: 500; flex-shrink: 0; }
      .rep-val { color: #1a202c; font-weight: 500; }
      .rep-findings { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }
      .rep-finding-highlight { background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
      .rep-finding-name { font-size: 18px; font-weight: 800; color: #0f766e; }
      .rep-badge { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 11px; font-weight: 700; }
      .rep-badge.low { background: #d1fae5; color: #065f46; }
      .rep-badge.moderate { background: #fef3c7; color: #92400e; }
      .rep-badge.high { background: #fee2e2; color: #7f1d1d; }
      .rep-badge.critical { background: #7f1d1d; color: #fff; }
      .rep-mri-wrap { display: flex; gap: 16px; align-items: flex-start; }
      .rep-mri-img { width: 140px; height: 140px; object-fit: contain; border: 1px solid #e2e8f0; border-radius: 8px; background: #0f172a; }
      .rep-disclaimer { font-size: 11px; color: #94a3b8; font-style: italic; border-top: 1px solid #e2e8f0; padding-top: 16px; margin-top: 24px; }
      .rep-footer { background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 16px 28px; display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; }
      .confidence-bar-wrap { height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; margin-top: 6px; }
      .confidence-bar { height: 100%; background: linear-gradient(90deg, #0891b2, #7c3aed); border-radius: 4px; }
    </style>
    </head><body>${printContent}</body></html>
  `);
  printWindow.document.close();
  printWindow.focus();
  setTimeout(() => { printWindow.print(); }, 500);
});

downloadReportBtn.addEventListener('click', () => {
  const htmlContent = buildFullReportPage();
  const blob = new Blob([htmlContent], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const patientName = document.getElementById('patientName').value || 'Patient';
  const date = new Date().toISOString().split('T')[0];
  a.download = `NeuroScan_Report_${patientName.replace(/\s+/g, '_')}_${date}.html`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Report downloaded!', 'success');
});

// ──────────────────────────────────────────────────────────
// BUILD REPORT HTML
// ──────────────────────────────────────────────────────────
function buildReportHTML() {
  const r = state.analysisResult;
  const tumor = TUMOR_DATABASE[r.tumorType];
  const level = tumor.levels[r.levelIndex];
  const levelKey = tumor.levelKeys[r.levelIndex];
  const confidencePct = Math.round(r.confidence * 100);
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-IN', { year: 'numeric', month: 'long', day: 'numeric' });
  const timeStr = now.toLocaleTimeString('en-IN');
  const reportId = 'NSR-' + Date.now().toString().slice(-8);

  const pName    = v('patientName')    || 'N/A';
  const pId      = v('patientId')      || 'N/A';
  const pAge     = v('patientAge')     || 'N/A';
  const pGender  = v('patientGender')  || 'N/A';
  const pDOB     = v('patientDOB')     ? new Date(v('patientDOB')).toLocaleDateString('en-IN') : 'N/A';
  const pPhone   = v('patientPhone')   || 'N/A';
  const pEmail   = v('patientEmail')   || 'N/A';
  const pAddr    = v('patientAddress') || 'N/A';
  const pDoc     = v('referringDoctor')|| 'N/A';
  const pHosp    = v('hospital')       || 'N/A';
  const pSymptoms= v('symptoms')       || 'None reported';

  const mriSrc = state.mriDataURL || '';

  const observationsHTML = tumor.observations
    .map(obs => `<li style="margin-bottom:6px; color:#475569; font-size:13px;">${obs}</li>`)
    .join('');

  const recommendationsHTML = tumor.recommendations
    .map(rec => `<li style="margin-bottom:6px; color:#475569; font-size:13px;">${rec}</li>`)
    .join('');

  return `
<div class="report-doc">
  <div class="rep-header">
    <div class="rep-logo">
      <svg width="36" height="36" viewBox="0 0 32 32" fill="none">
        <circle cx="16" cy="16" r="15" stroke="#67e8f9" stroke-width="2"/>
        <path d="M10 16 C10 10 22 10 22 16 C22 22 10 22 10 16Z" stroke="#67e8f9" stroke-width="1.5" fill="none"/>
        <circle cx="16" cy="16" r="3" fill="#67e8f9"/>
      </svg>
      <div>
        <div class="rep-logo-name">NeuroScan AI</div>
        <div class="rep-title" style="font-size:11px;color:#94a3b8;margin-top:2px;">Brain Tumor Detection System — AI Diagnostic Report</div>
      </div>
    </div>
    <div class="rep-meta">
      <div class="rep-meta-item"><span class="rep-meta-label">Report ID: </span><span class="rep-meta-val">${reportId}</span></div>
      <div class="rep-meta-item"><span class="rep-meta-label">Date: </span><span class="rep-meta-val">${dateStr}</span></div>
      <div class="rep-meta-item"><span class="rep-meta-label">Time: </span><span class="rep-meta-val">${timeStr}</span></div>
      <div class="rep-meta-item"><span class="rep-meta-label">Facility: </span><span class="rep-meta-val">${pHosp !== 'N/A' ? pHosp : 'NeuroScan AI Clinic'}</span></div>
    </div>
  </div>

  <div class="rep-body">

    <!-- Patient Info -->
    <div class="rep-section">
      <div class="rep-section-title">Patient Information</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 24px;">
        <div class="rep-row"><span class="rep-key">Patient Name</span><span class="rep-val">${pName}</span></div>
        <div class="rep-row"><span class="rep-key">Patient ID</span><span class="rep-val">${pId}</span></div>
        <div class="rep-row"><span class="rep-key">Age</span><span class="rep-val">${pAge} years</span></div>
        <div class="rep-row"><span class="rep-key">Gender</span><span class="rep-val">${pGender}</span></div>
        <div class="rep-row"><span class="rep-key">Date of Birth</span><span class="rep-val">${pDOB}</span></div>
        <div class="rep-row"><span class="rep-key">Phone</span><span class="rep-val">${pPhone}</span></div>
        <div class="rep-row"><span class="rep-key">Email</span><span class="rep-val">${pEmail}</span></div>
        <div class="rep-row"><span class="rep-key">Referring Doctor</span><span class="rep-val">${pDoc}</span></div>
        <div class="rep-row" style="grid-column:1/-1"><span class="rep-key">Address</span><span class="rep-val">${pAddr}</span></div>
        <div class="rep-row" style="grid-column:1/-1"><span class="rep-key">Clinical Symptoms</span><span class="rep-val">${pSymptoms}</span></div>
      </div>
    </div>

    <!-- MRI Scan -->
    <div class="rep-section">
      <div class="rep-section-title">MRI Scan Details</div>
      <div class="rep-mri-wrap">
        ${mriSrc ? `<img class="rep-mri-img" src="${mriSrc}" alt="MRI Scan" />` : '<div class="rep-mri-img" style="display:flex;align-items:center;justify-content:center;color:#64748b;font-size:12px;">No image</div>'}
        <div class="rep-mri-analysis">
          <div class="rep-row"><span class="rep-key">File Name</span><span class="rep-val">${state.mriFile ? state.mriFile.name : 'N/A'}</span></div>
          <div class="rep-row"><span class="rep-key">Scan Type</span><span class="rep-val">Brain MRI</span></div>
          <div class="rep-row"><span class="rep-key">Analysis Model</span><span class="rep-val">NeuroScan Deep CNN v1.0</span></div>
          <div class="rep-row" style="margin-top:12px"><span class="rep-key">AI Confidence</span></div>
          <div class="confidence-bar-wrap">
            <div class="confidence-bar" style="width:${confidencePct}%"></div>
          </div>
          <div style="font-size:22px;font-weight:800;color:#0891b2;margin-top:6px;">${confidencePct}%</div>
        </div>
      </div>
    </div>

    <!-- AI Diagnosis -->
    <div class="rep-section">
      <div class="rep-section-title">AI Diagnostic Findings</div>
      <div class="rep-findings">
        <div class="rep-finding-highlight">
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            <div>
              <div class="rep-finding-name">${tumor.fullName}</div>
              <div class="rep-finding-level" style="margin-top:6px;">
                Severity Level: <strong>${level}</strong> &nbsp;
                <span class="rep-badge ${levelKey}">${levelKey.toUpperCase()}</span>
              </div>
            </div>
          </div>
        </div>
        <p style="font-size:13px;color:#475569;line-height:1.65;margin-bottom:14px;">${tumor.description}</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
          <div>
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#0891b2;margin-bottom:8px;">Observations</div>
            <ul style="list-style:disc;padding-left:16px;">${observationsHTML}</ul>
          </div>
          <div>
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#7c3aed;margin-bottom:8px;">Recommendations</div>
            <ul style="list-style:disc;padding-left:16px;">${recommendationsHTML}</ul>
          </div>
        </div>
      </div>
    </div>

    <div class="rep-disclaimer">
      ⚠️ <strong>Important Disclaimer:</strong> This report is generated by an AI-assisted diagnostic tool and is intended for informational purposes only. 
      It does not constitute a definitive medical diagnosis. All findings must be reviewed and confirmed by a qualified radiologist or neurologist. 
      Clinical correlation and further imaging or biopsy may be required for accurate diagnosis and treatment planning. 
      NeuroScan AI is a decision-support tool only.
    </div>
  </div>

  <div class="rep-footer">
    <span>NeuroScan AI — Brain Tumor Detection System | Report: ${reportId}</span>
    <span>Generated: ${dateStr} at ${timeStr}</span>
  </div>
</div>
  `;
}

function buildFullReportPage() {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>NeuroScan AI – Medical Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
body{font-family:Inter,sans-serif;margin:0;padding:24px;background:#f8fafc;color:#1a202c;}
*{box-sizing:border-box;}
.report-doc{max-width:860px;margin:0 auto;background:#fff;border-radius:12px;box-shadow:0 4px 24px rgba(0,0,0,.12);overflow:hidden;}
.rep-header{background:linear-gradient(135deg,#0f172a,#1e293b);color:#fff;padding:32px;}
.rep-logo{display:flex;align-items:center;gap:12px;margin-bottom:16px;}
.rep-logo-name{font-size:20px;font-weight:800;color:#67e8f9;}
.rep-title{font-size:11px;color:#94a3b8;margin-top:2px;}
.rep-meta{display:flex;gap:32px;flex-wrap:wrap;}
.rep-meta-label{color:#94a3b8;font-size:12px;}
.rep-meta-val{color:#e2e8f0;font-weight:600;font-size:12px;}
.rep-body{padding:32px;}
.rep-section{margin-bottom:28px;}
.rep-section-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.7px;color:#0891b2;border-bottom:1px solid #e2e8f0;padding-bottom:8px;margin-bottom:14px;}
.rep-row{display:flex;margin-bottom:8px;font-size:13px;}
.rep-key{width:180px;color:#64748b;font-weight:500;flex-shrink:0;}
.rep-val{color:#1a202c;font-weight:500;}
.rep-findings{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;}
.rep-finding-highlight{background:#ecfdf5;border:1px solid #a7f3d0;border-radius:8px;padding:16px;margin-bottom:12px;}
.rep-finding-name{font-size:18px;font-weight:800;color:#0f766e;}
.rep-badge{display:inline-block;padding:3px 12px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;}
.rep-badge.low{background:#d1fae5;color:#065f46;}
.rep-badge.moderate{background:#fef3c7;color:#92400e;}
.rep-badge.high{background:#fee2e2;color:#7f1d1d;}
.rep-badge.critical{background:#7f1d1d;color:#fff;}
.rep-mri-wrap{display:flex;gap:16px;align-items:flex-start;}
.rep-mri-img{width:140px;height:140px;object-fit:contain;border:1px solid #e2e8f0;border-radius:8px;background:#0f172a;}
.confidence-bar-wrap{height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;margin-top:6px;}
.confidence-bar{height:100%;background:linear-gradient(90deg,#0891b2,#7c3aed);border-radius:4px;}
.rep-disclaimer{font-size:11px;color:#94a3b8;font-style:italic;border-top:1px solid #e2e8f0;padding-top:16px;margin-top:24px;line-height:1.7;}
.rep-footer{background:#f8fafc;border-top:1px solid #e2e8f0;padding:16px 32px;display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;}
ul{list-style:disc;padding-left:16px;}
li{margin-bottom:6px;color:#475569;font-size:13px;}
</style>
</head>
<body>${buildReportHTML()}</body></html>`;
}

// Helper: get form field value
function v(id) {
  const el = document.getElementById(id);
  return el ? el.value.trim() : '';
}

// ──────────────────────────────────────────────────────────
// TOAST
// ──────────────────────────────────────────────────────────
let toastTimer = null;
function showToast(message, type = 'default') {
  toast.textContent = message;
  toast.className = 'toast show ' + type;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.className = 'toast';
  }, 3500);
}

// ──────────────────────────────────────────────────────────
// DARK / LIGHT MODE TOGGLE
// ──────────────────────────────────────────────────────────
function applyTheme(theme) {
  document.getElementById('htmlRoot').setAttribute('data-theme', theme);
  const icon = document.getElementById('themeIcon');
  if (icon) icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
  localStorage.setItem('neuroscan-theme', theme);
}

function toggleTheme() {
  const current = document.getElementById('htmlRoot').getAttribute('data-theme') || 'dark';
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

// Restore saved theme on page load
(function() {
  const saved = localStorage.getItem('neuroscan-theme') || 'dark';
  applyTheme(saved);
})();

// ──────────────────────────────────────────────────────────
// INIT
// ──────────────────────────────────────────────────────────
updateChecklist();
console.log('%c🧠 NeuroScan AI', 'font-size:16px;font-weight:bold;color:#06b6d4');
console.log('%cBrain Tumor Detection System initialized.', 'color:#94a3b8');
console.log('%cTo connect a real AI model, see CONFIG.USE_REAL_MODEL and callRealModelAPI() in app.js', 'color:#f59e0b');

