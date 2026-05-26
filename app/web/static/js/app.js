/* BOM Pattern Detection — Frontend Logic */

const API = '';  // same origin

// ---- State ----
let patternFile = null;
let drawingFile = null;
let vizB64 = null;

// ---- DOM refs ----
const patternDropzone  = document.getElementById('patternDropzone');
const drawingDropzone  = document.getElementById('drawingDropzone');
const patternFileInput = document.getElementById('patternFile');
const drawingFileInput = document.getElementById('drawingFile');
const patternPreview   = document.getElementById('patternPreview');
const drawingPreview   = document.getElementById('drawingPreview');
const patternImg       = document.getElementById('patternImg');
const drawingImg       = document.getElementById('drawingImg');
const clearPatternBtn  = document.getElementById('clearPattern');
const clearDrawingBtn  = document.getElementById('clearDrawing');
const runBtn           = document.getElementById('runBtn');

const autoModeToggle   = document.getElementById('autoMode');
const manualSettings   = document.getElementById('manualSettings');
const nccSlider        = document.getElementById('nccThreshold');
const dinoSlider       = document.getElementById('dinoThreshold');
const nmsSlider        = document.getElementById('nmsThreshold');
const nccVal           = document.getElementById('nccVal');
const dinoVal          = document.getElementById('dinoVal');
const nmsVal           = document.getElementById('nmsVal');

const statusDot        = document.getElementById('statusDot');
const statusText       = document.getElementById('statusText');
const loadingOverlay   = document.getElementById('loadingOverlay');
const loadingStage     = document.getElementById('loadingStage');

const statsRow         = document.getElementById('statsRow');
const statDetections   = document.getElementById('statDetections');
const statAvgConf      = document.getElementById('statAvgConf');
const statBestDino     = document.getElementById('statBestDino');
const statTime         = document.getElementById('statTime');

const resultsSection   = document.getElementById('resultsSection');
const vizImage         = document.getElementById('vizImage');
const detectionsList   = document.getElementById('detectionsList');
const detectionCount   = document.getElementById('detectionCount');
const downloadBtn      = document.getElementById('downloadBtn');

// ---- Helpers ----
function setStatus(state, text) {
  statusDot.className = 'status-dot ' + state;
  statusText.textContent = text;
}

function showToast(msg, type = 'error') {
  let toast = document.querySelector('.toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.className = 'toast';
    document.body.appendChild(toast);
  }
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 4000);
}

function fileToDataURL(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.readAsDataURL(file);
  });
}

function checkRunReady() {
  runBtn.disabled = !(patternFile && drawingFile);
}

function setPreview(file, imgEl, previewEl, dropzone) {
  const url = URL.createObjectURL(file);
  imgEl.src = url;
  dropzone.querySelector('.upload-inner').style.display = 'none';
  previewEl.style.display = 'flex';
  dropzone.classList.add('has-file');
  // Remove file input pointer-events so user can't re-trigger accidentally
  dropzone.querySelector('.file-input').style.pointerEvents = 'none';
}

function clearPreview(imgEl, previewEl, dropzone, inputEl) {
  imgEl.src = '';
  previewEl.style.display = 'none';
  dropzone.querySelector('.upload-inner').style.display = 'flex';
  dropzone.classList.remove('has-file');
  dropzone.querySelector('.file-input').style.pointerEvents = 'auto';
  inputEl.value = '';
}

// ---- File Handlers ----
patternFileInput.addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  patternFile = f;
  setPreview(f, patternImg, patternPreview, patternDropzone);
  checkRunReady();
});

drawingFileInput.addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  drawingFile = f;
  setPreview(f, drawingImg, drawingPreview, drawingDropzone);
  checkRunReady();
});

clearPatternBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  patternFile = null;
  clearPreview(patternImg, patternPreview, patternDropzone, patternFileInput);
  checkRunReady();
});
clearDrawingBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  drawingFile = null;
  clearPreview(drawingImg, drawingPreview, drawingDropzone, drawingFileInput);
  checkRunReady();
});

// ---- Drag-and-drop ----
function setupDrop(zone, onFile) {
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) onFile(f);
  });
}

setupDrop(patternDropzone, (f) => {
  patternFile = f;
  setPreview(f, patternImg, patternPreview, patternDropzone);
  checkRunReady();
});
setupDrop(drawingDropzone, (f) => {
  drawingFile = f;
  setPreview(f, drawingImg, drawingPreview, drawingDropzone);
  checkRunReady();
});

// ---- Auto-mode toggle ----
autoModeToggle.addEventListener('change', () => {
  manualSettings.style.display = autoModeToggle.checked ? 'none' : 'block';
});

// ---- Slider display ----
nccSlider.addEventListener('input', () => { nccVal.textContent = parseFloat(nccSlider.value).toFixed(2); });
dinoSlider.addEventListener('input', () => { dinoVal.textContent = parseFloat(dinoSlider.value).toFixed(2); });
nmsSlider.addEventListener('input', () => { nmsVal.textContent = parseFloat(nmsSlider.value).toFixed(2); });

// ---- Run Detection ----
runBtn.addEventListener('click', async () => {
  if (!patternFile || !drawingFile) return;

  setStatus('running', 'Detecting…');
  loadingOverlay.style.display = 'flex';
  loadingStage.textContent = 'Preprocessing images…';

  const stageMessages = [
    'Running NCC template matching…',
    'Verifying with DINOv2…',
    'Applying NMS post-processing…',
    'Finalizing results…',
  ];
  let msgIdx = 0;
  const stageTimer = setInterval(() => {
    if (msgIdx < stageMessages.length) {
      loadingStage.textContent = stageMessages[msgIdx++];
    }
  }, 2000);

  try {
    const formData = new FormData();
    formData.append('pattern', patternFile, patternFile.name);
    formData.append('drawing', drawingFile, drawingFile.name);
    formData.append('mode', autoModeToggle.checked ? 'auto' : 'manual');
    formData.append('ncc_threshold', nccSlider.value);
    formData.append('cosine_threshold', dinoSlider.value);
    formData.append('final_nms_iou', nmsSlider.value);

    const resp = await fetch(`${API}/api/detect`, { method: 'POST', body: formData });
    clearInterval(stageTimer);
    loadingOverlay.style.display = 'none';

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `Server error ${resp.status}`);
    }

    const data = await resp.json();
    renderResults(data);
    setStatus('ready', 'Done');
    showToast(`Found ${data.total_detections} detection(s) in ${data.elapsed}s`, 'success');
  } catch (err) {
    clearInterval(stageTimer);
    loadingOverlay.style.display = 'none';
    setStatus('error', 'Error');
    showToast(err.message || 'Detection failed', 'error');
    console.error(err);
  }
});

// ---- Render Results ----
function confClass(conf) {
  if (conf >= 0.70) return 'high';
  if (conf >= 0.55) return 'mid';
  return 'low';
}

function renderResults(data) {
  const dets = data.detections || [];
  const n = data.total_detections;

  // Stats
  statsRow.style.display = 'grid';
  statDetections.textContent = n;

  if (n > 0) {
    const avgConf = dets.reduce((s, d) => s + (d.confidence || 0), 0) / n;
    const bestDino = Math.max(...dets.map(d => d.dino_score || d.cosine_score || 0));
    statAvgConf.textContent = avgConf.toFixed(3);
    statBestDino.textContent = bestDino.toFixed(3);
  } else {
    statAvgConf.textContent = '—';
    statBestDino.textContent = '—';
  }
  statTime.textContent = data.elapsed + 's';

  // Visualization
  if (data.visualization) {
    vizB64 = data.visualization;
    vizImage.src = 'data:image/png;base64,' + data.visualization;
  }

  // Detection list
  detectionCount.textContent = n;
  detectionsList.innerHTML = '';

  if (n === 0) {
    detectionsList.innerHTML = `
      <div class="empty-state">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="12" cy="12" r="10"/>
          <line x1="8" y1="12" x2="16" y2="12"/>
        </svg>
        <p>No patterns detected</p>
      </div>`;
  } else {
    dets.forEach((det, i) => {
      const conf = det.confidence || 0;
      const cls = confClass(conf);
      const ncc = det.ncc_score != null ? det.ncc_score.toFixed(3) : '—';
      const dino = (det.dino_score || det.cosine_score) != null
        ? (det.dino_score || det.cosine_score).toFixed(3) : '—';
      const scale = det.scale != null ? det.scale.toFixed(2) : '—';
      const bbox = det.bbox || {};
      const bboxStr = (bbox.x != null)
        ? `[${bbox.x}, ${bbox.y}, ${bbox.w}, ${bbox.h}]` : '—';

      const item = document.createElement('div');
      item.className = `detection-item ${cls}`;
      item.innerHTML = `
        <div class="detection-header">
          <div class="detection-num">
            <span class="num-badge">${i + 1}</span>
            Detection #${i + 1}
          </div>
          <div class="detection-conf">${(conf * 100).toFixed(1)}%</div>
        </div>
        <div class="detection-meta">
          <div class="meta-row"><span>NCC</span> ${ncc}</div>
          <div class="meta-row"><span>DINOv2</span> ${dino}</div>
          <div class="meta-row"><span>Scale</span> ${scale}</div>
          <div class="meta-row"><span>BBox</span> ${bboxStr}</div>
        </div>
        <div class="conf-bar-wrap">
          <div class="conf-bar" style="width:${Math.round(conf * 100)}%"></div>
        </div>
      `;
      detectionsList.appendChild(item);
    });
  }

  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---- Download ----
downloadBtn.addEventListener('click', () => {
  if (!vizB64) return;
  const a = document.createElement('a');
  a.href = 'data:image/png;base64,' + vizB64;
  a.download = 'detection_result.png';
  a.click();
});

// ---- Initial status check ----
(async () => {
  try {
    const r = await fetch('/api/health');
    if (r.ok) setStatus('ready', 'Ready');
  } catch {
    setStatus('', 'Offline');
  }
})();
