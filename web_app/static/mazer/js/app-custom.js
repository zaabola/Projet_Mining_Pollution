/**
 * EcoGuard AI Platform - Frontend JavaScript
 * Uses Mazer admin dashboard template
 */

function getCookie(name) {
    let v = null;
    if (document.cookie && document.cookie !== '') {
        document.cookie.split(';').forEach(c => {
            c = c.trim();
            if (c.substring(0, name.length + 1) === (name + '='))
                v = decodeURIComponent(c.substring(name.length + 1));
        });
    }
    return v;
}

function getCSRFToken() {
    const t = document.querySelector('meta[name="csrf-token"]');
    return t ? t.getAttribute('content') : getCookie('csrftoken');
}

/* ───── helpers ───── */
function show(el) { if (el) el.style.display = ''; }
function hide(el) { if (el) el.style.display = 'none'; }
function showById(id) { show(document.getElementById(id)); }
function hideById(id) { hide(document.getElementById(id)); }
function setHTML(id, html) { const e = document.getElementById(id); if (e) e.innerHTML = html; }

/* ───── Get the page mode ───── */
function getMode() {
    const el = document.getElementById('page-mode');
    return el ? el.value : 'ppe';
}

function getModelForMode(mode) {
    const map = { 'ppe': 'helmet', 'fish': 'fish', 'animaux': 'animaux', 'illegal_mining': 'ahmed', 'smoke': 'smoke' };
    return map[mode] || 'helmet';
}

/* ───── Conditionally show/hide video & webcam panels ───── */
function applyModeRestrictions() {
    const mode = getMode();

    // Video: only for PPE and Fish
    const videoCard = document.querySelector('.detection-video-card');
    if (videoCard) {
        if (mode === 'ppe' || mode === 'fish') {
            show(videoCard);
        } else {
            hide(videoCard);
        }
    }

    // Webcam: only for PPE
    const webcamCard = document.querySelector('.detection-webcam-card');
    if (webcamCard) {
        if (mode === 'ppe') {
            show(webcamCard);
        } else {
            hide(webcamCard);
        }
    }
}

/* ───── IMAGE DETECTION ───── */
function setupImageDetection() {
    const input = document.getElementById('image-input');
    const btn = document.getElementById('image-detect-btn');
    const conf = document.getElementById('image-confidence');
    const confDisplay = document.getElementById('image-conf-val');
    const heatmap = document.getElementById('image-heatmap');
    if (!btn) return;

    if (conf && confDisplay) conf.addEventListener('input', () => confDisplay.textContent = conf.value + '%');

    btn.addEventListener('click', async () => {
        if (!input || !input.files[0]) { setHTML('image-alert', '<div class="alert alert-danger">Please select an image first.</div>'); showById('image-alert'); return; }
        hideById('image-alert');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Processing...';
        try {
            const mode = getMode();
            const fd = new FormData();
            fd.append('image', input.files[0]);
            fd.append('model', getModelForMode(mode));
            fd.append('mode', mode);
            fd.append('confidence', conf ? conf.value / 100 : 0.25);
            fd.append('heatmap', heatmap ? heatmap.checked : false);

            const r = await fetch('/api/detect/image/', { method: 'POST', body: fd, headers: { 'X-CSRFToken': getCSRFToken() } });
            const d = await r.json();
            if (d.error) {
                setHTML('image-alert', `<div class="alert alert-danger">${d.error}</div>`);
                showById('image-alert');
            } else {
                const img = document.getElementById('image-result-img');
                if (img && d.annotated_image_url) { img.src = d.annotated_image_url; show(img); }

                let h = '';
                if (d.heatmap_url) {
                    h += `<div class="card mb-3"><div class="card-header"><h6 class="mb-0">Explainability Heatmap</h6></div><div class="card-body text-center"><img src="${d.heatmap_url}" style="max-width:100%;border-radius:8px;"></div></div>`;
                }

                if (mode === 'illegal_mining') {
                    if (d.detections && d.detections.length) {
                        h += '<div class="alert alert-danger"><i class="bi bi-exclamation-triangle-fill"></i> <strong>WARNING:</strong> Illegal mining detected in this area!</div>';
                    } else {
                        h += '<div class="alert alert-success"><i class="bi bi-check-circle-fill"></i> <strong>Safe:</strong> No illegal mining detected. Area is clear.</div>';
                    }
                } else {
                    if (d.detections && d.detections.length) {
                        h += `<div class="alert alert-info"><strong>${d.detections.length}</strong> object(s) detected</div>`;
                        h += '<div class="list-group">';
                        d.detections.forEach(det => {
                            h += `<div class="list-group-item d-flex justify-content-between align-items-center">${det.class_name}<span class="badge bg-primary rounded-pill">${(det.confidence * 100).toFixed(1)}%</span></div>`;
                        });
                        h += '</div>';
                    } else {
                        h += '<div class="alert alert-warning">No objects detected in this image.</div>';
                    }
                }
                setHTML('image-detections', h);
                showById('image-results');
            }
        } catch (e) {
            setHTML('image-alert', `<div class="alert alert-danger">Error: ${e.message}</div>`);
            showById('image-alert');
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-search"></i> Detect';
        }
    });
}

/* ───── VIDEO DETECTION ───── */
function setupVideoDetection() {
    const input = document.getElementById('video-input');
    const btn = document.getElementById('video-detect-btn');
    const stopBtn = document.getElementById('video-stop-btn');
    const conf = document.getElementById('video-confidence');
    const confDisplay = document.getElementById('video-conf-val');
    if (!btn) return;

    if (conf && confDisplay) conf.addEventListener('input', () => confDisplay.textContent = conf.value + '%');

    let abortController = null;

    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            if (abortController) abortController.abort();
            hide(stopBtn);
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-play-fill"></i> Start Video';
        });
    }

    btn.addEventListener('click', async () => {
        if (!input || !input.files[0]) { setHTML('video-alert', '<div class="alert alert-danger">Please select a video first.</div>'); showById('video-alert'); return; }
        hideById('video-alert');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Uploading...';
        if (stopBtn) show(stopBtn);
        abortController = new AbortController();

        try {
            const mode = getMode();
            const fd = new FormData();
            fd.append('video', input.files[0]);
            fd.append('confidence', conf ? conf.value / 100 : 0.25);
            fd.append('mode', mode);

            const uploadRes = await fetch('/api/upload/video/', { method: 'POST', body: fd, headers: { 'X-CSRFToken': getCSRFToken() }, signal: abortController.signal });
            const uploadData = await uploadRes.json();
            if (uploadData.error) { setHTML('video-alert', `<div class="alert alert-danger">${uploadData.error}</div>`); showById('video-alert'); return; }

            btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Streaming...';
            const streamUrl = `/api/stream/video/${uploadData.session_id}/?mode=${mode}`;
            const streamImg = document.getElementById('video-stream-img');
            if (streamImg) {
                streamImg.src = streamUrl;
                show(streamImg);
            }
            showById('video-results');
        } catch (e) {
            if (e.name !== 'AbortError') {
                setHTML('video-alert', `<div class="alert alert-danger">Error: ${e.message}</div>`);
                showById('video-alert');
            }
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-play-fill"></i> Start Video';
        }
    });
}

/* ───── WEBCAM DETECTION ───── */
function setupWebcamDetection() {
    const btn = document.getElementById('webcam-start-btn');
    const stopBtn = document.getElementById('webcam-stop-btn');
    const dur = document.getElementById('webcam-duration');
    const conf = document.getElementById('webcam-confidence');
    const confDisplay = document.getElementById('webcam-conf-val');
    if (!btn) return;

    if (conf && confDisplay) conf.addEventListener('input', () => confDisplay.textContent = conf.value + '%');

    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            const img = document.getElementById('webcam-stream-img');
            if (img) img.src = '';
            hide(stopBtn);
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-camera-video-fill"></i> Start Webcam';
        });
    }

    btn.addEventListener('click', async () => {
        hideById('webcam-alert');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Connecting...';
        if (stopBtn) show(stopBtn);

        try {
            const mode = getMode();
            const confVal = conf ? conf.value / 100 : 0.25;
            const durVal = dur ? dur.value : 30;
            const streamUrl = `/api/stream/webcam/?confidence=${confVal}&duration=${durVal}&mode=${mode}`;
            const img = document.getElementById('webcam-stream-img');
            if (img) {
                img.src = streamUrl;
                show(img);
            }
            showById('webcam-results');
            btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Streaming...';
        } catch (e) {
            setHTML('webcam-alert', `<div class="alert alert-danger">Error: ${e.message}</div>`);
            showById('webcam-alert');
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-camera-video-fill"></i> Start Webcam';
        }
    });
}

/* ───── ANIMAL COMPARE ───── */
function setupAnimalCompare() {
    const btn = document.getElementById('compare-btn');
    if (!btn) return;

    btn.addEventListener('click', async () => {
        const beforeInput = document.getElementById('before-input');
        const afterInput = document.getElementById('after-input');
        if (!beforeInput?.files[0] || !afterInput?.files[0]) {
            setHTML('compare-alert', '<div class="alert alert-danger">Please select both images.</div>');
            showById('compare-alert');
            return;
        }
        hideById('compare-alert');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Comparing...';

        try {
            const fd = new FormData();
            fd.append('image_before', beforeInput.files[0]);
            fd.append('image_after', afterInput.files[0]);
            const r = await fetch('/api/animaux/compare/', { method: 'POST', body: fd, headers: { 'X-CSRFToken': getCSRFToken() } });
            const d = await r.json();

            if (d.error) {
                setHTML('compare-alert', `<div class="alert alert-danger">${d.error}</div>`);
                showById('compare-alert');
            } else {
                const beforeImg = document.getElementById('result-before-img');
                const afterImg = document.getElementById('result-after-img');
                if (beforeImg && d.image_before) { beforeImg.src = d.image_before; show(beforeImg); }
                if (afterImg && d.image_after) { afterImg.src = d.image_after; show(afterImg); }

                let h = '';
                if (d.warning) {
                    h += `<div class="alert alert-danger"><i class="bi bi-exclamation-triangle-fill"></i> ${d.message}</div>`;
                } else {
                    h += `<div class="alert alert-success"><i class="bi bi-check-circle-fill"></i> ${d.message}</div>`;
                }
                if (d.count_before !== undefined) {
                    h += `<div class="alert alert-light"><strong>Before:</strong> ${d.count_before} animals &nbsp;|&nbsp; <strong>After:</strong> ${d.count_after} animals</div>`;
                }
                setHTML('compare-result-text', h);
                showById('compare-results');
            }
        } catch (e) {
            setHTML('compare-alert', `<div class="alert alert-danger">Error: ${e.message}</div>`);
            showById('compare-alert');
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-arrow-left-right"></i> Compare';
        }
    });
}

/* ───── INIT ───── */
document.addEventListener('DOMContentLoaded', () => {
    console.log('✅ EcoGuard AI Platform initializing...');

    // Apply mode restrictions (webcam: PPE only, video: PPE + fish only)
    applyModeRestrictions();

    setupImageDetection();
    setupVideoDetection();
    setupWebcamDetection();
    setupAnimalCompare();
    console.log('✅ EcoGuard AI Platform ready! Mode:', getMode());
});
