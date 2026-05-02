/**
 * PPE Detection Web App - Frontend JavaScript
 * Fixed: All DOM element access now happens after DOMContentLoaded
 */

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function getCSRFToken() {
    const token = document.querySelector('meta[name="csrf-token"]');
    return token ? token.getAttribute('content') : getCookie('csrftoken');
}

function showSpinner(id) { const e = document.getElementById(id); if(e) e.classList.add('show'); }
function hideSpinner(id) { const e = document.getElementById(id); if(e) e.classList.remove('show'); }
function showResults(id) { const e = document.getElementById(id); if(e) e.classList.add('show'); }
function showError(id, msg) { const e = document.getElementById(id); if(e) { e.textContent = msg; e.style.display = 'block'; } }
function hideError(id) { const e = document.getElementById(id); if(e) e.style.display = 'none'; }

function setupImageDetection() {
    const zone = document.getElementById('image-upload-zone');
    const input = document.getElementById('image-input');
    const display = document.getElementById('image-file-name');
    const btn = document.getElementById('image-detect-btn');
    const conf = document.getElementById('image-confidence');
    const model = document.getElementById('image-model');
    const heatmap = document.getElementById('image-heatmap');
    
    if (!zone) return;
    
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', (e) => {
        if (e.target.files[0]) display.textContent = `Selected: ${e.target.files[0].name}`;
    });
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            input.files = e.dataTransfer.files;
            display.textContent = `Selected: ${e.dataTransfer.files[0].name}`;
        }
    });
    
    btn.addEventListener('click', async () => {
        if (!input.files[0]) { showError('image-error-alert', 'Select image first'); return; }
        hideError('image-error-alert');
        showSpinner('image-spinner');
        btn.disabled = true;
        try {
            const fd = new FormData();
            fd.append('image', input.files[0]);
            fd.append('model', model.value);
            fd.append('confidence', conf.value / 100);
            fd.append('heatmap', heatmap.checked);
            const r = await fetch('/api/detect/image/', { method: 'POST', body: fd, headers: { 'X-CSRFToken': getCSRFToken() } });
            const d = await r.json();
            if (d.error) { showError('image-error-alert', d.error); } else {
                const res = document.getElementById('image-detections');
                const img = document.getElementById('image-result-img');
                if (d.annotated_image_url) img.src = d.annotated_image_url;
                let h = '';
                if (d.heatmap_url) h += `<div class="alert alert-info"><strong>Heatmap</strong><br><img src="${d.heatmap_url}" style="max-width:100%;max-height:300px;margin-top:10px;border-radius:8px;"></div>`;
                if (d.detections && d.detections.length) { h += `<strong>Detections: ${d.detections.length}</strong><br>`; d.detections.forEach(det => { h += `<div class="detection-result"><span class="class-name">${det.class_name}</span> - <span class="confidence">${(det.confidence*100).toFixed(2)}%</span></div>`; }); } else { h += '<div class="alert alert-warning">No objects</div>'; }
                res.innerHTML = h;
                showResults('image-results');
            }
        } catch (e) { showError('image-error-alert', `Error: ${e.message}`); } finally { hideSpinner('image-spinner'); btn.disabled = false; }
    });
}

function setupVideoDetection() {
    const zone = document.getElementById('video-upload-zone');
    const input = document.getElementById('video-input');
    const display = document.getElementById('video-file-name');
    const btn = document.getElementById('video-detect-btn');
    const conf = document.getElementById('video-confidence');
    const model = document.getElementById('video-model');
    
    if (!zone) return;
    
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', (e) => { if (e.target.files[0]) display.textContent = `Selected: ${e.target.files[0].name}`; });
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault(); zone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            input.files = e.dataTransfer.files;
            display.textContent = `Selected: ${e.dataTransfer.files[0].name}`;
        }
    });
    
    btn.addEventListener('click', async () => {
        if (!input.files[0]) { showError('video-error-alert', 'Select video first'); return; }
        hideError('video-error-alert');
        showSpinner('video-spinner');
        btn.disabled = true;
        try {
            const fd = new FormData();
            fd.append('video', input.files[0]);
            fd.append('confidence', conf.value / 100);
            const r = await fetch('/api/detect/video/', { method: 'POST', body: fd, headers: { 'X-CSRFToken': getCSRFToken() } });
            
            // Display streaming video
            const res = document.getElementById('video-detections');
            res.innerHTML = '<div class="alert alert-info"><strong>🎬 Processing Video...</strong><br><img id="video-stream" style="max-width:100%;border-radius:8px;" alt="Video Stream"></div>';
            showResults('video-results');
            
            await displayVideoStream(r, 'video-stream');
        } catch (e) { showError('video-error-alert', `Error: ${e.message}`); } finally { hideSpinner('video-spinner'); btn.disabled = false; }
    });
}

async function displayVideoStream(response, imgId) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const img = document.getElementById(imgId);
    let buffer = '';
    
    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // Find frame boundaries
            const frameStart = buffer.indexOf('\r\n\r\n');
            if (frameStart === -1) continue;
            
            const headerEnd = buffer.indexOf('\r\n\r\n');
            const contentLengthMatch = buffer.match(/Content-Length: (\d+)/);
            if (!contentLengthMatch) continue;
            
            const contentLength = parseInt(contentLengthMatch[1]);
            const frameDataStart = headerEnd + 4;
            const frameDataEnd = frameDataStart + contentLength;
            
            if (buffer.length < frameDataEnd) continue;
            
            const frameData = buffer.slice(frameDataStart, frameDataEnd);
            const base64Frame = btoa(String.fromCharCode.apply(null, new Uint8Array(frameData.split('').map(c => c.charCodeAt(0)))));
            img.src = `data:image/jpeg;base64,${base64Frame}`;
            
            // Extract detection data from header
            const dataMatch = buffer.match(/X-Detection-Data: ({.*?})\r\n/);
            if (dataMatch) {
                try {
                    const detection = JSON.parse(dataMatch[1]);
                    updateVideoStats(detection.detections);
                } catch (e) {}
            }
            
            buffer = buffer.slice(frameDataEnd);
        }
    } catch (e) {
        console.error('Stream error:', e);
    }
}

function updateVideoStats(detections) {
    const statsDiv = document.getElementById('video-stats') || createStatsDiv('video-stats', document.getElementById('video-detections'));
    if (detections) {
        statsDiv.innerHTML = `
            <div style="margin-top:10px;padding:10px;background:#f8f9fa;border-radius:5px;">
                <strong>👥 Persons: ${detections.total_persons}</strong> | 
                <span style="color:green;">✅ Safe: ${detections.safe_count}</span> | 
                <span style="color:red;">❌ Unsafe: ${detections.unsafe_count}</span>
            </div>
        `;
    }
}

function createStatsDiv(id, parent) {
    const div = document.createElement('div');
    div.id = id;
    parent.appendChild(div);
    return div;
}

function setupWebcamDetection() {
    const btn = document.getElementById('webcam-start-btn');
    const dur = document.getElementById('webcam-duration');
    const conf = document.getElementById('webcam-confidence');
    const model = document.getElementById('webcam-model');
    
    if (!btn) return;
    
    btn.addEventListener('click', async () => {
        hideError('webcam-error-alert');
        showSpinner('webcam-spinner');
        btn.disabled = true;
        try {
            const fd = new FormData();
            fd.append('confidence', conf.value / 100);
            fd.append('duration', dur.value);
            const r = await fetch('/api/detect/webcam/', { method: 'POST', body: fd, headers: { 'X-CSRFToken': getCSRFToken() } });
            
            // Display streaming video
            const res = document.getElementById('webcam-detections');
            res.innerHTML = '<div class="alert alert-info"><strong>📷 Webcam Stream...</strong><br><img id="webcam-stream" style="max-width:100%;border-radius:8px;" alt="Webcam Stream"></div>';
            showResults('webcam-results');
            
            hideSpinner('webcam-spinner');
            await displayVideoStream(r, 'webcam-stream');
        } catch (e) { showError('webcam-error-alert', `Error: ${e.message}`); } finally { hideSpinner('webcam-spinner'); btn.disabled = false; }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('✅ Initializing PPE Detection App...');
    setupImageDetection();
    setupVideoDetection();
    setupWebcamDetection();
    
    const setupSlider = (id, display) => { const s = document.getElementById(id); const d = document.getElementById(display); if(s && d) s.addEventListener('input', () => d.textContent = s.value); };
    setupSlider('image-confidence', 'image-conf-display');
    setupSlider('video-confidence', 'video-conf-display');
    setupSlider('webcam-confidence', 'webcam-conf-display');
    
    console.log('✅ PPE Detection App Ready!');
});
