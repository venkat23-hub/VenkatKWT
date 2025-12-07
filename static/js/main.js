/**
 * IIIT Sricity - Personal Keyword Spotting
 * No waveform | Pure WAV recording | Direct prediction
 */

const appState = {
    mode: 'upload',
    audioBlob: null,
    isRecording: false,
    audioContext: null,
    scriptNode: null,
    stream: null,
    recordedChunks: []
};

const API_BASE_URL = 'http://localhost:5000';

const predictBtn = document.getElementById('predict-btn');
const resetBtn = document.getElementById('reset-btn');
const startRecordBtn = document.getElementById('start-record');
const stopRecordBtn = document.getElementById('stop-record');
const resultsSection = document.getElementById('results-section');
const loadingSpinner = document.getElementById('loading-spinner');
const resultsContent = document.getElementById('results-content');
const errorMessage = document.getElementById('error-message');
const errorText = document.getElementById('error-text');

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('upload-mode').addEventListener('change', () => switchMode('upload'));
    document.getElementById('record-mode').addEventListener('change', () => switchMode('record'));
    document.getElementById('audio-file').addEventListener('change', handleUpload);
    startRecordBtn.addEventListener('click', startRecording);
    stopRecordBtn.addEventListener('click', stopRecording);
    predictBtn.addEventListener('click', predictKeyword);
    resetBtn.addEventListener('click', resetApp);

    switchMode('upload');
    predictBtn.disabled = true;
});

function switchMode(mode) {
    appState.mode = mode;
    document.getElementById('upload-interface').classList.toggle('visible', mode === 'upload');
    document.getElementById('record-interface').classList.toggle('visible', mode === 'record');
    resetApp();
}

function handleUpload(e) {
    const file = e.target.files[0];
    if (!file || file.size > 10*1024*1024) {
        showError('File too large or invalid');
        return;
    }
    appState.audioBlob = file;
    document.getElementById('file-info').style.display = 'block';
    document.getElementById('file-name').textContent = file.name;
    predictBtn.disabled = false;
}

async function startRecording() {
    try {
        appState.stream = await navigator.mediaDevices.getUserMedia({
            audio: { sampleRate: 16000, channelCount: 1 }
        });

        appState.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const source = appState.audioContext.createMediaStreamSource(appState.stream);
        const bufferSize = 4096;
        appState.scriptNode = appState.audioContext.createScriptProcessor(bufferSize, 1, 1);

        appState.scriptNode.onaudioprocess = e => {
            if (!appState.isRecording) return;
            const input = e.inputBuffer.getChannelData(0);
            appState.recordedChunks.push(new Float32Array(input));
        };

        source.connect(appState.scriptNode);
        appState.scriptNode.connect(appState.audioContext.destination);

        appState.isRecording = true;
        appState.recordedChunks = [];
        startRecordBtn.style.display = 'none';
        stopRecordBtn.style.display = 'flex';

        setTimeout(() => {
            if (appState.isRecording) stopRecording();
        }, 1200);

    } catch (err) {
        showError('Microphone access denied');
    }
}

function stopRecording() {
    if (!appState.isRecording) return;

    appState.isRecording = false;
    startRecordBtn.style.display = 'flex';
    stopRecordBtn.style.display = 'none';

    if (appState.stream) appState.stream.getTracks().forEach(t => t.stop());
    if (appState.scriptNode) appState.scriptNode.disconnect();
    if (appState.audioContext) appState.audioContext.close();

    const wavBlob = encodeWAV(appState.recordedChunks, 16000);
    appState.audioBlob = wavBlob;
    predictBtn.disabled = false;
}

function encodeWAV(samples, sampleRate) {
    const numSamples = samples.reduce((a, b) => a + b.length, 0);
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);

    const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, numSamples * 2, true);

    let offset = 44;
    for (const chunk of samples) {
        for (let i = 0; i < chunk.length; i++) {
            const s = Math.max(-1, Math.min(1, chunk[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

async function predictKeyword() {
    if (!appState.audioBlob) return;

    resultsSection.style.display = 'block';
    loadingSpinner.style.display = 'flex';
    resultsContent.style.display = 'none';
    errorMessage.style.display = 'none';

    const fd = new FormData();
    fd.append('audio', appState.audioBlob, 'audio.wav');

    try {
        const res = await fetch(`${API_BASE_URL}/predict`, { method: 'POST', body: fd });
        const data = await res.json();

        loadingSpinner.style.display = 'none';

        if (!data.success) {
            showError(data.error || 'Prediction failed');
            return;
        }

        document.getElementById('predicted-keyword').textContent = data.keyword;
        const conf = data.confidence;
        document.getElementById('confidence-score').textContent = `${conf}%`;
        document.getElementById('confidence-fill').style.width = `${conf}%`;

        if (data.spectrogram) {
            document.getElementById('spectrogram-image').src = data.spectrogram;
            document.getElementById('spectrogram-container').style.display = 'block';
        }

        resultsContent.style.display = 'block';
        predictBtn.disabled = true;

    } catch (err) {
        loadingSpinner.style.display = 'none';
        showError('Server not running');
    }
}

function showError(msg) {
    errorMessage.style.display = 'block';
    errorText.textContent = msg;
}

function resetApp() {
    appState.audioBlob = null;
    appState.recordedChunks = [];
    document.getElementById('audio-file').value = '';
    document.getElementById('file-info').style.display = 'none';
    predictBtn.disabled = true;
    resultsSection.style.display = 'none';
    startRecordBtn.style.display = 'flex';
    stopRecordBtn.style.display = 'none';
}