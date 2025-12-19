import base64
import os
import threading
import time
import uuid
from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO

from app import TrafficLightController


class ConfigUpdate(BaseModel):
    threshold: Optional[int] = None
    yellow_duration: Optional[float] = None
    conf: Optional[float] = None
    model: Optional[str] = None
    source: Optional[str] = None


MODEL_NANO_NAME = os.environ.get('MODEL_NANO', 'yolov8n.pt')
MODEL_MEDIUM_NAME = os.environ.get('MODEL_MEDIUM', 'yolov8m.pt')
MODEL_LARGE_NAME = os.environ.get('MODEL_LARGE', 'yolov8l.pt')
MODEL_NAME = os.environ.get('MODEL', MODEL_NANO_NAME)
CONFIDENCE = float(os.environ.get('CONF', '0.2'))
THRESHOLD = int(os.environ.get('THRESHOLD', '5'))
YELLOW_DURATION = float(os.environ.get('YELLOW_DURATION', '3.0'))
MIN_GREEN_TIME = float(os.environ.get('MIN_GREEN_TIME', '5.0'))
MIN_RED_TIME = float(os.environ.get('MIN_RED_TIME', '3.0'))
HYSTERESIS = int(os.environ.get('HYSTERESIS', '1'))
SMOOTHING_WINDOW = float(os.environ.get('SMOOTHING_WINDOW', '2.0'))
SESSION_TTL = float(os.environ.get('SESSION_TTL', '600'))
SERVER_CAPTURE = os.environ.get('SERVER_CAPTURE', '0') == '1'
COUNT_CLASSES = set(
    s.strip() for s in os.environ.get('COUNT_CLASSES', 'car,motorcycle,bus,truck').split(',') if s.strip()
)
ALLOWED_MODELS = [MODEL_NANO_NAME, MODEL_MEDIUM_NAME, MODEL_LARGE_NAME]

app = FastAPI()

# Shared state for server-side capture mode.
state: Dict = {
    'count': 0,
    'light': 'RED',
    'last_seen': 0.0,
    'running': False,
    'frame': None,  # latest JPEG bytes
    'source': '0',
    'threshold': THRESHOLD,
    'yellow_duration': YELLOW_DURATION,
    'conf': CONFIDENCE,
    'model': MODEL_NAME,
}
state_lock = threading.Lock()

models: Dict[str, YOLO] = {}
model_lock = threading.Lock()
model_init_lock = threading.Lock()

sessions: Dict[str, 'SessionState'] = {}
sessions_lock = threading.Lock()
last_cleanup = 0.0
cleanup_interval = 60.0


class SessionState:
    def __init__(self):
        self.controller = TrafficLightController(threshold=THRESHOLD, yellow_duration=YELLOW_DURATION)
        self.controller.set_timing(MIN_GREEN_TIME, MIN_RED_TIME, HYSTERESIS)
        self.counts = deque()
        self.last_seen = time.time()
        self.last_count = 0
        self.last_light = 'RED'

    def update(self, vehicle_count: int):
        now = time.time()
        self.last_seen = now
        self.counts.append((now, vehicle_count))
        while self.counts and (now - self.counts[0][0]) > SMOOTHING_WINDOW:
            self.counts.popleft()
        avg_count = int(round(sum(c for _, c in self.counts) / max(1, len(self.counts))))
        self.controller.update(avg_count)
        self.last_count = avg_count
        self.last_light = self.controller.get_state()
        return avg_count, self.last_light


def load_model(name: str):
    with model_init_lock:
        model = models.get(name)
        if model is None:
            model = YOLO(name)
            models[name] = model
    return model


def count_vehicles(result, vehicle_classes, names):
    count = 0
    try:
        cls_tensor = result.boxes.cls
        if cls_tensor is not None:
            for c in cls_tensor.cpu().numpy().astype(int):
                name = names.get(int(c), str(c))
                if name in vehicle_classes:
                    count += 1
            return count
    except Exception:
        pass

    try:
        for box in result.boxes.data.tolist():
            class_id = int(box[5])
            name = names.get(class_id, str(class_id))
            if name in vehicle_classes:
                count += 1
    except Exception:
        pass
    return count


def run_inference_with_conf(frame, model_name: str, conf: float):
    model_ref = load_model(model_name)
    with model_lock:
        results = model_ref.predict(source=frame, conf=conf, verbose=False)
    r = results[0]
    count = count_vehicles(r, COUNT_CLASSES, model_ref.names)
    return r, count


def run_inference(frame):
    with state_lock:
        conf = float(state.get('conf', CONFIDENCE))
        model_name = state.get('model', MODEL_NAME)
    return run_inference_with_conf(frame, model_name, conf)


def decode_image(data: bytes):
    if not data:
        return None
    np_data = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    return frame


def _cleanup_sessions_locked(now):
    global last_cleanup
    if (now - last_cleanup) < cleanup_interval:
        return
    expired = [sid for sid, s in sessions.items() if (now - s.last_seen) > SESSION_TTL]
    for sid in expired:
        sessions.pop(sid, None)
    last_cleanup = now


def get_or_create_session(session_id: str):
    now = time.time()
    with sessions_lock:
        session = sessions.get(session_id)
        if session is None:
            session = SessionState()
            sessions[session_id] = session
        session.last_seen = now
        _cleanup_sessions_locked(now)
    return session


def detection_loop(source: str = '0', conf: float = 0.35, threshold: int = 5, yellow_duration: float = 3.0):
    def open_capture(src):
        try:
            src_int = int(src)
            c = cv2.VideoCapture(src_int)
            if c.isOpened():
                return c
        except Exception:
            pass

        c = cv2.VideoCapture(src)
        if c.isOpened():
            return c

        common_paths = ['/video', '/video_feed', '/stream', '/mjpeg']
        for p in common_paths:
            try_url = src.rstrip('/') + p
            c = cv2.VideoCapture(try_url)
            if c.isOpened():
                return c
        return None

    cap = open_capture(source)
    if cap is None or not cap.isOpened():
        print('ERROR: Could not open video source for service:', source)
        cap = None

    controller = TrafficLightController(threshold=threshold, yellow_duration=yellow_duration)
    controller.set_timing(MIN_GREEN_TIME, MIN_RED_TIME, HYSTERESIS)
    counts_deque = deque()

    with state_lock:
        state['running'] = True
        state['threshold'] = int(threshold)
        state['yellow_duration'] = float(yellow_duration)
        state['conf'] = float(conf)

    current_source = source
    while True:
        with state_lock:
            desired_source = state.get('source', source)
            desired_threshold = state.get('threshold', threshold)
            desired_yellow = state.get('yellow_duration', yellow_duration)
        if desired_source != current_source:
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            cap = open_capture(desired_source)
            if cap is not None and cap.isOpened():
                print('Switched video source to', desired_source)
                current_source = desired_source
            else:
                print('Failed to open new source:', desired_source)
        if desired_threshold != controller.threshold:
            controller.threshold = int(desired_threshold)
        if desired_yellow != controller.yellow_duration:
            controller.yellow_duration = float(desired_yellow)

        if cap is None:
            time.sleep(0.5)
            continue

        ret, frame = cap.read()
        if not ret:
            cap2 = open_capture(current_source)
            if cap2 and cap2.isOpened():
                cap = cap2
                ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

        r, count = run_inference(frame)

        now = time.time()
        counts_deque.append((now, count))
        while counts_deque and (now - counts_deque[0][0]) > SMOOTHING_WINDOW:
            counts_deque.popleft()
        avg_count = int(round(sum(c for _, c in counts_deque) / max(1, len(counts_deque))))

        controller.update(avg_count)

        try:
            annotated = r.plot()
            ret2, jpeg = cv2.imencode('.jpg', annotated)
            jpeg_bytes = jpeg.tobytes() if ret2 else None
        except Exception:
            jpeg_bytes = None

        with state_lock:
            state['count'] = avg_count
            state['light'] = controller.get_state()
            state['last_seen'] = time.time()
            if jpeg_bytes:
                state['frame'] = jpeg_bytes
            state['source'] = current_source

        time.sleep(0.05)


@app.on_event('startup')
def startup_event():
    load_model(MODEL_NAME)
    if SERVER_CAPTURE:
        src = os.environ.get('SOURCE', '0')
        conf = float(os.environ.get('CONF', 0.35))
        threshold = int(os.environ.get('THRESHOLD', 5))
        yellow = float(os.environ.get('YELLOW_DURATION', 3.0))
        t = threading.Thread(target=detection_loop, args=(src, conf, threshold, yellow), daemon=True)
        t.start()


@app.get('/status')
def get_status():
    with state_lock:
        return {
            'count': state['count'],
            'light': state['light'],
            'last_seen': state['last_seen'],
            'running': state['running'],
            'source': state['source'],
        }


@app.get('/')
def index():
    html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Intelligent Traffic System</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet" />
    <style>
        :root {
            --bg1: #f7f1e4;
            --bg2: #d9efe8;
            --panel: #ffffff;
            --ink: #1b1b16;
            --muted: #5d5d57;
            --accent: #ff6b35;
            --accent-2: #0a6b6f;
            --shadow: rgba(15, 15, 15, 0.12);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: "Space Grotesk", sans-serif;
            color: var(--ink);
            background: radial-gradient(circle at 15% 15%, #fff6d8 0, transparent 55%),
                radial-gradient(circle at 90% 20%, #cfe9ff 0, transparent 45%),
                linear-gradient(120deg, var(--bg1), var(--bg2));
            min-height: 100vh;
        }

        .bg-blob {
            position: fixed;
            z-index: -1;
            width: 440px;
            height: 440px;
            border-radius: 50%;
            filter: blur(40px);
            opacity: 0.6;
        }

        .bg-blob.one {
            background: #ffd9b8;
            top: -140px;
            left: -120px;
        }

        .bg-blob.two {
            background: #b9f0da;
            bottom: -160px;
            right: -140px;
        }

        .app {
            max-width: 1240px;
            margin: 0 auto;
            padding: 28px 20px 36px;
        }

        header {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 20px;
        }

        header h1 {
            font-size: clamp(28px, 4vw, 40px);
            margin: 0;
            letter-spacing: -0.02em;
        }

        header p {
            margin: 0;
            color: var(--muted);
            max-width: 680px;
        }

        .grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 18px;
            margin-bottom: 18px;
        }

        .card {
            background: var(--panel);
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 18px 50px var(--shadow);
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .card h2 {
            margin: 0;
            font-size: 18px;
            color: var(--accent-2);
        }

        .media {
            position: relative;
            border-radius: 14px;
            overflow: hidden;
            background: #0b0b0b;
            border: 1px solid rgba(0, 0, 0, 0.08);
            min-height: 220px;
        }

        video,
        img {
            width: 100%;
            height: 100%;
            display: block;
            object-fit: cover;
            aspect-ratio: 16 / 9;
        }

        .flipped {
            transform: scaleX(-1);
        }

        .stats {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 16px;
            align-items: center;
        }

        .ai-layout {
            display: grid;
            gap: 16px;
            align-items: center;
            grid-template-columns: repeat(3, minmax(90px, 1fr));
            grid-template-areas:
                "media media media"
                "car count ped";
        }

        .ai-layout .media {
            grid-area: media;
        }

        .ai-layout .car-light {
            grid-area: car;
        }

        .ai-layout .ped-light {
            grid-area: ped;
        }

        .ai-layout .count-pill {
            grid-area: count;
        }

        .lights {
            display: grid;
            grid-template-columns: repeat(2, minmax(90px, 1fr));
            gap: 12px;
            align-items: center;
        }

        .light-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
        }

        .light-title {
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .metric {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .metric .label {
            font-size: 13px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .metric .value {
            font-size: 36px;
            font-weight: 600;
        }

        .metric-bottom {
            margin-top: 16px;
            display: flex;
            align-items: baseline;
            justify-content: center;
            gap: 10px;
            text-align: center;
        }

        .metric-bottom .label {
            font-size: 13px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .metric-bottom .value {
            font-size: 34px;
            font-weight: 600;
        }

        .light {
            background: #131313;
            border-radius: 18px;
            padding: 12px;
            display: grid;
            gap: 10px;
            place-items: center;
            min-width: 108px;
        }

        .bulb {
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: #3b3b3b;
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.7);
            transition: all 0.2s ease;
        }

        .light[data-light="RED"] .bulb.red {
            background: #ff4d3a;
            box-shadow: 0 0 18px rgba(255, 77, 58, 0.7);
        }

        .light[data-light="YELLOW"] .bulb.yellow {
            background: #ffd24a;
            box-shadow: 0 0 18px rgba(255, 210, 74, 0.7);
        }

        .light[data-light="GREEN"] .bulb.green {
            background: #46e07a;
            box-shadow: 0 0 18px rgba(70, 224, 122, 0.7);
        }

        .light-label {
            font-size: 12px;
            color: #f2f2f2;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 16px;
            padding: 12px 14px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
        }

        button {
            border: none;
            padding: 10px 16px;
            border-radius: 999px;
            background: var(--accent);
            color: #1d1d1b;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 10px 20px rgba(255, 107, 53, 0.3);
        }

        button.secondary {
            background: #e7f0ef;
            color: var(--ink);
            box-shadow: none;
        }

        button:disabled {
            cursor: not-allowed;
            opacity: 0.6;
        }

        button:not(:disabled):hover {
            transform: translateY(-1px);
        }

        .control {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        input[type="range"] {
            accent-color: var(--accent-2);
        }

        .status {
            margin-top: 12px;
            color: var(--muted);
        }

        .hint {
            margin-top: 6px;
            font-size: 13px;
            color: var(--muted);
        }

        .hidden {
            display: none;
        }

        .ghost {
            position: absolute;
            width: 1px;
            height: 1px;
            opacity: 0;
            pointer-events: none;
            left: -9999px;
            top: -9999px;
        }

        @media (max-width: 700px) {
            .controls {
                flex-direction: column;
                align-items: flex-start;
            }
        }

        @media (min-width: 900px) {
            .ai-layout {
                grid-template-columns: minmax(120px, 160px) minmax(0, 1fr) minmax(120px, 160px);
                grid-template-areas:
                    "car media ped"
                    "count count count";
            }
        }
    </style>
</head>
<body>
    <div class="bg-blob one"></div>
    <div class="bg-blob two"></div>
    <main class="app">
        <header>
            <h1>Intelligent Traffic System</h1>
            <p>Select a model and start detection.</p>
        </header>
        <section class="grid">
            <div class="card">
                <h2>AI View</h2>
                <div class="ai-layout">
                    <div class="light-card car-light">
                        <div class="light-title">Car</div>
                        <div class="light" id="carLight" data-light="RED">
                            <div class="bulb red"></div>
                            <div class="bulb yellow"></div>
                            <div class="bulb green"></div>
                            <div class="light-label" id="carLightText">RED</div>
                        </div>
                    </div>
                    <div class="media">
                        <img id="annotated" alt="Annotated detections" />
                    </div>
                    <div class="light-card ped-light">
                        <div class="light-title">Pedestrian</div>
                        <div class="light" id="pedLight" data-light="GREEN">
                            <div class="bulb red"></div>
                            <div class="bulb yellow"></div>
                            <div class="bulb green"></div>
                            <div class="light-label" id="pedLightText">GREEN</div>
                        </div>
                    </div>
                    <div class="metric-bottom count-pill">
                        <div class="label">Vehicles</div>
                        <div class="value" id="countValue">0</div>
                    </div>
                </div>
            </div>
        </section>
        <section class="controls">
            <button id="startBtn">Start detection</button>
            <button id="stopBtn" class="secondary" disabled>Stop</button>
            <button id="flipBtn" class="secondary">Flip view</button>
            <div class="control">
                <label for="fps">Send rate</label>
                <input id="fps" type="range" min="1" max="6" value="4" />
                <span id="fpsValue">4 fps</span>
            </div>
            <div class="control">
                <label for="threshold">Threshold</label>
                <input id="threshold" type="number" min="0" step="1" value="5" />
                <button id="applyThreshold" class="secondary" type="button">Apply</button>
            </div>
            <div class="control">
                <label for="modelSelect">Model</label>
                <select id="modelSelect">
                    <option value="yolov8n.pt">Nano</option>
                    <option value="yolov8m.pt">Medium</option>
                    <option value="yolov8l.pt">Large</option>
                </select>
                <button id="applyModel" class="secondary" type="button">Load</button>
            </div>
        </section>
        <div class="status" id="status">Idle. Click "Start detection" to begin.</div>
        <div class="hint">Camera access requires HTTPS (or localhost). For sharing on the internet, use a tunnel that provides HTTPS.</div>
    </main>
    <canvas id="capture" class="hidden"></canvas>
    <video id="camera" autoplay playsinline muted class="ghost"></video>
    <script>
        const video = document.getElementById('camera');
        const canvas = document.getElementById('capture');
        const annotated = document.getElementById('annotated');
        const countValue = document.getElementById('countValue');
        const carLight = document.getElementById('carLight');
        const carLightText = document.getElementById('carLightText');
        const pedLight = document.getElementById('pedLight');
        const pedLightText = document.getElementById('pedLightText');
        const statusEl = document.getElementById('status');
        const fpsInput = document.getElementById('fps');
        const fpsValue = document.getElementById('fpsValue');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const flipBtn = document.getElementById('flipBtn');
        const thresholdInput = document.getElementById('threshold');
        const applyThresholdBtn = document.getElementById('applyThreshold');
        const modelSelect = document.getElementById('modelSelect');
        const applyModelBtn = document.getElementById('applyModel');

        const placeholder = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';
        annotated.src = placeholder;

        let stream = null;
        let timer = null;
        let busy = false;
        let flipped = false;

        function setStatus(text) {
            statusEl.textContent = text;
        }

        function updateLight(state) {
            const carState = state || 'RED';
            carLight.dataset.light = carState;
            carLightText.textContent = carState;
            const pedState = carState === 'RED' ? 'GREEN' : 'RED';
            pedLight.dataset.light = pedState;
            pedLightText.textContent = pedState;
        }

        function applyFlip() {
            if (flipped) {
                video.classList.add('flipped');
                annotated.classList.add('flipped');
                flipBtn.textContent = 'Unflip view';
            } else {
                video.classList.remove('flipped');
                annotated.classList.remove('flipped');
                flipBtn.textContent = 'Flip view';
            }
        }

        function updateInterval() {
            const fps = Number(fpsInput.value || 4);
            fpsValue.textContent = fps + ' fps';
            if (timer) {
                clearInterval(timer);
                timer = setInterval(captureAndSend, 1000 / Math.max(fps, 1));
            }
        }

        function modelLabel(name) {
            const lower = name.toLowerCase();
            if (lower.includes('yolov8n')) {
                return 'Nano';
            }
            if (lower.includes('yolov8m')) {
                return 'Medium';
            }
            if (lower.includes('yolov8l')) {
                return 'Large';
            }
            return name;
        }

        async function loadConfig() {
            try {
                const res = await fetch('/config');
                if (!res.ok) {
                    return;
                }
                const data = await res.json();
                if (typeof data.threshold === 'number') {
                    thresholdInput.value = data.threshold;
                }
                if (Array.isArray(data.models) && data.models.length) {
                    modelSelect.innerHTML = '';
                    data.models.forEach((name) => {
                        const opt = document.createElement('option');
                        opt.value = name;
                        opt.textContent = modelLabel(name);
                        modelSelect.appendChild(opt);
                    });
                }
                if (typeof data.model === 'string') {
                    modelSelect.value = data.model;
                }
            } catch (err) {
                // ignore
            }
        }

        async function applyModel() {
            const selected = modelSelect.value;
            if (!selected) {
                return;
            }
            try {
                const res = await fetch('/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: selected })
                });
                if (!res.ok) {
                    const text = await res.text();
                    throw new Error(text || 'Server error');
                }
                setStatus('Model updated.');
            } catch (err) {
                setStatus('Error: ' + err.message);
            }
        }

        async function applyThreshold() {
            const raw = Number(thresholdInput.value);
            if (!Number.isFinite(raw) || raw < 0) {
                setStatus('Threshold must be 0 or higher.');
                return;
            }
            const payload = { threshold: Math.round(raw) };
            try {
                const res = await fetch('/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) {
                    const text = await res.text();
                    throw new Error(text || 'Server error');
                }
                setStatus('Threshold updated.');
            } catch (err) {
                setStatus('Error: ' + err.message);
            }
        }

        async function startCamera() {
            if (stream) {
                return;
            }
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { ideal: 'environment' } },
                    audio: false
                });
                video.srcObject = stream;
                await video.play();
                startBtn.disabled = true;
                stopBtn.disabled = false;
                setStatus('Camera active. Running detection...');
                const fps = Number(fpsInput.value || 4);
                timer = setInterval(captureAndSend, 1000 / Math.max(fps, 1));
            } catch (err) {
                setStatus('Camera access denied or unavailable.');
            }
        }

        function stopCamera() {
            if (timer) {
                clearInterval(timer);
                timer = null;
            }
            if (stream) {
                stream.getTracks().forEach((track) => track.stop());
                stream = null;
            }
            startBtn.disabled = false;
            stopBtn.disabled = true;
            setStatus('Camera stopped.');
        }

        async function captureAndSend() {
            if (!stream || busy || video.readyState < 2) {
                return;
            }
            const width = 640;
            const vWidth = video.videoWidth || width;
            const vHeight = video.videoHeight || (width * 0.75);
            const height = Math.round((width / vWidth) * vHeight);
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, width, height);
            busy = true;
            canvas.toBlob(async (blob) => {
                if (!blob) {
                    busy = false;
                    return;
                }
                try {
                    const res = await fetch('/infer', {
                        method: 'POST',
                        headers: { 'Content-Type': 'image/jpeg' },
                        body: blob
                    });
                    if (!res.ok) {
                        const text = await res.text();
                        throw new Error(text || 'Server error');
                    }
                    const data = await res.json();
                    if (data.image) {
                        annotated.src = 'data:image/jpeg;base64,' + data.image;
                    }
                    if (typeof data.count === 'number') {
                        countValue.textContent = data.count;
                    }
                    if (data.light) {
                        updateLight(data.light);
                    }
                    setStatus('Last update: ' + new Date().toLocaleTimeString());
                } catch (err) {
                    setStatus('Error: ' + err.message);
                } finally {
                    busy = false;
                }
            }, 'image/jpeg', 0.7);
        }

        fpsInput.addEventListener('input', updateInterval);
        startBtn.addEventListener('click', startCamera);
        stopBtn.addEventListener('click', stopCamera);
        flipBtn.addEventListener('click', () => {
            flipped = !flipped;
            applyFlip();
        });
        thresholdInput.addEventListener('change', applyThreshold);
        applyThresholdBtn.addEventListener('click', applyThreshold);
        modelSelect.addEventListener('change', applyModel);
        applyModelBtn.addEventListener('click', applyModel);
        loadConfig();
    </script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get('/snapshot')
def snapshot():
    with state_lock:
        frame = state.get('frame')
    if frame:
        return Response(content=frame, media_type='image/jpeg')
    return Response(status_code=204)


@app.get('/stream')
def stream():
    def gen():
        boundary = b'--frame\r\n'
        while True:
            with state_lock:
                frame = state.get('frame')
            if frame:
                yield boundary
                yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.1)

    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.post('/config')
def update_config(cfg: ConfigUpdate):
    updates = {}
    global THRESHOLD, YELLOW_DURATION, CONFIDENCE
    with state_lock:
        if cfg.source is not None:
            state['source'] = cfg.source
            updates['source'] = cfg.source
        if cfg.threshold is not None:
            THRESHOLD = max(0, int(cfg.threshold))
            state['threshold'] = THRESHOLD
            updates['threshold'] = THRESHOLD
        if cfg.yellow_duration is not None:
            YELLOW_DURATION = max(0.0, float(cfg.yellow_duration))
            state['yellow_duration'] = YELLOW_DURATION
            updates['yellow_duration'] = YELLOW_DURATION
        if cfg.conf is not None:
            CONFIDENCE = max(0.01, min(0.95, float(cfg.conf)))
            state['conf'] = CONFIDENCE
            updates['conf'] = CONFIDENCE
        if cfg.model is not None:
            if cfg.model not in ALLOWED_MODELS:
                raise HTTPException(status_code=400, detail='Unsupported model')
            state['model'] = cfg.model
            updates['model'] = cfg.model
            load_model(cfg.model)
    if 'threshold' in updates or 'yellow_duration' in updates:
        with sessions_lock:
            for session in sessions.values():
                if 'threshold' in updates:
                    session.controller.threshold = THRESHOLD
                if 'yellow_duration' in updates:
                    session.controller.yellow_duration = YELLOW_DURATION
    return {'ok': True, 'updated': updates or cfg.dict()}


@app.get('/config')
def get_config():
    with state_lock:
        return {
            'threshold': state.get('threshold', THRESHOLD),
            'yellow_duration': state.get('yellow_duration', YELLOW_DURATION),
            'conf': state.get('conf', CONFIDENCE),
            'model': state.get('model', MODEL_NAME),
            'models': ALLOWED_MODELS,
            'source': state.get('source', '0'),
        }


@app.post('/infer')
async def infer(request: Request, response: Response):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail='Empty body')

    frame = decode_image(body)
    if frame is None:
        raise HTTPException(status_code=400, detail='Invalid image')

    session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = uuid.uuid4().hex
        response.set_cookie('session_id', session_id, httponly=True, samesite='lax')

    session = get_or_create_session(session_id)
    with state_lock:
        conf = float(state.get('conf', CONFIDENCE))
        model_name = state.get('model', MODEL_NAME)
    r, raw_count = run_inference_with_conf(frame, model_name, conf)
    avg_count, light_state = session.update(raw_count)

    try:
        annotated = r.plot()
        ret, jpeg = cv2.imencode('.jpg', annotated)
        image_b64 = base64.b64encode(jpeg.tobytes()).decode('ascii') if ret else ''
    except Exception:
        image_b64 = ''

    return {
        'count': avg_count,
        'light': light_state,
        'image': image_b64,
    }


if __name__ == '__main__':
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run('service:app', host='0.0.0.0', port=args.port, log_level='info')
