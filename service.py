import base64
import os
import threading
import time
import uuid
from collections import deque
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO

from app import TrafficLightController


class ConfigUpdate(BaseModel):
    threshold: int | None = None
    yellow_duration: float | None = None
    source: str | None = None


MODEL_NAME = os.environ.get('MODEL', 'yolov8n.pt')
CONFIDENCE = float(os.environ.get('CONF', '0.35'))
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

app = FastAPI()

# Shared state for server-side capture mode.
state: Dict = {
    'count': 0,
    'light': 'RED',
    'last_seen': 0.0,
    'running': False,
    'frame': None,  # latest JPEG bytes
    'source': '0',
}
state_lock = threading.Lock()

model = None
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


def load_model():
    global model
    if model is None:
        with model_init_lock:
            if model is None:
                model = YOLO(MODEL_NAME)
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


def run_inference(frame):
    model_ref = load_model()
    with model_lock:
        results = model_ref.predict(source=frame, conf=CONFIDENCE, verbose=False)
    r = results[0]
    count = count_vehicles(r, COUNT_CLASSES, model_ref.names)
    return r, count


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

    current_source = source
    while True:
        with state_lock:
            desired_source = state.get('source', source)
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
    load_model()
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
    <title>Traffic Light Vehicle Counter</title>
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
            max-width: 1100px;
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
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
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

        @media (max-width: 700px) {
            .controls {
                flex-direction: column;
                align-items: flex-start;
            }
        }
    </style>
</head>
<body>
    <div class="bg-blob one"></div>
    <div class="bg-blob two"></div>
    <main class="app">
        <header>
            <h1>Traffic Light Vehicle Counter</h1>
            <p>Open your camera, point it at traffic, and the counter and light state will update in real time.</p>
        </header>
        <section class="grid">
            <div class="card">
                <h2>Live Camera</h2>
                <div class="media">
                    <video id="camera" autoplay playsinline muted></video>
                </div>
                <div class="hint">Your camera stream stays on this page; frames are sent to the server for detection.</div>
            </div>
            <div class="card">
                <h2>AI View</h2>
                <div class="media">
                    <img id="annotated" alt="Annotated detections" />
                </div>
                <div class="stats">
                    <div class="metric">
                        <div class="label">Vehicles</div>
                        <div class="value" id="countValue">0</div>
                    </div>
                    <div class="lights">
                        <div class="light-card">
                            <div class="light-title">Car</div>
                            <div class="light" id="carLight" data-light="RED">
                                <div class="bulb red"></div>
                                <div class="bulb yellow"></div>
                                <div class="bulb green"></div>
                                <div class="light-label" id="carLightText">RED</div>
                            </div>
                        </div>
                        <div class="light-card">
                            <div class="light-title">Pedestrian</div>
                            <div class="light" id="pedLight" data-light="GREEN">
                                <div class="bulb red"></div>
                                <div class="bulb yellow"></div>
                                <div class="bulb green"></div>
                                <div class="light-label" id="pedLightText">GREEN</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        <section class="controls">
            <button id="startBtn">Start camera</button>
            <button id="stopBtn" class="secondary" disabled>Stop</button>
            <button id="flipBtn" class="secondary">Flip view</button>
            <div class="control">
                <label for="fps">Send rate</label>
                <input id="fps" type="range" min="1" max="6" value="4" />
                <span id="fpsValue">4 fps</span>
            </div>
        </section>
        <div class="status" id="status">Idle. Click "Start camera" to begin.</div>
        <div class="hint">Camera access requires HTTPS (or localhost). For sharing on the internet, use a tunnel that provides HTTPS.</div>
    </main>
    <canvas id="capture" class="hidden"></canvas>
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
    with state_lock:
        if cfg.source is not None:
            state['source'] = cfg.source
    return {'ok': True, 'updated': cfg.dict()}


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
    r, raw_count = run_inference(frame)
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
