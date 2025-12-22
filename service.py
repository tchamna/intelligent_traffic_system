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
MODEL_NAME = os.environ.get('MODEL', MODEL_LARGE_NAME)
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
        self.counts = {name: deque() for name in ALLOWED_MODELS}
        self.last_seen = time.time()
        self.last_count = 0
        self.last_light = 'RED'

    def update_counts(self, counts: Dict[str, int], selected_model: str):
        now = time.time()
        self.last_seen = now
        avg_counts = {}
        for name, count in counts.items():
            bucket = self.counts.get(name)
            if bucket is None:
                bucket = deque()
                self.counts[name] = bucket
            bucket.append((now, count))
            while bucket and (now - bucket[0][0]) > SMOOTHING_WINDOW:
                bucket.popleft()
            avg_counts[name] = int(round(sum(c for _, c in bucket) / max(1, len(bucket))))
        selected_avg = avg_counts.get(selected_model, 0)
        self.controller.update(selected_avg)
        self.last_count = selected_avg
        self.last_light = self.controller.get_state()
        return avg_counts, self.last_light


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
            'model': state.get('model', MODEL_NAME),
            'threshold': state.get('threshold', THRESHOLD),
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
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&display=swap" rel="stylesheet" />
    <style>
        :root {
            --bg: #f3f4f6;
            --panel: #ffffff;
            --ink: #111827;
            --muted: #6b7280;
            --accent: #0ea5e9;
            --accent-2: #16a34a;
            --danger: #ef4444;
            --shadow: rgba(15, 23, 42, 0.08);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: "Sora", sans-serif;
            color: var(--ink);
            background: radial-gradient(circle at 20% 10%, #ffffff 0, transparent 45%),
                radial-gradient(circle at 85% 20%, #e0f2fe 0, transparent 50%),
                linear-gradient(180deg, #f8fafc, var(--bg));
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
            background: #e2e8f0;
            top: -140px;
            left: -120px;
        }

        .bg-blob.two {
            background: #e0f2fe;
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
            color: var(--ink);
        }

        .media {
            position: relative;
            border-radius: 14px;
            overflow: hidden;
            background: #0f172a;
            border: 1px solid rgba(15, 23, 42, 0.12);
            min-height: 220px;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
        }

        video,
        img {
            width: 100%;
            height: 100%;
            display: block;
            object-fit: cover;
            aspect-ratio: 16 / 9;
        }

        .media.is-live {
            outline: 2px solid rgba(14, 165, 233, 0.4);
            box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.12);
        }

        .media.is-processing {
            animation: live-glow 1.2s ease-in-out infinite;
        }

        .media-placeholder {
            position: absolute;
            inset: 0;
            display: grid;
            place-items: center;
            padding: 20px;
            text-align: center;
            color: #e2e8f0;
            font-size: 14px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background: radial-gradient(circle at 50% 30%, rgba(148, 163, 184, 0.25), transparent 55%);
        }

        .media-placeholder.hidden {
            display: none;
        }

        .live-badge {
            position: absolute;
            top: 12px;
            right: 12px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.72);
            color: #f8fafc;
            font-size: 11px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .live-badge .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 10px rgba(34, 197, 94, 0.8);
            animation: pulse 1.4s ease-in-out infinite;
        }

        .live-badge.hidden {
            display: none;
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
            flex-direction: column;
            align-items: center;
            gap: 6px;
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

        .count-list {
            display: flex;
            flex-direction: column;
            gap: 4px;
            font-size: 13px;
            width: 100%;
            max-width: 150px;
            z-index: 1;
        }

        .count-row {
            display: flex;
            justify-content: space-between;
            gap: 10px;
        }

        .count-name {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .count-value {
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
            animation: glow-red 1.8s ease-in-out infinite;
        }

        .light[data-light="YELLOW"] .bulb.yellow {
            background: #ffd24a;
            box-shadow: 0 0 18px rgba(255, 210, 74, 0.7);
            animation: glow-yellow 1.8s ease-in-out infinite;
        }

        .light[data-light="GREEN"] .bulb.green {
            background: #46e07a;
            box-shadow: 0 0 18px rgba(70, 224, 122, 0.7);
            animation: glow-green 1.8s ease-in-out infinite;
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

        .menu {
            border: none;
        }

        .menu summary {
            list-style: none;
            cursor: pointer;
            font-weight: 600;
            color: var(--ink);
            padding: 8px 14px;
            border-radius: 999px;
            background: #e2e8f0;
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
            transition: transform 0.2s ease, filter 0.2s ease;
        }

        .menu summary::-webkit-details-marker {
            display: none;
        }

        .menu[open] summary {
            transform: translateY(-1px);
            filter: brightness(1.02);
        }

        .menu-body {
            margin-top: 10px;
            display: grid;
            gap: 12px;
        }

        button {
            border: none;
            padding: 10px 16px;
            border-radius: 999px;
            background: linear-gradient(135deg, #0ea5e9, #38bdf8);
            color: #0f172a;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
            box-shadow: 0 12px 24px rgba(14, 165, 233, 0.25);
        }

        button.secondary {
            background: #e2e8f0;
            color: #0f172a;
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
        }

        button:disabled {
            cursor: not-allowed;
            opacity: 0.6;
        }

        button:not(:disabled):hover {
            transform: translateY(-1px);
            filter: brightness(1.02);
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

        .toast {
            position: fixed;
            right: 20px;
            bottom: 20px;
            background: rgba(15, 23, 42, 0.95);
            color: #f8fafc;
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 13px;
            letter-spacing: 0.01em;
            box-shadow: 0 16px 30px rgba(15, 23, 42, 0.25);
            z-index: 20;
            opacity: 0;
            transform: translateY(10px);
            transition: opacity 0.2s ease, transform 0.2s ease;
        }

        .toast.show {
            opacity: 1;
            transform: translateY(0);
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

        @media (max-width: 520px) {
            .ai-layout {
                grid-template-columns: minmax(88px, 1fr) minmax(120px, 1fr) minmax(88px, 1fr);
            }

            .light {
                min-width: 88px;
                padding: 10px;
            }

            .bulb {
                width: 46px;
                height: 46px;
            }
        }

        @keyframes pulse {
            0% {
                transform: scale(1);
                opacity: 0.8;
            }
            50% {
                transform: scale(1.2);
                opacity: 1;
            }
            100% {
                transform: scale(1);
                opacity: 0.8;
            }
        }

        @keyframes live-glow {
            0% {
                box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.15);
            }
            50% {
                box-shadow: 0 0 0 6px rgba(14, 165, 233, 0.2);
            }
            100% {
                box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.15);
            }
        }

        @keyframes glow-red {
            0% {
                box-shadow: 0 0 12px rgba(255, 77, 58, 0.4);
            }
            50% {
                box-shadow: 0 0 22px rgba(255, 77, 58, 0.8);
            }
            100% {
                box-shadow: 0 0 12px rgba(255, 77, 58, 0.4);
            }
        }

        @keyframes glow-yellow {
            0% {
                box-shadow: 0 0 12px rgba(255, 210, 74, 0.4);
            }
            50% {
                box-shadow: 0 0 22px rgba(255, 210, 74, 0.8);
            }
            100% {
                box-shadow: 0 0 12px rgba(255, 210, 74, 0.4);
            }
        }

        @keyframes glow-green {
            0% {
                box-shadow: 0 0 12px rgba(70, 224, 122, 0.4);
            }
            50% {
                box-shadow: 0 0 22px rgba(70, 224, 122, 0.8);
            }
            100% {
                box-shadow: 0 0 12px rgba(70, 224, 122, 0.4);
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
                    <div class="media" id="mediaBox">
                        <img id="annotated" alt="Annotated detections" />
                        <div class="media-placeholder" id="mediaPlaceholder">Camera idle. Press Start detection.</div>
                        <div class="live-badge hidden" id="liveBadge"><span class="dot"></span> Live</div>
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
                        <div class="count-list" id="countList"></div>
                    </div>
                </div>
            </div>
        </section>
        <section class="controls">
            <button id="toggleBtn">Start detection</button>
            <details class="menu">
                <summary>Settings</summary>
                <div class="menu-body">
                    <button id="switchBtn" class="secondary" type="button">Switch camera</button>
                    <div class="control">
                        <label for="fps">Send rate (fps)</label>
                        <input id="fps" type="range" min="1" max="6" value="4" />
                        <span id="fpsValue">4 fps</span>
                    </div>
                    <div class="control">
                        <label for="threshold">Threshold (vehicles)</label>
                        <input id="threshold" type="range" min="0" max="20" step="1" value="5" />
                        <span id="thresholdValue">5</span>
                    </div>
                    <div class="control">
                        <label for="modelSelect">Model</label>
                        <select id="modelSelect">
                            <option value="yolov8n.pt">Nano</option>
                            <option value="yolov8m.pt">Medium</option>
                            <option value="yolov8l.pt" selected>Large</option>
                        </select>
                    </div>
                </div>
            </details>
        </section>
        <div class="status" id="status">Idle. Click "Start detection" to begin.</div>
        <div class="hint" id="logicNote"></div>
        <div class="hint">Camera access requires HTTPS (or localhost). For sharing on the internet, use a tunnel that provides HTTPS.</div>
    </main>
    <canvas id="capture" class="hidden"></canvas>
    <video id="camera" autoplay playsinline muted class="ghost"></video>
    <div class="toast" id="toast"></div>
    <script>
        const video = document.getElementById('camera');
        const canvas = document.getElementById('capture');
        const annotated = document.getElementById('annotated');
        const countList = document.getElementById('countList');
        const mediaBox = document.getElementById('mediaBox');
        const mediaPlaceholder = document.getElementById('mediaPlaceholder');
        const liveBadge = document.getElementById('liveBadge');
        const carLight = document.getElementById('carLight');
        const carLightText = document.getElementById('carLightText');
        const pedLight = document.getElementById('pedLight');
        const pedLightText = document.getElementById('pedLightText');
        const statusEl = document.getElementById('status');
        const logicNote = document.getElementById('logicNote');
        const toastEl = document.getElementById('toast');
        const fpsInput = document.getElementById('fps');
        const fpsValue = document.getElementById('fpsValue');
        const toggleBtn = document.getElementById('toggleBtn');
        const switchBtn = document.getElementById('switchBtn');
        const thresholdInput = document.getElementById('threshold');
        const thresholdValue = document.getElementById('thresholdValue');
        const modelSelect = document.getElementById('modelSelect');

        const placeholder = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';
        annotated.src = placeholder;
        buildCountList(Array.from(modelSelect.options).map((opt) => opt.value));
        updateThresholdDisplay();
        setDetectionActive(false);

        let stream = null;
        let timer = null;
        let busy = false;
        let facingMode = 'environment';
        let thresholdTimer = null;

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

        function setDetectionActive(active) {
            if (active) {
                mediaBox.classList.add('is-live');
                mediaPlaceholder.classList.add('hidden');
                liveBadge.classList.remove('hidden');
            } else {
                mediaBox.classList.remove('is-live');
                mediaPlaceholder.classList.remove('hidden');
                liveBadge.classList.add('hidden');
            }
        }

        function setProcessing(active) {
            if (active) {
                mediaBox.classList.add('is-processing');
            } else {
                mediaBox.classList.remove('is-processing');
            }
        }

        async function restartCamera() {
            if (stream) {
                stream.getTracks().forEach((track) => track.stop());
                stream = null;
            }
            await startCamera();
        }

        function updateInterval() {
            const fps = Number(fpsInput.value || 4);
            fpsValue.textContent = fps + ' fps';
            if (timer) {
                clearInterval(timer);
                timer = setInterval(captureAndSend, 1000 / Math.max(fps, 1));
            }
        }

        function updateThresholdDisplay() {
            thresholdValue.textContent = thresholdInput.value;
        }

        function updateLogicNote() {
            const value = Number(thresholdInput.value || 0);
            const threshold = Number.isFinite(value) ? value : 0;
            logicNote.textContent =
                'Need at least ' + threshold +
                ' vehicles for the car light to switch green. If there are fewer than ' +
                threshold + ' vehicles, pedestrians can cross.';
        }

        function showToast(message, durationMs = 5000) {
            toastEl.textContent = message;
            toastEl.classList.add('show');
            setTimeout(() => {
                toastEl.classList.remove('show');
            }, durationMs);
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

        function buildCountList(models) {
            countList.innerHTML = '';
            models.forEach((name) => {
                const row = document.createElement('div');
                row.className = 'count-row';
                row.dataset.model = name;
                const label = document.createElement('span');
                label.className = 'count-name';
                label.textContent = modelLabel(name) + ':';
                const value = document.createElement('span');
                value.className = 'count-value';
                value.textContent = '0';
                row.appendChild(label);
                row.appendChild(value);
                countList.appendChild(row);
            });
        }

        function updateCounts(counts) {
            if (!counts) {
                return;
            }
            Object.entries(counts).forEach(([name, value]) => {
                const row = countList.querySelector(`.count-row[data-model="${name}"]`);
                if (row) {
                    const valueEl = row.querySelector('.count-value');
                    if (valueEl) {
                        valueEl.textContent = value;
                    }
                }
            });
        }


        async function loadConfig() {
            try {
                const res = await fetch('/config');
                if (!res.ok) {
                    return;
                }
                const data = await res.json();
                if (data.server_capture) {
                    toggleBtn.disabled = true;
                    switchBtn.disabled = true;
                    annotated.src = '/stream';
                    setDetectionActive(true);
                    setStatus('Server capture mode. Watching stream...');
                    pollStatus();
                }
                if (typeof data.threshold === 'number') {
                    thresholdInput.value = data.threshold;
                    thresholdValue.textContent = data.threshold;
                }
                if (Array.isArray(data.models) && data.models.length) {
                    modelSelect.innerHTML = '';
                    data.models.forEach((name) => {
                        const opt = document.createElement('option');
                        opt.value = name;
                        opt.textContent = modelLabel(name);
                        modelSelect.appendChild(opt);
                    });
                    buildCountList(data.models);
                }
                if (typeof data.model === 'string') {
                    modelSelect.value = data.model;
                }
                updateLogicNote();
                const t = Number(thresholdInput.value || 5);
                const display = Number.isFinite(t) ? t : 5;
                showToast('If there are fewer than ' + display + ' vehicles, pedestrians can cross.');
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
            setStatus('Requesting camera access...');
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { ideal: facingMode } },
                    audio: false
                });
                video.srcObject = stream;
                await video.play();
                toggleBtn.textContent = 'Stop detection';
                setDetectionActive(true);
                setStatus('Camera active. Running detection...');
                const fps = Number(fpsInput.value || 4);
                timer = setInterval(captureAndSend, 1000 / Math.max(fps, 1));
            } catch (err) {
                setDetectionActive(false);
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
            toggleBtn.textContent = 'Start detection';
            setDetectionActive(false);
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
            setProcessing(true);
            canvas.toBlob(async (blob) => {
                if (!blob) {
                    busy = false;
                    setProcessing(false);
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
                    if (data.counts) {
                        updateCounts(data.counts);
                    } else if (typeof data.count === 'number' && data.model) {
                        const fallback = {};
                        fallback[data.model] = data.count;
                        updateCounts(fallback);
                    }
                    if (data.light) {
                        updateLight(data.light);
                    }
                    const meta = [];
                    if (data.light) {
                        meta.push('Light: ' + data.light);
                    }
                    if (typeof data.selected_count === 'number' && typeof data.threshold === 'number' && data.model) {
                        meta.push(modelLabel(data.model) + ': ' + data.selected_count + '/' + data.threshold);
                    }
                    const prefix = meta.length ? meta.join(' | ') + ' | ' : '';
                    setStatus(prefix + 'Last update: ' + new Date().toLocaleTimeString());
                } catch (err) {
                    setStatus('Error: ' + err.message);
                } finally {
                    busy = false;
                    setProcessing(false);
                }
            }, 'image/jpeg', 0.7);
        }

        let statusTimer = null;
        async function pollStatus() {
            if (statusTimer) {
                return;
            }
            statusTimer = setInterval(async () => {
                try {
                    const res = await fetch('/status');
                    if (!res.ok) {
                        return;
                    }
                    const data = await res.json();
                    if (data.light) {
                        updateLight(data.light);
                    }
                    if (typeof data.count === 'number' && data.model) {
                        const fallback = {};
                        fallback[data.model] = data.count;
                        updateCounts(fallback);
                    }
                    const meta = [];
                    if (data.light) {
                        meta.push('Light: ' + data.light);
                    }
                    if (typeof data.count === 'number' && typeof data.threshold === 'number' && data.model) {
                        meta.push(modelLabel(data.model) + ': ' + data.count + '/' + data.threshold);
                    }
                    const prefix = meta.length ? meta.join(' | ') + ' | ' : '';
                    setStatus(prefix + 'Last update: ' + new Date().toLocaleTimeString());
                } catch (err) {
                    // ignore
                }
            }, 600);
        }

        fpsInput.addEventListener('input', updateInterval);
        thresholdInput.addEventListener('input', () => {
            updateThresholdDisplay();
            updateLogicNote();
            if (thresholdTimer) {
                clearTimeout(thresholdTimer);
            }
            thresholdTimer = setTimeout(applyThreshold, 250);
        });
        toggleBtn.addEventListener('click', () => {
            if (stream) {
                stopCamera();
            } else {
                startCamera();
            }
        });
        switchBtn.addEventListener('click', async () => {
            facingMode = facingMode === 'environment' ? 'user' : 'environment';
            if (stream) {
                setStatus('Switching camera...');
                await restartCamera();
                setStatus('Camera active. Running detection...');
            }
        });
        thresholdInput.addEventListener('change', applyThreshold);
        modelSelect.addEventListener('change', applyModel);
        loadConfig();
        updateLogicNote();
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
    if 'model' in updates:
        with sessions_lock:
            for session in sessions.values():
                session.controller = TrafficLightController(threshold=THRESHOLD, yellow_duration=YELLOW_DURATION)
                session.controller.set_timing(MIN_GREEN_TIME, MIN_RED_TIME, HYSTERESIS)
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
            'server_capture': SERVER_CAPTURE,
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
    counts = {}
    selected_result = None
    for name in ALLOWED_MODELS:
        r, count = run_inference_with_conf(frame, name, conf)
        counts[name] = count
        if name == model_name:
            selected_result = r
    if selected_result is None:
        selected_result = r
    avg_counts, light_state = session.update_counts(counts, model_name)

    try:
        annotated = selected_result.plot()
        ret, jpeg = cv2.imencode('.jpg', annotated)
        image_b64 = base64.b64encode(jpeg.tobytes()).decode('ascii') if ret else ''
    except Exception:
        image_b64 = ''

    return {
        'counts': avg_counts,
        'model': model_name,
        'selected_count': session.last_count,
        'threshold': session.controller.threshold,
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
