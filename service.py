import threading
import time
from typing import Dict

import cv2
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO

from app import TrafficLightController


class ConfigUpdate(BaseModel):
    threshold: int | None = None
    yellow_duration: float | None = None


app = FastAPI()

# shared state
state: Dict = {
    'count': 0,
    'light': 'RED',
    'last_seen': 0.0,
    'running': False,
}
state_lock = threading.Lock()


def detection_loop(source: str = '0', model_name: str = 'yolov8n.pt', conf: float = 0.35, threshold: int = 5, yellow_duration: float = 3.0):
    try:
        src_int = int(source)
        cap = cv2.VideoCapture(src_int)
    except Exception:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print('ERROR: Could not open video source for service:', source)
        return

    model = YOLO(model_name)
    vehicle_classes = set(['car', 'motorcycle', 'bus', 'truck'])
    controller = TrafficLightController(threshold=threshold, yellow_duration=yellow_duration)

    with state_lock:
        state['running'] = True

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        results = model.predict(source=frame, conf=conf, verbose=False)
        r = results[0]

        count = 0
        try:
            cls_tensor = r.boxes.cls
            if cls_tensor is not None:
                for c in cls_tensor.cpu().numpy().astype(int):
                    name = model.names.get(int(c), str(c))
                    if name in vehicle_classes:
                        count += 1
        except Exception:
            try:
                for box in r.boxes.data.tolist():
                    class_id = int(box[5])
                    name = model.names.get(class_id, str(class_id))
                    if name in vehicle_classes:
                        count += 1
            except Exception:
                pass

        controller.update(count)

        with state_lock:
            state['count'] = count
            state['light'] = controller.get_state()
            state['last_seen'] = time.time()

        # small sleep to avoid hogging CPU, adapt as needed
        time.sleep(0.05)


@app.on_event('startup')
def startup_event():
    # start background detection thread
    src = '0'
    t = threading.Thread(target=detection_loop, args=(src, 'yolov8n.pt', 0.35, 5, 3.0), daemon=True)
    t.start()


@app.get('/status')
def get_status():
    with state_lock:
        return {
            'count': state['count'],
            'light': state['light'],
            'last_seen': state['last_seen'],
            'running': state['running'],
        }


@app.post('/config')
def update_config(cfg: ConfigUpdate):
    # for now we only support updating controller settings by restarting thread externally
    # but echo back requested values
    return {'ok': True, 'requested': cfg.dict()}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('service:app', host='0.0.0.0', port=8000, log_level='info')
