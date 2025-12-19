import argparse
import time
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 vehicle counter + traffic light controller')
    parser.add_argument('--source', '-s', default='0',
                        help='Video source (0 for default webcam, or path/rtsp). For phone stream use an IP stream URL.')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='YOLOv8 model (pt file or model name)')
    parser.add_argument('--track', action='store_true', help='Enable tracker to assign IDs and improve counting')
    parser.add_argument('--classes', '-c', default='car,motorcycle,bus,truck',
                        help='Comma-separated list of class names to count (default: car,motorcycle,bus,truck)')
    parser.add_argument('--smoothing-window', type=float, default=2.0,
                        help='Seconds for moving-average smoothing of counts')
    parser.add_argument('--hysteresis', type=int, default=1,
                        help='Hysteresis value (in vehicles) to avoid rapid toggles')
    parser.add_argument('--min-green-time', type=float, default=5.0,
                        help='Minimum seconds to keep car light GREEN before allowing change')
    parser.add_argument('--min-red-time', type=float, default=3.0,
                        help='Minimum seconds to keep car light RED before allowing change')
    parser.add_argument('--threshold', '-t', type=int, default=5,
                        help='Vehicle count threshold to decide green vs yellow->red')
    parser.add_argument('--yellow-duration', type=float, default=3.0,
                        help='Seconds to show yellow before switching to red')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold')
    parser.add_argument('--display', action='store_true', help='Display annotated video window')
    return parser.parse_args()


class TrafficLightController:
    def __init__(self, threshold, yellow_duration=3.0):
        self.threshold = threshold
        self.yellow_duration = yellow_duration
        self.state = 'RED'  # RED, YELLOW, GREEN
        self._yellow_started = None
        self._last_state_change = time.time()
        self.min_green_time = 5.0
        self.min_red_time = 3.0
        self.hysteresis = 1

    def update(self, vehicle_count):
        now = time.time()
        # apply hysteresis thresholds
        up_threshold = self.threshold
        down_threshold = max(0, self.threshold - self.hysteresis)

        # enforce minimum times to avoid rapid toggles
        time_since_change = now - self._last_state_change

        if vehicle_count >= up_threshold:
            # request GREEN
            if self.state != 'GREEN':
                # if currently RED, allow immediate switch to GREEN
                self.state = 'GREEN'
                self._yellow_started = None
                self._last_state_change = now
        else:
            # request non-GREEN
            if self.state == 'GREEN':
                # only switch to yellow if min green time elapsed
                if time_since_change >= self.min_green_time:
                    self.state = 'YELLOW'
                    self._yellow_started = now
                    self._last_state_change = now
            elif self.state == 'YELLOW':
                if self._yellow_started and (now - self._yellow_started) >= self.yellow_duration:
                    # before switching to RED, ensure min_red_time logic applies afterwards
                    self.state = 'RED'
                    self._yellow_started = None
                    self._last_state_change = now
            else:
                # already RED, but prevent immediate flip back to GREEN until min_red_time elapses
                if self.state == 'RED' and vehicle_count >= up_threshold and (now - self._last_state_change) >= self.min_red_time:
                    self.state = 'GREEN'
                    self._last_state_change = now

    def get_state(self):
        return self.state

    def set_timing(self, min_green: float, min_red: float, hysteresis: int):
        self.min_green_time = float(min_green)
        self.min_red_time = float(min_red)
        self.hysteresis = int(hysteresis)


def draw_overlay(frame, count, state):
    h, w = frame.shape[:2]
    # count text (top-left)
    cv2.putText(frame, f'Vehicles: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # traffic light box (car) - place both lights on the right side, car first (left of pedestrian)
    box_w, box_h = 140, 200
    # position pair near right edge: car on the left, pedestrian to its right
    margin = 20
    car_x = w - (2 * box_w) - margin - 10
    car_y = 20
    cv2.rectangle(frame, (car_x, car_y), (car_x + box_w, car_y + box_h), (50, 50, 50), -1)
    # draw circles for R Y G for car
    centers = [(car_x + box_w // 2, car_y + 40), (car_x + box_w // 2, car_y + 100), (car_x + box_w // 2, car_y + 160)]
    colors = {'RED': (0, 0, 255), 'YELLOW': (0, 255, 255), 'GREEN': (0, 255, 0)}
    states = ['RED', 'YELLOW', 'GREEN']
    for i, s in enumerate(states):
        col = colors[s] if s == state else (80, 80, 80)
        cv2.circle(frame, centers[i], 20, col, -1)
    # center 'Car' label under the car panel
    car_label = 'Car'
    (tw, th), tb = cv2.getTextSize(car_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_x = car_x + (box_w - tw) // 2
    text_y = car_y + box_h + th + 6
    cv2.putText(frame, car_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # pedestrian light - same dimensions, placed to the right of car light
    pv_w, pv_h = box_w, box_h
    px = car_x + box_w + 10
    py = car_y
    cv2.rectangle(frame, (px, py), (px + pv_w, py + pv_h), (50, 50, 50), -1)
    # place pedestrian circles aligned with car light
    pcenters = [(px + pv_w // 2, py + 40), (px + pv_w // 2, py + 100), (px + pv_w // 2, py + 160)]
    pcolors = {'RED': (0, 0, 255), 'YELLOW': (0, 255, 255), 'GREEN': (0, 255, 0)}
    # pedestrian state is inverse of car: if car RED -> pedestrian GREEN
    pet_state = 'GREEN' if state == 'RED' else 'RED'
    for i, s in enumerate(['RED', 'YELLOW', 'GREEN']):
        col = pcolors[s] if s == pet_state else (80, 80, 80)
        cv2.circle(frame, pcenters[i], 20, col, -1)
    cv2.putText(frame, 'Pedestrian', (px + 6, py + pv_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    args = parse_args()

    source = args.source
    try:
        src_int = int(source)
        cap = cv2.VideoCapture(src_int)
    except Exception:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print('ERROR: Could not open video source:', source)
        return

    print('Loading model:', args.model)
    model = YOLO(args.model)

    vehicle_classes = set([s.strip() for s in args.classes.split(',') if s.strip()])

    controller = TrafficLightController(threshold=args.threshold, yellow_duration=args.yellow_duration)

    print('Press Q in the display window to quit.')

    if args.track:
        # Use built-in tracker to get consistent IDs and reduce double-counting
        stream = model.track(source=source, tracker='bytetrack', conf=args.conf, stream=True)
        tracked_ids = set()
        # maintain smoothing deque for tracked counts as well
        counts_deque = deque()
        for r in stream:
            # r is a Results object for one frame
            count = 0
            ids_in_frame = set()
            try:
                # prefer using tracked ids if available
                if hasattr(r.boxes, 'id') and r.boxes.id is not None:
                    ids = r.boxes.id.cpu().numpy().astype(int)
                    classes = r.boxes.cls.cpu().numpy().astype(int)
                    for tid, cid in zip(ids, classes):
                        name = model.names.get(int(cid), str(cid))
                        if name in vehicle_classes:
                            ids_in_frame.add(int(tid))
                    count = len(ids_in_frame)
                else:
                    # fallback to counting by class
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
            except Exception:
                pass

            # update controller with smoothed number
            now = time.time()
            counts_deque.append((now, count))
            while counts_deque and (now - counts_deque[0][0]) > args.smoothing_window:
                counts_deque.popleft()
            avg_count = int(round(sum(c for _, c in counts_deque) / max(1, len(counts_deque))))
            controller.set_timing(args.min_green_time, args.min_red_time, args.hysteresis)
            controller.update(avg_count)
            state = controller.get_state()

            annotated = r.plot()
            draw_overlay(annotated, avg_count, state)

            if args.display:
                cv2.imshow('YOLOv8 Vehicle Counter (tracked)', annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                print(f'Count={count}  Light={state}', end='\r')

        # end stream loop
    else:
        counts_deque = deque()
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Stream ended or cannot fetch frame')
                break

            # inference
            results = model.predict(source=frame, conf=args.conf, verbose=False)
            # results is a list (one item per image), take first
            r = results[0]

            # count vehicles (no tracker mode)
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

            # smoothing: append to deque and compute averaged count
            now = time.time()
            counts_deque.append((now, count))
            # remove old entries
            while counts_deque and (now - counts_deque[0][0]) > args.smoothing_window:
                counts_deque.popleft()
            # compute average
            avg_count = int(round(sum(c for _, c in counts_deque) / max(1, len(counts_deque))))

            controller.update(avg_count)
            controller.set_timing(args.min_green_time, args.min_red_time, args.hysteresis)
            state = controller.get_state()

            # annotate detections (using model's built-in plot)
            annotated = r.plot()

            draw_overlay(annotated, avg_count, state)

            if args.display:
                cv2.imshow('YOLOv8 Vehicle Counter', annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                # minimal console output
                print(f'AvgCount={avg_count}  Light={state}', end='\r')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
