# YOLOv8 Vehicle Counter + Traffic Light Controller

This project runs a YOLOv8 model on a webcam or video stream, counts vehicles, and simulates a traffic-light controller:

- If vehicle count >= threshold → GREEN
- If vehicle count < threshold → switch to YELLOW for a few seconds, then RED

Quick start

1. Install Python and uv (if not already):

   - Install uv: `pip install uv` or from https://github.com/astral-sh/uv
   - Install Python 3.11: `uv python install 3.11`

2. Create virtual environment and install dependencies:

```powershell
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

3. Run with your computer webcam (display annotated window):

```powershell
python app.py --display --source 0 --threshold 5 --yellow-duration 3
```

4. Use your phone camera (example using IP Webcam app on Android):

   - Start IP Webcam on your phone and find the video stream URL (e.g. `http://192.168.1.10:8080/video`).
   - Run:

```powershell
python app.py --display --source "http://192.168.1.10:8080/video" --threshold 5
```

5. Run the HTTP service (exposes `/status` endpoint for integration):

```powershell
python -m uvicorn service:app --host 0.0.0.0 --port 8000
```

   - Check status: `curl http://localhost:8000/status` (returns JSON with count, light state, etc.)
- Tune classes / counting logic for specific cameras
