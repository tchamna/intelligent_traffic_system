# YOLOv8 Vehicle Counter + Traffic Light Controller

This project runs a YOLOv8 model to count vehicles and drive a simulated traffic light state.
It supports a browser-first demo where anyone can open the page, allow camera access, and see live counts.

## Quick start (browser demo)

1. Install Python 3.10+ and dependencies.

```powershell
.\setup.ps1
```

2. Run the web service:

```powershell
python -m uvicorn service:app --host 0.0.0.0 --port 8000
```

3. Open `http://localhost:8000`, click "Start camera", and point at vehicles.

### Share on the internet (HTTPS required)

Browser camera access requires HTTPS (or localhost). To let anyone on the internet test:

- Example with Cloudflare Tunnel:
  `cloudflared tunnel --url http://localhost:8000`
- Example with ngrok:
  `ngrok http 8000`

Share the HTTPS URL that the tunnel provides.

## Optional: server-side capture mode

If you want the server to read a webcam or IP stream directly (without browser frames):

```powershell
$env:SERVER_CAPTURE = "1"
$env:SOURCE = "0"
python -m uvicorn service:app --host 0.0.0.0 --port 8000
```

This exposes:

- `GET /status` for JSON status
- `GET /stream` for MJPEG stream
- `GET /snapshot` for a single JPEG

## CLI demo (local window)

```powershell
python app.py --display --source 0 --threshold 5 --yellow-duration 3
```

Use `--source` with an IP camera URL if needed.

## Deploy to EC2 (GitHub Actions)

This repo includes a GitHub Actions workflow that SSHes into your EC2 instance and restarts the app.

1. Add these GitHub repo secrets:
   - `EC2_HOST`: `ec2-18-208-117-82.compute-1.amazonaws.com`
   - `EC2_USER`: `ec2-user`
   - `EC2_SSH_KEY`: contents of your `test-rag.pem`
   - `EC2_SSH_PORT`: `22` (optional)
   - `EC2_APP_PORT`: choose a free port (defaults to `8001` if unset)

2. Open the port in the EC2 security group and any OS firewall.

3. Push to `main` and the workflow will deploy and restart the service.
