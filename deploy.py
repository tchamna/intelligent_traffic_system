#!/usr/bin/env python3
"""
EC2 deploy helper for intelligent_traffic_system.

Usage:
  python deploy.py --deploy            # default
  python deploy.py --setup-secrets     # set GitHub secrets for Actions
  python deploy.py --trigger-workflow  # run GitHub Actions deploy workflow
  python deploy.py --push              # git push origin main

Environment:
  APP_PORT=auto (or unset) to pick a free port on the EC2 instance
  PORT_RANGE_START=8001, PORT_RANGE_END=8999 to control auto port range
  DOMAIN_NAME or NGINX_SERVER_NAME to control nginx server_name
  NGINX_SITE_NAME to control /etc/nginx/conf.d/<name>.conf
  NGINX_DISABLE_CONFLICTS=1 to auto-disable other configs with same server_name
  ENABLE_CERTBOT=0|1 to request HTTPS certs automatically
  CERTBOT_EMAIL=you@example.com (required when ENABLE_CERTBOT=1)
  NGINX_REDIRECT_TO_HTTPS=0|1 to force HTTPS when certbot runs
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse


def load_env_file(env_file=".env"):
    if not os.path.exists(env_file):
        return
    try:
        with open(env_file, "r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        print(f"Warning: could not read {env_file}: {exc}")


def get_env(key, default=None, required=False):
    value = os.getenv(key, default)
    if required and not value:
        print(f"ERROR: missing required environment variable: {key}")
        sys.exit(1)
    return value


def resolve_ssh_key(value):
    if not value:
        return None, None
    if os.path.exists(value):
        return value, None
    if "BEGIN" in value and "PRIVATE KEY" in value:
        tmp = tempfile.NamedTemporaryFile(delete=False, prefix="ec2_key_", suffix=".pem")
        tmp.write(value.encode("utf-8"))
        tmp.flush()
        tmp.close()
        return tmp.name, tmp.name
    return value, None


def git_remote_url():
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def repo_name_from_url(url):
    if not url:
        return None
    parsed = urlparse(url)
    path = parsed.path or url
    name = path.split("/")[-1].replace(".git", "")
    return name or None


def ssh(host, user, key_path, cmd):
    args = [
        "ssh",
        "-i",
        key_path,
        "-o",
        "StrictHostKeyChecking=no",
        f"{user}@{host}",
        cmd,
    ]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode == 0, result.stdout, result.stderr


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_remote_port_from_env(ec2_host, ec2_user, key_path):
    cmd = "if [ -f /etc/its.env ]; then grep -E '^PORT=' /etc/its.env | tail -1 | cut -d= -f2; fi"
    ok, out, _ = ssh(ec2_host, ec2_user, key_path, cmd)
    if not ok:
        return None
    return parse_int(out.strip())


def find_free_port_remote(ec2_host, ec2_user, key_path, start_port, end_port):
    script = f"""python3 - <<'PY'
import socket
start={start_port}
end={end_port}
for port in range(start, end + 1):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
        s.close()
        print(port)
        raise SystemExit(0)
    except OSError:
        s.close()
raise SystemExit(1)
PY"""
    ok, out, err = ssh(ec2_host, ec2_user, key_path, script)
    if not ok:
        print(err.strip() or out.strip())
        return None
    return parse_int(out.strip())


def get_listen_owner_remote(ec2_host, ec2_user, key_path, port):
    cmd = (
        f"owner=$(sudo ss -ltnp 2>/dev/null | awk '$4 ~ /:{port}$/ {{print $6}}' | head -n1); "
        f"if [ -z \"$owner\" ]; then "
        f"  owner=$(sudo netstat -ltnp 2>/dev/null | awk '$4 ~ /:{port}$/ {{print $7}}' | head -n1); "
        f"fi; "
        f"echo \"$owner\""
    )
    ok, out, _ = ssh(ec2_host, ec2_user, key_path, cmd)
    if not ok:
        return ""
    return out.strip()


def is_port_used_by_service_remote(ec2_host, ec2_user, key_path, port, service_name):
    cmd = f"systemctl show -p MainPID --value {service_name} 2>/dev/null || true"
    ok, out, _ = ssh(ec2_host, ec2_user, key_path, cmd)
    if not ok:
        return False
    pid = out.strip()
    if not pid or pid == "0":
        return False
    owner = get_listen_owner_remote(ec2_host, ec2_user, key_path, port)
    if not owner:
        return False
    if f"pid={pid}" in owner or owner.startswith(f"{pid}/"):
        return True
    return False


def is_port_in_use_remote(ec2_host, ec2_user, key_path, port):
    owner = get_listen_owner_remote(ec2_host, ec2_user, key_path, port)
    return bool(owner)


def setup_nginx(ec2_host, ec2_user, key_path, app_name, app_port, server_name):
    nginx_site = get_env("NGINX_SITE_NAME") or app_name
    config_path = f"/etc/nginx/conf.d/{nginx_site}.conf"
    server_name = server_name or ec2_host
    disable_conflicts = get_env("NGINX_DISABLE_CONFLICTS", "1") == "1"
    enable_certbot = get_env("ENABLE_CERTBOT", "0") == "1"
    certbot_email = get_env("CERTBOT_EMAIL", "").strip()
    redirect_https = get_env("NGINX_REDIRECT_TO_HTTPS", "0") == "1"

    install_cmd = (
        "if ! command -v nginx >/dev/null 2>&1; then "
        "if command -v yum >/dev/null 2>&1; then sudo yum install -y nginx; "
        "elif command -v apt-get >/dev/null 2>&1; then sudo apt-get update -y && sudo apt-get install -y nginx; "
        "else echo 'Nginx not installed and no package manager found.'; exit 1; fi; "
        "fi"
    )
    ok, out, err = ssh(ec2_host, ec2_user, key_path, install_cmd)
    if not ok:
        print(err.strip() or out.strip())
        sys.exit(1)

    if disable_conflicts and server_name not in ("_", "default_server"):
        conflict_cmd = (
            f"set -e; "
            f"matches=$(sudo grep -RIl \"server_name[[:space:]].*{server_name}\" /etc/nginx/conf.d || true); "
            f"ts=$(date +%Y%m%d%H%M%S); "
            f"for f in $matches; do "
            f"  if [ \"$f\" != \"{config_path}\" ]; then "
            f"    sudo mv \"$f\" \"${{f}}.bak.${{ts}}\"; "
            f"  fi; "
            f"done"
        )
        ok, out, err = ssh(ec2_host, ec2_user, key_path, conflict_cmd)
        if not ok:
            print(err.strip() or out.strip())
            sys.exit(1)

    nginx_conf = f"""server {{
    listen 80;
    server_name {server_name};
    client_max_body_size 25m;

    location /.well-known/acme-challenge/ {{
        root /var/www/certbot;
    }}

    location / {{
        proxy_pass http://127.0.0.1:{app_port};
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }}
}}
"""

    cmd = (
        f"cat > /tmp/{nginx_site}.conf << 'EOF'\n{nginx_conf}\nEOF\n"
        f"sudo mv /tmp/{nginx_site}.conf {config_path}\n"
        "sudo nginx -t\n"
        "sudo systemctl enable nginx\n"
        "sudo systemctl restart nginx\n"
        "sudo systemctl status nginx --no-pager -l"
    )
    ok, out, err = ssh(ec2_host, ec2_user, key_path, cmd)
    if not ok:
        print(err.strip() or out.strip())
        sys.exit(1)
    print(out.strip())

    if enable_certbot:
        if not certbot_email:
            print("ERROR: ENABLE_CERTBOT=1 requires CERTBOT_EMAIL.")
            sys.exit(1)
        certbot_install = (
            "if ! command -v certbot >/dev/null 2>&1; then "
            "if command -v yum >/dev/null 2>&1; then sudo yum install -y certbot python3-certbot-nginx; "
            "elif command -v apt-get >/dev/null 2>&1; then sudo apt-get update -y && sudo apt-get install -y certbot python3-certbot-nginx; "
            "else echo 'Certbot not installed and no package manager found.'; exit 1; fi; "
            "fi"
        )
        ok, out, err = ssh(ec2_host, ec2_user, key_path, certbot_install)
        if not ok:
            print(err.strip() or out.strip())
            sys.exit(1)

        redirect_flag = "--redirect" if redirect_https else ""
        certbot_cmd = (
            f"sudo certbot --nginx -d {server_name} "
            f"--non-interactive --agree-tos -m {certbot_email} {redirect_flag}"
        )
        ok, out, err = ssh(ec2_host, ec2_user, key_path, certbot_cmd)
        if not ok:
            print(err.strip() or out.strip())
            sys.exit(1)
        reload_cmd = "sudo nginx -t && sudo systemctl reload nginx"
        ok, out, err = ssh(ec2_host, ec2_user, key_path, reload_cmd)
        if not ok:
            print(err.strip() or out.strip())
            sys.exit(1)


def deploy():
    load_env_file(".env")

    ec2_host = get_env("EC2_HOST") or get_env("EC2_IP", required=True)
    ec2_user = get_env("EC2_USER", "ec2-user")
    ssh_key_raw = get_env("SSH_KEY_PATH") or get_env("EC2_SSH_KEY", required=True)
    repo_url = get_env("GIT_REPO") or git_remote_url()
    if not repo_url:
        print("ERROR: missing GIT_REPO and no git remote origin found.")
        sys.exit(1)

    app_port_raw = get_env("APP_PORT")
    app_name = repo_name_from_url(repo_url)
    app_dir = get_env("APP_DIR") or f"/home/{ec2_user}/apps/{app_name}"
    service_name = get_env("SERVICE_NAME", "its.service")
    port_range_start = parse_int(get_env("PORT_RANGE_START", "8001")) or 8001
    port_range_end = parse_int(get_env("PORT_RANGE_END", "8999")) or 8999

    model_name = get_env("MODEL", "yolov8n.pt")
    conf = get_env("CONF", "0.2")
    threshold = get_env("THRESHOLD", "5")
    yellow_duration = get_env("YELLOW_DURATION", "3.0")
    smoothing = get_env("SMOOTHING_WINDOW", "2.0")
    min_green = get_env("MIN_GREEN_TIME", "5.0")
    min_red = get_env("MIN_RED_TIME", "3.0")
    hysteresis = get_env("HYSTERESIS", "1")
    torch_index_url = get_env("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu").strip()

    key_path, temp_key = resolve_ssh_key(ssh_key_raw)
    if not key_path:
        print("ERROR: SSH key path is missing or invalid.")
        sys.exit(1)

    app_port = None
    if app_port_raw and app_port_raw not in ("auto", "0"):
        app_port = parse_int(app_port_raw)
        if app_port is not None:
            if is_port_in_use_remote(ec2_host, ec2_user, key_path, app_port) and not is_port_used_by_service_remote(
                ec2_host, ec2_user, key_path, app_port, service_name
            ):
                print(f"ERROR: APP_PORT {app_port} is already in use on the EC2 instance.")
                print("Set APP_PORT=auto (or unset) to allow auto-selection.")
                sys.exit(1)
    if app_port is None:
        app_port = get_remote_port_from_env(ec2_host, ec2_user, key_path)
        if app_port is not None:
            if is_port_in_use_remote(ec2_host, ec2_user, key_path, app_port) and not is_port_used_by_service_remote(
                ec2_host, ec2_user, key_path, app_port, service_name
            ):
                app_port = None
    if app_port is None:
        app_port = find_free_port_remote(ec2_host, ec2_user, key_path, port_range_start, port_range_end)
    if app_port is None:
        print("ERROR: could not find a free port on the EC2 instance.")
        sys.exit(1)

    server_name = get_env("NGINX_SERVER_NAME") or get_env("DOMAIN_NAME") or ec2_host

    print("Deploy configuration")
    print(f"- EC2 host: {ec2_host}")
    print(f"- EC2 user: {ec2_user}")
    print(f"- Repo: {repo_url}")
    print(f"- App dir: {app_dir}")
    print(f"- Port: {app_port}")
    print(f"- Service: {service_name}")
    print(f"- Nginx server_name: {server_name}")

    try:
        cmd = (
            f"set -e; "
            f"mkdir -p /home/{ec2_user}/apps; "
            f"if [ -d {app_dir}/.git ]; then "
            f"cd {app_dir} && git fetch origin main && git checkout main && git pull --ff-only origin main; "
            f"else git clone {repo_url} {app_dir}; fi"
        )
        ok, out, err = ssh(ec2_host, ec2_user, key_path, cmd)
        if not ok:
            print(err.strip() or out.strip())
            sys.exit(1)

        pip_env = "PIP_NO_CACHE_DIR=1"
        if torch_index_url:
            pip_env = f"{pip_env} PIP_EXTRA_INDEX_URL={torch_index_url}"

        cmd = (
            f"set -e; "
            f"cd {app_dir}; "
            f"if [ ! -d .venv ]; then python3 -m venv .venv; fi; "
            f". .venv/bin/activate; "
            f"rm -rf ~/.cache/pip; "
            f"{pip_env} pip install --upgrade pip; "
            f"{pip_env} pip install --prefer-binary -r requirements.txt"
        )
        ok, out, err = ssh(ec2_host, ec2_user, key_path, cmd)
        if not ok:
            print(err.strip() or out.strip())
            sys.exit(1)

        env_content = "\n".join(
            [
                f"PORT={app_port}",
                f"MODEL={model_name}",
                f"CONF={conf}",
                f"THRESHOLD={threshold}",
                f"YELLOW_DURATION={yellow_duration}",
                f"SMOOTHING_WINDOW={smoothing}",
                f"MIN_GREEN_TIME={min_green}",
                f"MIN_RED_TIME={min_red}",
                f"HYSTERESIS={hysteresis}",
            ]
        ) + "\n"

        cmd = f"sudo tee /etc/its.env >/dev/null <<'EOF'\n{env_content}EOF"
        ok, out, err = ssh(ec2_host, ec2_user, key_path, cmd)
        if not ok:
            print(err.strip() or out.strip())
            sys.exit(1)

        service = f"""[Unit]
Description=Intelligent Traffic System
After=network.target

[Service]
Type=simple
User={ec2_user}
WorkingDirectory={app_dir}
EnvironmentFile=/etc/its.env
ExecStart={app_dir}/.venv/bin/uvicorn service:app --host 0.0.0.0 --port ${{PORT}}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
"""

        cmd = (
            f"cat > /tmp/{service_name} << 'EOF'\n{service}\nEOF\n"
            f"sudo mv /tmp/{service_name} /etc/systemd/system/{service_name}\n"
            f"sudo systemctl daemon-reload\n"
            f"sudo systemctl enable {service_name}\n"
            f"sudo systemctl restart {service_name}\n"
            f"sudo systemctl status {service_name} --no-pager -l"
        )
        ok, out, err = ssh(ec2_host, ec2_user, key_path, cmd)
        if not ok:
            print(err.strip() or out.strip())
            sys.exit(1)

        print(out.strip())
        setup_nginx(ec2_host, ec2_user, key_path, app_name, app_port, server_name)
        print(f"Deployed. Access: http://{server_name}/")
        print(f"Direct port: http://{ec2_host}:{app_port}/")
        print("Make sure ports 80 and the app port are open in the EC2 security group.")
    finally:
        if temp_key:
            try:
                os.unlink(temp_key)
            except OSError:
                pass


def setup_github_secrets():
    load_env_file(".env")
    result = subprocess.run(
        ["gh", "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        print("ERROR: gh CLI not installed or not in PATH.")
        sys.exit(1)

    repo_url = git_remote_url()
    if not repo_url or "github.com" not in repo_url:
        print("ERROR: not a GitHub repository or remote origin missing.")
        sys.exit(1)

    repo = repo_url.split("/")[-1].replace(".git", "")
    owner = repo_url.split("/")[-2]
    full_repo = f"{owner}/{repo}"

    ec2_host = get_env("EC2_HOST") or get_env("EC2_IP")
    ec2_user = get_env("EC2_USER", "ec2-user")
    ssh_key_raw = get_env("SSH_KEY_PATH") or get_env("EC2_SSH_KEY")
    ssh_port = get_env("EC2_SSH_PORT", "22")
    app_port_raw = get_env("APP_PORT", "")
    app_port = app_port_raw if app_port_raw.isdigit() else ""
    port_range_start = get_env("PORT_RANGE_START", "")
    port_range_end = get_env("PORT_RANGE_END", "")
    nginx_server = get_env("NGINX_SERVER_NAME") or get_env("DOMAIN_NAME") or ""
    nginx_site = get_env("NGINX_SITE_NAME") or ""
    nginx_disable_conflicts = get_env("NGINX_DISABLE_CONFLICTS", "")
    enable_certbot = get_env("ENABLE_CERTBOT", "")
    certbot_email = get_env("CERTBOT_EMAIL", "")
    nginx_redirect_https = get_env("NGINX_REDIRECT_TO_HTTPS", "")

    key_path, temp_key = resolve_ssh_key(ssh_key_raw)
    if key_path and os.path.exists(key_path):
        key_content = Path(key_path).read_text(encoding="utf-8", errors="ignore")
    else:
        key_content = ssh_key_raw

    secrets = {
        "EC2_HOST": ec2_host,
        "EC2_USER": ec2_user,
        "EC2_SSH_KEY": key_content,
        "EC2_SSH_PORT": ssh_port,
        "EC2_APP_PORT": app_port,
        "EC2_PORT_RANGE_START": port_range_start,
        "EC2_PORT_RANGE_END": port_range_end,
        "NGINX_SERVER_NAME": nginx_server,
        "NGINX_SITE_NAME": nginx_site,
        "NGINX_DISABLE_CONFLICTS": nginx_disable_conflicts,
        "ENABLE_CERTBOT": enable_certbot,
        "CERTBOT_EMAIL": certbot_email,
        "NGINX_REDIRECT_TO_HTTPS": nginx_redirect_https,
    }

    try:
        for name, value in secrets.items():
            if not value:
                print(f"Skipping {name} (empty)")
                continue
            proc = subprocess.run(
                ["gh", "secret", "set", name, "--repo", full_repo],
                input=str(value),
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
            )
            if proc.returncode != 0:
                print(f"Failed to set {name}: {proc.stderr.strip()}")
            else:
                print(f"Set {name}")
        print(f"Secrets updated for {full_repo}")
    finally:
        if temp_key:
            try:
                os.unlink(temp_key)
            except OSError:
                pass


def trigger_workflow():
    result = subprocess.run(
        ["gh", "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        print("ERROR: gh CLI not installed or not in PATH.")
        sys.exit(1)
    run = subprocess.run(
        ["gh", "workflow", "run", "deploy-ec2.yml", "--ref", "main"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if run.returncode != 0:
        print(run.stderr.strip())
        sys.exit(1)
    print(run.stdout.strip() or "Workflow triggered.")


def push_changes():
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if status.stdout.strip():
        print("ERROR: working tree is dirty. Commit or stash before pushing.")
        sys.exit(1)
    run = subprocess.run(
        ["git", "push", "origin", "main"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if run.returncode != 0:
        print(run.stderr.strip())
        sys.exit(1)
    print(run.stdout.strip() or "Pushed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true", help="deploy directly over SSH")
    parser.add_argument("--setup-secrets", action="store_true", help="set GitHub secrets via gh CLI")
    parser.add_argument("--trigger-workflow", action="store_true", help="run the GitHub Actions deploy workflow")
    parser.add_argument("--push", action="store_true", help="git push origin main")
    args = parser.parse_args()

    if args.setup_secrets:
        setup_github_secrets()
        return
    if args.trigger_workflow:
        trigger_workflow()
        return
    if args.push:
        push_changes()
        return
    if args.deploy or (not any(vars(args).values())):
        deploy()
        return


if __name__ == "__main__":
    main()
