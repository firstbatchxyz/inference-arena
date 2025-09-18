import json
import re
import time
from typing import Optional, Tuple
import requests


def ssh_exec(ssh_client, command: str, timeout: int = 120) -> Tuple[int, str, str]:
    """Execute SSH command and return exit status, stdout, stderr"""
    stdin, stdout, stderr = ssh_client.exec_command(command, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    exit_status = stdout.channel.recv_exit_status()
    return exit_status, out, err


def start_mlx_server(
    ssh_client,
    model_path: str,
    port: int = 8080,
) -> None:
    """Start MLX OpenAI server with specified configuration"""
    print(f"Starting MLX server with model {model_path}...")

    command = f"nohup bash -c 'python3 -m venv env && source env/bin/activate && pip install transformers[sentencepiece] && pip install mlx-lm && mlx_lm.server --port {port} --model {model_path}' > ~/mlx_server.log 2>&1 &"

    ssh_exec(ssh_client, command, timeout=60)
    print("MLX server launch command executed")


def wait_for_mlx_server(
    ssh_client, public_url: str, model_path: str, max_wait_time: int = 600
) -> Optional[str]:
    """
    Wait for MLX server to be ready and return the model ID
    Returns None if server doesn't become ready in time
    """
    print("Waiting for MLX server to become ready...")
    deadline = time.time() + max_wait_time
    ready_model_id = None

    while time.time() < deadline:
        try:
            r = requests.get(f"{public_url}/v1/models", timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("data"):
                    # Use the first model id if present; fallback to model_path
                    ready_model_id = data["data"][0].get("id") or model_path
                    print(f"MLX server is ready with model: {ready_model_id}")
                    return ready_model_id
        except Exception:
            pass
        time.sleep(2)

    # Print logs if server didn't start
    print("Server did not become ready in time. Checking logs...")
    code, out, err = ssh_exec(
        ssh_client, "tail -n 200 ~/mlx_server.log || true", timeout=10
    )
    if out:
        print("Server logs:", out)
    if err:
        print("Server errors:", err)

    return None


def install_ngrok(ssh_client):
    """Install ngrok on the remote machine"""
    print("Installing ngrok...")
    install_cmd = (
        "(command -v ngrok >/dev/null 2>&1) || ("
        "cd ~ && curl -sL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-arm64.zip -o ngrok.zip && "
        "unzip -o ngrok.zip >/dev/null 2>&1 && rm -f ngrok.zip && chmod +x ~/ngrok)"
    )
    ssh_exec(ssh_client, install_cmd, timeout=600)
    print("ngrok installed successfully")


def setup_ngrok_tunnel(
    ssh_client, port: int, ngrok_auth_token: Optional[str] = None
) -> Optional[str]:
    """
    Set up ngrok tunnel and return the public URL
    """
    print(f"Setting up ngrok tunnel for port {port}...")

    # Install ngrok if not present
    install_ngrok(ssh_client)

    # Kill any existing ngrok processes
    ssh_exec(ssh_client, "pkill -f 'ngrok' || true", timeout=20)
    ssh_exec(ssh_client, "rm -f ~/ngrok.log", timeout=10)

    # Set auth token if provided
    if ngrok_auth_token:
        ssh_exec(
            ssh_client,
            f"~/ngrok config add-authtoken {ngrok_auth_token} || ngrok config add-authtoken {ngrok_auth_token}",
            timeout=30,
        )

    # Start ngrok tunnel
    start_cmd = f'~/ngrok http {port} --host-header="localhost:{port}" --log=stdout > ~/ngrok.log 2>&1 &'
    ssh_exec(ssh_client, start_cmd, timeout=10)

    # Wait for tunnel to be established and extract URL
    deadline = time.time() + 120
    tunnel_url = None

    pattern_list = [
        r"(https?://[a-zA-Z0-9-]+\.ngrok[-a-z0-9]*\.(?:io|app))",
        r"(https?://[a-zA-Z0-9-]+\.ngrok-free\..*)",
        r"(https?://[a-zA-Z0-9-]+\.ngrok\.io)",
        r"url=([^\s]+\.ngrok[-a-z0-9]*\.(?:io|app))",
        r"Forwarding\s+(https?://[^\s]+\.ngrok[-a-z0-9]*\.(?:io|app))",
    ]

    while time.time() < deadline:
        # Try ngrok local API first
        code, out, err = ssh_exec(
            ssh_client,
            "curl -s http://localhost:4040/api/tunnels 2>/dev/null || echo '{}'",
            timeout=10,
        )
        if out and out.strip() and out.strip() != "{}":
            try:
                api = json.loads(out)
                for t in api.get("tunnels") or []:
                    pub = t.get("public_url")
                    if pub and pub.startswith("https://"):
                        tunnel_url = pub
                        break
            except Exception:
                pass

        # Fallback to log parsing
        if not tunnel_url:
            code, out, err = ssh_exec(
                ssh_client, "cat ~/ngrok.log 2>/dev/null || echo 'no_log'", timeout=10
            )
            if out and out.strip() != "no_log":
                for p in pattern_list:
                    m = re.search(p, out)
                    if m:
                        tunnel_url = m.group(1)
                        break

        if tunnel_url:
            print(f"ngrok tunnel established: {tunnel_url}")
            return tunnel_url

        time.sleep(3)

    print("Failed to establish ngrok tunnel")
    return None


def verify_tunnel_connectivity(
    ssh_client, tunnel_url: str, endpoint: str = "/v1/models"
) -> bool:
    """Verify that the tunnel is working by checking the models endpoint"""
    print(f"Verifying tunnel connectivity to {tunnel_url}{endpoint}...")
    try:
        response = requests.get(f"{tunnel_url}{endpoint}", timeout=10)
        if response.status_code == 200:
            print("Tunnel connectivity verified successfully")
            return True
        else:
            print(f"Tunnel returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Failed to verify tunnel connectivity: {e}")
        return False


def cleanup_ngrok(ssh_client):
    """Clean up ngrok processes"""
    print("Cleaning up ngrok processes...")
    ssh_exec(ssh_client, "pkill -f 'ngrok' || true", timeout=20)
    ssh_exec(ssh_client, "rm -f ~/ngrok.log", timeout=10)
