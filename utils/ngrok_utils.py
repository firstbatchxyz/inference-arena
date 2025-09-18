import json
import os
import re
import time
from lightning_sdk import Studio


def install_ngrok(studio: Studio):
    """Install ngrok on the Lightning AI Studio"""
    studio.run(
        "curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && "
        "echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list && "
        "sudo apt update && sudo apt install -y ngrok"
    )


def setup_ngrok_tunnel(studio: Studio, port: int) -> str:
    """Set up ngrok tunnel and return the public URL"""
    # Clean up existing tunnel processes and start ngrok
    studio.run("pkill -f 'ngrok' || true")
    time.sleep(2)
    studio.run("rm -f /tmp/tunnel.log")
    studio.run(
        f'ngrok config add-authtoken {os.getenv("NGROK_AUTH_TOKEN")} && ngrok http {port} --host-header="localhost:{port}" --log=stdout > /tmp/tunnel.log 2>&1 &'
    )

    time.sleep(10)

    # Extract tunnel URL from ngrok
    tunnel_url = None
    max_tunnel_wait = 60
    tunnel_wait = 0

    while tunnel_wait < max_tunnel_wait:
        time.sleep(5)
        tunnel_wait += 5

        try:
            try:
                ngrok_api_response = studio.run(
                    "curl -s http://localhost:4040/api/tunnels 2>/dev/null || echo '{}'"
                )

                if ngrok_api_response and ngrok_api_response != "{}":
                    try:
                        tunnels_data = json.loads(ngrok_api_response)
                        if "tunnels" in tunnels_data and tunnels_data["tunnels"]:
                            for tunnel in tunnels_data["tunnels"]:
                                if tunnel.get("proto") == "https":
                                    tunnel_url = tunnel.get("public_url")
                                    break
                            if not tunnel_url and tunnels_data["tunnels"]:
                                tunnel_url = tunnels_data["tunnels"][0].get(
                                    "public_url"
                                )
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass

            if not tunnel_url:
                tunnel_logs = studio.run(
                    "cat /tmp/tunnel.log 2>/dev/null || echo 'no_log'"
                )

                if "no_log" not in tunnel_logs and tunnel_logs.strip():
                    url_patterns = [
                        r"(https?://[a-zA-Z0-9-]+\.ngrok[-a-z0-9]*\.(?:io|app))",
                        r"(https?://[a-zA-Z0-9-]+\.ngrok-free\.app)",
                        r"(https?://[a-zA-Z0-9-]+\.ngrok\.io)",
                        r"url=([^\s]+\.ngrok[-a-z0-9]*\.(?:io|app))",
                        r"Forwarding\s+(https?://[^\s]+\.ngrok[-a-z0-9]*\.(?:io|app))",
                    ]

                    for pattern in url_patterns:
                        url_match = re.search(pattern, tunnel_logs, re.IGNORECASE)
                        if url_match:
                            if url_match.groups():
                                tunnel_url = url_match.group(1).strip()
                            else:
                                tunnel_url = url_match.group(0).strip()
                            break

            if tunnel_url:
                break

        except Exception:
            pass

    if not tunnel_url:
        studio.run("ps aux | grep ngrok")
        studio.run("cat /tmp/tunnel.log")
        studio.run("ngrok version")
        raise Exception("Failed to establish tunnel URL after multiple attempts")

    return tunnel_url


def verify_tunnel_connectivity(
    studio: Studio, tunnel_url: str, health_endpoint: str = "/health"
):
    """Verify that the tunnel is working by testing connectivity"""
    test_result = studio.run(
        f"curl -s {tunnel_url}{health_endpoint} -m 15 --connect-timeout 10 || echo 'tunnel_not_ready'"
    )
    print(f"Tunnel URL: {tunnel_url}")
    if "tunnel_not_ready" in test_result:
        studio.run("ps aux | grep ngrok")
        studio.run("cat /tmp/tunnel.log | tail -20")
        raise Exception("Tunnel URL not responding after establishment")


def cleanup_ngrok(studio: Studio):
    """Clean up ngrok processes"""
    studio.run("pkill -f 'ngrok' || true")
