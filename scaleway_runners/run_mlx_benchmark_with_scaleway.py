import asyncio
import datetime
import os
import sys
import time
from typing import Optional
import paramiko
from scaleway import Client
from scaleway.applesilicon.v1alpha1 import ApplesiliconV1Alpha1API
from scaleway_core.bridge.zone import ZONE_FR_PAR_1

from clients import GuideLLMBenchmarkClient, Mongo
from utils.tokenizer_utils import HF_TOKENIZER
from utils.mlx_utils import (
    start_mlx_server,
    wait_for_mlx_server,
    setup_ngrok_tunnel,
    verify_tunnel_connectivity,
    cleanup_ngrok,
)


async def create_mlx_scaleway_server(
    server_id: str,
    model_path: str,
    model_type: str = "lm",
    port: int = 8000,
    context_length: Optional[int] = None,
    config_name: Optional[str] = None,
    quantize: Optional[int] = None,
    max_concurrency: int = 1,
    llm_parameter_size: str = "",
    llm_common_name: str = "",
    gpu_count: int = 1,
    gpu_id: str = "",
):
    """
    Create and benchmark MLX OpenAI server on Scaleway Apple Silicon instance
    """

    # Initialize mongo client
    mongo_url = os.getenv("MONGODB_URL")
    if not mongo_url:
        print(
            "Warning: MONGODB_URL is not set. Results will not be persisted.",
            file=sys.stderr,
        )
    mongo_client = Mongo(mongo_url) if mongo_url else None

    # Initialize Scaleway client
    client = Client(
        access_key=os.getenv("SCW_ACCESS_KEY"),
        secret_key=os.getenv("SCW_SECRET_KEY"),
        default_project_id=os.getenv("SCW_PROJECT_ID"),
        default_region=os.getenv("SCW_REGION"),
        default_zone=ZONE_FR_PAR_1,
    )
    apple_silicon_api = ApplesiliconV1Alpha1API(client)

    time_before_server_creation = datetime.datetime.now()

    # Fetch server SSH details
    print(f"Fetching Scaleway server details for {server_id}...")
    resp = apple_silicon_api.get_server(server_id=server_id)
    ssh_hostname = getattr(resp, "ip", None)
    ssh_username = getattr(resp, "ssh_username", None)
    ssh_password = getattr(resp, "sudo_password", None)

    if not ssh_hostname or not ssh_username or not ssh_password:
        raise RuntimeError(
            "Missing SSH connection details (ip/ssh_username/sudo_password)"
        )

    # Establish SSH connection
    print(f"Connecting to server at {ssh_hostname}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=ssh_hostname,
        username=ssh_username,
        password=ssh_password,
        port=22,
        look_for_keys=False,
        allow_agent=False,
        timeout=60,
    )

    try:

        upload_llm_start_time = datetime.datetime.now()

        start_mlx_server(
            ssh,
            model_path=model_path,
        )

        # Set up ngrok tunnel
        ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
        tunnel_url = setup_ngrok_tunnel(ssh, port, ngrok_auth_token)

        if not tunnel_url:
            raise RuntimeError("Failed to establish ngrok tunnel")

        # Wait for server to be ready
        ready_model_id = wait_for_mlx_server(
            ssh, tunnel_url, model_path, max_wait_time=600
        )

        if not ready_model_id:
            raise RuntimeError("MLX server did not become ready in time")

        mlx_server_readiness_time = datetime.datetime.now()
        upload_llm_end_time = datetime.datetime.now()

        # Verify tunnel connectivity
        if not verify_tunnel_connectivity(ssh, tunnel_url, "/v1/models"):
            raise RuntimeError("Tunnel connectivity verification failed")

        # Calculate timing metrics
        time_taken_to_start_server = (
            mlx_server_readiness_time - time_before_server_creation
        )
        time_taken_to_start_server_seconds = int(
            time_taken_to_start_server.total_seconds()
        )

        time_taken_to_upload_llm = upload_llm_end_time - upload_llm_start_time
        time_taken_to_upload_llm_seconds = int(time_taken_to_upload_llm.total_seconds())

        # Persist pod details
        pod_id = f"scaleway-{server_id}-{int(time.time())}"
        pod_details = {
            "pod_id": pod_id,
            "pod_url": tunnel_url,
            "server_id": server_id,
            "server_ip": ssh_hostname,
            "time_taken_to_start_server": time_taken_to_start_server_seconds,
            "time_taken_to_upload_llm": time_taken_to_upload_llm_seconds,
            "llm_id": ready_model_id,
            "gpu_id": gpu_id or "apple-silicon",
            "volume_in_gb": 0,
            "container_disk_in_gb": 0,
            "port": port,
            "created_at": datetime.datetime.now(),
            "inference_name": "MLX-OpenAI-Server",
            "server_type": "Scaleway-AppleSilicon",
            "llm_parameter_size": llm_parameter_size,
            "llm_common_name": llm_common_name or model_path,
            "gpu_count": gpu_count,
            "model_type": model_type,
            "context_length": context_length,
            "config_name": config_name,
            "quantize": quantize,
            "max_concurrency": max_concurrency,
        }

        if mongo_client:
            mongo_client.insert_one("pod_benchmarks", pod_details)

        # Run benchmarks
        benchmark_start_time = datetime.datetime.now()

        # Get tokenizer for the model if available
        processor = HF_TOKENIZER.get(ready_model_id, None)

        client = GuideLLMBenchmarkClient(
            base_url=tunnel_url,
            model=ready_model_id,
            mongo_url=mongo_url,
            processor=processor,
            text_completions_path="/v1/completions",
            max_retries=5,
            base_delay=3.0,
            max_delay=120.0,
            timeout=600.0,
            health_check_timeout=90.0,
        )

        # Run concurrent benchmark with 1-6 concurrent rate requests
        for benchmark_report_rate in range(6):
            try:
                report, path = await client.run_benchmark(
                    max_seconds=30,
                    mongo_query={},
                    rate_type="concurrent",
                    output_path="benchmark_results.json",
                    rate=float(benchmark_report_rate + 1),
                    disable_token_counting=True,  # MLX doesn't always return token counts
                )

                if report and report.benchmarks:
                    benchmark_report = report.benchmarks[0]
                    result_data = client._extract_benchmark_metrics(
                        benchmark_report, pod_id, benchmark_report_rate + 1
                    )
                    if mongo_client:
                        mongo_client.insert_one("benchmark_results", result_data)
                else:
                    print(
                        f"Failed to get benchmark results for rate {benchmark_report_rate+1}"
                    )
                    break

            except Exception as e:
                print(f"Error in benchmark rate {benchmark_report_rate+1}: {e}")
                break

            # Add delay between concurrent benchmarks to prevent overloading the server
            time.sleep(5)

        # Run throughput benchmark
        try:
            report, path = await client.run_benchmark(
                max_seconds=60,
                mongo_query={},
                rate_type="throughput",
                output_path="benchmark_results_throughput.json",
                disable_token_counting=True,
            )

            if report and report.benchmarks:
                benchmark_report = report.benchmarks[0]
                result_data = client._extract_benchmark_metrics(
                    benchmark_report, pod_id, None, "throughput"
                )
                if mongo_client:
                    mongo_client.insert_one("benchmark_results", result_data)
            else:
                print("Failed to get throughput benchmark results")

        except Exception as e:
            print(f"Error in throughput benchmark: {e}")

        benchmark_duration = datetime.datetime.now() - benchmark_start_time

        # Scaleway cost is not exposed via API, set to 0
        pod_cost = 0.0

        # Update pod details with benchmark duration
        if mongo_client:
            mongo_client.update_one(
                "pod_benchmarks",
                {"pod_id": pod_id},
                {
                    "$set": {
                        "pod_cost": pod_cost,
                        "benchmark_duration": benchmark_duration.total_seconds(),
                    }
                },
            )

        print(f"Scaleway MLX server {pod_id} benchmark completed successfully")

    finally:
        # Cleanup: stop server and ngrok
        print("Cleaning up remote resources...")
        cleanup_ngrok(ssh)
        ssh.close()
        print("Cleanup completed")
