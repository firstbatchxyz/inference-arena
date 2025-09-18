import datetime
import os
import time

from lightning_sdk import Studio
from clients import GuideLLMBenchmarkClient, Mongo
from utils.tokenizer_utils import HF_TOKENIZER
from utils.sglang_utils import build_sglang_serve_args, get_sglang_environment_vars
from utils.ngrok_utils import (
    install_ngrok,
    setup_ngrok_tunnel,
    verify_tunnel_connectivity,
    cleanup_ngrok,
)


async def create_sglang_lightning_ai_server(
    studio_name: str,
    teamspace: str,
    org: str,
    llm_id: str,
    port: int = 30000,
    llm_parameter_size: str = "",
    llm_common_name: str = "",
    gpu_count: int = 1,
    gpu_id: str = "",
):
    """
    Create and benchmark SGLang server on Lightning AI Studio
    """

    # Initialize mongo client
    mongo_client = Mongo(os.getenv("MONGODB_URL"))

    # Initialize Lightning AI Studio
    studio = Studio(
        name=studio_name,
        teamspace=teamspace,
        org=org,
    )
    ## SDK not working for selecting gpu and start with that. That is why for now you should create studio manually.

    time_before_server_creation = datetime.datetime.now()

    # Set environment variables
    env_vars = get_sglang_environment_vars(llm_id, gpu_count)
    env_commands = []
    for key, value in env_vars.items():
        env_commands.append(f"export {key}={value}")

    studio.run(" && ".join(env_commands))

    ## Install uv for faster installation
    studio.run("curl -LsSf https://astral.sh/uv/install.sh | sh")

    # Install SGLang
    studio.run('uv pip install "sglang[all]>=0.5.2rc0"')

    # Install ngrok for creating public tunnel
    install_ngrok(studio)

    # Build SGLang serve arguments
    serve_args = build_sglang_serve_args(llm_id, port, gpu_count)
    sglang_command = f"uv run python -m sglang.launch_server --model-path {serve_args}"

    # Start SGLang server in background
    studio.run_and_detach(sglang_command)

    # Wait for SGLang server to be ready

    while True:
        try:
            print("Checking SGLang server health...")
            health_check = studio.run(
                f"curl -s http://localhost:{port}/health || echo 'not_ready'"
            )
            print(health_check)
            if "not_ready" not in health_check:
                # Verify model is loaded by checking models endpoint
                models_response = studio.run(
                    f"curl -s http://localhost:{port}/v1/models"
                )
                if llm_id in models_response:
                    print(
                        f"SGLang server is ready at http://localhost:{port} with model {llm_id}"
                    )
                    sglang_server_readiness_time = datetime.datetime.now()
                    break
        except Exception:
            pass

        time.sleep(1)

    time_taken_to_start_sglang_server = (
        sglang_server_readiness_time - time_before_server_creation
    )
    time_taken_to_start_sglang_server_seconds = int(
        time_taken_to_start_sglang_server.total_seconds()
    )

    # Set up ngrok tunnel
    tunnel_url = setup_ngrok_tunnel(studio, port)

    # Verify tunnel is working
    verify_tunnel_connectivity(studio, tunnel_url)

    # Generate unique server ID
    server_id = f"lightning-ai-{int(time.time())}"

    pod_details = {
        "pod_id": server_id,  # Use server_id as pod_id equivalent
        "pod_url": tunnel_url,
        "time_taken_to_start_server": time_taken_to_start_sglang_server_seconds,
        "time_taken_to_upload_llm": time_taken_to_start_sglang_server_seconds,
        "llm_id": llm_id,
        "gpu_id": "lightning-ai-gpu",  # Lightning AI doesn't expose specific GPU IDs
        "volume_in_gb": 0,  # Lightning AI manages storage
        "container_disk_in_gb": 0,  # Lightning AI manages storage
        "port": port,
        "created_at": datetime.datetime.now(),
        "inference_name": "SGLang",
        "server_type": "Lightning AI",
        "llm_parameter_size": llm_parameter_size,
        "llm_common_name": llm_common_name,
        "gpu_count": gpu_count,
    }

    mongo_client.insert_one("pod_benchmarks", pod_details)

    benchmark_start_time = datetime.datetime.now()

    # Initialize benchmark client
    client = GuideLLMBenchmarkClient(
        base_url=tunnel_url,
        model=llm_id,
        mongo_url=os.getenv("MONGODB_URL"),
        processor=HF_TOKENIZER.get(llm_id, None),
        text_completions_path="/v1/completions",
        max_retries=3,
        base_delay=2.0,
        max_delay=60.0,
        timeout=300.0,
        health_check_timeout=45.0,
    )

    # Run benchmark with 1-6 concurrent rate requests
    for benchmark_report_rate in range(6):
        try:
            report, path = await client.run_benchmark(
                max_seconds=15,  # SGLang uses 15s like RunPod version
                mongo_query={},
                rate_type="concurrent",
                output_path="benchmark_results.json",
                rate=float(benchmark_report_rate + 1),
            )

            if report and report.benchmarks:
                benchmark_report = report.benchmarks[0]
                result_data = client._extract_benchmark_metrics(
                    benchmark_report, server_id, benchmark_report_rate + 1
                )
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
            output_path="benchmark_results.json",
        )

        if report and report.benchmarks:
            benchmark_report = report.benchmarks[0]
            result_data = client._extract_benchmark_metrics(
                benchmark_report, server_id, None, "throughput"
            )
            mongo_client.insert_one("benchmark_results", result_data)
        else:
            print("Failed to get throughput benchmark results")

    except Exception as e:
        print(f"Error in throughput benchmark: {e}")

    benchmark_duration = datetime.datetime.now() - benchmark_start_time

    pod_cost = 0.0  # Lightning AI cost is not exposed via API

    # After benchmark is completed update the pod details with pod cost and benchmark duration
    mongo_client.update_one(
        "pod_benchmarks",
        {"pod_id": server_id},
        {
            "$set": {
                "pod_cost": pod_cost,
                "benchmark_duration": benchmark_duration.total_seconds(),
            }
        },
    )

    # Clean up tunnel processes
    cleanup_ngrok(studio)

    print(f"Lightning AI SGLang server {server_id} benchmark completed successfully")
