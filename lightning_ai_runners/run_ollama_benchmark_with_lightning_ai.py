import datetime
import json
import os
import time

from lightning_sdk import Studio
from clients import GuideLLMBenchmarkClient, Mongo
from utils.ollama_utils import (
    get_ollama_environment_vars,
)
from utils.tokenizer_utils import HF_TOKENIZER
from utils.ngrok_utils import (
    install_ngrok,
    setup_ngrok_tunnel,
    verify_tunnel_connectivity,
    cleanup_ngrok,
)

#### INCOMPLETE CONT.


async def create_ollama_lightning_ai_server(
    studio_name: str,
    teamspace: str,
    org: str,
    llm_id: str,
    port: int = 11434,
    llm_parameter_size: str = "",
    llm_common_name: str = "",
    gpu_count: int = 1,
    gpu_id: str = "",
    auto_managed: bool = False,
):
    """
    Create and benchmark Ollama server on Lightning AI Studio
    """

    print(f"Using Ollama on port {port}")
    # Initialize mongo client
    mongo_client = Mongo(os.getenv("MONGODB_URL"))

    if auto_managed:
        try:
            studio = Studio(name=studio_name)
            print(f"Connected to auto-managed studio: {studio_name}")
        except Exception as e:
            print(f"Failed to connect to auto-managed studio, trying with teamspace: {e}")
            studio = Studio(name=studio_name, teamspace=teamspace, org=org)
    else:
        studio = Studio(name=studio_name, teamspace=teamspace, org=org)

    time_before_server_creation = datetime.datetime.now()

    # Set environment variables
    env_vars = get_ollama_environment_vars(gpu_count)
    # Set the host to use the default Ollama port
    env_vars["OLLAMA_HOST"] = "0.0.0.0:11434"

    env_commands = []
    for key, value in env_vars.items():
        env_commands.append(f"export {key}={value}")

    studio.run(" && ".join(env_commands))

    # Install Ollama
    studio.run("curl -fsSL https://ollama.com/install.sh | sh")

    # Install ngrok for creating public tunnel
    install_ngrok(studio)

    # Kill any existing Ollama processes and start fresh
    studio.run("pkill -f 'ollama' || true")
    time.sleep(2)

    # Also kill any process using the Ollama port
    studio.run("fuser -k 11434/tcp || true")
    time.sleep(2)

    # Start Ollama server in background using serve command
    studio.run("nohup ollama serve > /tmp/ollama.log 2>&1 &")

    # Wait for Ollama server to be ready
    time.sleep(10)

    # Start Ollama model in background (pull first if needed)
    studio.run(f"nohup ollama pull {llm_id} > /tmp/ollama_pull.log 2>&1 &")

    # Wa
    while True:
        try:
            # Check if server is responding
            health_check = studio.run(
                "curl -s http://localhost:11434/api/tags || echo 'not_ready'"
            )
            if "not_ready" not in health_check and "models" in health_check:
                print("Ollama server is ready at http://localhost:11434")
                ollama_server_readiness_time = datetime.datetime.now()
                break

        except Exception as e:
            print(f"Health check error: {e}")

        time.sleep(1)

    time_taken_to_start_ollama_server = (
        ollama_server_readiness_time - time_before_server_creation
    )
    time_taken_to_start_ollama_server_seconds = int(
        time_taken_to_start_ollama_server.total_seconds()
    )

    # Set up ngrok tunnel
    tunnel_url = setup_ngrok_tunnel(studio, port)

    # Verify tunnel is working with Ollama-specific endpoint
    verify_tunnel_connectivity(studio, tunnel_url, "/api/tags")

    # Convert HuggingFace model ID to Ollama format
    upload_llm_start_time = datetime.datetime.now()

    # Use studio.run to pull model instead of direct API call
    pull_command = f"curl -X POST {tunnel_url}/api/pull -d '{{\"model\": \"{llm_id}\"}}' -H 'Content-Type: application/json'"
    studio.run(pull_command)

    print("Model pull command executed")

    while True:
        try:
            # Check if model is available using ollama list command
            models_response = studio.run("curl -s http://localhost:11434/api/tags")
            print(models_response, "models_response", type(models_response))

            # Parse the JSON string response
            models_data = json.loads(models_response)

            # Check if the model is in the list
            if "models" in models_data:
                for model in models_data["models"]:
                    if model.get("name") == llm_id or model.get("model") == llm_id:
                        model_available = True
                        print(f"Model {llm_id} is loaded and responding")
                        break

            if model_available:
                break
        except Exception as e:
            print(f"Error checking model availability: {e}")
            pass

        time.sleep(10)

    if not model_available:
        raise Exception(f"Failed to load model {llm_id}.")

    print("Model loaded successfully")

    upload_llm_end_time = datetime.datetime.now()
    time_taken_to_upload_llm = upload_llm_end_time - upload_llm_start_time
    time_taken_to_upload_llm_seconds = int(time_taken_to_upload_llm.total_seconds())

    # Generate unique server ID
    server_id = f"lightning-ai-{int(time.time())}"

    pod_details = {
        "pod_id": server_id,  # Use server_id as pod_id equivalent
        "pod_url": tunnel_url,
        "time_taken_to_start_server": time_taken_to_start_ollama_server_seconds,
        "time_taken_to_upload_llm": time_taken_to_upload_llm_seconds,
        "llm_id": llm_id,
        "gpu_id": "lightning-ai-gpu",  # Lightning AI doesn't expose specific GPU IDs
        "volume_in_gb": 0,  # Lightning AI manages storage
        "container_disk_in_gb": 0,  # Lightning AI manages storage
        "port": 11434,  # Fixed Ollama port
        "created_at": datetime.datetime.now(),
        "inference_name": "Ollama",
        "server_type": "Lightning AI",
        "llm_parameter_size": llm_parameter_size,
        "llm_common_name": llm_common_name,
        "gpu_count": gpu_count,
    }

    mongo_client.insert_one("pod_benchmarks", pod_details)

    benchmark_start_time = datetime.datetime.now()
    print("Starting benchmark...")
    # Initialize benchmark client with HTTP/2 disabled for tunnel stability
    client = GuideLLMBenchmarkClient(
        base_url=tunnel_url,
        model=llm_id,
        mongo_url=os.getenv("MONGODB_URL"),
        processor=HF_TOKENIZER.get(
            llm_id, None
        ),  # Keep original HF tokenizer for metrics
        text_completions_path="/api/generate",
        max_retries=5,
        base_delay=3.0,
        max_delay=120.0,
        timeout=600.0,
        health_check_timeout=90.0,
    )

    # Run benchmark with 1-6 concurrent rate requests
    for benchmark_report_rate in range(6):
        try:
            report, path = await client.run_benchmark(
                max_seconds=30,  # Ollama uses 30s like RunPod version
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

    # Clean up tunnel and Ollama processes
    cleanup_ngrok(studio)

    print(f"Lightning AI Ollama server {server_id} benchmark completed successfully")
