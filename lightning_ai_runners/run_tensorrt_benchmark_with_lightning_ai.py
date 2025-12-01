import base64
import datetime
import os
import time

from lightning_sdk import Studio
from clients import GuideLLMBenchmarkClient, Mongo
from utils.tokenizer_utils import HF_TOKENIZER
from utils.tensorrt_utils import (
    build_tensorrt_serve_args,
    get_compatible_tensorrt_image,
    get_tensorrt_environment_vars,
)
from utils.ngrok_utils import (
    install_ngrok,
    setup_ngrok_tunnel,
    verify_tunnel_connectivity,
    cleanup_ngrok,
)


async def create_tensorrt_lightning_ai_server(
    studio_name: str,
    teamspace: str,
    org: str,
    llm_id: str,
    port: int = 8000,
    llm_parameter_size: str = "",
    llm_common_name: str = "",
    gpu_count: int = 1,
    gpu_id: str = "",
    moe_backend: str | None = None,
    kv_cache_free_gpu_memory_fraction: float | None = None,
    enable_attention_dp: bool | None = None,
    ep_size: int | None = None,
    pp_size: int | None = None,
    auto_managed: bool = False,
):

    # Initialize mongo client
    mongo_client = Mongo(os.getenv("MONGODB_URL"))

    # Connect to Lightning AI Studio
    if auto_managed:
        # For auto-managed studios, try simple connection first (avoid auth issues)
        try:
            studio = Studio(name=studio_name)
            print(f"Connected to auto-managed studio: {studio_name}")
        except Exception as e:
            print(f"Failed to connect to auto-managed studio, trying with teamspace: {e}")
            studio = Studio(name=studio_name, teamspace=teamspace, org=org)
    else:
        # For manually created studios, use full teamspace/org specification
        studio = Studio(name=studio_name, teamspace=teamspace, org=org)

    time_before_server_creation = datetime.datetime.now()

    # Get compatible TensorRT-LLM Docker image
    image_name = get_compatible_tensorrt_image(gpu_id, llm_id)
    print(f"Using TensorRT-LLM Docker image: {image_name}")

    # Get environment variables for the container
    env_vars = get_tensorrt_environment_vars(llm_id, gpu_count)

    # Build serve arguments
    serve_args, yaml_config_content = build_tensorrt_serve_args(
        llm_id,
        port,
        gpu_count,
        moe_backend=moe_backend,
        kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
        enable_attention_dp=enable_attention_dp,
        ep_size=ep_size,
        pp_size=pp_size,
    )

    # Pull the Docker image
    print(f"Pulling Docker image: {image_name}")
    studio.run(f"docker pull {image_name}")
    print("Docker image pulled successfully")

    # Build the Docker command that will run inside the container
    # When optional parameters are provided via CLI, it creates a YAML config file that TensorRT-LLM can read at startup.
    if yaml_config_content:
        config_b64 = base64.b64encode(yaml_config_content.encode('utf-8')).decode('ascii')
        docker_cmd = f"sh -c 'echo {config_b64} | base64 -d > /tmp/trtllm_config.yaml && trtllm-serve {serve_args}'"
    else:
        docker_cmd = f"trtllm-serve {serve_args}"

    print(f"TensorRT-LLM Docker command: {docker_cmd}")

    # Build environment variables string for Docker (quote values to handle special characters)
    env_vars_str = " ".join([f"-e {key}='{value}'" for key, value in env_vars.items()])

    # Create HuggingFace cache directory on host
    studio.run("mkdir -p /tmp/hf_cache || true")

    # Run Docker container with GPU access, port mapping, and environment variables
    container_name = f"tensorrt-llm-{int(time.time())}"
    # docker_cmd already includes sh -c wrapper if YAML config is needed
    docker_run_cmd = (
        f"docker run -d "
        f"--name {container_name} "
        f"--gpus all "
        f"-p {port}:{port} "
        f"{env_vars_str} "
        f"-v /tmp/hf_cache:/root/.cache/huggingface "
        f"{image_name} "
        f"{docker_cmd}"
    )

    print(f"Starting TensorRT-LLM Docker container: {container_name}")
    studio.run(docker_run_cmd)
    
    # Verify container is running
    container_status = studio.run(f"docker ps --filter name={container_name} --format '{{{{.Status}}}}'")
    if "Up" not in container_status:
        # Check container logs if it failed to start
        logs = studio.run(f"docker logs {container_name} 2>&1 | tail -50")
        raise Exception(f"Container {container_name} failed to start. Logs: {logs}")
    print("TensorRT-LLM Docker container started successfully")

    # Install ngrok for creating public tunnel
    install_ngrok(studio)
    print("ngrok installed successfully")

    # Wait for TensorRT-LLM server to be ready
    print("Waiting for TensorRT-LLM server to be ready...")
    while True:
        try:
            # Try /health endpoint first (faster check)
            health_check = studio.run(
                f"curl -s http://localhost:{port}/health || echo 'not_ready'"
            )
            if "not_ready" not in health_check:
                # Then verify model is loaded via /v1/models
                models_response = studio.run(
                    f"curl -s http://localhost:{port}/v1/models || echo 'not_ready'"
                )
                if "not_ready" not in models_response:
                    # Check if the model is in the response
                    if llm_id in models_response or '"data"' in models_response:
                        print(
                            f"TensorRT-LLM server is ready at http://localhost:{port} with model {llm_id}"
                        )
                        tensorrt_server_readiness_time = datetime.datetime.now()
                        break
        except Exception:
            pass

        time.sleep(2)

    time_taken_to_start_tensorrt_server = (
        tensorrt_server_readiness_time - time_before_server_creation
    )
    time_taken_to_start_tensorrt_server_seconds = int(
        time_taken_to_start_tensorrt_server.total_seconds()
    )

    time_taken_to_upload_llm = time_taken_to_start_tensorrt_server_seconds

    # Set up ngrok tunnel
    tunnel_url = setup_ngrok_tunnel(studio, port)

    # Verify tunnel is working
    verify_tunnel_connectivity(studio, tunnel_url)

    # Generate unique server ID
    server_id = f"lightning-ai-{int(time.time())}"

    pod_details = {
        "pod_id": server_id,  # Use server_id as pod_id equivalent
        "pod_url": tunnel_url,
        "time_taken_to_start_server": time_taken_to_start_tensorrt_server_seconds,
        "time_taken_to_upload_llm": time_taken_to_upload_llm,
        "llm_id": llm_id,
        "gpu_id": "lightning-ai-gpu",  # Lightning AI doesn't expose specific GPU IDs
        "volume_in_gb": 0,  # Lightning AI manages storage
        "container_disk_in_gb": 0,  # Lightning AI manages storage
        "port": port,
        "created_at": datetime.datetime.now(),
        "inference_name": "TensorRT-LLM",
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
        chat_completions_path="/v1/chat/completions",
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
                max_seconds=30,
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

    # Stop and remove Docker container
    print(f"Stopping Docker container: {container_name}")
    studio.run(f"docker stop {container_name} || true")
    studio.run(f"docker rm {container_name} || true")
    print(f"Docker container {container_name} cleaned up successfully")

    print(f"Lightning AI TensorRT-LLM server {server_id} benchmark completed successfully")

