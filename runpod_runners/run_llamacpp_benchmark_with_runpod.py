import datetime
import os
import time

import requests
import runpod

from clients import GuideLLMBenchmarkClient, Mongo, RunpodClient
from utils.llamacpp_utils import (
    build_llamacpp_runpod_docker_args,
    get_compatible_llamacpp_image,
    get_llamacpp_environment_vars,
)
from utils.tokenizer_utils import HF_TOKENIZER


async def create_llamacpp_pod(
    gpu_id: str,
    volume_in_gb: int,
    container_disk_in_gb: int,
    llm_id: str,
    port: int = 8000,
    llm_parameter_size: str | None = None,
    llm_common_name: str | None = None,
    gpu_count: int = 1,
    quantization: str = "Q4_0",
    model_path: str = "",
):
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    runpod.api_key = os.getenv("RUNPOD_API_KEY")
    runpod_graphql_client = RunpodClient(api_key=os.getenv("RUNPOD_API_KEY"))

    time_before_pod_creation = datetime.datetime.now()

    # Build docker args using utility function - downloads model and runs server
    docker_cmd = build_llamacpp_runpod_docker_args(
        llm_id, port, gpu_count, quantization, model_path
    )

    # Choose Docker image based on CUDA compatibility
    image_name = get_compatible_llamacpp_image(gpu_id, llm_id)

    # Get environment variables with multi-GPU support
    env_vars = get_llamacpp_environment_vars(gpu_count)

    # Create pod with GPU and the full setup command
    pod = runpod.create_pod(
        name="llamacpp-pod",
        image_name=image_name,
        gpu_type_id=gpu_id,
        env=env_vars,
        gpu_count=gpu_count,
        volume_in_gb=volume_in_gb,
        container_disk_in_gb=container_disk_in_gb,
        volume_mount_path="/workspace",
        ports=f"{port}/http",
        docker_args=docker_cmd,
    )

    pod_id = pod["id"]
    pod_url = f"https://{pod_id}-{port}.proxy.runpod.net"

    # Check if llama.cpp server is ready
    max_wait_time = 20 * 60  # 20 minutes
    start_wait = time.time()

    while time.time() - start_wait < max_wait_time:
        try:
            response = requests.get(f"{pod_url}/v1/models", timeout=30)
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("data"):
                    print(f"LLamaCPP server is ready at {pod_url}")
                    server_readiness_time = datetime.datetime.now()
                    break
        except Exception as e:
            elapsed = int(time.time() - start_wait)
            print(f"Waiting for server to be ready... ({elapsed}s elapsed) - {e}")
            time.sleep(5)
    else:
        raise TimeoutError(
            f"Server did not become ready within {max_wait_time} seconds"
        )

    time_taken_to_start_server_seconds = int(
        (server_readiness_time - time_before_pod_creation).total_seconds()
    )

    print(f"Server started in {time_taken_to_start_server_seconds} seconds")

    pod_details = {
        "pod_id": pod_id,
        "pod_url": pod_url,
        "time_taken_to_start_server": time_taken_to_start_server_seconds,
        "time_taken_to_upload_llm": time_taken_to_start_server_seconds,
        "llm_id": llm_id,
        "gpu_id": gpu_id,
        "volume_in_gb": volume_in_gb,
        "container_disk_in_gb": container_disk_in_gb,
        "port": port,
        "created_at": datetime.datetime.now(),
        "inference_name": "LLAMACPP",
        "server_type": "Runpod",
        "llm_parameter_size": llm_parameter_size,
        "llm_common_name": llm_common_name,
        "gpu_count": gpu_count,
    }

    mongo_client.insert_one("pod_benchmarks", pod_details)

    benchmark_start_time = datetime.datetime.now()

    # Initialize benchmark client
    # For llama.cpp, the model name in /v1/models is the HF repo name
    # So use model_path instead of llm_id for the model parameter
    model_name_for_client = model_path

    client = GuideLLMBenchmarkClient(
        base_url=pod_url,
        model=model_name_for_client,
        mongo_url=os.getenv("MONGODB_URL"),
        processor=HF_TOKENIZER.get(llm_id, None),
        text_completions_path="/v1/completions",
        max_retries=3,
        base_delay=2.0,
        max_delay=60.0,
        timeout=300.0,
        health_check_timeout=45.0,
    )

    # Run concurrent benchmarks (rates 1-6)
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
                    benchmark_report, pod_id, benchmark_report_rate + 1
                )
                mongo_client.insert_one("benchmark_results", result_data)
            else:
                print(
                    f"Failed to get benchmark results for rate {benchmark_report_rate + 1}"
                )
                break

        except Exception as e:
            print(f"Error in benchmark rate {benchmark_report_rate + 1}: {e}")
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
                benchmark_report, pod_id, None, "throughput"
            )
            mongo_client.insert_one("benchmark_results", result_data)
        else:
            print("Failed to get throughput benchmark results")

    except Exception as e:
        print(f"Error in throughput benchmark: {e}")

    benchmark_end_time = datetime.datetime.now()

    pod_cost = runpod_graphql_client.calculate_used_balance(pod_id)

    benchmark_duration = benchmark_end_time - benchmark_start_time

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

    # Stop the pod
    runpod.terminate_pod(pod_id)
    print(f"Pod {pod_id} terminated successfully")
