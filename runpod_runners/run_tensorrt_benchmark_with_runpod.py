import datetime
import os
import time

import requests
import runpod

from clients import GuideLLMBenchmarkClient, Mongo, RunpodClient
from utils.tensorrt_utils import (
    build_tensorrt_serve_args,
    get_compatible_tensorrt_image,
    get_tensorrt_environment_vars,
)


async def create_tensorrt_pod(
    gpu_id: str,
    volume_in_gb: int,
    container_disk_in_gb: int,
    llm_id: str,
    port: int = 8000,
    llm_parameter_size: str | None = None,
    llm_common_name: str | None = None,
    gpu_count: int = 1,
):
   
    ## Initialize mongo client
    mongo_client = Mongo(os.getenv("MONGODB_URL"))

    ## Apply api key to runpod
    runpod.api_key = os.getenv("RUNPOD_API_KEY")

    ## Also apply api key to runpod graphql client for calculating used balance
    runpod_graphql_client = RunpodClient(api_key=os.getenv("RUNPOD_API_KEY"))

    time_before_pod_creation = datetime.datetime.now()

    # Build serve args using utility function
    serve_args = build_tensorrt_serve_args(llm_id, port, gpu_count)

    # Choose Docker image based on CUDA compatibility
    image_name = get_compatible_tensorrt_image(gpu_id, llm_id)

    # Get environment variables
    env_vars = get_tensorrt_environment_vars(llm_id, gpu_count)

    # Create a pod in runpod with TensorRT-LLM image
    # The serve_args are prefixed with "trtllm-serve" to form the full docker command
    pod = runpod.create_pod(
        name="tensorrt-pod",
        image_name=image_name,
        gpu_type_id=gpu_id,
        env=env_vars,
        docker_args=f"trtllm-serve {serve_args}",
        volume_in_gb=volume_in_gb,
        container_disk_in_gb=container_disk_in_gb,
        volume_mount_path="/root/.cache/huggingface",
        ports=f"{port}/http",
        gpu_count=gpu_count,
    )

    pod_id = pod["id"]

    pod_url = f"https://{pod_id}-{port}.proxy.runpod.net"

    # Check if TensorRT-LLM server is ready
    # trtllm-serve provides OpenAI-compatible endpoints: /v1/models, /v1/completions, /v1/chat/completionsa and also supports /health, /metrics, /version endpoints
    print("Waiting for TensorRT-LLM server to be ready...")
    while True:
        try:
            # Try /health endpoint first (faster check)
            health_response = requests.get(pod_url + "/health", timeout=5)
            if health_response.status_code == 200:
                # Then verify model is loaded via /v1/models
                response = requests.get(pod_url + "/v1/models", timeout=10)
                response_data = response.json()
                # Check if the model is loaded
                if response_data.get("data") and len(response_data.get("data", [])) > 0:
                    model_info = response_data["data"][0]
                    if model_info.get("id") == llm_id or llm_id in model_info.get("id", ""):
                        print(f"TensorRT-LLM server is ready at {pod_url} with model {llm_id}")
                        tensorrt_server_readiness_time = datetime.datetime.now()
                        break
            time.sleep(2)
        except Exception:
            # Server might still be starting up
            time.sleep(2)

    time_taken_to_start_server = (
        tensorrt_server_readiness_time - time_before_pod_creation
    )

    time_taken_to_start_server_seconds = int(
        time_taken_to_start_server.total_seconds()
    )

    # Note: trtllm-serve handles model download and conversion automatically
    time_taken_to_upload_llm = time_taken_to_start_server_seconds

    pod_details = {
        "pod_id": pod_id,
        "pod_url": pod_url,
        "time_taken_to_start_server": time_taken_to_start_server_seconds,
        "time_taken_to_upload_llm": time_taken_to_upload_llm,
        "llm_id": llm_id,
        "gpu_id": gpu_id,
        "volume_in_gb": volume_in_gb,
        "container_disk_in_gb": container_disk_in_gb,
        "port": port,
        "created_at": datetime.datetime.now(),
        "inference_name": "TensorRT-LLM",
        "server_type": "Runpod",
        "llm_parameter_size": llm_parameter_size,
        "llm_common_name": llm_common_name,
        "gpu_count": gpu_count,
    }

    mongo_client.insert_one("pod_benchmarks", pod_details)

    benchmark_start_time = datetime.datetime.now()

    # Initialize benchmark client
    client = GuideLLMBenchmarkClient(
        base_url=pod_url,
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
                    benchmark_report, pod_id, benchmark_report_rate + 1
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
                benchmark_report, pod_id, None, "throughput"
            )
            mongo_client.insert_one("benchmark_results", result_data)
        else:
            print("Failed to get throughput benchmark results")

    except Exception as e:
        print(f"Error in throughput benchmark: {e}")

    benchmark_duration = datetime.datetime.now() - benchmark_start_time

    pod_cost = runpod_graphql_client.calculate_used_balance(pod_id)

    # After benchmark is completed update the pod details with pod cost and benchmark duration
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