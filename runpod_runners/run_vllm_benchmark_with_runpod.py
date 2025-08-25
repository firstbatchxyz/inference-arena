import argparse
import datetime
import os
import sys
import time

import requests
import runpod

from clients import GuideLLMBenchmarkClient, Mongo, RunpodClient
from utils.vllm_utils import (
    build_vllm_docker_args,
    get_compatible_vllm_image,
    get_vllm_environment_vars,
)


async def create_vllm_pod(
    gpu_id: str,
    volume_in_gb: int,
    container_disk_in_gb: int,
    llm_id: str,
    port: int = 8000,
    llm_parameter_size: str = "",
    llm_common_name: str = "",
    gpu_count: int = 1,
):

    ## Initialize mongo client
    mongo_client = Mongo(os.getenv("MONGODB_URL"))

    ## Apply api key to runpod
    runpod.api_key = os.getenv("RUNPOD_API_KEY")

    ## Also apply api key to runpod graphql client for calculating used balance
    runpod_graphql_client = RunpodClient(api_key=os.getenv("RUNPOD_API_KEY"))

    time_before_pod_creation = datetime.datetime.now()

    # Build docker args using utility function
    docker_args = build_vllm_docker_args(llm_id, port, gpu_count)

    # Choose Docker image based on CUDA compatibility and model requirements or etc.
    image_name = get_compatible_vllm_image(gpu_id, llm_id)

    env_vars = get_vllm_environment_vars(llm_id, llm_parameter_size, gpu_id, gpu_count)

    # Create a pod in runpod with vLLM image
    pod = runpod.create_pod(
        name="vllm-pod",
        image_name=image_name,
        gpu_type_id=gpu_id,
        env=env_vars,
        docker_args=docker_args,
        volume_in_gb=volume_in_gb,
        container_disk_in_gb=container_disk_in_gb,
        volume_mount_path="/root/.cache/huggingface",
        ports=f"{port}/http",
        gpu_count=gpu_count,
    )

    pod_id = pod["id"]

    pod_url = f"https://{pod_id}-{port}.proxy.runpod.net"

    # Check if vLLM server is ready. Guidellm client also do that but it is calculating vLLM server readiness time
    while True:
        try:
            response = requests.get(pod_url + "/v1/models")
            response = response.json()
            if response.get("data")[0].get("id") == llm_id:
                print(f"vLLM server is ready at {pod_url} with model {llm_id}")
                vllm_server_readiness_time = datetime.datetime.now()
                break
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)

    time_taken_to_start_vllm_server = (
        vllm_server_readiness_time - time_before_pod_creation
    )

    time_taken_to_start_vllm_server_seconds = int(
        time_taken_to_start_vllm_server.total_seconds()
    )

    pod_details = {
        "pod_id": pod_id,
        "pod_url": pod_url,
        "time_taken_to_start_server": time_taken_to_start_vllm_server_seconds,
        "time_taken_to_upload_llm": time_taken_to_start_vllm_server_seconds,
        "llm_id": llm_id,
        "gpu_id": gpu_id,
        "volume_in_gb": volume_in_gb,
        "container_disk_in_gb": container_disk_in_gb,
        "port": port,
        "created_at": datetime.datetime.now(),
        "inference_name": "VLLM",
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
