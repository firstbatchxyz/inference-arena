import datetime
import os
import time

import requests
import runpod

from clients.guidellm_benchmark_client import GuideLLMBenchmarkClient
from clients.mongo_client import Mongo
from clients.runpod_client import RunpodClient
from utils.ollama_utils import OLLAMA_TO_HF_TOKENIZER
from utils.vllm_utils import (
    build_vllm_docker_args,
    get_compatible_vllm_image,
    get_optimal_vllm_config,
    get_vllm_environment_vars,
)


def create_vllm_pod(
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

    ## Also apply api key to runpod graphql client
    runpod_graphql_client = RunpodClient(api_key=os.getenv("RUNPOD_API_KEY"))

    ## Create a pod in runpod

    time_before_pod_creation = datetime.datetime.now()

    ## Get optimal vLLM configuration for this model
    vllm_config = get_optimal_vllm_config(llm_id, llm_parameter_size, gpu_id)

    # Build docker args using utility function
    docker_args = build_vllm_docker_args(llm_id, port, vllm_config)

    # Choose Docker image based on CUDA compatibility and model requirements
    image_name = get_compatible_vllm_image(gpu_id, llm_id)

    # Get environment variables using utility function
    env_vars = get_vllm_environment_vars(llm_id, llm_parameter_size, gpu_id, gpu_count)

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

    ## Get the pod id
    pod_id = pod["id"]

    ## Get pod url

    pod_url = f"https://{pod_id}-{port}.proxy.runpod.net"

    # Check if vLLM server is ready
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

    ## Calculate the time taken to start the vLLM server

    time_taken_to_start_vllm_server = (
        vllm_server_readiness_time - time_before_pod_creation
    )

    print(f"Time taken to start the pod: {time_taken_to_start_vllm_server}")

    ## Convert to time taken to seconds
    time_taken_to_start_vllm_server_seconds = int(
        time_taken_to_start_vllm_server.total_seconds()
    )

    print(
        f"Time taken to start the vLLM server: {time_taken_to_start_vllm_server_seconds}"
    )

    ## For vLLM, the model is loaded automatically when the container starts

    ## Insert the pod details into the database
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

    ## Benchmark timer

    benchmark_start_time = datetime.datetime.now()

    # Enhanced client with retry mechanisms and robust error handling
    client = GuideLLMBenchmarkClient(
        base_url=pod_url,
        model=llm_id,
        mongo_url=os.getenv("MONGODB_URL"),
        text_completions_path="/v1/completions",  # vLLM uses OpenAI-compatible API
        processor=OLLAMA_TO_HF_TOKENIZER.get(
            llm_id, None
        ),  # Use None if tokenizer not found
        max_retries=3,
        base_delay=2.0,
        max_delay=60.0,
        timeout=300.0,
        health_check_timeout=45.0,
    )

    # Wait for server to be fully ready before benchmarking
    if not client.wait_for_server_ready(max_wait_time=600.0, check_interval=10.0):
        raise RuntimeError("vLLM server failed to become ready within timeout")

    # Run benchmark suite with enhanced error handling
    concurrent_rates = list(range(1, 10))  # Test rates 1-9

    suite_results = client.run_benchmark_suite(
        pod_id=pod_id,
        concurrent_rates=concurrent_rates,
        include_throughput=True,
        max_seconds_per_test=30,
    )

    # Log suite summary to database
    mongo_client.insert_one(
        "benchmark_suite_results",
        {
            "pod_id": pod_id,
            "suite_results": suite_results,
            "framework": "vLLM",
            "timestamp": time.time(),
        },
    )

    # Legacy compatibility: run individual benchmarks if suite fails
    if suite_results["successful_tests"] == 0:
        print("Benchmark suite failed, falling back to individual benchmarks...")

        for benchmark_report_rate in range(6):
            try:
                report, path = client.run_sync_with_retry(
                    max_seconds=15,
                    mongo_query={},
                    rate_type="concurrent",
                    output_path="benchmark_results.json",
                    rate=benchmark_report_rate + 1,
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

            # Add delay between concurrent benchmarks to prevent Rich LiveError
            time.sleep(2)

        # Add delay before throughput benchmark to prevent Rich LiveError
        time.sleep(2)

        # Throughput test
        try:
            report, path = client.run_sync_with_retry(
                max_seconds=60,  # Added timeout for throughput test
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

    ## Calculate the benchmark duration

    benchmark_duration = datetime.datetime.now() - benchmark_start_time

    ## Calculate pod cost

    pod_cost = runpod_graphql_client.calculate_used_balance(pod_id)

    ## Log the pod cost to related pod log

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

    ### Stop the pod
    runpod.terminate_pod(pod_id)
    print(f"Pod {pod_id} terminated successfully")


def main():
    """Main function to handle command line execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Run vLLM benchmark with Runpod")
    parser.add_argument(
        "--gpu_id", type=str, required=True, help="GPU type ID (e.g., NVIDIA H200)"
    )
    parser.add_argument(
        "--volume_in_gb", type=int, required=True, help="Volume size in GB"
    )
    parser.add_argument(
        "--container_disk_in_gb",
        type=int,
        required=True,
        help="Container disk size in GB",
    )
    parser.add_argument("--llm_id", type=str, required=True, help="LLM model ID")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port number (default: 8000)"
    )
    parser.add_argument(
        "--llm_parameter_size", type=str, default="", help="LLM parameter size"
    )
    parser.add_argument(
        "--llm_common_name", type=str, default="", help="LLM common name"
    )
    parser.add_argument(
        "--gpu_count", type=int, default=1, help="Number of GPUs (default: 1)"
    )

    args = parser.parse_args()

    try:
        create_vllm_pod(
            gpu_id=args.gpu_id,
            volume_in_gb=args.volume_in_gb,
            container_disk_in_gb=args.container_disk_in_gb,
            llm_id=args.llm_id,
            port=args.port,
            llm_parameter_size=args.llm_parameter_size,
            llm_common_name=args.llm_common_name,
            gpu_count=args.gpu_count,
        )
        print("vLLM benchmark completed successfully!")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        import sys

        sys.exit(1)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
