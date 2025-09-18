import datetime
import os
import time

import requests
import runpod

from clients import GuideLLMBenchmarkClient, Mongo, RunpodClient
from utils.ollama_utils import (
    get_ollama_environment_vars,
    pull_model,
    verify_model_availability,
)
from utils.tokenizer_utils import HF_TOKENIZER


async def create_ollama_pod(
    gpu_id: str,
    volume_in_gb: int,
    container_disk_in_gb: int,
    llm_id: str,
    port: int = 11434,
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

    env_vars = get_ollama_environment_vars(gpu_count)

    ## Create a pod in runpod with ollama
    pod = runpod.create_pod(
        name="ollama-pod",
        image_name="ollama/ollama:latest",
        gpu_type_id=gpu_id,
        env=env_vars,
        gpu_count=gpu_count,
        ports=f"{port}/http",
        volume_in_gb=volume_in_gb,
        container_disk_in_gb=container_disk_in_gb,
    )

    pod_id = pod["id"]

    pod_url = f"https://{pod_id}-{port}.proxy.runpod.net"

    # Check if ollama server is ready Guidellm client also do that but it is calculating ollama server readiness time

    while True:
        try:
            response = requests.get(pod_url + "/api/tags")
            if response.status_code == 200:
                print(f"Ollama server is ready at {pod_url}")
                ollama_server_readiness_time = datetime.datetime.now()
                break
            else:
                time.sleep(1)
        except Exception as e:
            print(f"Error checking ollama server: {e}")
            time.sleep(1)

    time_taken_to_start_ollama_server = (
        ollama_server_readiness_time - time_before_pod_creation
    )

    time_taken_to_start_ollama_server_in_seconds = int(
        time_taken_to_start_ollama_server.total_seconds()
    )

    upload_llm_to_pod_start_time = datetime.datetime.now()

    ## Pull the model from ollama. Ollama server can upload the model via api request
    pull_model(pod_url, llm_id)

    model_available = verify_model_availability(pod_url, llm_id)

    if model_available:

        upload_llm_to_pod_end_time = datetime.datetime.now()

        time_taken_to_upload_llm = (
            upload_llm_to_pod_end_time - upload_llm_to_pod_start_time
        )

        time_taken_to_upload_llm_seconds = int(time_taken_to_upload_llm.total_seconds())

        pod_details = {
            "pod_id": pod_id,
            "pod_url": pod_url,
            "time_taken_to_start_server": time_taken_to_start_ollama_server_in_seconds,
            "time_taken_to_upload_llm": time_taken_to_upload_llm_seconds,
            "llm_id": llm_id,
            "gpu_id": gpu_id,
            "volume_in_gb": volume_in_gb,
            "container_disk_in_gb": container_disk_in_gb,
            "port": port,
            "created_at": datetime.datetime.now(),
            "inference_name": "Ollama",
            "server_type": "Runpod",
            "llm_parameter_size": llm_parameter_size,
            "llm_common_name": llm_common_name,
            "gpu_count": gpu_count,
        }

        mongo_client.insert_one("pod_benchmarks", pod_details)

    benchmark_start_time = datetime.datetime.now()

    client = GuideLLMBenchmarkClient(
        base_url=pod_url,
        model=llm_id,
        mongo_url=os.getenv("MONGODB_URL"),
        text_completions_path="/api/generate",
        processor=HF_TOKENIZER.get(llm_id, None),
        max_retries=5,
        base_delay=3.0,
        max_delay=120.0,
        timeout=600.0,
        health_check_timeout=90.0,
    )

    for benchmark_report_rate in range(6):
        try:

            # Run benchmark with 1-6 concurrent rate requests
            report, path = await client.run_benchmark(
                max_seconds=30,
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
