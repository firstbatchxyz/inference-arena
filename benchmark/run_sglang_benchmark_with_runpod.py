import runpod
import os
import datetime
from typing import Optional
from runpod_client import RunpodClient
import requests
import time
from ollama_utils import pull_model,verify_model_availability
from mongo_client import Mongo
from enhanced_guidellm_client import EnhancedGuideLLMBenchmarkClient
from tokenizer_utils import OLLAMA_TO_HF_TOKENIZER
from sglang_utils import get_optimal_sglang_config, build_sglang_docker_args, get_compatible_sglang_image

def create_sglang_pod(gpu_id: str, volume_in_gb: int, container_disk_in_gb: int, llm_id:str, port:int=30000,llm_parameter_size:Optional[str]=None,llm_common_name:Optional[str]=None,gpu_count:int=1):

    ## Initialize mongo client
    mongo_client=Mongo(os.getenv("MONGODB_URL"))
    
    ## Apply api key to runpod
    runpod.api_key = os.getenv("RUNPOD_API_KEY")

    ## Also apply api key to runpod graphql client
    runpod_graphql_client= RunpodClient(api_key=os.getenv("RUNPOD_API_KEY"))

    ## Get optimal SGLang configuration based on model characteristics
    sglang_config = get_optimal_sglang_config(
        llm_id=llm_id,
        llm_parameter_size=llm_parameter_size or "",
        gpu_id=gpu_id,
        gpu_count=gpu_count
    )
    
    # Build docker args dynamically based on optimal configuration
    docker_args = build_sglang_docker_args(llm_id, port, sglang_config)
    
    ## Create a pod in runpod

    time_before_pod_creation=datetime.datetime.now()

    ## Create pod with SGLang image
    ## Note: Requires HF_TOKEN environment variable for private models and tokenizer access
    
    # Choose Docker image based on CUDA compatibility
    image_name = get_compatible_sglang_image(gpu_id)
    print(f"Using SGLang image: {image_name} for GPU: {gpu_id}")

    # Enhanced environment variables for better HuggingFace integration
    # Set CUDA_VISIBLE_DEVICES based on GPU count for multi-GPU support
    if gpu_count > 1:
        cuda_devices = ",".join(str(i) for i in range(gpu_count))
        print(f"Multi-GPU setup: Setting CUDA_VISIBLE_DEVICES={cuda_devices} for {gpu_count} GPUs")
    else:
        cuda_devices = "0"
        print(f"Single GPU setup: Setting CUDA_VISIBLE_DEVICES={cuda_devices}")
    
    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),  # Primary HF token (new standard)
        "HUGGING_FACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),  # Fallback compatibility
        "HF_HOME": "/root/.cache/huggingface",  # Centralized HF cache
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/transformers",  # Transformers cache
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",  # Datasets cache
        "TOKENIZERS_PARALLELISM": "false",  # Disable tokenizer parallelism to avoid warnings
        "CUDA_VISIBLE_DEVICES": cuda_devices,  # Expose appropriate GPUs based on gpu_count
        "CUDA_LAUNCH_BLOCKING": "1",  # Enable better CUDA error reporting
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",  # Cleaner logs
        "HF_HUB_DISABLE_EXPERIMENTAL_WARNING": "1",  # Disable experimental warnings
        "SGLANG_USE_MODELSCOPE": "false",  # Use HuggingFace by default
    }
    
    # Add multi-GPU specific environment variables
    if gpu_count > 1:
        env_vars.update({
            "NCCL_DEBUG": "INFO",  # Enable NCCL debugging for multi-GPU issues
            "NCCL_IB_DISABLE": "1",  # Disable InfiniBand for RunPod compatibility
            "NCCL_P2P_DISABLE": "1",  # Disable P2P for RunPod stability
            "PYTHONUNBUFFERED": "1",  # Unbuffered output for better logging
            "TORCH_NCCL_BLOCKING_WAIT": "1",  # Better error reporting for NCCL
        })
        print(f"Added multi-GPU environment variables for {gpu_count} GPUs")
    
    # Try to create pod with error handling for Docker image issues
    pod = None
    image_fallbacks = [
        image_name,  # Primary choice
        "lmsysorg/sglang:latest",  # Fallback to official latest
        "lmsysorg/sglang:dev",  # Fallback to official dev version
    ]
    
    for attempt, fallback_image in enumerate(image_fallbacks):
        try:
            print(f"Attempt {attempt + 1}: Trying Docker image: {fallback_image}")
            pod = runpod.create_pod(
                name="sglang-pod",
                image_name=fallback_image,
                gpu_type_id=gpu_id,
                env=env_vars,
                docker_args=docker_args,
                gpu_count=gpu_count,
                volume_in_gb=volume_in_gb,
                container_disk_in_gb=container_disk_in_gb,
                volume_mount_path="/root/.cache/huggingface",
                ports=f"{port}/http"
            )
            print(f"Successfully created pod with image: {fallback_image}")
            break
        except Exception as e:
            print(f"Failed to create pod with image {fallback_image}: {str(e)}")
            if "manifest" in str(e).lower() and "not found" in str(e).lower():
                print(f"Docker image {fallback_image} not found, trying next fallback...")
                continue
            else:
                # If it's not a manifest/image issue, re-raise the exception
                raise e
    
    if pod is None:
        raise RuntimeError(f"Failed to create pod with any of the fallback images: {image_fallbacks}")

    ## Get the pod id
    pod_id=pod["id"]

    ## Get pod url

    pod_url=f"https://{pod_id}-{port}.proxy.runpod.net"

    # Check if SGLang server is ready
    while True:
        try:
            response=requests.get(pod_url+"/v1/models")
            response=response.json()
            if response.get("data")[0].get("id")==llm_id:
                print(f"SGLang server is ready at {pod_url} with model {llm_id}")
                sglang_server_readiness_time=datetime.datetime.now()
                break
            else:
                time.sleep(1)
        except Exception as e:
            time.sleep(1)

    ## Calculate the time taken to start the SGLang server

    time_taken_to_start_sglang_server=sglang_server_readiness_time-time_before_pod_creation

    print(f"Time taken to start the pod: {time_taken_to_start_sglang_server}")

    ## Convert to time taken to seconds
    time_taken_to_start_sglang_server_seconds=int(time_taken_to_start_sglang_server.total_seconds())

    print(f"Time taken to start the SGLang server: {time_taken_to_start_sglang_server_seconds}")

    ## For SGLang, the model is loaded automatically when the container starts
    

    ## Insert the pod details into the database
    pod_details={
        "pod_id":pod_id,
        "pod_url":pod_url,
        "time_taken_to_start_server":time_taken_to_start_sglang_server_seconds,
        "time_taken_to_upload_llm":time_taken_to_start_sglang_server_seconds,
        "llm_id":llm_id,
        "gpu_id":gpu_id,
        "volume_in_gb":volume_in_gb,
        "container_disk_in_gb":container_disk_in_gb,
        "port":port,
        "created_at":datetime.datetime.now(),
        "inference_name":"SGLang",
        "server_type":"Runpod",
        "llm_parameter_size":llm_parameter_size,
        "llm_common_name":llm_common_name,
        "sglang_config": sglang_config,
        "gpu_count":gpu_count
    }

    mongo_client.insert_one("pod_benchmarks",pod_details)


    ## Benchmark timer

    benchmark_start_time=datetime.datetime.now()

    client = EnhancedGuideLLMBenchmarkClient(
        base_url=pod_url,
        model=llm_id,
        mongo_url=os.getenv("MONGODB_URL"),
        text_completions_path="/v1/completions",  # SGLang uses OpenAI-compatible API
        processor=OLLAMA_TO_HF_TOKENIZER.get(llm_id, None),  # Use None if tokenizer not found
        max_retries=3,
        base_delay=2.0,
        max_delay=60.0,
        timeout=300.0,
        health_check_timeout=45.0
    )

    # Wait for server to be fully ready before benchmarking
    if not client.wait_for_server_ready(max_wait_time=600.0, check_interval=10.0):
        raise RuntimeError("SGLang server failed to become ready within timeout")

    # Run benchmark suite with enhanced error handling
    concurrent_rates = list(range(1, 10))  # Test rates 1-9
    
    suite_results = client.run_benchmark_suite(
        pod_id=pod_id,
        concurrent_rates=concurrent_rates,
        include_throughput=True,
        max_seconds_per_test=30
    )
    
    # Log suite summary to database
    mongo_client.insert_one("benchmark_suite_results", {
        "pod_id": pod_id,
        "suite_results": suite_results,
        "framework": "SGLang",
        "timestamp": time.time()
    })

    # Legacy compatibility: run individual benchmarks if suite fails
    if suite_results['successful_tests'] == 0:
        print("⚠️ Benchmark suite failed, falling back to individual benchmarks...")
        
        for benchmark_report_rate in range(9):
            try:
                report, path = client.run_sync_with_retry(
                    max_seconds=30,
                    mongo_query={},  
                    rate_type="concurrent",
                    output_path="benchmark_results.json",
                    rate=benchmark_report_rate+1
                )
                
                if report and report.benchmarks:
                    benchmark_report = report.benchmarks[0]
                    result_data = client._extract_benchmark_metrics(benchmark_report, pod_id, benchmark_report_rate+1)
                    mongo_client.insert_one("benchmark_results", result_data)
                else:
                    print(f"❌ Failed to get benchmark results for rate {benchmark_report_rate+1}")
                    break
                    
            except Exception as e:
                print(f"❌ Error in benchmark rate {benchmark_report_rate+1}: {e}")
                break

        # Throughput test
        try:
            report, path = client.run_sync_with_retry(
                mongo_query={},  
                rate_type="throughput",
                output_path="benchmark_results.json"
            )
            
            if report and report.benchmarks:
                benchmark_report = report.benchmarks[0]
                result_data = client._extract_benchmark_metrics(benchmark_report, pod_id, None, "throughput")
                mongo_client.insert_one("benchmark_results", result_data)
                
        except Exception as e:
            print(f"❌ Error in throughput benchmark: {e}")
    


    ## Log prompts to new collection IT IS GIVES ERROR DUE TO PROMPT LENGTH
    # for request in benchmark_report.requests:
    #     mongo_client.insert_one("benchmark_results_by_prompts",
    #                             {
    #                                 "pod_id":pod_id,
    #                                 "requests":request
    #                             })
   
    ## Calculate the benchmark duration

    benchmark_duration=datetime.datetime.now()-benchmark_start_time


    ## Calculate pod cost

    pod_cost=runpod_graphql_client.calculate_used_balance(pod_id)

    ## Log the pod cost to related pod log

    mongo_client.update_one("pod_benchmarks",
                            {"pod_id":pod_id},
                            {"$set":{"pod_cost":pod_cost,"benchmark_duration":benchmark_duration.total_seconds()}})

    ### Stop the pod
    runpod.terminate_pod(pod_id)
    print(f"✅ Pod {pod_id} terminated successfully")


def main():
    """Main function to handle command line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SGLang benchmark with Runpod')
    parser.add_argument('--gpu_id', type=str, required=True, help='GPU type ID (e.g., NVIDIA H200)')
    parser.add_argument('--volume_in_gb', type=int, required=True, help='Volume size in GB')
    parser.add_argument('--container_disk_in_gb', type=int, required=True, help='Container disk size in GB')
    parser.add_argument('--llm_id', type=str, required=True, help='LLM model ID')
    parser.add_argument('--port', type=int, default=30000, help='Port number (default: 30000)')
    parser.add_argument('--llm_parameter_size', type=str, default='', help='LLM parameter size')
    parser.add_argument('--llm_common_name', type=str, default='', help='LLM common name')
    parser.add_argument('--gpu_count', type=int, default=1, help='Number of GPUs (default: 1)')
    
    args = parser.parse_args()
    
    try:
        create_sglang_pod(
            gpu_id=args.gpu_id,
            volume_in_gb=args.volume_in_gb,
            container_disk_in_gb=args.container_disk_in_gb,
            llm_id=args.llm_id,
            port=args.port,
            llm_parameter_size=args.llm_parameter_size,
            llm_common_name=args.llm_common_name,
            gpu_count=args.gpu_count
        )
        print("🎉 SGLang benchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()