import runpod
import os
import datetime
from runpod_client import RunpodClient
import requests
import time
from ollama_utils import pull_model,verify_model_availability
from mongo_client import Mongo
from enhanced_guidellm_client import EnhancedGuideLLMBenchmarkClient
from tokenizer_utils import OLLAMA_TO_HF_TOKENIZER


def create_ollama_pod(gpu_id: str, volume_in_gb: int, container_disk_in_gb: int, llm_id:str, port:int=11434,llm_parameter_size:str|None=None,llm_common_name:str|None=None,gpu_count:int=1):

    ## Initialize mongo client
    mongo_client=Mongo(os.getenv("MONGODB_URL"))
    
    ## Apply api key to runpod
    runpod.api_key = os.getenv("RUNPOD_API_KEY")

    ## Also apply api key to runpod graphql client
    runpod_graphql_client= RunpodClient(api_key=os.getenv("RUNPOD_API_KEY"))

    ## Create a pod in runpod

    time_before_pod_creation=datetime.datetime.now()


    pod=runpod.create_pod(
        name="ollama-pod",
        image_name="ollama/ollama:latest",
        gpu_type_id=gpu_id,
        env={
            "OLLAMA_HOST": "0.0.0.0",
            "OLLAMA_FLASH_ATTENTION": "true", # Enable flash attention if available
            "OLLAMA_GPU_OVERHEAD": "1GiB",   # Reserve some GPU memory
            "OLLAMA_LOAD_TIMEOUT": "10m0s"   # Increase load timeout for large models
        },
        gpu_count=gpu_count,
        ports=f"{port}/http",
        volume_in_gb=volume_in_gb,  # 100GB disk space for large models
        container_disk_in_gb=container_disk_in_gb  # Additional container disk space
    )

    ## Get the pod id
    pod_id=pod["id"]

    ## Get pod url

    pod_url=f"https://{pod_id}-{port}.proxy.runpod.net"

    ## Check if ollama server is ready
    while True:
        try:
            response=requests.get(pod_url+"/api/tags")
            if response.status_code==200:
                print(f"Ollama server is ready at {pod_url}")
                ollama_server_readiness_time=datetime.datetime.now()
                break
            else:
                time.sleep(1)
        except Exception as e:
            print(f"Error checking ollama server: {e}")
            time.sleep(1)

    ## Calculate the time taken to start the ollama server

    time_taken_to_start_ollama_server=ollama_server_readiness_time-time_before_pod_creation

    print(f"Time taken to start the pod: {time_taken_to_start_ollama_server}")

    ## Convert to time taken to seconds
    time_taken_to_start_ollama_server_seconds=int(time_taken_to_start_ollama_server.total_seconds())

    print(f"Time taken to start the ollama server: {time_taken_to_start_ollama_server_seconds}")

    ## Upload LLM to pod

    upload_llm_to_pod_start_time=datetime.datetime.now()

    ## Pull the model from ollama
    pull_model(pod_url,llm_id)

    ## Verify the model is available
    model_available=verify_model_availability(pod_url,llm_id)

    if model_available:

        upload_llm_to_pod_end_time=datetime.datetime.now()

        ## Calculate the time taken to upload the model
        time_taken_to_upload_llm=upload_llm_to_pod_end_time-upload_llm_to_pod_start_time

        print(f"Time taken to upload the model: {time_taken_to_upload_llm}")

        ## Calculate the time taken to upload the model
        time_taken_to_upload_llm_seconds=int(time_taken_to_upload_llm.total_seconds())

        print(f"Time taken to upload the model: {time_taken_to_upload_llm_seconds}")

        ## Insert the pod details into the database
        pod_details={
            "pod_id":pod_id,
            "pod_url":pod_url,
            "time_taken_to_start_server":time_taken_to_start_ollama_server_seconds,
            "time_taken_to_upload_llm":time_taken_to_upload_llm_seconds,
            "llm_id":llm_id,
            "gpu_id":gpu_id,
            "volume_in_gb":volume_in_gb,
            "container_disk_in_gb":container_disk_in_gb,
            "port":port,
            "created_at":datetime.datetime.now(),
            "inference_name":"Ollama",
            "server_type":"Runpod",
            "llm_parameter_size":llm_parameter_size,
            "llm_common_name":llm_common_name,
            "gpu_count":gpu_count
        }

        mongo_client.insert_one("pod_benchmarks",pod_details)


    ## Benchmark timer

    benchmark_start_time=datetime.datetime.now()

    client = EnhancedGuideLLMBenchmarkClient(
        base_url=pod_url,
        model=llm_id,
        mongo_url=os.getenv("MONGODB_URL"),
        text_completions_path="/api/generate",  # Ollama uses different endpoint
        processor=OLLAMA_TO_HF_TOKENIZER.get(llm_id, None),
        max_retries=3,
        base_delay=2.0,
        max_delay=60.0,
        timeout=300.0,
        health_check_timeout=45.0
    )

    # Wait for server to be fully ready before benchmarking
    if not client.wait_for_server_ready(max_wait_time=600.0, check_interval=10.0):
        raise RuntimeError("Ollama server failed to become ready within timeout")

    # Run benchmark suite with enhanced error handling - reduced rates for Ollama
    concurrent_rates = list(range(1, 7))  # Test rates 1-6 (Ollama is typically slower)
    
    suite_results = client.run_benchmark_suite(
        pod_id=pod_id,
        concurrent_rates=concurrent_rates,
        include_throughput=True,
        max_seconds_per_test=15  # Shorter tests for Ollama
    )
    
    # Log suite summary to database
    mongo_client.insert_one("benchmark_suite_results", {
        "pod_id": pod_id,
        "suite_results": suite_results,
        "framework": "Ollama",
        "timestamp": time.time()
    })

    # Legacy compatibility: run individual benchmarks if suite fails
    if suite_results['successful_tests'] == 0:
        print("⚠️ Benchmark suite failed, falling back to individual benchmarks...")
        
        for benchmark_report_rate in range(6):
            try:
                report, path = client.run_sync_with_retry(
                    max_seconds=15,
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
            
            # Add delay between concurrent benchmarks to prevent Rich LiveError
            time.sleep(2)

        # Add delay before throughput benchmark to prevent Rich LiveError
        time.sleep(2)

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
    
    parser = argparse.ArgumentParser(description='Run Ollama benchmark with Runpod')
    parser.add_argument('--gpu_id', type=str, required=True, help='GPU type ID (e.g., NVIDIA H200)')
    parser.add_argument('--volume_in_gb', type=int, required=True, help='Volume size in GB')
    parser.add_argument('--container_disk_in_gb', type=int, required=True, help='Container disk size in GB')
    parser.add_argument('--llm_id', type=str, required=True, help='LLM model ID')
    parser.add_argument('--port', type=int, default=11434, help='Port number (default: 11434)')
    parser.add_argument('--llm_parameter_size', type=str, default='', help='LLM parameter size')
    parser.add_argument('--llm_common_name', type=str, default='', help='LLM common name')
    parser.add_argument('--gpu_count', type=int, default=1, help='Number of GPUs (default: 1)')
    
    args = parser.parse_args()
    
    try:
        create_ollama_pod(
            gpu_id=args.gpu_id,
            volume_in_gb=args.volume_in_gb,
            container_disk_in_gb=args.container_disk_in_gb,
            llm_id=args.llm_id,
            port=args.port,
            llm_parameter_size=args.llm_parameter_size,
            llm_common_name=args.llm_common_name,
            gpu_count=args.gpu_count
        )
        print("🎉 Ollama benchmark completed successfully!")
        
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




   