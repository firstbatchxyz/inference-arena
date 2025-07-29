import os
import datetime
import time
from mongo_client import Mongo
from guidellm_benchmark_client import GuideLLMBenchmarkClient
from tokenizer_utils import OLLAMA_TO_HF_TOKENIZER
import traceback

def run_benchmark_on_existing_pod(pod_id: str):
    """
    Run GuideLLM benchmark on an existing pod by retrieving pod details from MongoDB.
    
    Args:
        pod_id (str): The ID of the existing pod to benchmark
    """
    
    # Initialize MongoDB client
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    # Retrieve pod details from MongoDB
    pod_details = mongo_client.find_one("pod_benchmarks", {"pod_id": pod_id})
    
    if not pod_details:
        raise ValueError(f"Pod with ID {pod_id} not found in pod_benchmarks collection")
    
    print(f"Found pod details: {pod_details}")
    
    # Extract necessary information from pod details
    pod_url = pod_details["pod_url"]
    llm_id = pod_details["llm_id"]
    
    print(f"Running benchmark on pod: {pod_id}")
    print(f"Pod URL: {pod_url}")
    print(f"LLM ID: {llm_id}")
    
    # Check if the pod is still accessible
    try:
        import requests
        response = requests.get(f"{pod_url}/api/tags", timeout=10)
        if response.status_code != 200:
            raise Exception(f"Pod is not accessible. Status code: {response.status_code}")
        print("Pod is accessible and ready for benchmarking")
    except Exception as e:
        raise Exception(f"Failed to connect to pod: {e}")
    
    # Initialize benchmark client
    client = GuideLLMBenchmarkClient(
        base_url=pod_url,
        model=llm_id,
        mongo_url=os.getenv("MONGODB_URL"),
        text_completions_path="/api/generate",
        processor=OLLAMA_TO_HF_TOKENIZER.get(llm_id)
    )
    
    # Start benchmark timer
    benchmark_start_time = datetime.datetime.now()
    
    print("Starting concurrent rate benchmarks...")
    
    # Run concurrent rate benchmarks (same as original)
    for benchmark_report_rate in range(3):  # Reduced from 9 to 3 for large model
        print(f"Running concurrent benchmark with rate {benchmark_report_rate + 1}")
        
        report, path = client.run_sync(
            max_seconds=600,  # Increased to 10 minutes for large model
            mongo_query={},
            rate_type="concurrent",
            output_path="benchmark_results.json",
            rate=benchmark_report_rate + 1,
        )
        
        benchmark_report = report.benchmarks[0]
        
        # Log the benchmark report to the database
        mongo_client.insert_one("benchmark_results", {
            "pod_id": pod_id,
            "benchmark_type": benchmark_report.args.profile.type_,
            "rate": mongo_client.safe_get_metric(benchmark_report.args.profile, "streams[0]"),
            "max_number": benchmark_report.args.max_number,
            "warmup_number": benchmark_report.args.warmup_number,
            "benchmark_duration": benchmark_report.run_stats.end_time - benchmark_report.run_stats.start_time if benchmark_report.run_stats.end_time and benchmark_report.run_stats.start_time else None,
            "total_requests": benchmark_report.run_stats.requests_made.total,
            "successful_requests": benchmark_report.run_stats.requests_made.successful,
            "requests_per_second": benchmark_report.metrics.requests_per_second.total.mean,
            "request_concurrency": benchmark_report.metrics.request_concurrency.total.mean,
            "request_latency": benchmark_report.metrics.request_latency.total.mean,
            "prompt_token_count": benchmark_report.metrics.prompt_token_count.total.mean,
            "output_token_count": benchmark_report.metrics.output_token_count.total.mean,
            "time_to_first_token_ms": benchmark_report.metrics.time_to_first_token_ms.total.mean,
            "time_per_output_token_ms": benchmark_report.metrics.time_per_output_token_ms.total.mean,
            "inter_token_latency_ms": benchmark_report.metrics.inter_token_latency_ms.total.mean,
            "output_tokens_per_second": benchmark_report.metrics.output_tokens_per_second.total.mean,
            "tokens_per_second": benchmark_report.metrics.tokens_per_second.total.mean,
            "benchmark_run_timestamp": datetime.datetime.now(),
        })
        
        print(f"Completed concurrent benchmark with rate {benchmark_report_rate + 1}")
    
    print("Starting throughput benchmark...")
    
    # Run throughput benchmark
    report, path = client.run_sync(
        mongo_query={},
        rate_type="throughput",
        max_requests=50,  # Reduced from 200 for large model
        max_seconds=1200,  # Add timeout for throughput test
        output_path="benchmark_results.json",
    )
    
    benchmark_report = report.benchmarks[0]
    
    mongo_client.insert_one("benchmark_results", {
        "pod_id": pod_id,
        "benchmark_type": "throughput",
        "max_number": benchmark_report.args.max_number,
        "warmup_number": benchmark_report.args.warmup_number,
        "benchmark_duration": benchmark_report.run_stats.end_time - benchmark_report.run_stats.start_time if benchmark_report.run_stats.end_time and benchmark_report.run_stats.start_time else None,
        "total_requests": benchmark_report.run_stats.requests_made.total,
        "successful_requests": benchmark_report.run_stats.requests_made.successful,
        "requests_per_second": benchmark_report.metrics.requests_per_second.total.mean,
        "request_concurrency": benchmark_report.metrics.request_concurrency.total.mean,
        "request_latency": benchmark_report.metrics.request_latency.total.mean,
        "prompt_token_count": benchmark_report.metrics.prompt_token_count.total.mean,
        "output_token_count": benchmark_report.metrics.output_token_count.total.mean,
        "time_to_first_token_ms": benchmark_report.metrics.time_to_first_token_ms.total.mean,
        "time_per_output_token_ms": benchmark_report.metrics.time_per_output_token_ms.total.mean,
        "inter_token_latency_ms": benchmark_report.metrics.inter_token_latency_ms.total.mean,
        "output_tokens_per_second": benchmark_report.metrics.output_tokens_per_second.total.mean,
        "tokens_per_second": benchmark_report.metrics.tokens_per_second.total.mean,
        "benchmark_run_timestamp": datetime.datetime.now(),
    })
    
    print("Completed throughput benchmark")
    
    # Calculate the benchmark duration
    benchmark_duration = datetime.datetime.now() - benchmark_start_time
    
    print(f"Total benchmark duration: {benchmark_duration}")
    
    # Update the pod_benchmarks collection with the new benchmark run
    mongo_client.update_one(
        "pod_benchmarks",
        {"pod_id": pod_id},
        {
            "$set": {
                "last_benchmark_run": datetime.datetime.now(),
                "last_benchmark_duration": benchmark_duration.total_seconds(),
                "total_benchmark_runs": mongo_client.safe_get_metric(
                    mongo_client.find_one("pod_benchmarks", {"pod_id": pod_id}), 
                    "total_benchmark_runs", 
                    0
                ) + 1
            }
        }
    )
    
    print(f"Successfully completed benchmark on pod {pod_id}")
    print(f"Results saved to MongoDB benchmark_results collection")


def list_available_pods():
    """
    List all available pods from the pod_benchmarks collection.
    """
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    pods = list(mongo_client.find_many("pod_benchmarks", {}))
    
    if not pods:
        print("No pods found in pod_benchmarks collection")
        return
    
    print("Available pods:")
    print("-" * 80)
    for pod in pods:
        print(f"Pod ID: {pod['pod_id']}")
        print(f"Pod URL: {pod['pod_url']}")
        print(f"LLM ID: {pod['llm_id']}")
        print(f"GPU ID: {pod['gpu_id']}")
        print(f"Created at: {pod['created_at']}")
        if 'last_benchmark_run' in pod:
            print(f"Last benchmark run: {pod['last_benchmark_run']}")
        print("-" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python run_benchmark_on_existing_pod.py <pod_id>")
        print("Or use 'list' to see available pods:")
        print("python run_benchmark_on_existing_pod.py list")
        sys.exit(1)
    
    if sys.argv[1] == "list":
        list_available_pods()
    else:
        pod_id = sys.argv[1]
        try:
            run_benchmark_on_existing_pod(pod_id)
        except Exception as e:
            ### Also log line of error code
            print(f"Error running benchmark: {e}")
            print(f"Error line: {traceback.format_exc()}")
            sys.exit(1) 