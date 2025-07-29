#!/usr/bin/env python3
"""
Verilen pod_id için veritabanından eksik benchmark'ları bulup çalıştıran script.
Missing benchmark finder and runner for given pod_id.
"""

import os
import sys
import datetime
from typing import Set, List, Dict, Any
import traceback
import requests
import time

from mongo_client import Mongo
from guidellm_benchmark_client import GuideLLMBenchmarkClient
from tokenizer_utils import OLLAMA_TO_HF_TOKENIZER


def get_completed_benchmarks(mongo_client: Mongo, pod_id: str) -> Set[str]:
    """
    Belirtilen pod_id için tamamlanmış benchmark'ları döndür.
    Returns completed benchmarks for given pod_id.
    """
    # Find all benchmark results for this pod
    completed_results = list(mongo_client.find_many("benchmark_results", {"pod_id": pod_id}))
    
    completed_benchmarks = set()
    for result in completed_results:
        benchmark_type = result.get("benchmark_type", "")
        
        if benchmark_type == "concurrent":
            rate = result.get("rate", 0)
            completed_benchmarks.add(f"concurrent_{rate}")
        elif benchmark_type == "throughput":
            completed_benchmarks.add("throughput")
    
    return completed_benchmarks


def get_expected_benchmarks() -> Set[str]:
    """
    Beklenen tüm benchmark'ları döndür.
    Returns all expected benchmarks.
    """
    expected = set()
    
    # Concurrent benchmarks with rates 1-9
    for rate in range(1, 10):
        expected.add(f"concurrent_{rate}")
    
    # Throughput benchmark
    expected.add("throughput")
    
    return expected


def find_missing_benchmarks(mongo_client: Mongo, pod_id: str) -> List[str]:
    """
    Eksik benchmark'ları bulup döndür.
    Find and return missing benchmarks.
    """
    completed = get_completed_benchmarks(mongo_client, pod_id)
    expected = get_expected_benchmarks()
    
    missing = expected - completed
    return sorted(list(missing))


def check_pod_accessibility(pod_url: str, inference_name: str) -> bool:
    """
    Pod'un erişilebilir olup olmadığını kontrol et.
    Check if pod is accessible.
    """
    try:
        if inference_name.lower() == "ollama":
            # Ollama için /api/tags endpoint'ini kontrol et
            response = requests.get(f"{pod_url}/api/tags", timeout=10)
        elif inference_name.lower() in ["vllm", "sglang"]:
            # vLLM ve SGLang için /v1/models endpoint'ini kontrol et
            response = requests.get(f"{pod_url}/v1/models", timeout=10)
        else:
            print(f"Warning: Unknown inference engine: {inference_name}")
            response = requests.get(pod_url, timeout=10)
        
        return response.status_code == 200
    except Exception as e:
        print(f"Pod accessibility check failed: {e}")
        return False


def run_concurrent_benchmark(client: GuideLLMBenchmarkClient, mongo_client: Mongo, pod_id: str, rate: int):
    """
    Belirtilen rate ile concurrent benchmark çalıştır.
    Run concurrent benchmark with specified rate.
    """
    print(f"Running concurrent benchmark with rate {rate}")
    
    try:
        report, path = client.run_sync(
            max_seconds=600,  # 10 minutes timeout
            mongo_query={},
            rate_type="concurrent",
            output_path="benchmark_results.json",
            rate=rate,
        )
        
        benchmark_report = report.benchmarks[0]
        
        # Log the benchmark report to the database
        benchmark_data = {
            "pod_id": pod_id,
            "benchmark_type": benchmark_report.args.profile.type_,
            "rate": mongo_client.safe_get_metric(benchmark_report.args.profile, "streams[0]") if hasattr(benchmark_report.args.profile, "streams") else rate,
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
        }
        
        mongo_client.insert_one("benchmark_results", benchmark_data)
        print(f"✅ Completed concurrent benchmark with rate {rate}")
        
    except Exception as e:
        print(f"❌ Failed concurrent benchmark with rate {rate}: {e}")
        traceback.print_exc()


def run_throughput_benchmark(client: GuideLLMBenchmarkClient, mongo_client: Mongo, pod_id: str):
    """
    Throughput benchmark çalıştır.
    Run throughput benchmark.
    """
    print("Running throughput benchmark")
    
    try:
        report, path = client.run_sync(
            mongo_query={},
            rate_type="throughput",
            output_path="benchmark_results.json",
        )
        
        benchmark_report = report.benchmarks[0]
        
        benchmark_data = {
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
        }
        
        mongo_client.insert_one("benchmark_results", benchmark_data)
        print("✅ Completed throughput benchmark")
        
    except Exception as e:
        print(f"❌ Failed throughput benchmark: {e}")
        traceback.print_exc()


def run_missing_benchmarks(pod_id: str):
    """
    Verilen pod_id için eksik benchmark'ları bulup çalıştır.
    Find and run missing benchmarks for given pod_id.
    """
    print(f"🔍 Searching for missing benchmarks for pod: {pod_id}")
    
    # MongoDB connection
    mongo_url = os.getenv("MONGODB_URL")
    if not mongo_url:
        raise ValueError("MONGODB_URL environment variable is required")
    
    mongo_client = Mongo(mongo_url)
    
    try:
        # Get pod details from database
        pod_details = mongo_client.find_one("pod_benchmarks", {"pod_id": pod_id})
        
        if not pod_details:
            raise ValueError(f"Pod with ID {pod_id} not found in pod_benchmarks collection")
        
        print(f"📋 Pod details found:")
        print(f"  Pod URL: {pod_details['pod_url']}")
        print(f"  LLM ID: {pod_details['llm_id']}")
        print(f"  GPU ID: {pod_details['gpu_id']}")
        print(f"  Inference Engine: {pod_details['inference_name']}")
        
        # Check pod accessibility
        print(f"🔗 Checking pod accessibility...")
        if not check_pod_accessibility(pod_details['pod_url'], pod_details['inference_name']):
            raise Exception("Pod is not accessible. Please check if the pod is running.")
        
        print("✅ Pod is accessible")
        
        # Find missing benchmarks
        missing_benchmarks = find_missing_benchmarks(mongo_client, pod_id)
        
        if not missing_benchmarks:
            print("🎉 No missing benchmarks found! All benchmarks are completed.")
            return
        
        print(f"📊 Found {len(missing_benchmarks)} missing benchmarks:")
        for benchmark in missing_benchmarks:
            print(f"  - {benchmark}")
        
        # Setup benchmark client based on inference engine
        inference_name = pod_details['inference_name'].lower()
        llm_id = pod_details['llm_id']
        
        if inference_name == "ollama":
            text_completions_path = "/api/generate"
            processor = OLLAMA_TO_HF_TOKENIZER.get(llm_id)
        elif inference_name in ["vllm", "sglang"]:
            text_completions_path = "/v1/completions"
            processor = OLLAMA_TO_HF_TOKENIZER.get(llm_id, None)
        else:
            print(f"⚠️  Warning: Unknown inference engine {inference_name}, using default settings")
            text_completions_path = "/v1/completions"
            processor = None
        
        # Initialize benchmark client
        client = GuideLLMBenchmarkClient(
            base_url=pod_details['pod_url'],
            model=llm_id,
            mongo_url=mongo_url,
            text_completions_path=text_completions_path,
            processor=processor
        )
        
        # Start running missing benchmarks
        start_time = datetime.datetime.now()
        print(f"🚀 Starting missing benchmarks at {start_time}")
        
        # Run concurrent benchmarks
        concurrent_missing = [b for b in missing_benchmarks if b.startswith("concurrent_")]
        for benchmark in concurrent_missing:
            rate = int(benchmark.split("_")[1])
            run_concurrent_benchmark(client, mongo_client, pod_id, rate)
            time.sleep(2)  # Small delay between benchmarks
        
        # Run throughput benchmark
        if "throughput" in missing_benchmarks:
            run_throughput_benchmark(client, mongo_client, pod_id)
        
        # Calculate total duration
        end_time = datetime.datetime.now()
        total_duration = end_time - start_time
        
        print(f"🎯 All missing benchmarks completed!")
        print(f"⏱️  Total duration: {total_duration}")
        
        # Update pod_benchmarks collection
        mongo_client.update_one(
            "pod_benchmarks",
            {"pod_id": pod_id},
            {
                "$set": {
                    "last_missing_benchmark_run": datetime.datetime.now(),
                    "last_missing_benchmark_duration": total_duration.total_seconds(),
                }
            }
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        mongo_client.close()


def list_pod_benchmark_status():
    """
    Tüm pod'ların benchmark durumunu listele.
    List benchmark status for all pods.
    """
    mongo_url = os.getenv("MONGODB_URL")
    if not mongo_url:
        raise ValueError("MONGODB_URL environment variable is required")
    
    mongo_client = Mongo(mongo_url)
    
    try:
        pods = list(mongo_client.find_many("pod_benchmarks", {}))
        
        if not pods:
            print("No pods found in database")
            return
        
        print("📊 Pod Benchmark Status:")
        print("=" * 100)
        
        for pod in pods:
            pod_id = pod['pod_id']
            missing = find_missing_benchmarks(mongo_client, pod_id)
            completed = get_completed_benchmarks(mongo_client, pod_id)
            
            print(f"Pod ID: {pod_id}")
            print(f"LLM: {pod.get('llm_id', 'N/A')}")
            print(f"Inference: {pod.get('inference_name', 'N/A')}")
            print(f"Completed: {len(completed)}/10 benchmarks")
            
            if missing:
                print(f"Missing: {', '.join(missing)}")
            else:
                print("Status: ✅ All benchmarks completed")
            
            print("-" * 100)
            
    finally:
        mongo_client.close()


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} <pod_id>                 # Run missing benchmarks for specific pod")
        print(f"  {sys.argv[0]} --list                   # List all pods and their benchmark status")
        print(f"  {sys.argv[0]} --help                   # Show this help")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "--help":
        print("Eksik Benchmark Çalıştırıcı / Missing Benchmark Runner")
        print("=" * 60)
        print("Bu script verilen pod_id için eksik benchmark'ları bulup çalıştırır.")
        print("This script finds and runs missing benchmarks for given pod_id.")
        print()
        print("Kullanım / Usage:")
        print(f"  {sys.argv[0]} <pod_id>     # Specific pod için eksik benchmark'ları çalıştır")
        print(f"  {sys.argv[0]} --list       # Tüm pod'ların benchmark durumunu listele")
        print()
        print("Örnek / Example:")
        print(f"  {sys.argv[0]} abc123def456")
        print()
        print("Gerekli Environment Variables:")
        print("  MONGODB_URL - MongoDB connection string")
        
    elif command == "--list":
        list_pod_benchmark_status()
        
    else:
        pod_id = command
        run_missing_benchmarks(pod_id)


if __name__ == "__main__":
    main()
