import asyncio
import datetime
import os
import shlex
import subprocess
import time

import aiohttp
import modal

from clients import GuideLLMBenchmarkClient, ModalClient, Mongo
from utils.tokenizer_utils import HF_TOKENIZER
from utils.vllm_utils import build_vllm_serve_args


# Get the parent directory (inference-arena)
import pathlib
_inference_arena_dir = pathlib.Path(__file__).parent.parent

# Modal image setup for vLLM
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
        "pymongo>=4.6.0",
        "guidellm>=0.2.1",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir(
        str(_inference_arena_dir / "utils"),
        remote_path="/app/inference-arena/utils",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )
    .add_local_dir(
        str(_inference_arena_dir / "clients"),
        remote_path="/app/inference-arena/clients",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )
    .env({"PYTHONPATH": "/app/inference-arena"})
)


# Modal volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

MINUTES = 60  # seconds
VLLM_PORT = 8000


async def create_vllm_modal_server(
    llm_id: str,
    port: int = 8000,
    llm_parameter_size: str = "",
    llm_common_name: str = "",
    gpu_id: str = "",
    gpu_count: int = 1,
    quantization: str = "",
    fast_boot: bool = True,
):
    
    # Initialize mongo client
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    # Initialize Modal client for cost calculation
    modal_client = ModalClient()
    
    # Track start time - assume GPU billing starts when function begins execution
    # This is safer (slightly overestimates) and simpler than trying to pinpoint exact allocation time
    start_time = datetime.datetime.now()
    
    # Format GPU for Modal: "GPU_TYPE:COUNT" (e.g., "A100:1", "H100:2")
    gpu_spec = f"{gpu_id}:{gpu_count}"
    
    # Create a unique app name to avoid conflicts
    app_name = f"vllm-benchmark-{int(time.time())}"
    deployment_app = modal.App(app_name)
    
    @deployment_app.function(
        image=vllm_image,
        gpu=gpu_spec,  
        scaledown_window=15 * MINUTES,
        timeout=10 * MINUTES,
        volumes={
            "/root/.cache/huggingface": hf_cache_vol,
            "/root/.cache/vllm": vllm_cache_vol,
        },
        serialized=True,
        max_containers=1, 
    )

    @modal.concurrent(max_inputs=32)  
    @modal.web_server(port=port, startup_timeout=10 * MINUTES) 
    def serve_vllm():

        import subprocess
        import sys
        
        # Add /app/inference-arena to Python path to ensure imports work
        inference_arena_path = "/app/inference-arena"
        if inference_arena_path not in sys.path:
            sys.path.insert(0, inference_arena_path)
        
        
        from utils.vllm_utils import build_vllm_serve_args
        serve_args = build_vllm_serve_args(llm_id, port, gpu_count, quantization)
        
        # Build command: vllm serve + serve_args + fast_boot option
        # Using shlex.split() to properly parse the string arguments
        cmd = ["vllm", "serve"] + shlex.split(serve_args)
        
        # Add fast boot option (not handled by utility function)
        if fast_boot:
            cmd.append("--enforce-eager")
        else:
            cmd.append("--no-enforce-eager")
        
        print(f"Starting vLLM with command: {' '.join(cmd)}")
        
        # Start vLLM server - Modal's web_server decorator keeps container alive
        subprocess.Popen(cmd)
    
    # Deploy the app using Modal's run context
    print(f"Deploying Modal app '{app_name}' with GPU {gpu_spec}...")
    
    with deployment_app.run():
        # Get the web URL for the deployed function
        web_url = serve_vllm.get_web_url()
        print(f"Modal server deployed at: {web_url}")
        
        # Wait for vLLM server to be ready using async HTTP 
        max_wait_time = 10 * MINUTES
        start_wait = time.time()
        print(f"Waiting for vLLM server to be ready ...")
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_wait < max_wait_time:
                try:
                    # Check if vLLM server is ready by verifying model is loaded
                    async with session.get(f"{web_url}/v1/models", timeout=aiohttp.ClientTimeout(total=30)) as models_resp:
                        if models_resp.status == 200:
                            models_data = await models_resp.json()
                            if models_data.get("data") and any(
                                model.get("id") == llm_id for model in models_data.get("data", [])
                            ):
                                print(f"vLLM server is ready at {web_url} with model {llm_id}")
                                vllm_server_readiness_time = datetime.datetime.now()
                                break
                except Exception as e:
                    elapsed = int(time.time() - start_wait)
                    print(f"Waiting for server to be ready... ({elapsed}s elapsed) - {e}")
                    await asyncio.sleep(5)  # Check every 5 seconds
            else:
                raise TimeoutError(f"Server did not become ready within {max_wait_time} seconds")
        
        time_taken_to_start_vllm_server = (
            vllm_server_readiness_time - start_time
        )
        time_taken_to_start_vllm_server_seconds = int(
            time_taken_to_start_vllm_server.total_seconds()
        )
        
        # Generate unique server ID
        server_id = f"modal-{int(time.time())}"
        
        pod_details = {
            "pod_id": server_id,
            "pod_url": web_url,
            "time_taken_to_start_server": time_taken_to_start_vllm_server_seconds,
            "time_taken_to_upload_llm": time_taken_to_start_vllm_server_seconds,
            "llm_id": llm_id,
            "gpu_id": gpu_id,
            "volume_in_gb": 0,  # Modal manages storage via volumes
            "container_disk_in_gb": 0,  # Modal manages storage
            "port": port,
            "created_at": datetime.datetime.now(),
            "inference_name": "VLLM",
            "server_type": "Modal",
            "llm_parameter_size": llm_parameter_size,
            "llm_common_name": llm_common_name,
            "gpu_count": gpu_count,
        }
        
        mongo_client.insert_one("pod_benchmarks", pod_details)
        
        benchmark_start_time = datetime.datetime.now()
        
        # Initialize benchmark client
        client = GuideLLMBenchmarkClient(
            base_url=web_url,
            model=llm_id,
            mongo_url=os.getenv("MONGODB_URL"),
            processor=HF_TOKENIZER.get(llm_id, None),
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
            await asyncio.sleep(5)
        
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
        
        # Calculate total runtime from function start to benchmark completion
        # This is a conservative estimate (slightly overestimates) but simpler and safer
        total_runtime = (datetime.datetime.now() - start_time).total_seconds()
        
        # Calculate Modal cost based on GPU pricing and runtime
        pod_cost = modal_client.calculate_cost(gpu_id, gpu_count, total_runtime)
        
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
        
        print(f"Modal server {server_id} benchmark completed successfully")
        
       