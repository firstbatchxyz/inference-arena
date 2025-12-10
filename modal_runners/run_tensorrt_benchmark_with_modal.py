import asyncio
import base64
import datetime
import os
import shlex
import subprocess
import time

import aiohttp
import modal

from clients import GuideLLMBenchmarkClient, ModalClient, Mongo
from utils.tokenizer_utils import HF_TOKENIZER
from utils.tensorrt_utils import build_tensorrt_serve_args, get_tensorrt_environment_vars


# Get the parent directory (inference-arena)
import pathlib
_inference_arena_dir = pathlib.Path(__file__).parent.parent


# Note: Local Python must also be 3.12 when using serialized=True
tensorrt_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install(
        "openmpi-bin",
        "libopenmpi-dev",
        "git",
        "git-lfs",
        "wget",
    )
    .pip_install(
        "tensorrt-llm==1.0.0",  
        "flashinfer-python==0.2.5", 
        "cuda-python==12.9.1",
        "onnx==1.19.1",  
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    .uv_pip_install(
        "huggingface_hub==0.36.0",  
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .pip_install(
        "transformers==4.53.1",  
        "pymongo>=4.6.0",
        "guidellm>=0.2.1",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "pyyaml>=6.0",  
    )
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

MINUTES = 60  # seconds
TENSORRT_PORT = 8000


async def create_tensorrt_modal_server(
    llm_id: str,
    port: int = 8000,
    llm_parameter_size: str = "",
    llm_common_name: str = "",
    gpu_id: str = "",
    gpu_count: int = 1,
    moe_backend: str | None = None,
    kv_cache_free_gpu_memory_fraction: float | None = None,
    enable_attention_dp: bool | None = None,
    ep_size: int | None = None,
    pp_size: int | None = None,
):
    
    # Initialize mongo client
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    # Initialize Modal client for cost calculation
    modal_client = ModalClient()
    
    start_time = datetime.datetime.now()
    
    # Format GPU for Modal: "GPU_TYPE:COUNT" (e.g., "A100:1", "H100:2")
    gpu_spec = f"{gpu_id}:{gpu_count}"
    
    # Get TensorRT-LLM environment variables
    env_vars = get_tensorrt_environment_vars(llm_id, gpu_count)
    
    # Build serve arguments and optional YAML config
    serve_args, yaml_config_content = build_tensorrt_serve_args(
        llm_id,
        port,
        gpu_count,
        moe_backend=moe_backend,
        kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
        enable_attention_dp=enable_attention_dp,
        ep_size=ep_size,
        pp_size=pp_size,
    )
    
    # Create a unique app name to avoid conflicts
    app_name = f"tensorrt-benchmark-{int(time.time())}"
    deployment_app = modal.App(app_name)
    
    @deployment_app.function(
        image=tensorrt_image,
        gpu=gpu_spec,
        scaledown_window=15 * MINUTES,
        timeout=20 * MINUTES,  
        volumes={
            "/root/.cache/huggingface": hf_cache_vol,
        },
        serialized=True,  
        max_containers=1,
        env=env_vars,  
    )

    @modal.concurrent(max_inputs=32)
    @modal.web_server(port=port, startup_timeout=20 * MINUTES)
    def serve_tensorrt():
        import subprocess
        import sys
        
        # Add /app/inference-arena to Python path to ensure imports work
        inference_arena_path = "/app/inference-arena"
        if inference_arena_path not in sys.path:
            sys.path.insert(0, inference_arena_path)
        
        from utils.tensorrt_utils import build_tensorrt_serve_args
        
        # Rebuild serve args inside the function (needed for proper parsing)
        # The function parameters are captured via closure
        serve_args_str, yaml_config_content = build_tensorrt_serve_args(
            llm_id,
            port,
            gpu_count,
            moe_backend=moe_backend,
            kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
            enable_attention_dp=enable_attention_dp,
            ep_size=ep_size,
            pp_size=pp_size,
        )
        
        # If YAML config is needed, write it to file first
        # The serve_args_str already includes --extra_llm_api_options /tmp/trtllm_config.yaml
        if yaml_config_content:
            config_path = "/tmp/trtllm_config.yaml"
            with open(config_path, "w") as f:
                f.write(yaml_config_content)
            print(f"Created YAML config file at {config_path}")
            print(f"YAML config content:\n{yaml_config_content}")
        
        # Parse the serve args string (this will split --extra_llm_api_options /tmp/trtllm_config.yaml into separate args)
        parsed_args = shlex.split(serve_args_str)
        
        # Build the final command: trtllm-serve + parsed args
        cmd = ["trtllm-serve"] + parsed_args
        
        print(f"Starting TensorRT-LLM with command: {' '.join(cmd)}")
        
        # Start TensorRT-LLM server - Modal's web_server decorator keeps container alive
        subprocess.Popen(cmd)
    
    # Deploy the app using Modal's run context
    print(f"Deploying Modal app '{app_name}' with GPU {gpu_spec}...")
    
    with deployment_app.run():
        # Get the web URL for the deployed function
        web_url = serve_tensorrt.get_web_url()
        print(f"Modal server deployed at: {web_url}")
        
        # Wait for TensorRT-LLM server to be ready using async HTTP
        max_wait_time = 20 * MINUTES  
        start_wait = time.time()
        print(f"Waiting for TensorRT-LLM server to be ready...")
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_wait < max_wait_time:
                try:
                    # Try /health endpoint first (faster check)
                    async with session.get(f"{web_url}/health", timeout=aiohttp.ClientTimeout(total=30)) as health_resp:
                        if health_resp.status == 200:
                            # Then verify model is loaded via /v1/models
                            async with session.get(f"{web_url}/v1/models", timeout=aiohttp.ClientTimeout(total=30)) as models_resp:
                                if models_resp.status == 200:
                                    models_data = await models_resp.json()
                                    if models_data.get("data") and any(
                                        model.get("id") == llm_id or llm_id in model.get("id", "")
                                        for model in models_data.get("data", [])
                                    ):
                                        print(f"TensorRT-LLM server is ready at {web_url} with model {llm_id}")
                                        tensorrt_server_readiness_time = datetime.datetime.now()
                                        break
                except Exception as e:
                    elapsed = int(time.time() - start_wait)
                    print(f"Waiting for server to be ready... ({elapsed}s elapsed) - {e}")
                    await asyncio.sleep(5)  # Check every 5 seconds
            else:
                raise TimeoutError(f"Server did not become ready within {max_wait_time} seconds")
        
        time_taken_to_start_tensorrt_server = (
            tensorrt_server_readiness_time - start_time
        )
        time_taken_to_start_tensorrt_server_seconds = int(
            time_taken_to_start_tensorrt_server.total_seconds()
        )
        
        # Generate unique server ID
        server_id = f"modal-tensorrt-{int(time.time())}"
        
        pod_details = {
            "pod_id": server_id,
            "pod_url": web_url,
            "time_taken_to_start_server": time_taken_to_start_tensorrt_server_seconds,
            "time_taken_to_upload_llm": time_taken_to_start_tensorrt_server_seconds,
            "llm_id": llm_id,
            "gpu_id": gpu_id,
            "volume_in_gb": 0,  # Modal manages storage via volumes
            "container_disk_in_gb": 0,  # Modal manages storage
            "port": port,
            "created_at": datetime.datetime.now(),
            "inference_name": "TensorRT-LLM",
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

