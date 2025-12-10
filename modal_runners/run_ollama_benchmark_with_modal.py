import asyncio
import datetime
import os
import subprocess
import time

import aiohttp
import modal

from clients import GuideLLMBenchmarkClient, ModalClient, Mongo
from utils.ollama_utils import get_ollama_environment_vars, pull_model, verify_model_availability
from utils.tokenizer_utils import HF_TOKENIZER


# Get the parent directory (inference-arena)
import pathlib
_inference_arena_dir = pathlib.Path(__file__).parent.parent

# Ollama version to install
OLLAMA_VERSION = "0.6.5"

# Directory for Ollama models within the container and volume
MODEL_DIR = "/ollama_models"

# Modal image setup for Ollama
ollama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
    .uv_pip_install(
        "pymongo>=4.6.0",
        "guidellm>=0.2.1",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
    )
    .run_commands(
        "echo 'Installing Ollama...'",
        f"OLLAMA_VERSION={OLLAMA_VERSION} curl -fsSL https://ollama.com/install.sh | sh",
        "echo 'Ollama installed at $(which ollama)'",
        f"mkdir -p {MODEL_DIR}",
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
ollama_models_vol = modal.Volume.from_name("ollama-models-store", create_if_missing=True)

MINUTES = 60  # seconds
OLLAMA_PORT = 11434


async def create_ollama_modal_server(
    llm_id: str,
    port: int = 11434,
    llm_parameter_size: str = "",
    llm_common_name: str = "",
    gpu_id: str = "",
    gpu_count: int = 1,
):
    
    # Initialize mongo client
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    # Initialize Modal client for cost calculation
    modal_client = ModalClient()
    
    # Track start time - assume GPU billing starts when function begins execution
    start_time = datetime.datetime.now()
    
    # Format GPU for Modal: "GPU_TYPE:COUNT" (e.g., "A100:1", "H100:2")
    gpu_spec = f"{gpu_id}:{gpu_count}"
    
    # Get Ollama environment variables
    env_vars = get_ollama_environment_vars(gpu_count)
    # Set the host to use the specified port
    env_vars["OLLAMA_HOST"] = f"0.0.0.0:{port}"
    env_vars["OLLAMA_MODELS"] = MODEL_DIR
    
    # Create a unique app name to avoid conflicts
    app_name = f"ollama-benchmark-{int(time.time())}"
    deployment_app = modal.App(app_name)
    
    @deployment_app.function(
        image=ollama_image,
        gpu=gpu_spec,
        scaledown_window=15 * MINUTES,
        timeout=20 * MINUTES,
        volumes={MODEL_DIR: ollama_models_vol},
        serialized=True,
        max_containers=1,
    )
    @modal.concurrent(max_inputs=1)  
    @modal.web_server(port=port, startup_timeout=20 * MINUTES)
    def serve_ollama():
        import subprocess
        import sys
        
        # Add /app/inference-arena to Python path to ensure imports work
        inference_arena_path = "/app/inference-arena"
        if inference_arena_path not in sys.path:
            sys.path.insert(0, inference_arena_path)
        
        print("Starting Ollama setup...")
        print(f"Starting Ollama server on port {port}...")
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # Start Ollama server - Modal's web_server decorator keeps container alive
        cmd = ["ollama", "serve"]
        print(f"Starting Ollama with command: {' '.join(cmd)}")
        subprocess.Popen(cmd)
        
        print("Ollama server started. Model will be pulled via API after server is ready.")
    
    # Deploy the app using Modal's run context
    print(f"Deploying Modal app '{app_name}' with GPU {gpu_spec}...")
    
    with deployment_app.run():
        # Get the web URL for the deployed function
        web_url = serve_ollama.get_web_url()
        print(f"Modal server deployed at: {web_url}")
        
        # Wait for Ollama server to be ready using async HTTP
        max_wait_time = 10 * MINUTES
        start_wait = time.time()
        print(f"Waiting for Ollama server to be ready...")
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_wait < max_wait_time:
                try:
                    # Check if Ollama server is ready by verifying /api/tags endpoint
                    async with session.get(f"{web_url}/api/tags", timeout=aiohttp.ClientTimeout(total=30)) as tags_resp:
                        if tags_resp.status == 200:
                            print(f"Ollama server is ready at {web_url}")
                            ollama_server_readiness_time = datetime.datetime.now()
                            break
                except Exception as e:
                    elapsed = int(time.time() - start_wait)
                    print(f"Waiting for server to be ready... ({elapsed}s elapsed) - {e}")
                    await asyncio.sleep(5)  # Check every 5 seconds
            else:
                raise TimeoutError(f"Server did not become ready within {max_wait_time} seconds")
        
        time_taken_to_start_ollama_server = (
            ollama_server_readiness_time - start_time
        )
        time_taken_to_start_ollama_server_seconds = int(
            time_taken_to_start_ollama_server.total_seconds()
        )
        
        # Pull model if not already available (using API)
        upload_llm_start_time = datetime.datetime.now()
        
        # Use API to pull model if needed
        async with aiohttp.ClientSession() as session:
            # Check if model is already available
            async with session.get(f"{web_url}/api/tags", timeout=aiohttp.ClientTimeout(total=30)) as tags_resp:
                if tags_resp.status == 200:
                    tags_data = await tags_resp.json()
                    models = tags_data.get("models", [])
                    model_names = [model.get("name", "") for model in models]
                    
                    model_available = False
                    for model_name in model_names:
                        if llm_id in model_name or model_name in llm_id:
                            model_available = True
                            break
                    
                    if not model_available:
                        print(f"Pulling model {llm_id} via API...")
                        # Use the utility function for pulling
                        pull_success = pull_model(web_url, llm_id)
                        if not pull_success:
                            print(f"Warning: Model pull may have failed for {llm_id}")
        
        # Verify model availability
        model_available = verify_model_availability(web_url, llm_id)
        
        if not model_available:
            raise RuntimeError(f"Model {llm_id} is not available after pull")
        
        upload_llm_end_time = datetime.datetime.now()
        time_taken_to_upload_llm = (
            upload_llm_end_time - upload_llm_start_time
        )
        time_taken_to_upload_llm_seconds = int(
            time_taken_to_upload_llm.total_seconds()
        )
        
        # Generate unique server ID
        server_id = f"modal-ollama-{int(time.time())}"
        
        pod_details = {
            "pod_id": server_id,
            "pod_url": web_url,
            "time_taken_to_start_server": time_taken_to_start_ollama_server_seconds,
            "time_taken_to_upload_llm": time_taken_to_upload_llm_seconds,
            "llm_id": llm_id,
            "gpu_id": gpu_id,
            "volume_in_gb": 0,  # Modal manages storage via volumes
            "container_disk_in_gb": 0,  # Modal manages storage
            "port": port,
            "created_at": datetime.datetime.now(),
            "inference_name": "Ollama",
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
            text_completions_path="/api/generate",
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

