import asyncio
import datetime
import os
import subprocess
import time
from pathlib import Path

import aiohttp
import modal

from clients import GuideLLMBenchmarkClient, ModalClient, Mongo
from utils.llamacpp_utils import (
    build_llamacpp_serve_args,
    get_llamacpp_environment_vars,
    parse_model_path,
)
from utils.tokenizer_utils import HF_TOKENIZER

import pathlib

_inference_arena_dir = pathlib.Path(__file__).parent.parent

MINUTES = 60

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

llamacpp_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands("git clone https://github.com/ggerganov/llama.cpp")
    .run_commands(
        "cd llama.cpp && "
        "cmake -B build "
        "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON"
    )
    .run_commands(
        "cd llama.cpp && "
        "cmake --build build --config Release -j --clean-first --target llama-quantize llama-cli llama-server"
    )
    .run_commands("cp llama.cpp/build/bin/llama-* /usr/local/bin/")
    .uv_pip_install(
        "huggingface-hub==0.36.0",
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

hf_cache_vol = modal.Volume.from_name(
    "llamacpp-huggingface-cache", create_if_missing=True
)
llamacpp_cache_vol = modal.Volume.from_name("llamacpp-cache", create_if_missing=True)


async def create_llamacpp_modal_server(
    llm_id: str,
    port: int = 8000,
    llm_parameter_size: str = "",
    llm_common_name: str = "",
    gpu_id: str = "",
    gpu_count: int = 1,
    quantization: str = "Q4_0",
    model_path: str = "",
):
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    modal_client = ModalClient()
    start_time = datetime.datetime.now()

    gpu_spec = f"{gpu_id}:{gpu_count}"
    app_name = f"llamacpp-benchmark-{int(time.time())}"
    deployment_app = modal.App(app_name)

    env_vars = get_llamacpp_environment_vars(gpu_count)
    serve_args_str, repo_id = build_llamacpp_serve_args(
        llm_id, port, gpu_count, quantization, model_path
    )
    _, model_filename, model_pattern = parse_model_path(
        model_path, llm_id, quantization
    )

    @deployment_app.function(
        image=llamacpp_image,
        gpu=gpu_spec,
        scaledown_window=15 * MINUTES,
        timeout=30 * MINUTES,
        volumes={
            "/root/.cache/huggingface": hf_cache_vol,
            "/root/.cache/llama.cpp": llamacpp_cache_vol,
        },
        env=env_vars,
        serialized=True,
        max_containers=1,
    )
    @modal.concurrent(max_inputs=32)
    @modal.web_server(port=port, startup_timeout=20 * MINUTES)
    def serve_llamacpp():
        import sys
        from huggingface_hub import snapshot_download

        inference_arena_path = "/app/inference-arena"
        if inference_arena_path not in sys.path:
            sys.path.insert(0, inference_arena_path)

        cache_dir = "/root/.cache/llama.cpp"
        print(f"Downloading model from {repo_id} with pattern: {model_pattern}")

        snapshot_download(
            repo_id=repo_id,
            local_dir=cache_dir,
            allow_patterns=model_pattern
            if isinstance(model_pattern, list)
            else [model_pattern],
        )

        hf_cache_vol.commit()

        model_file_path = f"{cache_dir}/{model_filename}"
        if not Path(model_file_path).exists():
            gguf_files = list(Path(cache_dir).rglob("*.gguf"))
            if gguf_files:
                model_file_path = str(gguf_files[0])
                print(f"Using found GGUF file: {model_file_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_file_path}")

        n_gpu_layers = -1 if gpu_count > 0 else 0
        cmd = [
            "llama-server",
            "--model",
            model_file_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--n_gpu_layers",
            str(n_gpu_layers),
        ]

        print(f"Starting LLamaCPP with command: {' '.join(cmd)}")
        subprocess.Popen(cmd)

    with deployment_app.run():
        web_url = serve_llamacpp.get_web_url()

        max_wait_time = 20 * MINUTES
        start_wait = time.time()

        async with aiohttp.ClientSession() as session:
            while time.time() - start_wait < max_wait_time:
                try:
                    async with session.get(
                        f"{web_url}/v1/models", timeout=aiohttp.ClientTimeout(total=30)
                    ) as models_resp:
                        if models_resp.status == 200:
                            models_data = await models_resp.json()
                            if models_data.get("data"):
                                print(f"LLamaCPP server is ready at {web_url}")
                                server_readiness_time = datetime.datetime.now()
                                break
                except Exception as e:
                    elapsed = int(time.time() - start_wait)
                    print(
                        f"Waiting for server to be ready... ({elapsed}s elapsed) - {e}"
                    )
                    await asyncio.sleep(5)
            else:
                raise TimeoutError(
                    f"Server did not become ready within {max_wait_time} seconds"
                )

        time_taken_to_start_server_seconds = int(
            (server_readiness_time - start_time).total_seconds()
        )
        server_id = f"modal-llamacpp-{int(time.time())}"

        pod_details = {
            "pod_id": server_id,
            "pod_url": web_url,
            "time_taken_to_start_server": time_taken_to_start_server_seconds,
            "time_taken_to_upload_llm": time_taken_to_start_server_seconds,
            "llm_id": llm_id,
            "gpu_id": gpu_id,
            "volume_in_gb": 0,
            "container_disk_in_gb": 0,
            "port": port,
            "created_at": datetime.datetime.now(),
            "inference_name": "LLAMACPP",
            "server_type": "Modal",
            "llm_parameter_size": llm_parameter_size,
            "llm_common_name": llm_common_name,
            "gpu_count": gpu_count,
        }

        mongo_client.insert_one("pod_benchmarks", pod_details)
        benchmark_start_time = datetime.datetime.now()

        # Query the server to get the actual model name
        actual_model_name = None
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{web_url}/v1/models", timeout=aiohttp.ClientTimeout(total=10)
                ) as models_resp:
                    if models_resp.status == 200:
                        models_data = await models_resp.json()
                        if models_data.get("data") and len(models_data["data"]) > 0:
                            actual_model_name = models_data["data"][0]["id"]
            except Exception:
                pass

        # Use the actual model name returned by the server, fallback to llm_id
        model_name_for_client = actual_model_name or llm_id

        client = GuideLLMBenchmarkClient(
            base_url=web_url,
            model=model_name_for_client,
            mongo_url=os.getenv("MONGODB_URL"),
            processor=HF_TOKENIZER.get(llm_id, None),
            text_completions_path="/v1/completions",
            max_retries=3,
            base_delay=2.0,
            max_delay=60.0,
            timeout=300.0,
            health_check_timeout=45.0,
        )

        # Run concurrent benchmarks (rates 1-6)
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
                        f"Failed to get benchmark results for rate {benchmark_report_rate + 1}"
                    )
                    break

            except Exception as e:
                print(f"Error in benchmark rate {benchmark_report_rate + 1}: {e}")
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

        benchmark_end_time = datetime.datetime.now()
        total_runtime_seconds = (
            benchmark_end_time - benchmark_start_time
        ).total_seconds()

        pod_cost = modal_client.calculate_cost(gpu_id, gpu_count, total_runtime_seconds)

        benchmark_duration = benchmark_end_time - benchmark_start_time

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
