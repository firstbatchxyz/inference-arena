import datetime
import json
import os
import time
from lightning_sdk import Studio
from clients import GuideLLMBenchmarkClient, Mongo
from utils.tokenizer_utils import HF_TOKENIZER
from utils.ngrok_utils import (
    install_ngrok,
    setup_ngrok_tunnel,
    verify_tunnel_connectivity,
    cleanup_ngrok,
)


async def create_lmstudio_lightning_ai_server(
    studio_name: str,
    teamspace: str,
    org: str,
    llm_id: str,
    port: int = 1234,
    llm_parameter_size: str = "",
    llm_common_name: str = "",
    gpu_count: int = 1,
    gpu_id: str = "",
):
    """
    Create and benchmark LM Studio server on Lightning AI Studio (headless setup using AppImage + lms CLI)
    """

    # Initialize mongo client
    mongo_client = Mongo(os.getenv("MONGODB_URL"))

    # Initialize Lightning AI Studio
    studio = Studio(name=studio_name, teamspace=teamspace, org=org)

    time_before_server_creation = datetime.datetime.now()

    # Install dependencies and LM Studio AppImage, then bootstrap lms CLI
    install_commands = [
        "sudo apt-get update",
        "sudo apt-get install -y wget curl xvfb fuse libnspr4 libnss3 libatk-bridge2.0-0 libgtk-3-0 libasound2 || true",
        "wget https://installers.lmstudio.ai/linux/x64/0.3.25-2/LM-Studio-0.3.25-2-x64.AppImage -O $HOME/LMStudio",
        "chmod +x $HOME/LMStudio",
        "xvfb-run --auto-servernum $HOME/LMStudio --no-sandbox >/tmp/lmstudio_first_run.log 2>&1 || true",
        "mkdir -p $HOME/.lmstudio/.internal",
        'bash -lc \'cat > $HOME/.lmstudio/.internal/http-server-config.json <<EOF\n{\n"path": "\'$HOME\'/LMStudio",\n"argv": ["\'$HOME\'/LMStudio"],\n"cwd": "\'$HOME\'"\n}\nEOF\'',
    ]

    studio.run(" && ".join(install_commands))

    # Install ngrok for public tunnel
    install_ngrok(studio)
    print("ngrok installed successfully")

    # Try to start the server once; fallback attempted later if health check fails
    studio.run("./.lmstudio/bin/lms server start")

    print("Waiting for LM Studio server to be ready...")
    while True:
        try:
            # Check models endpoint
            health_check = studio.run(
                f"curl -s http://localhost:{port}/v1/models -m 10 || echo 'not_ready'"
            )
            print("Health check:", health_check)
            if "not_ready" not in health_check and health_check.strip() not in (
                "",
                "null",
            ):
                try:
                    _ = json.loads(health_check)
                    lmstudio_server_readiness_time = datetime.datetime.now()
                    print(f"LM Studio server is ready at http://localhost:{port}")
                    break
                except Exception:
                    pass
            time.sleep(1)
        except Exception:
            pass

    upload_llm_start_time = datetime.datetime.now()

    studio.run(f"echo y | ./.lmstudio/bin/lms get {llm_id}")
    # Wait until model appears in /v1/models
    max_model_wait = 900
    model_wait = 0
    while model_wait < max_model_wait:
        try:
            models_response = studio.run(
                f"curl -s http://localhost:{port}/v1/models -m 20 || true"
            )
            if models_response:
                try:
                    models_json = json.loads(models_response)
                    data = models_json.get("data") or []
                    if any((item.get("id") == llm_id) for item in data):
                        break
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(10)
        model_wait += 10

    upload_llm_end_time = datetime.datetime.now()

    time_taken_to_start_server = (
        lmstudio_server_readiness_time - time_before_server_creation
    )
    time_taken_to_start_server_seconds = int(time_taken_to_start_server.total_seconds())

    time_taken_to_upload_llm = upload_llm_end_time - upload_llm_start_time
    time_taken_to_upload_llm_seconds = int(time_taken_to_upload_llm.total_seconds())

    # Set up ngrok tunnel
    tunnel_url = setup_ngrok_tunnel(studio, port)

    # Verify tunnel is working (use OpenAI models endpoint)
    verify_tunnel_connectivity(studio, tunnel_url, "/v1/models")

    # Persist pod details
    server_id = f"lightning-ai-{int(time.time())}"
    pod_details = {
        "pod_id": server_id,
        "pod_url": tunnel_url,
        "time_taken_to_start_server": time_taken_to_start_server_seconds,
        "time_taken_to_upload_llm": time_taken_to_upload_llm_seconds,
        "llm_id": llm_id,
        "gpu_id": "lightning-ai-gpu",
        "volume_in_gb": 0,
        "container_disk_in_gb": 0,
        "port": port,
        "created_at": datetime.datetime.now(),
        "inference_name": "LMStudio",
        "server_type": "Lightning AI",
        "llm_parameter_size": llm_parameter_size,
        "llm_common_name": llm_common_name,
        "gpu_count": gpu_count,
    }

    mongo_client.insert_one("pod_benchmarks", pod_details)

    # Run benchmarks
    benchmark_start_time = datetime.datetime.now()

    client = GuideLLMBenchmarkClient(
        base_url=tunnel_url,
        model=llm_id,
        mongo_url=os.getenv("MONGODB_URL"),
        processor=HF_TOKENIZER.get(llm_id, None),
        text_completions_path="/v1/completions",
        max_retries=5,
        base_delay=3.0,
        max_delay=120.0,
        timeout=600.0,
        health_check_timeout=90.0,
    )

    # Run concurrent benchmark with 1-6 concurrent rate requests
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
                benchmark_report, server_id, None, "throughput"
            )
            mongo_client.insert_one("benchmark_results", result_data)
        else:
            print("Failed to get throughput benchmark results")

    except Exception as e:
        print(f"Error in throughput benchmark: {e}")

    benchmark_duration = datetime.datetime.now() - benchmark_start_time

    pod_cost = 0.0  # Lightning AI cost is not exposed via API

    # Update pod details
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

    # Cleanup tunnel processes
    cleanup_ngrok(studio)

    print(f"Lightning AI LM Studio server {server_id} benchmark completed successfully")
