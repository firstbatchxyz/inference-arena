import os


def get_compatible_vllm_image(gpu_id: str = "", model_id: str = "") -> str:
    """
    Determine the appropriate vLLM Docker image based on CUDA compatibility and model requirements.
    RunPod typically uses CUDA 12.6, so we need to use compatible vLLM versions.
    """
    if "gpt-oss" in model_id.lower():
        return "vllm/vllm-openai:gptoss"
    else:
        return "vllm/vllm-openai:latest"


def build_vllm_docker_args(llm_id: str, port: int, gpu_count: int) -> str:
    """
    Build vLLM docker arguments based on the optimal configuration.
    """
    # Build docker args dynamically based on optimal configuration
    docker_args_list = [
        f"--model {llm_id}",
        f"--port {port}",
        "--host 0.0.0.0",
        f"--tensor-parallel-size={gpu_count}",
    ]

    if "glm" in llm_id.lower():
        docker_args_list.extend(
            [
                "--tool-call-parser=glm45",
                "--reasoning-parser=glm45",
                "--gpu-memory-utilization=0.55",
                "--max-model-len=4096",
                "--enforce-eager",
            ]
        )

    elif "kimi-k2" in llm_id.lower():
        docker_args_list.extend(
            [
                "--served-model-name=kimi-k2",
                "--trust-remote-code",
                "--enable-auto-tool-choice",
                "--tool-call-parser=kimi_k2",
            ]
        )

    # Filter out empty strings and join all arguments into a single string
    docker_args = " ".join([arg for arg in docker_args_list if arg.strip()])

    return docker_args


def get_vllm_environment_vars(
    llm_id: str, llm_parameter_size: str, gpu_id: str, gpu_count: int
) -> dict:
    """
    Get vLLM environment variables based on model and GPU configuration.
    """
    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/transformers",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
    }

    if "gpt-oss" in llm_id.lower():
        env_vars["EXTRA_INDEX_URL"] = "https://wheels.vllm.ai/gpt-oss/"
        env_vars["PYTORCH_INDEX_URL"] = "https://download.pytorch.org/whl/nightly/cu128"
        env_vars["INDEX_STRATEGY"] = "unsafe-best-match"

    elif "glm" in llm_id.lower():
        env_vars["EXTRA_INDEX_URL"] = "https://wheels.vllm.ai/glm/"
        env_vars["PYTORCH_INDEX_URL"] = "https://download.pytorch.org/whl/nightly/cu128"
        env_vars["INDEX_STRATEGY"] = "unsafe-best-match"

    return env_vars
