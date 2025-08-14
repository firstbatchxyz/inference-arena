import os


def get_compatible_vllm_image(gpu_id: str = "", model_id: str = "") -> str:
    """
    Determine the appropriate vLLM Docker image based on CUDA compatibility and model requirements.
    RunPod typically uses CUDA 12.6, so we need to use compatible vLLM versions.
    """
    if "glm" in model_id.lower():
        return "vllm/vllm-openai:gptoss"
    else:
        return "vllm/vllm-openai:latest"

def build_vllm_docker_args(llm_id: str, port: int, vllm_config: dict) -> str:
    """
    Build vLLM docker arguments based on the optimal configuration.
    """
    # Build docker args dynamically based on optimal configuration
    docker_args_list = [
        f"--model {llm_id}",
        f"--port {port}",
        "--host 0.0.0.0",
    ]

    # Add additional arguments
    docker_args_list.extend(vllm_config["additional_args"])

    # Filter out empty strings and join all arguments into a single string
    docker_args = " ".join([arg for arg in docker_args_list if arg.strip()])

    return docker_args


def get_vllm_environment_vars(
    llm_id: str, llm_parameter_size: str, gpu_id: str, gpu_count: int
) -> dict:
    """
    Get vLLM environment variables based on model and GPU configuration.
    """
    # Enhanced environment variables for better HuggingFace integration
    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),  # Primary HF token (new standard)
        "HUGGING_FACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),  # Fallback compatibility
        "HF_HOME": "/root/.cache/huggingface",  # Centralized HF cache
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/transformers",  # Transformers cache
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",  # Datasets cache
    }


    return env_vars


def get_optimal_vllm_config(
    llm_id: str, llm_parameter_size: str = "", gpu_id: str = ""
) -> dict:
    """
    """
    config = {
        "gpu_memory_utilization": 0.85,
        "dtype": "auto",
        "additional_args": [],
    }

    return config
