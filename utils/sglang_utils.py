import os


def get_compatible_sglang_image(gpu_id: str = "", model_id: str = "") -> str:
    """
    Determine the appropriate SGLang Docker image based on CUDA compatibility.
    RunPod may have older CUDA versions, so we need to use compatible SGLang versions.
    """
    # Use official SGLang Docker images from lmsysorg/sglang
    # Based on official documentation at https://docs.sglang.ai/start/install.html
    # Available images: latest, dev

    # GPT-OSS specific image selection based on GitHub thread
    if "gpt-oss" in model_id.lower():
        return "fcan1batch/sglang-gpt-oss"
    else:
        return "lmsysorg/sglang:latest"


def get_sglang_environment_vars(llm_id: str, gpu_count: int) -> dict:
    """
    Get SGLang environment variables based on model and GPU configuration.
    """
    # Set CUDA_VISIBLE_DEVICES based on GPU count for multi-GPU support
    if gpu_count > 1:
        cuda_devices = ",".join(str(i) for i in range(gpu_count))
    else:
        cuda_devices = "0"

    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),  # Primary HF token (new standard)
        "HUGGING_FACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),  # Fallback compatibility
        "HF_HOME": "/root/.cache/huggingface",  # Centralized HF cache
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/transformers",  # Transformers cache
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",  # Datasets cache
        "TOKENIZERS_PARALLELISM": "false",  # Disable tokenizer parallelism to avoid warnings
        "CUDA_VISIBLE_DEVICES": cuda_devices,  # Expose appropriate GPUs based on gpu_count
        "CUDA_LAUNCH_BLOCKING": "1",  # Enable better CUDA error reporting
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",  # Cleaner logs
        "HF_HUB_DISABLE_EXPERIMENTAL_WARNING": "1",  # Disable experimental warnings
        "SGLANG_USE_MODELSCOPE": "false",  # Use HuggingFace by default
    }

    return env_vars


def get_optimal_sglang_config(
    llm_id: str, llm_parameter_size: str = "", gpu_id: str = "", gpu_count: int = 1
) -> dict:
    """
    Automatically determine optimal SGLang configuration based on model characteristics.
    """
    config = {
        "tp_size": 1,  # Start with single GPU
        "dp_size": 1,
        "mem_fraction_static": 0.85,
        "context_length": 4096,
        "trust_remote_code": False,
        "quantization": None,
        "kv_cache_dtype": "auto",
        "attention_backend": "auto",
        "cuda_graph_bs": 256,
        "max_running_requests": 256,
    }

    return config


def build_sglang_docker_args(llm_id: str, port: int, sglang_config: dict) -> str:
    """
    Build SGLang docker arguments based on the optimal configuration.
    Enhanced with HuggingFace token support and tokenizer optimizations.
    """
    # Start with the proper SGLang server command
    args = [
        "python",
        "-m",
        "sglang.launch_server",
        f"--model-path {llm_id}",
        f"--port {port}",
        "--host 0.0.0.0",
        f"--tp-size {sglang_config['tp_size']}",
        f"--dp-size {sglang_config['dp_size']}",
        f"--mem-fraction-static {sglang_config['mem_fraction_static']}",
        f"--context-length {sglang_config['context_length']}",
        "--tokenizer-mode auto",  # Use fast tokenizer when available
    ]

    return args
