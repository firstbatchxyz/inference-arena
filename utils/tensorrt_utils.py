import os
from typing import Any


def get_compatible_tensorrt_image(gpu_id: str = "", model_id: str = "") -> str:
    model_id_lower = model_id.lower()
    if "gpt-oss" in model_id_lower or "gpt/oss" in model_id_lower:
        return "nvcr.io/nvidia/tensorrt-llm/release:gpt-oss-dev"
    else:
        return "nvcr.io/nvidia/tensorrt-llm/release:1.0.0"


def _add_config_if_provided(config_dict: dict, key: str, value: Any) -> None:
    """Helper to add config only if value is provided in CLI for extra parameters"""
    if value is not None:
        config_dict[key] = value


def build_tensorrt_serve_args(
    llm_id: str,
    port: int,
    gpu_count: int,
    moe_backend: str | None = None,
    kv_cache_free_gpu_memory_fraction: float | None = None,
    enable_attention_dp: bool | None = None,
    ep_size: int | None = None,
) -> tuple[str, str | None]:
    """
    Builds TensorRT-LLM serve arguments and optional YAML configuration, optional parameters are only included in the YAML config if explicitly provided via CLI.
    Different models support different optional parameters:
    - Qwen3-30B-A3B: Only supports ep_size (expert parallelism for MoE)
    - GPT-OSS-120B: Supports all optional parameters (moe_backend, kv_cache, attention_dp, ep_size)
    """
    try:
        import yaml
    except ImportError:
        yaml = None

    # Base arguments that all models use
    serve_args_list = [
        llm_id, 
        f"--port {port}",
        "--host 0.0.0.0",
        "--backend pytorch",
    ]

    if gpu_count > 1:
        serve_args_list.append(f"--tp_size {gpu_count}")

    config_dict = {}
    yaml_config_content = None
    llm_id_lower = llm_id.lower()
    
    if "qwen3" in llm_id_lower:
        serve_args_list.append("--trust_remote_code")
        
        if ("qwen3-30b-a3b" in llm_id_lower or "qwen/qwen3-30b-a3b" in llm_id_lower) and ep_size is not None:
            config_dict["moe_expert_parallel_size"] = ep_size
    
    elif "gpt-oss" in llm_id_lower or "gpt/oss" in llm_id_lower:
        serve_args_list.append("--trust_remote_code")
        
        # MoE backend selection (TRITON/CUTLASS/TRTLLM)
        if moe_backend:
            config_dict["moe_config"] = {"backend": moe_backend}
        
        # Memory and performance tuning options
        _add_config_if_provided(config_dict, "kv_cache_free_gpu_memory_fraction", kv_cache_free_gpu_memory_fraction)
        _add_config_if_provided(config_dict, "enable_attention_dp", enable_attention_dp)
        _add_config_if_provided(config_dict, "moe_expert_parallel_size", ep_size)

    # Generate YAML config file if any optional parameters were provided
    if config_dict:
        if yaml is None:
            raise ImportError("PyYAML is required when using moe_backend, kv_cache_free_gpu_memory_fraction, enable_attention_dp, or ep_size. Install with: pip install pyyaml")
        
        yaml_config_content = yaml.dump(config_dict, default_flow_style=False)
        config_path = "/tmp/trtllm_config.yaml"
        serve_args_list.append(f"--extra_llm_api_options {config_path}")
    
    serve_args = " ".join([arg for arg in serve_args_list if arg.strip()])
    print(f"TensorRT-LLM serve args: trtllm-serve {serve_args}")
    return serve_args, yaml_config_content


def get_tensorrt_environment_vars(llm_id: str, gpu_count: int) -> dict[str, str]:
    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),
        "HF_HOME": "/root/.cache/huggingface",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
    }

    # Configure CUDA device visibility for multi-GPU tensor parallelism
    if gpu_count > 1:
        cuda_devices = ",".join(str(i) for i in range(gpu_count))
        env_vars["CUDA_VISIBLE_DEVICES"] = cuda_devices

    return env_vars