import os


def get_compatible_tensorrt_image(gpu_id: str = "", model_id: str = "") -> str:
    """
        gpu_id and model_id(not used currently but kept for consistency) 
        Returns official NVIDIA TensorRT-LLM image,1.0.0 is compatible with older drivers (550+)
    """
    return "nvcr.io/nvidia/tensorrt-llm/release:1.0.0"


def build_tensorrt_serve_args(llm_id: str, port: int, gpu_count: int) -> str:
    """
    Builds arguments for trtllm serve command by following the official NVIDIA pattern: trtllm-serve "model-id" --port X --host Y
    Args:
        llm_id:  model identifier
        port: Port number for the server
        gpu_count: no. of GPUs for tensor parallelism
        
    Returns:
        Command-line arguments string for trtllm-serve
    """

    serve_args_list = [
        llm_id, 
        f"--port {port}",
        "--host 0.0.0.0",
    ]

    # Add tensor parallelism if using multiple GPUs
    if gpu_count > 1:
        serve_args_list.append(f"--tp_size {gpu_count}")

    if "qwen3" in llm_id.lower():
        serve_args_list.append("--trust_remote_code")
    
    # serve_args_list.append("--max-input-len 4096")

    serve_args = " ".join([arg for arg in serve_args_list if arg.strip()])
    print(f"TensorRT-LLM serve args: trtllm-serve {serve_args}")
    return serve_args


def get_tensorrt_environment_vars(llm_id: str, gpu_count: int) -> dict[str, str]:
    
    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),
        "HF_HOME": "/root/.cache/huggingface",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
    }

    # Add CUDA device visibility if using multiple GPUs
    if gpu_count > 1:
        cuda_devices = ",".join(str(i) for i in range(gpu_count))
        env_vars["CUDA_VISIBLE_DEVICES"] = cuda_devices

    return env_vars