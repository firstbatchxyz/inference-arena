import os


def get_compatible_sglang_image(gpu_id: str = "", model_id: str = "") -> str:
    """
    Determine the appropriate SGLang Docker image based on model id.
    """
    if "gpt-oss" in model_id.lower():
        return "fcan1batch/sglang-gpt-oss"
    elif "grok-2" in model_id.lower():
        return "fatihbugrakdogan/sglang:grok-2"
    else:
        return "lmsysorg/sglang:latest"


def get_sglang_environment_vars(llm_id: str, gpu_count: int) -> dict:
    """
    Get SGLang environment variables based on model and GPU configuration.
    """

    if gpu_count > 1:
        cuda_devices = ",".join(str(i) for i in range(gpu_count))
    else:
        cuda_devices = "0"

    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/transformers",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": cuda_devices,
        "CUDA_LAUNCH_BLOCKING": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_DISABLE_EXPERIMENTAL_WARNING": "1",
        "SGLANG_USE_MODELSCOPE": "false",
    }

    return env_vars


def build_sglang_docker_args(llm_id: str, port: int, gpu_count: int) -> str:
    """
    Build SGLang docker arguments based on the optimal configuration.
    """

    args = [
        "python",
        "-m",
        "sglang.launch_server",
        f"--model-path {llm_id}",
        f"--port {port}",
        "--host 0.0.0.0",
        "--tp-size 1",
        "--dp-size 1",
        "--mem-fraction-static 0.85",
        "--context-length 4096",
        "--tokenizer-mode auto",
    ]
    if "grok-2" in llm_id.lower():
        args.append(f"--tp {gpu_count}")
        args.append("--quantization fp8")
        args.append("--attention-backend triton")
        args.append("--tokenizer-path Xenova/grok-1-tokenizer")

    docker_args = " ".join([arg for arg in args if arg.strip()])
    print(docker_args)
    return docker_args


def build_sglang_serve_args(llm_id: str, port: int, gpu_count: int) -> str:
    """
    Build SGLang serve arguments for Lightning AI (command line format).
    """

    args = [
        llm_id,  # model-path is the first positional argument
        f"--port {port}",
        "--host 0.0.0.0",
        f"--tp-size {gpu_count}",
        f"--dp-size {gpu_count}",
        "--mem-fraction-static 0.85",
        "--context-length 4096",
        "--tokenizer-mode auto",
    ]

    if "grok-2" in llm_id.lower():
        args.append("--quantization fp8")
        args.append("--attention-backend triton")
        args.append("--tokenizer-path Xenova/grok-1-tokenizer")

    serve_args = " ".join([arg for arg in args if arg.strip()])
    print(f"SGLang serve args: {serve_args}")
    return serve_args
