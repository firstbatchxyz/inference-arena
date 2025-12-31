import os

def get_compatible_llamacpp_image(gpu_id: str = "", model_id: str = "") -> str:
    # using official llamacpp image
    return "ghcr.io/ggml-org/llama.cpp:server-cuda"


def get_gpu_layer_args(gpu_count: int) -> list[str]:
    return ["--n_gpu_layers", "-1" if gpu_count > 0 else "0"]


def get_host_port_args(port: int) -> list[str]:
    return ["--host", "0.0.0.0", "--port", str(port)]


def get_model_pattern_str(model_pattern) -> str:
    if isinstance(model_pattern, list):
        return model_pattern[0] if model_pattern else "*.gguf"
    return model_pattern


def build_llamacpp_docker_args(
    repo_id: str, model_filename: str, model_pattern_str: str, port: int
) -> str:
    args_list = [
        f"--hf_model_repo_id {repo_id}",
        f"--model {model_pattern_str}",
        "--host 0.0.0.0",
        f"--port {port}",
        "--n_gpu_layers -1",
    ]

    serve_args = " ".join([arg for arg in args_list if arg.strip()])

    docker_cmd = (
        "apt-get update && apt-get install -y build-essential cmake git && "
        "CMAKE_ARGS='-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=\"70;75;80;86\"' pip install llama-cpp-python[cuda] && "
        f"python3 -m llama_cpp.server {serve_args}"
    )

    return docker_cmd


def get_llamacpp_environment_vars(gpu_count: int = 1) -> dict:
    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),
    }

    if gpu_count > 1:
        cuda_devices = ",".join(str(i) for i in range(gpu_count))
        env_vars["CUDA_VISIBLE_DEVICES"] = cuda_devices

    return env_vars


def build_llamacpp_serve_args(
    llm_id: str, port: int, gpu_count: int, quantization: str = "", model_path: str = ""
) -> tuple[str, str]:
    repo_id, model_filename, model_pattern = parse_model_path(
        model_path, llm_id, quantization
    )

    args = ["-hf", repo_id] + get_host_port_args(port) + get_gpu_layer_args(gpu_count)

    serve_args = " ".join(args)
    return serve_args, repo_id


def build_llamacpp_runpod_docker_args(
    llm_id: str, port: int, gpu_count: int, quantization: str = "", model_path: str = ""
) -> str:
    repo_id, _, model_pattern = parse_model_path(model_path, llm_id, quantization)

    args_list = [
        f"-hf {repo_id}",
        "--host 0.0.0.0",
        f"--port {port}",
    ] + get_gpu_layer_args(gpu_count)

    docker_args = " ".join(args_list)

    return docker_args


def parse_model_path(model_path: str, llm_id: str, quantization: str):
    if model_path:
        if ":" in model_path:
            repo_id, model_filename = model_path.split(":", 1)
            model_pattern = [model_filename]
        else:
            repo_id = model_path
            if repo_id.endswith("-GGUF"):
                model_basename = repo_id.split("/")[-1].replace("-GGUF", "")
                if quantization:
                    model_filename = f"{model_basename}-{quantization}.gguf"
                    model_pattern = [model_filename]
                else:
                    model_filename = f"{model_basename}.gguf"
                    model_pattern = ["*.gguf"]
            else:
                model_filename = f"{llm_id}.gguf"
                model_pattern = ["*.gguf"]
    else:
        repo_id = f"{llm_id}-GGUF"
        model_filename = f"{llm_id}-{quantization}.gguf"
        model_pattern = f"*{quantization}*"

    return repo_id, model_filename, model_pattern
