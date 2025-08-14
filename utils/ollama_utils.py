import json
import time

import requests

# Model name mappings from Ollama to HuggingFace
OLLAMA_TO_HF_TOKENIZER = {
    "llama3.1:70b": "meta-llama/Meta-Llama-3.1-70B",
    "llama3.1:8b": "meta-llama/Meta-Llama-3.1-8B",
    "llama3:70b": "meta-llama/Meta-Llama-3-70B",
    "llama3:8b": "meta-llama/Meta-Llama-3-8B",
    "llama2:70b": "meta-llama/Llama-2-70b-hf",
    "llama2:13b": "meta-llama/Llama-2-13b-hf",
    "llama2:7b": "meta-llama/Llama-2-7b-hf",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "qwen2:72b": "Qwen/Qwen2-72B",
    "qwen2:7b": "Qwen/Qwen2-7B",
    "qwen3:32b": "Qwen/Qwen3-32B",
    "falcon:180b": "tiiuae/falcon-180B",
    "qwen3-coder:30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "deepseek-v3:671b": "deepseek-ai/DeepSeek-V3",
    "gpt-oss:20b": "openai/gpt-oss-20b",
    "gpt-oss:12b": "openai/gpt-oss-12b",
    "lmsys/gpt-oss-120b-bf16": "lmsys/gpt-oss-120b-bf16",
    "lmsys/gpt-oss-20b-bf16": "lmsys/gpt-oss-20b-bf16",
    "qwen3:235b": "Qwen/Qwen3-235B-A22B",
    "zai-org/GLM-4.5-Air-FP8": "zai-org/GLM-4.5-Air-FP8"
}

# Configuration constants
MAX_RETRIES = 120  # 10 minutes max wait time
RETRY_DELAY = 5  # seconds between retries
TEST_TIMEOUT = 60  # seconds for model test


def get_ollama_environment_vars(gpu_count: int) -> dict[str, str]:
    """Get Ollama environment variables based on GPU configuration."""
    cuda_devices = ",".join(str(i) for i in range(gpu_count)) if gpu_count > 1 else "0"

    env_vars = {
        "OLLAMA_HOST": "0.0.0.0",
        "CUDA_VISIBLE_DEVICES": cuda_devices,
        "OLLAMA_NUM_PARALLEL": str(max(1, gpu_count * 2)),
        "OLLAMA_MAX_LOADED_MODELS": "0",
        "OLLAMA_LOAD_TIMEOUT": "30m0s",
        "OLLAMA_KEEP_ALIVE": "10m",
        "OLLAMA_ORIGINS": "*",
        "OLLAMA_TIMEOUT": "900s",
        "OLLAMA_MAX_QUEUE": "512",
        "OLLAMA_MAX_CONCURRENT": str(max(4, gpu_count * 2)),
        "OLLAMA_VRAM_RECOVERY_TIMEOUT": "30s",
        "OLLAMA_PRELOAD_TIMEOUT": "1800s",
        "OLLAMA_UNLOAD_TIMEOUT": "60s",
        "CUDA_LAUNCH_BLOCKING": "0",
    }

    # Add multi-GPU specific settings
    if gpu_count > 1:
        env_vars.update({
            "NCCL_DEBUG": "INFO",
            "NCCL_IB_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
            "PYTHONUNBUFFERED": "1",
            "TORCH_NCCL_BLOCKING_WAIT": "1",
        })

    return env_vars


def _test_model(pod_url: str, llm_id: str) -> bool:
    """Test if a model is fully loaded and responsive."""
    try:
        response = requests.post(
            f"{pod_url}/api/generate",
            json={
                "model": llm_id,
                "prompt": "Test",
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=TEST_TIMEOUT,
        )
        return response.status_code == 200
    except Exception:
        return False


def _get_available_models(pod_url: str) -> list:
    """Get list of available models from Ollama API."""
    try:
        response = requests.get(f"{pod_url}/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get("models", [])
            return [model.get("name", "") for model in models]
    except Exception:
        pass
    return []


def verify_model_availability(pod_url: str, llm_id: str) -> bool:
    """
    Verify if a model is available and fully loaded on the Ollama server.

    Args:
        pod_url: The URL of the Ollama server
        llm_id: The model identifier to check

    Returns:
        True if model is available and responsive, False otherwise
    """
    for retry in range(MAX_RETRIES):
        model_names = _get_available_models(pod_url)

        # Check if model exists (handle both with and without tag)
        for model_name in model_names:
            if llm_id in model_name or model_name in llm_id:
                # Test if model is fully loaded
                if _test_model(pod_url, llm_id):
                    return True
                return False

        # Wait before retrying if not last attempt
        if retry < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)

    return False


def pull_model(pod_url: str, llm_id: str) -> bool:
    """
    Pull a model from the Ollama registry.

    Args:
        pod_url: The URL of the Ollama server
        llm_id: The model identifier to pull

    Returns:
        True if model was successfully pulled, False otherwise
    """
    try:
        with requests.post(
            f"{pod_url}/api/pull",
            json={"model": llm_id},
            stream=True
        ) as response:
            if response.status_code != 200:
                # Retry the pull once on failure
                return pull_model(pod_url, llm_id)

            # Monitor download progress
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                    status = data.get("status", "")

                    # Check if download is complete
                    if status == "success" or "success" in str(data):
                        time.sleep(1)  # Wait for model to be fully loaded
                        return True

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        print(f"Error pulling model: {e}")
        return False

    return False


