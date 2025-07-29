import re
import os

def get_compatible_sglang_image(gpu_id: str = "") -> str:
    """
    Determine the appropriate SGLang Docker image based on CUDA compatibility.
    RunPod may have older CUDA versions, so we need to use compatible SGLang versions.
    """
    # Use official SGLang Docker images from lmsysorg/sglang
    # Based on official documentation at https://docs.sglang.ai/start/install.html
    # Available images: latest, dev
    
    if "B200" in gpu_id:
        # B200 might need more recent version, use latest
        return "lmsysorg/sglang:latest"
    else:
        # For most GPUs, use the stable latest version
        return "lmsysorg/sglang:latest"

def get_optimal_sglang_config(llm_id: str, llm_parameter_size: str = "", gpu_id: str = "", gpu_count: int = 1) -> dict:
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
        "kv_cache_dtype": "auto"
    }
    
    # Extract parameter count from model ID or parameter size
    param_count = 0
    if llm_parameter_size:
        # Extract numeric value from parameter size (e.g., "70b" -> 70)
        size_match = re.search(r'(\d+)', llm_parameter_size.lower())
        if size_match:
            param_count = int(size_match.group(1))
    
    # If no explicit parameter size, try to extract from model ID
    if param_count == 0:
        model_size_patterns = [
            (r'(\d+)b(?:-|$|:)', lambda x: int(x)),  # "70b", "7b-instruct"
            (r'(\d+\.\d+)b(?:-|$|:)', lambda x: int(float(x))),  # "3.1b"
            (r'-(\d+)b(?:-|$)', lambda x: int(x)),  # "-70b-"
            (r'llama-?3\.1-(\d+)b', lambda x: int(x)),  # "llama-3.1-70b"
        ]
        
        for pattern, converter in model_size_patterns:
            match = re.search(pattern, llm_id.lower())
            if match:
                param_count = converter(match.group(1))
                break
    
    # Check if it's a MoE (Mixture of Experts) model
    is_moe = any(keyword in llm_id.lower() for keyword in ['mixtral', 'moe', 'deepseek-r1', 'qwen3:235b'])
    
    # Adjust configuration based on model size and type
    if param_count >= 400:  # 400B+ models (e.g., Llama 3.1 405B)
        config.update({
            "tp_size": min(8, gpu_count),  # Use up to 8 GPUs or available GPU count
            "mem_fraction_static": 0.75 if gpu_count > 1 else 0.60,  # Higher memory usage for multi-GPU
            "context_length": 8192,
            "kv_cache_dtype": "fp8"
        })
    elif param_count >= 180:  # 180B+ models (e.g., Falcon 180B)
        config.update({
            "tp_size": min(4, gpu_count),  # Use up to 4 GPUs or available GPU count
            "mem_fraction_static": 0.80 if gpu_count > 1 else 0.65,  # Higher memory usage for multi-GPU
            "context_length": 8192,
            "kv_cache_dtype": "fp8"
        })
    elif param_count >= 70:  # 70B+ models
        # For 70B models, be very conservative with memory allocation
        # SGLang needs substantial memory for KV cache pool after model loading
        if "H100" in gpu_id or "A100" in gpu_id or "H200" in gpu_id:
            config.update({
                "tp_size": min(2, gpu_count),  # Use up to 2 GPUs for 70B models
                "mem_fraction_static": 0.75 if gpu_count > 1 else 0.55,  # Higher memory usage for multi-GPU
                "context_length": 4096 if gpu_count > 1 else 2048,  # Larger context with multiple GPUs
                "kv_cache_dtype": "fp8"  # Use compressed KV cache to save memory
            })
        else:
            config.update({
                "tp_size": min(2, gpu_count),  # Use up to 2 GPUs for other hardware
                "mem_fraction_static": 0.75 if gpu_count > 1 else 0.60,  # Higher memory usage for multi-GPU
                "context_length": 4096 if gpu_count > 1 else 2048
            })
    elif param_count >= 30:  # 30B+ models
        config.update({
            "tp_size": min(2, gpu_count),  # Use up to 2 GPUs for 30B+ models
            "mem_fraction_static": 0.90 if gpu_count > 1 else 0.80,  # Higher memory usage for multi-GPU
            "context_length": 16384
        })
    elif param_count >= 13:  # 13B+ models
        config.update({
            "tp_size": 1,  # Single GPU is usually sufficient for 13B models
            "mem_fraction_static": 0.85,
            "context_length": 32768
        })
    else:  # Small models (7B and below)
        config.update({
            "tp_size": 1,  # Single GPU is sufficient for small models
            "mem_fraction_static": 0.90,
            "context_length": 32768
        })
    
    # Special handling for MoE models
    if is_moe:
        config["mem_fraction_static"] *= 0.8  # MoE models need more memory overhead
        # MoE models can benefit from multiple GPUs for better performance
        if gpu_count > 1 and param_count >= 30:
            config["tp_size"] = min(2, gpu_count)
        else:
            config["tp_size"] = 1
    
    # Model-specific optimizations
    if "qwen" in llm_id.lower():
        config["trust_remote_code"] = True
        if param_count >= 72:
            config["context_length"] = min(config["context_length"], 8192)  # Qwen large models
    
    if "gemma" in llm_id.lower():
        config["trust_remote_code"] = True
        if param_count <= 4:
            config["context_length"] = 8192  # Gemma small models
    
    if "deepseek" in llm_id.lower():
        config["trust_remote_code"] = True
        config["mem_fraction_static"] *= 0.7  # DeepSeek models need more memory
        # DeepSeek models can benefit from multiple GPUs for large models
        if gpu_count > 1 and param_count >= 70:
            config["tp_size"] = min(2, gpu_count)
        else:
            config["tp_size"] = 1
    
    if "falcon" in llm_id.lower():
        config["trust_remote_code"] = True
    
    if "llama" in llm_id.lower() and "meta-llama" in llm_id:
        # Meta's Llama models often work better with specific settings
        if param_count >= 70:
            # Llama 70B+ models need even more conservative memory settings
            config["mem_fraction_static"] = min(config["mem_fraction_static"], 0.65 if gpu_count > 1 else 0.50)
            config["context_length"] = min(config["context_length"], 2048 if gpu_count == 1 else 4096)
            config["kv_cache_dtype"] = "fp8"  # Force compressed KV cache
    
    # GPU-specific optimizations
    if "H100" in gpu_id or "B200" in gpu_id or "H200" in gpu_id:
        # Latest GPUs support better features
        if param_count >= 70:
            config["kv_cache_dtype"] = "auto"  # Let SGLang choose optimal format
    
    # Multi-GPU configuration: Use tensor parallelism when multiple GPUs are available
    if gpu_count > 1:
        # Ensure we don't exceed available GPU count
        config["tp_size"] = min(config["tp_size"], gpu_count)
        config["dp_size"] = 1  # Keep data parallelism at 1 for simplicity
        
        # Increase memory fraction for multi-GPU setups since memory is distributed
        # Each GPU only needs to hold a portion of the model
        current_mem_fraction = config["mem_fraction_static"]
        config["mem_fraction_static"] = min(0.95, current_mem_fraction + 0.1)
        
        print(f"Multi-GPU configuration: Using {config['tp_size']} GPUs with tensor parallelism")
        print(f"Memory fraction per GPU: {config['mem_fraction_static']:.2f}")
    else:
        # Single GPU setup - keep conservative settings to avoid CUDA device errors
        config["tp_size"] = 1
        config["dp_size"] = 1
    
    return config

def build_sglang_docker_args(llm_id: str, port: int, sglang_config: dict) -> str:
    """
    Build SGLang docker arguments based on the optimal configuration.
    Enhanced with HuggingFace token support and tokenizer optimizations.
    """
    # Start with the proper SGLang server command
    args = [
        "python", "-m", "sglang.launch_server",
        f"--model-path {llm_id}",
        f"--port {port}",
        "--host 0.0.0.0",
        f"--tp-size {sglang_config['tp_size']}",
        f"--dp-size {sglang_config['dp_size']}",
        f"--mem-fraction-static {sglang_config['mem_fraction_static']}",
        f"--context-length {sglang_config['context_length']}",
        "--tokenizer-mode auto",  # Use fast tokenizer when available
    ]
    
    # Multi-GPU specific optimizations
    if sglang_config.get('tp_size', 1) > 1:
        # Add multi-GPU specific arguments for better stability
        args.extend([
            "--disable-cuda-graph",  # Disable CUDA graphs for multi-GPU stability
            "--chunked-prefill-size 8192",
            "--tp 2",
        ])
        print(f"Added multi-GPU optimizations for tp_size={sglang_config['tp_size']}")
    
    # Add optional parameters
    if sglang_config.get("trust_remote_code"):
        args.append("--trust-remote-code")
    
    if sglang_config.get("quantization"):
        args.append(f"--quantization {sglang_config['quantization']}")
    
    if sglang_config.get("kv_cache_dtype") and sglang_config["kv_cache_dtype"] != "auto":
        args.append(f"--kv-cache-dtype {sglang_config['kv_cache_dtype']}")
    
    # Enable chunked prefill for better memory efficiency on large models
    if sglang_config.get("context_length", 0) >= 8192:
        args.append("--chunked-prefill-size 4096")
    
    # Add HuggingFace token support if available
    # Note: HF_TOKEN will be passed via environment variables, but we can also
    # add it to args for explicit authentication (SGLang supports both methods)
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token:
        # SGLang doesn't have explicit --hf-token flag, but it reads from env vars
        # We'll rely on environment variables for token authentication
        pass
    
    # Additional optimizations for better performance and memory management
    # Using only well-supported SGLang arguments
    
    # Extra memory optimizations for large models (70B+)
    if llm_id and any(size in llm_id.lower() for size in ['70b', '72b', '180b', '405b']):
        # Use smaller prefill chunks to reduce memory peaks
        if sglang_config.get("context_length", 0) >= 2048:
            args.append("--chunked-prefill-size 1024")
        else:
            args.append("--chunked-prefill-size 512")
    
    # Model-specific optimizations
    if "deepseek" in llm_id.lower():
        # DeepSeek models - keep only core supported arguments
        pass  # No special arguments needed that aren't already set
    
    if "qwen" in llm_id.lower() or "gemma" in llm_id.lower():
        # These models often need trust-remote-code
        if "--trust-remote-code" not in " ".join(args):
            args.append("--trust-remote-code")
    
    return " ".join(args) 