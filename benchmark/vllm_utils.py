import re

def get_compatible_vllm_image(gpu_id: str = "", model_id: str = "") -> str:
    """
    Determine the appropriate vLLM Docker image based on CUDA compatibility and model requirements.
    RunPod typically uses CUDA 12.6, so we need to use compatible vLLM versions.
    """
    # Use stable vLLM version that's compatible with CUDA 12.6
    # vLLM v0.6.0 and v0.6.1.post2 are known to work with CUDA 12.6
    
    # Gemma 2 models have better support in newer versions
    if "gemma-2" in model_id.lower():
        # Use v0.5.5 which has better Gemma 2 support and is compatible with CUDA 12.6
        return "vllm/vllm-openai:v0.5.5"
    elif "B200" in gpu_id:
        # B200 requires special handling - use a more recent version that might support it
        # but still compatible with CUDA 12.6
        return "vllm/vllm-openai:v0.6.1.post2"
    else:
        # For most GPUs, use stable version compatible with CUDA 12.6
        return "vllm/vllm-openai:v0.6.0"

def get_optimal_vllm_config(llm_id: str, llm_parameter_size: str = "", gpu_id: str = "") -> dict:
    """
    Automatically determine optimal vLLM configuration based on model characteristics.
    
    This function now includes model-specific context length detection to prevent
    setting max_model_len higher than what the model actually supports.
    
    If you encounter max_model_len errors, the environment variable 
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 is automatically set to allow overriding.
    """
    config = {
        "gpu_memory_utilization": 0.85,
        "max_model_len": 4096,
        "dtype": "auto",
        "additional_args": []
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
    
    # Model-specific context length detection
    # This helps avoid setting max_model_len higher than what the model supports
    def get_model_max_context(model_id: str) -> int:
        """Get the maximum context length supported by specific models"""
        model_lower = model_id.lower()
        
        # Llama models context lengths
        if "llama-3" in model_lower or "meta-llama/llama-3" in model_lower:
            print(f"🔍 Detected Llama-3 model, setting max context to 8192")
            return 8192  # Llama 3 models support 8192 context
        elif "llama-3.1" in model_lower:
            print(f"🔍 Detected Llama-3.1 model, setting max context to 131072")
            return 131072  # Llama 3.1 supports very long context
        elif "llama-2" in model_lower:
            print(f"🔍 Detected Llama-2 model, setting max context to 4096")
            return 4096  # Llama 2 models support 4096 context
        
        # Qwen models
        elif "qwen2.5" in model_lower:
            if param_count >= 72:
                print(f"🔍 Detected large Qwen2.5 model ({param_count}B), using conservative context 8192")
                return 8192  # Conservative for large Qwen models
            else:
                print(f"🔍 Detected Qwen2.5 model, setting max context to 32768")
                return 32768  # Qwen2.5 smaller models support long context
        elif "qwen" in model_lower:
            print(f"🔍 Detected Qwen model, using conservative context 8192")
            return 8192  # Conservative for older Qwen models
        
        # Gemma models
        elif "gemma-2" in model_lower:
            print(f"🔍 Detected Gemma-2 model, setting max context to 8192")
            return 8192  # Gemma 2 has good context support
        elif "gemma" in model_lower:
            print(f"🔍 Detected Gemma model, setting max context to 8192")
            return 8192  # Original Gemma models
        
        # DeepSeek models
        elif "deepseek" in model_lower:
            print(f"🔍 Detected DeepSeek model, using conservative context 4096")
            return 4096  # Conservative for DeepSeek models
        
        # Mixtral and MoE models
        elif "mixtral" in model_lower or is_moe:
            print(f"🔍 Detected MoE/Mixtral model, setting max context to 32768")
            return 32768  # Mixtral supports long context
        
        # Default fallback based on parameter size
        if param_count >= 70:
            print(f"🔍 Large model ({param_count}B), using conservative context 4096")
            return 4096  # Conservative for very large models
        elif param_count >= 30:
            print(f"🔍 Medium model ({param_count}B), setting context to 8192")
            return 8192  # Medium models
        elif param_count >= 7:
            print(f"🔍 Standard model ({param_count}B), setting context to 8192")
            return 8192  # 7B+ models typically support 8K
        else:
            print(f"🔍 Small model ({param_count}B), using context 4096")
            return 4096  # Small models
    
    # Get model-specific max context
    model_max_context = get_model_max_context(llm_id)
    print(f"📏 Model max context determined: {model_max_context} tokens for {llm_id}")
    
    # Adjust configuration based on model size and type
    if param_count >= 400:  # 400B+ models (e.g., Llama 3.1 405B)
        config.update({
            "gpu_memory_utilization": 0.60,
            "max_model_len": min(8192, model_max_context),
            "dtype": "bfloat16",
            "additional_args": ["--tensor-parallel-size", "8", "--pipeline-parallel-size", "2"]
        })
    elif param_count >= 180:  # 180B+ models (e.g., Falcon 180B)
        config.update({
            "gpu_memory_utilization": 0.65,
            "max_model_len": min(8192, model_max_context),
            "dtype": "bfloat16",
            "additional_args": ["--tensor-parallel-size", "4"]
        })
    elif param_count >= 72:  # 72B+ models (like Qwen2.5-72B)
        config.update({
            "gpu_memory_utilization": 0.65,  # Very conservative for single H200
            "max_model_len": min(1024, model_max_context),  # Minimal context to ensure KV cache fits
            "dtype": "bfloat16",
            "additional_args": ["--enforce-eager", "--disable-log-requests", "--kv-cache-dtype", "fp8", "--enable-chunked-prefill"]
        })
    elif param_count >= 70:  # 70B models
        config.update({
            "gpu_memory_utilization": 0.95,  # Very aggressive - give almost all memory to model
            "max_model_len": min(512, model_max_context),  # Minimal context to leave room for KV cache
            "dtype": "bfloat16",  # Use explicit precision
            "additional_args": ["--enforce-eager", "--disable-log-requests", "--tensor-parallel-size", "2", "--max-num-seqs", "8"]  # Removed FP8 for compatibility
        })
    elif param_count >= 30:  # 30B+ models
        config.update({
            "gpu_memory_utilization": 0.80,
            "max_model_len": min(16384, model_max_context),
            "dtype": "auto"
        })
    elif param_count >= 13:  # 13B+ models
        config.update({
            "gpu_memory_utilization": 0.85,
            "max_model_len": min(16384, model_max_context),  # Respect model limits
            "dtype": "auto"
        })
    else:  # Small models (7B and below)
        # Be more conservative for very small models like 2B
        if param_count <= 3:
            config.update({
                "gpu_memory_utilization": 0.80,  # More conservative for small models
                "max_model_len": min(4096, model_max_context),  # Reasonable context for 2B models
                "dtype": "auto"
            })
        else:
            config.update({
                "gpu_memory_utilization": 0.90,
                "max_model_len": min(8192, model_max_context),  # Respect model's actual context limit
                "dtype": "auto"
            })
    
    # Log the final max_model_len decision
    print(f"⚙️  Final max_model_len set to: {config['max_model_len']} tokens")
    print(f"📝 Note: VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 is set to allow override if needed")
    
    # Special handling for MoE models
    if is_moe:
        config["gpu_memory_utilization"] *= 0.8  # MoE models need more memory overhead
        config["additional_args"].append("--enforce-eager")  # Better for MoE models
    
    # Model-specific optimizations
    if "qwen" in llm_id.lower():
        config["additional_args"].append("--trust-remote-code")
        if param_count >= 72:
            # Extra aggressive optimization for 72B+ Qwen models on single GPU
            config["max_model_len"] = min(config["max_model_len"], 1024)
            config["gpu_memory_utilization"] = min(config["gpu_memory_utilization"], 0.60)
            if "--quantization" not in " ".join(config["additional_args"]):
                config["additional_args"].extend(["--quantization", "fp8"])
    
    if "gemma" in llm_id.lower():
        config["additional_args"].append("--trust-remote-code")
        if "gemma-2" in llm_id.lower():
            # Gemma 2 has sliding window attention issues in some vLLM versions
            # Cap to safe context length that works with sliding window
            config["max_model_len"] = min(4096, model_max_context)  # Safe length for Gemma 2
            config["additional_args"].extend(["--enforce-eager", "--disable-sliding-window"])
        elif param_count <= 4:
            config["max_model_len"] = min(8192, model_max_context)  # Original Gemma small models
    
    if "deepseek" in llm_id.lower():
        config["additional_args"].extend(["--trust-remote-code", "--enforce-eager"])
        config["gpu_memory_utilization"] *= 0.7  # DeepSeek models need more memory
    
    if "mistral" in llm_id.lower():
        # Mistral models use sliding window attention
        config["additional_args"].extend(["--trust-remote-code"])
        # Don't disable sliding window for Mistral models - they need it
        # Ensure reasonable context length for sliding window
        config["max_model_len"] = min(8192, model_max_context)
    
    if "llama" in llm_id.lower() and "meta-llama" in llm_id and param_count >= 70:
        # Llama 70B+ models need the most aggressive memory settings possible
        config["gpu_memory_utilization"] = 0.85  # Conservative but allow for tensor parallelism
        config["max_model_len"] = min(1024, model_max_context)  # Small but reasonable context
        # Use tensor parallelism to split model across multiple GPUs
        config["additional_args"] = ["--enforce-eager", "--disable-log-requests", "--kv-cache-dtype", "fp8", 
                                   "--tensor-parallel-size", "2", "--max-num-seqs", "16", "--block-size", "16"]
        print(f"🔥 Applied tensor parallelism (2 GPUs) for Llama {param_count}B model")
    
    if "falcon" in llm_id.lower():
        config["additional_args"].append("--trust-remote-code")
        if param_count >= 180:
            config["dtype"] = "bfloat16"
    
    # GPU-specific optimizations
    if "H100" in gpu_id or "H200" in gpu_id:
        # Latest compatible GPUs support better precision
        if config["dtype"] == "auto":
            config["dtype"] = "bfloat16"
    elif "B200" in gpu_id:
        # B200 has PyTorch compatibility issues with standard vLLM image
        # Use more conservative settings
        config["gpu_memory_utilization"] *= 0.9
        config["additional_args"].extend(["--enforce-eager", "--disable-custom-all-reduce"])
        if config["dtype"] == "auto":
            config["dtype"] = "float16"  # Use float16 instead of bfloat16 for compatibility
    
    return config
