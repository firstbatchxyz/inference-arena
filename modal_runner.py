import argparse
import asyncio
import sys

from modal_runners.run_vllm_benchmark_with_modal import (
    create_vllm_modal_server,
)
from modal_runners.run_ollama_benchmark_with_modal import (
    create_ollama_modal_server,
)
from modal_runners.run_sglang_benchmark_with_modal import (
    create_sglang_modal_server,
)
from modal_runners.run_tensorrt_benchmark_with_modal import (
    create_tensorrt_modal_server,
)
from modal_runners.run_llamacpp_benchmark_with_modal import (
    create_llamacpp_modal_server,
)


async def main():
    """Main function to handle command line execution"""

    parser = argparse.ArgumentParser(description="Run benchmark with Modal")
    parser.add_argument("--llm_id", type=str, required=True, help="LLM model ID")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port number (default: 8000)"
    )
    parser.add_argument(
        "--llm_parameter_size", type=str, default="", help="LLM parameter size"
    )
    parser.add_argument(
        "--llm_common_name", type=str, default="", help="LLM common name"
    )
    parser.add_argument(
        "--gpu_count", type=int, default=1, help="Number of GPUs (default: 1)"
    )
    parser.add_argument(
        "--gpu_id", type=str, required=True, help="GPU type ID (e.g., H100, A100, L4)"
    )
    parser.add_argument(
        "--inference_engine",
        type=str,
        required=True,
        help="Inference engine to use (vllm, ollama, sglang, tensorrt, llamacpp)",
    )
    parser.add_argument(
        "--quantization", type=str, default="", help="Quantization (default: )"
    )
    parser.add_argument(
        "--fast_boot",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Enable fast boot mode (enforce-eager) for faster startup (default: true)",
    )
    # TensorRT-LLM optional parameters
    parser.add_argument(
        "--moe_backend",
        type=str,
        default=None,
        help="MoE backend for TensorRT-LLM (TRITON/CUTLASS/TRTLLM)",
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        default=None,
        help="Expert parallelism size for MoE models in TensorRT-LLM",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=None,
        help="Pipeline parallelism size for TensorRT-LLM",
    )
    parser.add_argument(
        "--kv_cache_free_gpu_memory_fraction",
        type=float,
        default=None,
        help="KV cache free GPU memory fraction for TensorRT-LLM",
    )
    parser.add_argument(
        "--enable_attention_dp",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Enable attention data parallelism for TensorRT-LLM",
    )
    # llama.cpp specific parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="GGUF model path as 'repo/filename' or just 'repo' (e.g., 'Aldaris/Qwen3-14B-Q4_K_M-GGUF:Qwen3-14B-Q4_K_M.gguf')",
    )

    args = parser.parse_args()

    try:
        if args.inference_engine == "vllm":
            # Convert string boolean to actual boolean
            fast_boot = args.fast_boot.lower() == "true"
            
            await create_vllm_modal_server(
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_id=args.gpu_id,
                gpu_count=args.gpu_count,
                quantization=args.quantization,
                fast_boot=fast_boot,
            )
            print("vLLM benchmark completed successfully!")
        elif args.inference_engine == "ollama":
            await create_ollama_modal_server(
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_id=args.gpu_id,
                gpu_count=args.gpu_count,
            )
            print("Ollama benchmark completed successfully!")
        elif args.inference_engine == "sglang":
            await create_sglang_modal_server(
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_id=args.gpu_id,
                gpu_count=args.gpu_count,
            )
            print("SGLang benchmark completed successfully!")
        elif args.inference_engine == "tensorrt":
            # Convert enable_attention_dp string to boolean if provided
            enable_attention_dp = None
            if args.enable_attention_dp:
                enable_attention_dp = args.enable_attention_dp.lower() == "true"
            
            await create_tensorrt_modal_server(
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_id=args.gpu_id,
                gpu_count=args.gpu_count,
                moe_backend=args.moe_backend,
                kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
                enable_attention_dp=enable_attention_dp,
                ep_size=args.ep_size,
                pp_size=args.pp_size,
            )
            print("TensorRT-LLM benchmark completed successfully!")
        elif args.inference_engine == "llamacpp":
            await create_llamacpp_modal_server(
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_id=args.gpu_id,
                gpu_count=args.gpu_count,
                quantization=args.quantization,
                model_path=args.model_path,
            )
            print("llama.cpp benchmark completed successfully!")
        else:
            print("Invalid inference engine. Currently only 'vllm', 'ollama', 'sglang', 'tensorrt', and 'llamacpp' are supported for Modal.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())