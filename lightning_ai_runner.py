from lightning_ai_runners.run_vllm_benchmark_with_lightning_ai import (
    create_vllm_lightning_ai_server,
)
from lightning_ai_runners.run_sglang_benchmark_with_lightning_ai import (
    create_sglang_lightning_ai_server,
)
from lightning_ai_runners.run_ollama_benchmark_with_lightning_ai import (
    create_ollama_lightning_ai_server,
)
from lightning_ai_runners.run_lmstudio_benchmark_with_lightning_ai import (
    create_lmstudio_lightning_ai_server,
)
from lightning_ai_runners.run_tensorrt_benchmark_with_lightning_ai import (
    create_tensorrt_lightning_ai_server,
)
from utils.lightning_studio_utils import (
    create_studio_if_needed,
    stop_studio_if_created,
    is_auto_management_supported,
    convert_gpu_id_to_lightning_format,
)
import argparse
import sys
import asyncio
import traceback


async def main():
    """Main function to handle command line execution"""

    parser = argparse.ArgumentParser(description="Run benchmark with Lightning AI")
    parser.add_argument(
        "--studio_name",
        type=str,
        required=True,
        help="Studio name",
    )
    parser.add_argument("--teamspace", type=str, required=True, help="Teamspace")
    parser.add_argument("--org", type=str, required=True, help="Org")
    parser.add_argument("--llm_id", type=str, required=True, help="LLM model ID")
    parser.add_argument(
        "--port", type=int, default=11434, help="Port number (default: 8000)"
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
        "--gpu_id", type=str, default="", help="GPU type ID (default: )"
    )
    parser.add_argument(
        "--inference_engine",
        type=str,
        required=True,
        help="Inference engine to use (vllm, sglang, ollama, lmstudio, tensorrt)",
    )
    parser.add_argument(
        "--quantization", type=str, default="", help="Quantization (default: )"
    )
    parser.add_argument(
        "--moe_backend",
        type=str,
        default=None,
        help="MoE backend for TensorRT-LLM (TRITON/CUTLASS/TRTLLM)",
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
        help="Enable attention data parallelism for TensorRT-LLM (true/false)",
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        default=None,
        help="Expert parallelism size for TensorRT-LLM",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=None,
        help="Pipeline parallelism size for TensorRT-LLM",
    )
    
    parser.add_argument(
        "--studio_create_timeout",
        type=int,
        default=300,
        help="Timeout for studio creation in seconds (default: 300)",
    )

    args = parser.parse_args()
    
    # Auto-detect platform and manage studio lifecycle if supported
    studio_was_created = False
    auto_manage = is_auto_management_supported()
    
    if auto_manage:
        # Convert GPU ID and count to Lightning AI format
        lightning_gpu_type = convert_gpu_id_to_lightning_format(args.gpu_id, args.gpu_count)
        
        print(f"Auto-managing studio '{args.studio_name}' with GPU: {lightning_gpu_type}")
        success, message, was_created = create_studio_if_needed(
            studio_name=args.studio_name,
            teamspace=args.teamspace,
            gpu_type=lightning_gpu_type,
            timeout=args.studio_create_timeout
        )
        
        if not success:
            print(f"Failed to create/start studio: {message}")
            sys.exit(1)
        
        studio_was_created = was_created
        print(f"Studio ready: {message}")
    else:
        print("Platform does not support auto studio management, assuming studio is already running")

    try:
        if args.inference_engine == "vllm":
            await create_vllm_lightning_ai_server(
                studio_name=args.studio_name,
                teamspace=args.teamspace,
                org=args.org,
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_count=args.gpu_count,
                gpu_id=args.gpu_id,
                auto_managed=auto_manage,
            )
        elif args.inference_engine == "sglang":
            await create_sglang_lightning_ai_server(
                studio_name=args.studio_name,
                teamspace=args.teamspace,
                org=args.org,
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_count=args.gpu_count,
                gpu_id=args.gpu_id,
                auto_managed=auto_manage,
            )
        elif args.inference_engine == "ollama":
            await create_ollama_lightning_ai_server(
                studio_name=args.studio_name,
                teamspace=args.teamspace,
                org=args.org,
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_count=args.gpu_count,
                gpu_id=args.gpu_id,
                auto_managed=auto_manage,
            )
        elif args.inference_engine == "lmstudio":
            await create_lmstudio_lightning_ai_server(
                studio_name=args.studio_name,
                teamspace=args.teamspace,
                org=args.org,
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_count=args.gpu_count,
                gpu_id=args.gpu_id,
                auto_managed=auto_manage,
            )
        elif args.inference_engine == "tensorrt":
            # Convert string boolean to actual boolean
            enable_attention_dp = None
            if args.enable_attention_dp is not None:
                enable_attention_dp = args.enable_attention_dp.lower() == "true"
            
            await create_tensorrt_lightning_ai_server(
                studio_name=args.studio_name,
                teamspace=args.teamspace,
                org=args.org,
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_count=args.gpu_count,
                gpu_id=args.gpu_id,
                moe_backend=args.moe_backend,
                kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
                enable_attention_dp=enable_attention_dp,
                ep_size=args.ep_size,
                pp_size=args.pp_size,
                auto_managed=auto_manage,
            )
        else:
            print("Invalid inference engine")
            sys.exit(1)
            
        # Studio cleanup after successful benchmark
        if auto_manage:
            print("Stopping studio after benchmark completion...")
            success, message = stop_studio_if_created(
                studio_name=args.studio_name,
                teamspace=args.teamspace,
                was_created=studio_was_created
            )
            
            if success:
                print(f"Studio cleanup: {message}")
            else:
                print(f"Studio cleanup warning: {message}")
                
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        
        # Cleanup studio if it was auto-managed
        if auto_manage:
            print("Cleaning up studio after interruption...")
            success, message = stop_studio_if_created(
                studio_name=args.studio_name,
                teamspace=args.teamspace,
                was_created=studio_was_created
            )
            
            if success:
                print(f"Studio cleanup: {message}")
            else:
                print(f"Studio cleanup warning: {message}")
        
        sys.exit(1)
    except Exception as e:
        # Get the traceback information
        tb = traceback.extract_tb(e.__traceback__)

        # Get the last frame (where the error occurred)
        if tb:
            last_frame = tb[-1]
            error_file = last_frame.filename
            error_line = last_frame.lineno
            error_func = last_frame.name

            print(f"Benchmark failed: {e}")
            print(f"Error occurred in file: {error_file}")
            print(f"Error occurred at line: {error_line}")
            print(f"Error occurred in function: {error_func}")
            print("\nFull traceback:")
            traceback.print_exc()
        else:
            print(f"Benchmark failed: {e}")
            traceback.print_exc()
        
        # Cleanup studio if it was auto-managed
        if auto_manage:
            print("Cleaning up studio after error...")
            success, message = stop_studio_if_created(
                studio_name=args.studio_name,
                teamspace=args.teamspace,
                was_created=studio_was_created
            )
            
            if success:
                print(f"Studio cleanup: {message}")
            else:
                print(f"Studio cleanup warning: {message}")

        sys.exit(1)


if __name__ == "__main__":

    asyncio.run(main())
