import argparse
import asyncio
import sys

from runpod_runners.run_ollama_benchmark_with_runpod import create_ollama_pod
from runpod_runners.run_vllm_benchmark_with_runpod import create_vllm_pod
from runpod_runners.run_sglang_benchmark_with_runpod import create_sglang_pod
from runpod_runners.run_tensorrt_benchmark_with_runpod import create_tensorrt_pod


# Support for running the script with python run_<inference-engine>_benchmark_with_runpod.py --gpu_id <gpu_id> --volume_in_gb <volume_in_gb> --container_disk_in_gb <container_disk_in_gb> --llm_id <llm_id> --port <port> --llm_parameter_size <llm_parameter_size> --llm_common_name <llm_common_name> --gpu_count <gpu_count>


async def main():
    """Main function to handle command line execution"""

    parser = argparse.ArgumentParser(description="Run benchmark with Runpod")
    parser.add_argument(
        "--inference_engine",
        type=str,
        required=True,
        help="Inference engine to use (ollama, vllm, sglang, tensorrt)",
    )
    parser.add_argument(
        "--gpu_id", type=str, required=True, help="GPU type ID (e.g., NVIDIA H200)"
    )
    parser.add_argument(
        "--volume_in_gb", type=int, required=True, help="Volume size in GB"
    )
    parser.add_argument(
        "--container_disk_in_gb",
        type=int,
        required=True,
        help="Container disk size in GB",
    )
    parser.add_argument("--llm_id", type=str, required=True, help="LLM model ID")
    parser.add_argument(
        "--port", type=int, default=11434, help="Port number (default: 11434)"
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
        "--moe_backend",
        type=str,
        default=None,
        choices=["TRITON", "CUTLASS", "TRTLLM"],
        help="MoE backend for TensorRT-LLM (TRITON, CUTLASS, or TRTLLM). For GPT-OSS: all backends supported. For KimiK2: only TRTLLM supported (v1.2.0rc2+). Qwen3 models use PyTorch backend by default.",
    )
    parser.add_argument(
        "--kv_cache_free_gpu_memory_fraction",
        type=float,
        default=None,
        help="Fraction of free GPU memory to allocate for KV cache.",
    )
    parser.add_argument(
        "--enable_attention_dp",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Enable attention data parallelism. Use 'true' for throughput mode, 'false' for latency mode. Default: false (latency mode).",
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        default=None,
        help="Expert parallelism size for MoE models",
    )
  


    args = parser.parse_args()

    try:
        if args.inference_engine == "ollama":
            await create_ollama_pod(
                gpu_id=args.gpu_id,
                volume_in_gb=args.volume_in_gb,
                container_disk_in_gb=args.container_disk_in_gb,
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_count=args.gpu_count,
            )
            print("Ollama benchmark completed successfully!")

        elif args.inference_engine == "vllm":
            await create_vllm_pod(
                gpu_id=args.gpu_id,
                volume_in_gb=args.volume_in_gb,
                container_disk_in_gb=args.container_disk_in_gb,
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_count=args.gpu_count,
            )
            print("vLLM benchmark completed successfully!")


        

        elif args.inference_engine == "tensorrt":
            # Convert string boolean to actual boolean
            enable_attention_dp = None
            if args.enable_attention_dp is not None:
                enable_attention_dp = args.enable_attention_dp.lower() == "true"
            
            await create_tensorrt_pod(
                gpu_id=args.gpu_id,
                volume_in_gb=args.volume_in_gb,
                container_disk_in_gb=args.container_disk_in_gb,
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_count=args.gpu_count,
                moe_backend=args.moe_backend,
                kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
                enable_attention_dp=enable_attention_dp,
                ep_size=args.ep_size,
            )
            print("TensorRT-LLM benchmark completed successfully!")



        elif args.inference_engine == "sglang":
            await create_sglang_pod(
                gpu_id=args.gpu_id,
                volume_in_gb=args.volume_in_gb,
                container_disk_in_gb=args.container_disk_in_gb,
                llm_id=args.llm_id,
                port=args.port,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_count=args.gpu_count,
            )
            print("SGLang benchmark completed successfully!")
        else:
            print("Invalid inference engine")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":

    asyncio.run(main())