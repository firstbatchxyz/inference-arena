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
        help="Inference engine to use (vllm, sglang, ollama)",
    )
    parser.add_argument(
        "--quantization", type=str, default="", help="Quantization (default: )"
    )

    args = parser.parse_args()

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
            )
        else:
            print("Invalid inference engine")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
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

        sys.exit(1)


if __name__ == "__main__":

    asyncio.run(main())
