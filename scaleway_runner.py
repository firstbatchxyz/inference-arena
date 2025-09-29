from scaleway_runners.run_mlx_benchmark_with_scaleway import create_mlx_scaleway_server
import argparse
import sys
import asyncio
import traceback


async def main():
    """Main function to handle command line execution"""

    parser = argparse.ArgumentParser(description="Run benchmark with Scaleway")
    parser.add_argument(
        "--server_id",
        type=str,
        default="9e3d8d31-e1ea-4dd8-9f63-0ea3766a072d",
        help="Scaleway Apple server ID",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-4B-MLX-4bit",
        help="Model path (default: Qwen/Qwen3-4B-MLX-4bit)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="lm",
        help="Model type (default: lm)",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port number (default: 8080)"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=None,
        help="Context length",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Config name",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        default=None,
        help="Quantization bits",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=1,
        help="Maximum concurrency (default: 1)",
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
        default="mlx",
        help="Inference engine to use (default: mlx)",
    )

    args = parser.parse_args()

    try:
        if args.inference_engine == "mlx":
            await create_mlx_scaleway_server(
                server_id=args.server_id,
                model_path=args.model_path,
                model_type=args.model_type,
                port=args.port,
                context_length=args.context_length,
                config_name=args.config_name,
                quantize=args.quantize,
                max_concurrency=args.max_concurrency,
                llm_parameter_size=args.llm_parameter_size,
                llm_common_name=args.llm_common_name,
                gpu_count=args.gpu_count,
                gpu_id=args.gpu_id,
            )
        else:
            print(f"Invalid inference engine: {args.inference_engine}")
            print("Currently supported: mlx")
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
